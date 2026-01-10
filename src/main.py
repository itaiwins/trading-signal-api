"""
FastAPI application and route definitions.

This is the main entry point for the Trading Signal API.
It defines all HTTP endpoints and handles request/response processing.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.cache import get_cache_manager
from src.config import get_settings, TICKER_TO_COINGECKO_ID
from src.data import DataFetchError, get_data_client
from src.indicators import calculate_all_indicators, indicator_result_to_model
from src.models import (
    ErrorResponse,
    HealthResponse,
    IndicatorResponse,
    TradingSignalResponse,
)
from src.signals import generate_signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events for the application.
    """
    # Startup
    logger.info("Starting Trading Signal API...")
    settings = get_settings()
    logger.info(f"Cache TTL - OHLCV: {settings.cache_ttl_ohlcv}s, Price: {settings.cache_ttl_price}s")

    yield

    # Shutdown
    logger.info("Shutting down Trading Signal API...")


# Initialize FastAPI application
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    description="""
## Trading Signal API

A REST API that serves cryptocurrency trading signals based on technical indicators.

### Features
- Real-time trading signals (BUY/SELL/HOLD) with confidence scores
- Technical indicator calculations (RSI, MACD, EMA crossovers)
- Support for major cryptocurrencies
- Built-in caching to respect API rate limits

### How Signals Work
The API combines multiple technical indicators to generate signals:
- **RSI (40% weight)**: Identifies overbought/oversold conditions
- **MACD (35% weight)**: Measures trend direction and momentum
- **EMA Crossover (25% weight)**: Confirms trend direction

### Supported Cryptocurrencies
BTC, ETH, SOL, ADA, DOT, AVAX, MATIC, LINK, UNI, ATOM, XRP, DOGE, and many more.

### Rate Limiting
The API caches responses to avoid hitting external API rate limits.
- OHLCV data: 5-minute cache
- Price data: 1-minute cache
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for browser compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DataFetchError)
async def data_fetch_error_handler(request, exc: DataFetchError) -> JSONResponse:
    """
    Handle DataFetchError exceptions globally.

    Maps different error types to appropriate HTTP status codes.
    """
    status_code = exc.status_code or 500

    if status_code == 404:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="not_found",
                message=exc.message,
                ticker=exc.ticker
            ).model_dump()
        )
    elif status_code == 429:
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                error="rate_limited",
                message=exc.message,
                ticker=exc.ticker
            ).model_dump()
        )
    else:
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error="fetch_error",
                message=exc.message,
                ticker=exc.ticker
            ).model_dump()
        )


# =============================================================================
# API Endpoints
# =============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint"
)
async def health_check() -> HealthResponse:
    """
    Check the health status of the API.

    Returns basic service information including version and current timestamp.
    Useful for monitoring and load balancer health checks.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(timezone.utc)
    )


@app.get(
    "/signal/{ticker}",
    response_model=TradingSignalResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Ticker not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Trading Signals"],
    summary="Get trading signal for a cryptocurrency"
)
async def get_trading_signal(
    ticker: str = Path(
        ...,
        description="Cryptocurrency ticker symbol (e.g., BTC, ETH, SOL)",
        min_length=2,
        max_length=10,
        examples=["BTC", "ETH", "SOL"]
    )
) -> TradingSignalResponse:
    """
    Generate a trading signal for the specified cryptocurrency.

    This endpoint fetches historical price data, calculates technical indicators
    (RSI, MACD, EMA crossovers), and generates a composite trading signal.

    **Signal Types:**
    - **BUY**: Indicators suggest the asset is undervalued or in an uptrend
    - **SELL**: Indicators suggest the asset is overvalued or in a downtrend
    - **HOLD**: No strong signal in either direction

    **Confidence Score:**
    A value from 0.0 to 1.0 indicating how strongly the indicators agree.
    Higher confidence means the indicators are aligned in the same direction.

    **Note:** This is for educational/informational purposes only.
    Not financial advice. Always do your own research.
    """
    ticker_upper = ticker.upper()
    logger.info(f"Generating signal for {ticker_upper}")

    # Fetch OHLCV data
    client = get_data_client()
    df = await client.fetch_ohlcv(ticker_upper)

    # Get current price (use the most recent close price)
    current_price = float(df["close"].iloc[-1])

    # Calculate indicators
    indicator_result = calculate_all_indicators(df)
    indicators = indicator_result_to_model(indicator_result)

    # Generate trading signal
    signal_result = generate_signal(indicator_result)

    return TradingSignalResponse(
        ticker=ticker_upper,
        signal=signal_result.signal,
        confidence=signal_result.confidence,
        price=round(current_price, 2),
        indicators=indicators,
        timestamp=datetime.now(timezone.utc)
    )


@app.get(
    "/indicators/{ticker}",
    response_model=IndicatorResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Ticker not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Technical Analysis"],
    summary="Get raw technical indicators for a cryptocurrency"
)
async def get_indicators(
    ticker: str = Path(
        ...,
        description="Cryptocurrency ticker symbol",
        min_length=2,
        max_length=10,
        examples=["BTC", "ETH", "SOL"]
    ),
    days: int = Query(
        default=90,
        ge=7,
        le=365,
        description="Number of days of historical data to use"
    )
) -> IndicatorResponse:
    """
    Get raw technical indicator values for the specified cryptocurrency.

    Returns the calculated values for RSI, MACD, and EMA indicators
    without generating a trading signal. Useful for custom analysis
    or building your own signal generation logic.

    **Indicators Returned:**
    - **RSI**: Relative Strength Index (0-100)
    - **MACD**: Moving Average Convergence Divergence (value, signal, histogram)
    - **EMA Cross**: Short/Long EMA crossover state (BULLISH/BEARISH/NEUTRAL)
    """
    ticker_upper = ticker.upper()
    logger.info(f"Fetching indicators for {ticker_upper} ({days} days)")

    # Fetch OHLCV data
    client = get_data_client()
    df = await client.fetch_ohlcv(ticker_upper, days=days)

    # Get current price
    current_price = float(df["close"].iloc[-1])

    # Calculate indicators
    indicator_result = calculate_all_indicators(df)
    indicators = indicator_result_to_model(indicator_result)

    return IndicatorResponse(
        ticker=ticker_upper,
        price=round(current_price, 2),
        indicators=indicators,
        ohlcv_data_points=len(df),
        timestamp=datetime.now(timezone.utc)
    )


@app.get(
    "/tickers",
    tags=["Reference"],
    summary="List supported cryptocurrency tickers"
)
async def list_supported_tickers() -> dict:
    """
    Get a list of all supported cryptocurrency ticker symbols.

    These are pre-mapped tickers that are guaranteed to work with the API.
    Other tickers may work but are not guaranteed.
    """
    return {
        "supported_tickers": sorted(TICKER_TO_COINGECKO_ID.keys()),
        "count": len(TICKER_TO_COINGECKO_ID)
    }


@app.get(
    "/cache/stats",
    tags=["System"],
    summary="Get cache statistics"
)
async def get_cache_stats() -> dict:
    """
    Get statistics about the cache system.

    Returns hit/miss rates and cache sizes for monitoring purposes.
    """
    cache = get_cache_manager()
    return cache.get_stats()


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError) -> JSONResponse:
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="validation_error",
            message=str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred. Please try again later."
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
