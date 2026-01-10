"""
External API data fetching module.

Handles all communication with the CoinGecko API to fetch OHLCV
(candlestick) data and current prices for cryptocurrencies.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd

from src.cache import get_cache_manager
from src.config import get_coingecko_id, get_settings
from src.models import OHLCVData

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Custom exception for data fetching errors."""

    def __init__(self, message: str, ticker: str, status_code: Optional[int] = None):
        self.message = message
        self.ticker = ticker
        self.status_code = status_code
        super().__init__(self.message)


class CoinGeckoClient:
    """
    Async client for CoinGecko API.

    Handles fetching OHLCV data and current prices with proper
    error handling, rate limiting awareness, and caching.
    """

    def __init__(self) -> None:
        """Initialize the CoinGecko client."""
        self.settings = get_settings()
        self.base_url = self.settings.coingecko_base_url
        self.cache = get_cache_manager()

    def _get_headers(self) -> dict[str, str]:
        """
        Get request headers, including API key if configured.

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": "TradingSignalAPI/1.0"
        }

        # Add API key if configured (for higher rate limits)
        if self.settings.coingecko_api_key:
            headers["x-cg-demo-api-key"] = self.settings.coingecko_api_key

        return headers

    async def fetch_ohlcv(
        self,
        ticker: str,
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data for a cryptocurrency.

        Uses CoinGecko's market_chart endpoint which provides more reliable
        data points than the OHLC endpoint for the free tier.

        Args:
            ticker: Cryptocurrency ticker symbol (e.g., "BTC")
            days: Number of days of historical data (default from config)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            DataFetchError: If API request fails or data is invalid
        """
        days = days or self.settings.ohlcv_days
        ticker_upper = ticker.upper()

        # Check cache first
        cached_data = self.cache.get_ohlcv(ticker_upper, days)
        if cached_data is not None:
            logger.info(f"Using cached OHLCV data for {ticker_upper}")
            return cached_data

        # Convert ticker to CoinGecko ID
        coin_id = get_coingecko_id(ticker_upper)
        logger.info(f"Fetching OHLCV data for {ticker_upper} (CoinGecko ID: {coin_id})")

        # Use market_chart endpoint for more reliable data
        # This returns price, market_cap, and volume data points
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": str(days),
            "interval": "daily"  # Get daily data points
        }

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._get_headers()
                )

                if response.status_code == 404:
                    raise DataFetchError(
                        f"Cryptocurrency '{ticker_upper}' not found",
                        ticker_upper,
                        404
                    )

                if response.status_code == 429:
                    raise DataFetchError(
                        "Rate limit exceeded. Please try again later.",
                        ticker_upper,
                        429
                    )

                if response.status_code != 200:
                    raise DataFetchError(
                        f"API request failed with status {response.status_code}",
                        ticker_upper,
                        response.status_code
                    )

                data = response.json()

        except httpx.TimeoutException:
            raise DataFetchError(
                "Request timed out while fetching data",
                ticker_upper
            )
        except httpx.RequestError as e:
            raise DataFetchError(
                f"Network error: {str(e)}",
                ticker_upper
            )

        # Validate response data
        if not data or "prices" not in data:
            raise DataFetchError(
                "Invalid response format from API",
                ticker_upper
            )

        prices = data["prices"]
        volumes = data.get("total_volumes", [])

        if len(prices) < 26:  # Need minimum data for MACD calculation
            raise DataFetchError(
                f"Insufficient data points ({len(prices)}). Need at least 26.",
                ticker_upper
            )

        # Convert to DataFrame
        # market_chart format: [[timestamp_ms, price], ...]
        df = pd.DataFrame(prices, columns=["timestamp", "close"])

        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # For OHLC-like data, we'll use close price for all since this endpoint
        # only provides price data. This is acceptable for our indicators
        # (RSI, MACD, EMA all work on close prices)
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]

        # Add volume data if available
        if volumes and len(volumes) == len(prices):
            df["volume"] = [v[1] for v in volumes]
        else:
            df["volume"] = 0.0

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop any rows with NaN values
        df = df.dropna()

        # Sort by timestamp (oldest first)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Cache the result
        self.cache.set_ohlcv(ticker_upper, days, df)

        logger.info(f"Fetched {len(df)} data points for {ticker_upper}")
        return df

    async def fetch_current_price(self, ticker: str) -> float:
        """
        Fetch the current price for a cryptocurrency.

        Args:
            ticker: Cryptocurrency ticker symbol (e.g., "BTC")

        Returns:
            Current price in USD

        Raises:
            DataFetchError: If API request fails
        """
        ticker_upper = ticker.upper()

        # Check cache first
        cached_price = self.cache.get_price(ticker_upper)
        if cached_price is not None:
            logger.debug(f"Using cached price for {ticker_upper}")
            return cached_price

        # Convert ticker to CoinGecko ID
        coin_id = get_coingecko_id(ticker_upper)

        url = f"{self.base_url}/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd"
        }

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._get_headers()
                )

                if response.status_code != 200:
                    raise DataFetchError(
                        f"Failed to fetch price (status {response.status_code})",
                        ticker_upper,
                        response.status_code
                    )

                data = response.json()

        except httpx.TimeoutException:
            raise DataFetchError(
                "Request timed out while fetching price",
                ticker_upper
            )
        except httpx.RequestError as e:
            raise DataFetchError(
                f"Network error: {str(e)}",
                ticker_upper
            )

        # Extract price from response
        if coin_id not in data or "usd" not in data[coin_id]:
            raise DataFetchError(
                f"Price data not found for {ticker_upper}",
                ticker_upper
            )

        price = float(data[coin_id]["usd"])

        # Cache the price
        self.cache.set_price(ticker_upper, price)

        return price


# Global client instance
_client: Optional[CoinGeckoClient] = None


def get_data_client() -> CoinGeckoClient:
    """
    Get or create the global CoinGecko client instance.

    Returns:
        CoinGeckoClient: Global client instance
    """
    global _client
    if _client is None:
        _client = CoinGeckoClient()
    return _client
