"""
Pydantic models for request/response validation.

This module defines all data structures used throughout the API,
ensuring type safety and automatic validation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class EMACrossType(str, Enum):
    """EMA crossover signal types."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class MACDData(BaseModel):
    """MACD indicator values."""
    value: float = Field(..., description="MACD line value")
    signal: float = Field(..., description="Signal line value")
    histogram: float = Field(..., description="MACD histogram value")


class IndicatorData(BaseModel):
    """Collection of technical indicator values."""
    rsi: float = Field(..., description="Relative Strength Index (0-100)")
    macd: MACDData = Field(..., description="MACD indicator values")
    ema_cross: EMACrossType = Field(..., description="EMA crossover signal")
    ema_short: float = Field(..., description="Short-term EMA value")
    ema_long: float = Field(..., description="Long-term EMA value")


class TradingSignalResponse(BaseModel):
    """Response model for the /signal endpoint."""
    ticker: str = Field(..., description="Cryptocurrency ticker symbol")
    signal: SignalType = Field(..., description="Trading signal recommendation")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)"
    )
    price: float = Field(..., description="Current price in USD")
    indicators: IndicatorData = Field(..., description="Technical indicator values")
    timestamp: datetime = Field(..., description="Timestamp of the analysis")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ticker": "BTC",
                "signal": "BUY",
                "confidence": 0.72,
                "price": 97234.50,
                "indicators": {
                    "rsi": 42.3,
                    "macd": {"value": 234.5, "signal": 180.2, "histogram": 54.3},
                    "ema_cross": "BULLISH",
                    "ema_short": 96500.0,
                    "ema_long": 95200.0
                },
                "timestamp": "2025-01-10T14:30:00Z"
            }
        }
    }


class IndicatorResponse(BaseModel):
    """Response model for the /indicators endpoint."""
    ticker: str = Field(..., description="Cryptocurrency ticker symbol")
    price: float = Field(..., description="Current price in USD")
    indicators: IndicatorData = Field(..., description="Technical indicator values")
    ohlcv_data_points: int = Field(..., description="Number of OHLCV data points used")
    timestamp: datetime = Field(..., description="Timestamp of the analysis")


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server timestamp")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    ticker: Optional[str] = Field(None, description="Ticker that caused the error")


class OHLCVData(BaseModel):
    """OHLCV (candlestick) data model."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
