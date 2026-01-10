"""
Technical indicator calculations.

This module implements core technical analysis indicators used for
generating trading signals. All calculations are performed on pandas
DataFrames for efficiency and clarity.

Indicators implemented:
- RSI (Relative Strength Index): Momentum oscillator measuring speed of price changes
- MACD (Moving Average Convergence Divergence): Trend-following momentum indicator
- EMA (Exponential Moving Average): Weighted moving average with recent price emphasis
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import get_settings
from src.models import EMACrossType, IndicatorData, MACDData

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Container for calculated indicator values."""
    rsi: float
    macd_value: float
    macd_signal: float
    macd_histogram: float
    ema_short: float
    ema_long: float
    ema_cross: EMACrossType


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions. Values range from 0 to 100:
    - RSI > 70: Potentially overbought (bearish signal)
    - RSI < 30: Potentially oversold (bullish signal)

    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss

    Args:
        prices: Series of closing prices
        period: Lookback period (default 14)

    Returns:
        Series of RSI values
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate exponential moving averages of gains and losses
    # Using Wilder's smoothing method (equivalent to EMA with alpha = 1/period)
    avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    # Handle edge cases:
    # - If avg_losses is 0 (all gains): RSI = 100
    # - If avg_gains is 0 (all losses): RSI = 0
    # - If both are 0: RSI = 50 (neutral)
    rsi = pd.Series(index=prices.index, dtype=float)

    for i in range(len(prices)):
        avg_gain = avg_gains.iloc[i]
        avg_loss = avg_losses.iloc[i]

        if pd.isna(avg_gain) or pd.isna(avg_loss):
            rsi.iloc[i] = 50.0  # Not enough data yet
        elif avg_loss == 0:
            rsi.iloc[i] = 100.0 if avg_gain > 0 else 50.0
        elif avg_gain == 0:
            rsi.iloc[i] = 0.0
        else:
            rs = avg_gain / avg_loss
            rsi.iloc[i] = 100 - (100 / (1 + rs))

    return rsi


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA).

    EMA gives more weight to recent prices, making it more responsive
    to new information than a simple moving average.

    Formula:
        EMA = Price(today) * k + EMA(yesterday) * (1-k)
        where k = 2 / (period + 1)

    Args:
        prices: Series of closing prices
        period: Number of periods for EMA calculation

    Returns:
        Series of EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD).

    MACD shows the relationship between two EMAs of prices:
    - MACD Line: Fast EMA - Slow EMA
    - Signal Line: EMA of MACD Line
    - Histogram: MACD Line - Signal Line

    Trading signals:
    - MACD crosses above Signal: Bullish
    - MACD crosses below Signal: Bearish
    - Positive histogram: Bullish momentum
    - Negative histogram: Bearish momentum

    Args:
        prices: Series of closing prices
        fast_period: Period for fast EMA (default 12)
        slow_period: Period for slow EMA (default 26)
        signal_period: Period for signal line EMA (default 9)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    # Calculate fast and slow EMAs
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    # MACD line is the difference between fast and slow EMAs
    macd_line = ema_fast - ema_slow

    # Signal line is EMA of MACD line
    signal_line = calculate_ema(macd_line, signal_period)

    # Histogram is the difference between MACD and Signal
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def determine_ema_cross(ema_short: float, ema_long: float) -> EMACrossType:
    """
    Determine the EMA crossover state.

    EMA crossovers are a simple but effective trend indicator:
    - Short EMA > Long EMA: Bullish (uptrend)
    - Short EMA < Long EMA: Bearish (downtrend)
    - Very close values: Neutral (no clear trend)

    Args:
        ema_short: Short-term EMA value
        ema_long: Long-term EMA value

    Returns:
        EMACrossType indicating the crossover state
    """
    # Calculate percentage difference
    pct_diff = ((ema_short - ema_long) / ema_long) * 100

    # Use a small threshold to avoid noise (0.1% difference)
    threshold = 0.1

    if pct_diff > threshold:
        return EMACrossType.BULLISH
    elif pct_diff < -threshold:
        return EMACrossType.BEARISH
    else:
        return EMACrossType.NEUTRAL


def calculate_all_indicators(df: pd.DataFrame) -> IndicatorResult:
    """
    Calculate all technical indicators from OHLCV data.

    This is the main function that computes RSI, MACD, and EMA crossovers
    from the provided price data. It uses the most recent values for
    generating trading signals.

    Args:
        df: DataFrame with columns including 'close' prices

    Returns:
        IndicatorResult with all calculated indicator values

    Raises:
        ValueError: If DataFrame doesn't have required columns or data
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    if len(df) < 26:  # Need at least 26 periods for MACD slow EMA
        raise ValueError(f"Insufficient data: need at least 26 rows, got {len(df)}")

    settings = get_settings()
    close_prices = df["close"]

    # Calculate RSI
    rsi_series = calculate_rsi(close_prices, settings.rsi_period)
    current_rsi = float(rsi_series.iloc[-1])

    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(
        close_prices,
        settings.macd_fast_period,
        settings.macd_slow_period,
        settings.macd_signal_period
    )
    current_macd = float(macd_line.iloc[-1])
    current_signal = float(signal_line.iloc[-1])
    current_histogram = float(histogram.iloc[-1])

    # Calculate EMAs
    ema_short_series = calculate_ema(close_prices, settings.ema_short_period)
    ema_long_series = calculate_ema(close_prices, settings.ema_long_period)
    current_ema_short = float(ema_short_series.iloc[-1])
    current_ema_long = float(ema_long_series.iloc[-1])

    # Determine EMA crossover state
    ema_cross = determine_ema_cross(current_ema_short, current_ema_long)

    logger.debug(
        f"Indicators calculated: RSI={current_rsi:.2f}, "
        f"MACD={current_macd:.2f}, EMA Cross={ema_cross.value}"
    )

    return IndicatorResult(
        rsi=round(current_rsi, 2),
        macd_value=round(current_macd, 4),
        macd_signal=round(current_signal, 4),
        macd_histogram=round(current_histogram, 4),
        ema_short=round(current_ema_short, 2),
        ema_long=round(current_ema_long, 2),
        ema_cross=ema_cross
    )


def indicator_result_to_model(result: IndicatorResult) -> IndicatorData:
    """
    Convert IndicatorResult to Pydantic model for API response.

    Args:
        result: Calculated indicator values

    Returns:
        IndicatorData Pydantic model
    """
    return IndicatorData(
        rsi=result.rsi,
        macd=MACDData(
            value=result.macd_value,
            signal=result.macd_signal,
            histogram=result.macd_histogram
        ),
        ema_cross=result.ema_cross,
        ema_short=result.ema_short,
        ema_long=result.ema_long
    )
