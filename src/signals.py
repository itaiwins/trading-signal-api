"""
Signal generation logic.

This module combines multiple technical indicators to generate
trading signals with confidence scores. The approach uses a
weighted scoring system that considers RSI, MACD, and EMA crossovers.

Signal Generation Philosophy:
-----------------------------
Rather than relying on a single indicator, we combine multiple indicators
to reduce false signals and increase confidence. Each indicator contributes
to an overall score:

1. RSI (40% weight): Good for identifying overbought/oversold conditions
2. MACD (35% weight): Good for trend direction and momentum
3. EMA Cross (25% weight): Good for confirming trend direction

The final signal is determined by the composite score:
- Score > 0.2: BUY signal
- Score < -0.2: SELL signal
- Otherwise: HOLD signal

The confidence score (0-1) reflects how strongly the indicators agree.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

from src.config import get_settings
from src.indicators import IndicatorResult
from src.models import EMACrossType, SignalType

logger = logging.getLogger(__name__)


# Signal weights for composite calculation
# These can be tuned based on backtesting results
WEIGHT_RSI = 0.40
WEIGHT_MACD = 0.35
WEIGHT_EMA = 0.25


@dataclass
class SignalResult:
    """Container for signal generation results."""
    signal: SignalType
    confidence: float
    component_scores: dict[str, float]


def calculate_rsi_score(rsi: float) -> float:
    """
    Calculate a normalized score from RSI value.

    RSI Score Interpretation:
    - RSI < 30 (oversold): Strong buy signal (+1.0)
    - RSI 30-40: Moderate buy signal (+0.5 to 0)
    - RSI 40-60: Neutral (0)
    - RSI 60-70: Moderate sell signal (0 to -0.5)
    - RSI > 70 (overbought): Strong sell signal (-1.0)

    The scoring uses linear interpolation within each zone
    to provide gradual transitions.

    Args:
        rsi: RSI value (0-100)

    Returns:
        Score from -1.0 (strong sell) to +1.0 (strong buy)
    """
    settings = get_settings()
    oversold = settings.rsi_oversold   # Default: 30
    overbought = settings.rsi_overbought  # Default: 70
    neutral_low = 40
    neutral_high = 60

    if rsi <= oversold:
        # Strong oversold - strong buy signal
        # Scale from 1.0 at RSI=0 to 0.7 at RSI=30
        return 1.0 - (rsi / oversold) * 0.3

    elif rsi <= neutral_low:
        # Approaching neutral from oversold - moderate buy
        # Linear interpolation from 0.7 at RSI=30 to 0 at RSI=40
        return 0.7 * (neutral_low - rsi) / (neutral_low - oversold)

    elif rsi <= neutral_high:
        # Neutral zone - no strong signal
        return 0.0

    elif rsi <= overbought:
        # Approaching overbought - moderate sell
        # Linear interpolation from 0 at RSI=60 to -0.7 at RSI=70
        return -0.7 * (rsi - neutral_high) / (overbought - neutral_high)

    else:
        # Strong overbought - strong sell signal
        # Scale from -0.7 at RSI=70 to -1.0 at RSI=100
        return -0.7 - (rsi - overbought) / (100 - overbought) * 0.3


def calculate_macd_score(macd_value: float, macd_signal: float, histogram: float) -> float:
    """
    Calculate a normalized score from MACD values.

    MACD Score Interpretation:
    - MACD above signal line + positive histogram: Buy signal
    - MACD below signal line + negative histogram: Sell signal
    - Magnitude of histogram indicates strength

    We normalize the histogram relative to the MACD value to get
    a percentage-based score that works across different price scales.

    Args:
        macd_value: Current MACD line value
        macd_signal: Current signal line value
        histogram: MACD histogram (macd - signal)

    Returns:
        Score from -1.0 (strong sell) to +1.0 (strong buy)
    """
    # Determine direction based on crossover
    if abs(macd_value) < 0.0001:  # Avoid division by zero
        # If MACD is near zero, use histogram sign only
        if histogram > 0:
            return 0.3
        elif histogram < 0:
            return -0.3
        return 0.0

    # Normalize histogram relative to MACD magnitude
    # This gives us a percentage-based strength indicator
    histogram_pct = histogram / abs(macd_value)

    # Clamp to reasonable range and scale
    # histogram_pct of 0.5 (50%) would be a very strong signal
    score = max(-1.0, min(1.0, histogram_pct * 2))

    return score


def calculate_ema_score(ema_cross: EMACrossType, ema_short: float, ema_long: float) -> float:
    """
    Calculate a normalized score from EMA crossover state.

    EMA Score Interpretation:
    - Bullish crossover (short > long): Buy signal
    - Bearish crossover (short < long): Sell signal
    - Strength based on percentage difference between EMAs

    A larger gap between EMAs indicates a stronger trend.

    Args:
        ema_cross: Current EMA crossover state
        ema_short: Short-term EMA value
        ema_long: Long-term EMA value

    Returns:
        Score from -1.0 (strong sell) to +1.0 (strong buy)
    """
    if ema_long == 0:  # Avoid division by zero
        return 0.0

    # Calculate percentage difference
    pct_diff = ((ema_short - ema_long) / ema_long) * 100

    # Scale the score based on percentage difference
    # A 2% difference would be a fairly strong signal
    if ema_cross == EMACrossType.BULLISH:
        # Bullish: positive score, strength based on gap
        return min(1.0, pct_diff / 2.0)

    elif ema_cross == EMACrossType.BEARISH:
        # Bearish: negative score, strength based on gap
        return max(-1.0, pct_diff / 2.0)

    else:
        # Neutral
        return 0.0


def generate_signal(indicators: IndicatorResult) -> SignalResult:
    """
    Generate a trading signal from technical indicators.

    This function combines RSI, MACD, and EMA crossover indicators
    using a weighted scoring system to produce a single trading
    recommendation with confidence level.

    The algorithm:
    1. Calculate individual scores for each indicator (-1 to +1)
    2. Apply weights to get composite score
    3. Determine signal based on score thresholds
    4. Calculate confidence based on agreement between indicators

    Args:
        indicators: Calculated technical indicator values

    Returns:
        SignalResult with signal type, confidence, and component scores
    """
    # Calculate individual component scores
    rsi_score = calculate_rsi_score(indicators.rsi)
    macd_score = calculate_macd_score(
        indicators.macd_value,
        indicators.macd_signal,
        indicators.macd_histogram
    )
    ema_score = calculate_ema_score(
        indicators.ema_cross,
        indicators.ema_short,
        indicators.ema_long
    )

    # Calculate weighted composite score
    composite_score = (
        rsi_score * WEIGHT_RSI +
        macd_score * WEIGHT_MACD +
        ema_score * WEIGHT_EMA
    )

    # Determine signal based on composite score
    # Using thresholds to avoid weak signals
    BUY_THRESHOLD = 0.2
    SELL_THRESHOLD = -0.2

    if composite_score > BUY_THRESHOLD:
        signal = SignalType.BUY
    elif composite_score < SELL_THRESHOLD:
        signal = SignalType.SELL
    else:
        signal = SignalType.HOLD

    # Calculate confidence score
    # Confidence is based on:
    # 1. Magnitude of composite score (stronger = more confident)
    # 2. Agreement between indicators (all same direction = more confident)
    confidence = calculate_confidence(
        composite_score,
        rsi_score,
        macd_score,
        ema_score
    )

    # Store component scores for debugging/transparency
    component_scores = {
        "rsi": round(rsi_score, 3),
        "macd": round(macd_score, 3),
        "ema": round(ema_score, 3),
        "composite": round(composite_score, 3)
    }

    logger.info(
        f"Signal generated: {signal.value} with confidence {confidence:.2f} "
        f"(RSI: {rsi_score:.2f}, MACD: {macd_score:.2f}, EMA: {ema_score:.2f})"
    )

    return SignalResult(
        signal=signal,
        confidence=round(confidence, 2),
        component_scores=component_scores
    )


def calculate_confidence(
    composite: float,
    rsi_score: float,
    macd_score: float,
    ema_score: float
) -> float:
    """
    Calculate confidence score based on signal strength and indicator agreement.

    Confidence factors:
    1. Composite magnitude: Stronger composite = higher base confidence
    2. Indicator agreement: Indicators pointing same direction = bonus

    Args:
        composite: Weighted composite score
        rsi_score: RSI component score
        macd_score: MACD component score
        ema_score: EMA component score

    Returns:
        Confidence score from 0.0 to 1.0
    """
    # Base confidence from composite magnitude (0 to 0.6)
    # A composite of 0.5 would give base confidence of 0.3
    base_confidence = min(0.6, abs(composite) * 0.6)

    # Agreement bonus (0 to 0.4)
    # Check if all indicators agree on direction
    scores = [rsi_score, macd_score, ema_score]
    positive_count = sum(1 for s in scores if s > 0)
    negative_count = sum(1 for s in scores if s < 0)

    # Perfect agreement: all 3 same direction
    if positive_count == 3 or negative_count == 3:
        agreement_bonus = 0.4
    # Strong agreement: 2 same direction, 1 neutral
    elif positive_count == 2 or negative_count == 2:
        agreement_bonus = 0.25
    # Mixed signals
    else:
        agreement_bonus = 0.1

    confidence = base_confidence + agreement_bonus

    # Ensure confidence is in valid range
    return max(0.0, min(1.0, confidence))
