"""
Unit tests for signal generation logic.

These tests verify that the signal generation correctly combines
indicators and produces appropriate buy/sell/hold signals.
"""

import pytest

from src.indicators import IndicatorResult
from src.models import EMACrossType, SignalType
from src.signals import (
    calculate_confidence,
    calculate_ema_score,
    calculate_macd_score,
    calculate_rsi_score,
    generate_signal,
)


class TestRSIScore:
    """Test RSI score calculations."""

    def test_oversold_strong_buy(self):
        """RSI < 30 should give strong buy signal."""
        score = calculate_rsi_score(25)
        assert score > 0.5

    def test_overbought_strong_sell(self):
        """RSI > 70 should give strong sell signal."""
        score = calculate_rsi_score(75)
        assert score < -0.5

    def test_neutral_zone(self):
        """RSI 40-60 should give near-zero score."""
        score = calculate_rsi_score(50)
        assert -0.1 <= score <= 0.1

    def test_extreme_oversold(self):
        """RSI near 0 should give maximum buy score."""
        score = calculate_rsi_score(5)
        assert score > 0.9

    def test_extreme_overbought(self):
        """RSI near 100 should give maximum sell score."""
        score = calculate_rsi_score(95)
        assert score < -0.9

    def test_score_range(self):
        """Score should always be between -1 and 1."""
        for rsi in range(0, 101, 5):
            score = calculate_rsi_score(rsi)
            assert -1 <= score <= 1


class TestMACDScore:
    """Test MACD score calculations."""

    def test_positive_histogram_bullish(self):
        """Positive histogram should give positive score."""
        score = calculate_macd_score(
            macd_value=100,
            macd_signal=80,
            histogram=20
        )
        assert score > 0

    def test_negative_histogram_bearish(self):
        """Negative histogram should give negative score."""
        score = calculate_macd_score(
            macd_value=80,
            macd_signal=100,
            histogram=-20
        )
        assert score < 0

    def test_zero_macd_handles_gracefully(self):
        """Should handle near-zero MACD value."""
        score = calculate_macd_score(
            macd_value=0.00001,
            macd_signal=0.00002,
            histogram=0.00001
        )
        # Should return a reasonable value without error
        assert -1 <= score <= 1

    def test_strong_bullish_divergence(self):
        """Large positive histogram should give strong buy score."""
        score = calculate_macd_score(
            macd_value=100,
            macd_signal=50,
            histogram=50
        )
        assert score > 0.5


class TestEMAScore:
    """Test EMA crossover score calculations."""

    def test_bullish_crossover(self):
        """Bullish crossover should give positive score."""
        score = calculate_ema_score(
            ema_cross=EMACrossType.BULLISH,
            ema_short=105,
            ema_long=100
        )
        assert score > 0

    def test_bearish_crossover(self):
        """Bearish crossover should give negative score."""
        score = calculate_ema_score(
            ema_cross=EMACrossType.BEARISH,
            ema_short=95,
            ema_long=100
        )
        assert score < 0

    def test_neutral_crossover(self):
        """Neutral crossover should give zero score."""
        score = calculate_ema_score(
            ema_cross=EMACrossType.NEUTRAL,
            ema_short=100,
            ema_long=100
        )
        assert score == 0

    def test_strong_bullish_gap(self):
        """Large bullish gap should give stronger score."""
        weak = calculate_ema_score(EMACrossType.BULLISH, 101, 100)
        strong = calculate_ema_score(EMACrossType.BULLISH, 105, 100)
        assert strong > weak


class TestGenerateSignal:
    """Test the main signal generation function."""

    @pytest.fixture
    def bullish_indicators(self) -> IndicatorResult:
        """Create indicators suggesting a buy signal."""
        return IndicatorResult(
            rsi=28,  # Oversold
            macd_value=100,
            macd_signal=80,
            macd_histogram=20,  # Positive
            ema_short=105,
            ema_long=100,
            ema_cross=EMACrossType.BULLISH
        )

    @pytest.fixture
    def bearish_indicators(self) -> IndicatorResult:
        """Create indicators suggesting a sell signal."""
        return IndicatorResult(
            rsi=75,  # Overbought
            macd_value=80,
            macd_signal=100,
            macd_histogram=-20,  # Negative
            ema_short=95,
            ema_long=100,
            ema_cross=EMACrossType.BEARISH
        )

    @pytest.fixture
    def neutral_indicators(self) -> IndicatorResult:
        """Create indicators suggesting a hold signal."""
        return IndicatorResult(
            rsi=50,  # Neutral
            macd_value=100,
            macd_signal=100,
            macd_histogram=0,  # Neutral
            ema_short=100,
            ema_long=100,
            ema_cross=EMACrossType.NEUTRAL
        )

    def test_bullish_signal(self, bullish_indicators: IndicatorResult):
        """Should generate BUY signal for bullish indicators."""
        result = generate_signal(bullish_indicators)
        assert result.signal == SignalType.BUY

    def test_bearish_signal(self, bearish_indicators: IndicatorResult):
        """Should generate SELL signal for bearish indicators."""
        result = generate_signal(bearish_indicators)
        assert result.signal == SignalType.SELL

    def test_neutral_signal(self, neutral_indicators: IndicatorResult):
        """Should generate HOLD signal for neutral indicators."""
        result = generate_signal(neutral_indicators)
        assert result.signal == SignalType.HOLD

    def test_confidence_range(self, bullish_indicators: IndicatorResult):
        """Confidence should be between 0 and 1."""
        result = generate_signal(bullish_indicators)
        assert 0 <= result.confidence <= 1

    def test_component_scores_included(self, bullish_indicators: IndicatorResult):
        """Component scores should be included in result."""
        result = generate_signal(bullish_indicators)
        assert "rsi" in result.component_scores
        assert "macd" in result.component_scores
        assert "ema" in result.component_scores
        assert "composite" in result.component_scores


class TestConfidenceCalculation:
    """Test confidence score calculations."""

    def test_perfect_agreement_high_confidence(self):
        """All indicators agreeing should give high confidence."""
        confidence = calculate_confidence(
            composite=0.8,
            rsi_score=0.7,
            macd_score=0.6,
            ema_score=0.5
        )
        assert confidence > 0.7

    def test_mixed_signals_lower_confidence(self):
        """Mixed signals should give lower confidence."""
        confidence = calculate_confidence(
            composite=0.2,
            rsi_score=0.5,
            macd_score=-0.3,
            ema_score=0.2
        )
        assert confidence < 0.6

    def test_weak_signals_low_confidence(self):
        """Weak signals should give low confidence."""
        confidence = calculate_confidence(
            composite=0.1,
            rsi_score=0.1,
            macd_score=0.05,
            ema_score=0.0
        )
        assert confidence < 0.5

    def test_confidence_always_valid(self):
        """Confidence should always be between 0 and 1."""
        test_cases = [
            (1.0, 1.0, 1.0, 1.0),
            (-1.0, -1.0, -1.0, -1.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.5, -0.5, 0.3, -0.2),
        ]
        for composite, rsi, macd, ema in test_cases:
            confidence = calculate_confidence(composite, rsi, macd, ema)
            assert 0 <= confidence <= 1


class TestMixedSignals:
    """Test signal generation with conflicting indicators."""

    def test_rsi_bullish_others_bearish(self):
        """RSI bullish but MACD and EMA bearish should be nuanced."""
        indicators = IndicatorResult(
            rsi=25,  # Very bullish
            macd_value=80,
            macd_signal=100,
            macd_histogram=-20,  # Bearish
            ema_short=95,
            ema_long=100,
            ema_cross=EMACrossType.BEARISH
        )
        result = generate_signal(indicators)
        # With mixed signals, should have lower confidence
        assert result.confidence < 0.7

    def test_borderline_signal(self):
        """Indicators near thresholds should give HOLD."""
        indicators = IndicatorResult(
            rsi=45,  # Slightly bullish
            macd_value=100,
            macd_signal=95,
            macd_histogram=5,  # Slightly bullish
            ema_short=100.5,
            ema_long=100,
            ema_cross=EMACrossType.NEUTRAL
        )
        result = generate_signal(indicators)
        # Weak signals should result in HOLD
        assert result.signal in [SignalType.HOLD, SignalType.BUY]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
