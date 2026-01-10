"""
Unit tests for technical indicator calculations.

These tests verify the mathematical correctness of RSI, MACD, and EMA
calculations using known values and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_all_indicators,
    determine_ema_cross,
)
from src.models import EMACrossType


class TestRSI:
    """Test suite for RSI calculations."""

    def test_rsi_overbought(self):
        """RSI should be high (>70) after sustained gains."""
        # Create a series with strong, consistent gains
        # Need more data points and larger gains to overcome EMA smoothing
        prices = pd.Series([100 * (1.03 ** i) for i in range(30)])  # 3% daily gains
        rsi = calculate_rsi(prices, period=14)

        # RSI should be above 70 for sustained uptrend
        assert rsi.iloc[-1] > 70

    def test_rsi_oversold(self):
        """RSI should be low (<30) after sustained losses."""
        # Create a series with strong, consistent losses
        prices = pd.Series([100 * (0.97 ** i) for i in range(30)])  # 3% daily losses
        rsi = calculate_rsi(prices, period=14)

        # RSI should be below 30 for sustained downtrend
        assert rsi.iloc[-1] < 30

    def test_rsi_neutral(self):
        """RSI should be around 50 for sideways movement."""
        # Create a series of alternating prices (sideways)
        prices = pd.Series([100, 101, 99, 100, 101, 99, 100, 101, 99, 100,
                          101, 99, 100, 101, 99, 100, 101, 99, 100, 101])
        rsi = calculate_rsi(prices, period=14)

        # RSI should be between 40 and 60 for sideways movement
        assert 40 <= rsi.iloc[-1] <= 60

    def test_rsi_range(self):
        """RSI should always be between 0 and 100."""
        # Test with random prices
        np.random.seed(42)
        prices = pd.Series(np.random.uniform(50, 150, 100))
        rsi = calculate_rsi(prices, period=14)

        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_rsi_short_series(self):
        """RSI should handle series shorter than period."""
        prices = pd.Series([100, 102, 101, 103, 102])
        rsi = calculate_rsi(prices, period=14)

        # Should still return values (filled with 50 for initial periods)
        assert len(rsi) == len(prices)


class TestEMA:
    """Test suite for EMA calculations."""

    def test_ema_tracks_trend(self):
        """EMA should follow the general price trend."""
        # Uptrend
        prices = pd.Series([100 + i for i in range(50)])
        ema = calculate_ema(prices, period=12)

        # EMA should be increasing
        assert ema.iloc[-1] > ema.iloc[-10]

    def test_ema_responds_to_changes(self):
        """EMA should respond faster than SMA to price changes."""
        prices = pd.Series([100] * 20 + [120] * 10)
        ema = calculate_ema(prices, period=10)

        # After the jump, EMA should be above 100 but below 120
        assert 100 < ema.iloc[-1] < 120

    def test_ema_length(self):
        """EMA output should have same length as input."""
        prices = pd.Series(range(100))
        ema = calculate_ema(prices, period=20)

        assert len(ema) == len(prices)


class TestMACD:
    """Test suite for MACD calculations."""

    def test_macd_positive_in_uptrend(self):
        """MACD should be positive in a strong uptrend."""
        prices = pd.Series([100 + i * 1.5 for i in range(50)])
        macd, signal, histogram = calculate_macd(prices)

        # MACD line should be positive in uptrend
        assert macd.iloc[-1] > 0

    def test_macd_negative_in_downtrend(self):
        """MACD should be negative in a strong downtrend."""
        prices = pd.Series([100 - i * 1.5 for i in range(50)])
        macd, signal, histogram = calculate_macd(prices)

        # MACD line should be negative in downtrend
        assert macd.iloc[-1] < 0

    def test_macd_histogram_calculation(self):
        """Histogram should equal MACD minus Signal."""
        np.random.seed(42)
        prices = pd.Series(np.random.uniform(90, 110, 100))
        macd, signal, histogram = calculate_macd(prices)

        # Verify histogram = macd - signal
        expected_histogram = macd - signal
        pd.testing.assert_series_equal(histogram, expected_histogram)

    def test_macd_lengths(self):
        """All MACD outputs should have same length as input."""
        prices = pd.Series(range(100))
        macd, signal, histogram = calculate_macd(prices)

        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)


class TestEMACross:
    """Test suite for EMA crossover detection."""

    def test_bullish_crossover(self):
        """Should detect bullish when short EMA > long EMA."""
        result = determine_ema_cross(ema_short=105.0, ema_long=100.0)
        assert result == EMACrossType.BULLISH

    def test_bearish_crossover(self):
        """Should detect bearish when short EMA < long EMA."""
        result = determine_ema_cross(ema_short=95.0, ema_long=100.0)
        assert result == EMACrossType.BEARISH

    def test_neutral_crossover(self):
        """Should detect neutral when EMAs are very close."""
        result = determine_ema_cross(ema_short=100.05, ema_long=100.0)
        assert result == EMACrossType.NEUTRAL


class TestCalculateAllIndicators:
    """Test suite for the combined indicator calculation."""

    @pytest.fixture
    def sample_ohlcv_data(self) -> pd.DataFrame:
        """Generate sample OHLCV data for testing."""
        np.random.seed(42)
        n = 100

        # Generate somewhat realistic price data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, n)
        close_prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": close_prices * np.random.uniform(0.99, 1.01, n),
            "high": close_prices * np.random.uniform(1.0, 1.02, n),
            "low": close_prices * np.random.uniform(0.98, 1.0, n),
            "close": close_prices,
            "volume": np.random.uniform(1000, 10000, n)
        })

        return df

    def test_calculate_all_returns_result(self, sample_ohlcv_data: pd.DataFrame):
        """Should return an IndicatorResult with all values."""
        result = calculate_all_indicators(sample_ohlcv_data)

        assert result is not None
        assert isinstance(result.rsi, float)
        assert isinstance(result.macd_value, float)
        assert isinstance(result.macd_signal, float)
        assert isinstance(result.macd_histogram, float)
        assert isinstance(result.ema_short, float)
        assert isinstance(result.ema_long, float)
        assert isinstance(result.ema_cross, EMACrossType)

    def test_rsi_in_valid_range(self, sample_ohlcv_data: pd.DataFrame):
        """RSI should be between 0 and 100."""
        result = calculate_all_indicators(sample_ohlcv_data)

        assert 0 <= result.rsi <= 100

    def test_insufficient_data_raises_error(self):
        """Should raise ValueError with insufficient data."""
        df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104]  # Only 5 data points
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_all_indicators(df)

    def test_missing_close_column_raises_error(self):
        """Should raise ValueError without close column."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101]
        })

        with pytest.raises(ValueError, match="close"):
            calculate_all_indicators(df)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_constant_prices(self):
        """Should handle constant prices gracefully."""
        prices = pd.Series([100.0] * 50)
        rsi = calculate_rsi(prices, period=14)

        # RSI should be 50 (neutral) for constant prices
        # Due to no gains or losses, it defaults to neutral
        assert 40 <= rsi.iloc[-1] <= 60

    def test_single_large_move(self):
        """Should handle sudden large price moves."""
        # Use small variations before the big move to avoid constant-price edge case
        prices = pd.Series([100.0 + (i % 2) * 0.1 for i in range(40)] + [200.0])
        rsi = calculate_rsi(prices, period=14)

        # After a large gain, RSI should be elevated (the spike may be dampened by EMA)
        assert rsi.iloc[-1] > 50  # Should be above neutral after a big gain

    def test_very_small_prices(self):
        """Should handle very small price values."""
        prices = pd.Series([0.001 + i * 0.0001 for i in range(50)])
        rsi = calculate_rsi(prices, period=14)
        ema = calculate_ema(prices, period=12)
        macd, signal, hist = calculate_macd(prices)

        # Should produce valid results
        assert not np.isnan(rsi.iloc[-1])
        assert not np.isnan(ema.iloc[-1])
        assert not np.isnan(macd.iloc[-1])

    def test_large_prices(self):
        """Should handle large price values (like BTC)."""
        prices = pd.Series([50000 + i * 100 for i in range(50)])
        rsi = calculate_rsi(prices, period=14)
        ema = calculate_ema(prices, period=12)
        macd, signal, hist = calculate_macd(prices)

        # Should produce valid results
        assert not np.isnan(rsi.iloc[-1])
        assert not np.isnan(ema.iloc[-1])
        assert not np.isnan(macd.iloc[-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
