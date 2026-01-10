"""
Caching layer for API responses.

Implements in-memory caching using cachetools to reduce external API calls
and improve response times. This prevents hitting rate limits and provides
a better user experience.
"""

import hashlib
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from cachetools import TTLCache

from src.config import get_settings

logger = logging.getLogger(__name__)

# Type variable for generic cache decorator
T = TypeVar("T")


class CacheManager:
    """
    Manages multiple TTL caches for different data types.

    Each cache type has its own TTL (time-to-live) to balance
    freshness requirements with API rate limiting.
    """

    def __init__(self) -> None:
        """Initialize cache manager with configured settings."""
        settings = get_settings()

        # Separate caches for different data types with appropriate TTLs
        self._ohlcv_cache: TTLCache[str, Any] = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=settings.cache_ttl_ohlcv
        )
        self._price_cache: TTLCache[str, Any] = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=settings.cache_ttl_price
        )
        self._signal_cache: TTLCache[str, Any] = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=settings.cache_ttl_ohlcv  # Signals use OHLCV TTL
        )

        # Track cache statistics for monitoring
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0
        }

    def _generate_key(self, prefix: str, *args: Any, **kwargs: Any) -> str:
        """
        Generate a unique cache key from prefix and arguments.

        Args:
            prefix: Cache type prefix (e.g., "ohlcv", "price")
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key

        Returns:
            Unique hash-based cache key
        """
        key_parts = [prefix] + [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = ":".join(key_parts)

        # Use hash for consistent key length
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_ohlcv(self, ticker: str, days: int) -> Optional[Any]:
        """
        Get cached OHLCV data for a ticker.

        Args:
            ticker: Cryptocurrency ticker symbol
            days: Number of days of data

        Returns:
            Cached data if available, None otherwise
        """
        key = self._generate_key("ohlcv", ticker.upper(), days)
        result = self._ohlcv_cache.get(key)

        if result is not None:
            self._stats["hits"] += 1
            logger.debug(f"Cache HIT for OHLCV: {ticker}")
        else:
            self._stats["misses"] += 1
            logger.debug(f"Cache MISS for OHLCV: {ticker}")

        return result

    def set_ohlcv(self, ticker: str, days: int, data: Any) -> None:
        """
        Cache OHLCV data for a ticker.

        Args:
            ticker: Cryptocurrency ticker symbol
            days: Number of days of data
            data: OHLCV data to cache
        """
        key = self._generate_key("ohlcv", ticker.upper(), days)
        self._ohlcv_cache[key] = data
        self._stats["sets"] += 1
        logger.debug(f"Cached OHLCV data for: {ticker}")

    def get_price(self, ticker: str) -> Optional[float]:
        """
        Get cached current price for a ticker.

        Args:
            ticker: Cryptocurrency ticker symbol

        Returns:
            Cached price if available, None otherwise
        """
        key = self._generate_key("price", ticker.upper())
        result = self._price_cache.get(key)

        if result is not None:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1

        return result

    def set_price(self, ticker: str, price: float) -> None:
        """
        Cache current price for a ticker.

        Args:
            ticker: Cryptocurrency ticker symbol
            price: Current price to cache
        """
        key = self._generate_key("price", ticker.upper())
        self._price_cache[key] = price
        self._stats["sets"] += 1

    def get_signal(self, ticker: str) -> Optional[Any]:
        """
        Get cached signal data for a ticker.

        Args:
            ticker: Cryptocurrency ticker symbol

        Returns:
            Cached signal data if available, None otherwise
        """
        key = self._generate_key("signal", ticker.upper())
        return self._signal_cache.get(key)

    def set_signal(self, ticker: str, signal_data: Any) -> None:
        """
        Cache signal data for a ticker.

        Args:
            ticker: Cryptocurrency ticker symbol
            signal_data: Signal data to cache
        """
        key = self._generate_key("signal", ticker.upper())
        self._signal_cache[key] = signal_data
        self._stats["sets"] += 1

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dictionary with cache statistics
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "hit_rate": round(hit_rate, 3),
            "ohlcv_cache_size": len(self._ohlcv_cache),
            "price_cache_size": len(self._price_cache),
            "signal_cache_size": len(self._signal_cache)
        }

    def clear_all(self) -> None:
        """Clear all caches. Useful for testing or manual refresh."""
        self._ohlcv_cache.clear()
        self._price_cache.clear()
        self._signal_cache.clear()
        logger.info("All caches cleared")


# Global cache manager instance
cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.

    Returns:
        CacheManager: Global cache manager
    """
    return cache_manager
