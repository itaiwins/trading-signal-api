"""
Configuration management using environment variables.

This module centralizes all configuration settings, making it easy to
adjust behavior without code changes.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings have sensible defaults for development but can be
    overridden via environment variables or .env file.
    """

    # API Settings
    app_name: str = "Trading Signal API"
    app_version: str = "1.0.0"
    debug: bool = False

    # External API Settings
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    coingecko_api_key: Optional[str] = None  # Optional: for higher rate limits

    # Cache Settings (in seconds)
    cache_ttl_ohlcv: int = 300  # 5 minutes for OHLCV data
    cache_ttl_price: int = 60   # 1 minute for current price
    cache_max_size: int = 100   # Maximum number of cached items

    # Technical Indicator Settings
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    ema_short_period: int = 12
    ema_long_period: int = 26

    # Signal Generation Settings
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Data Fetching Settings
    ohlcv_days: int = 90  # Days of historical data to fetch

    # Rate Limiting
    request_timeout: int = 30  # seconds

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Using lru_cache ensures settings are only loaded once,
    improving performance and consistency.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Mapping of common ticker symbols to CoinGecko IDs
# CoinGecko uses full names/IDs rather than ticker symbols
TICKER_TO_COINGECKO_ID: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "SHIB": "shiba-inu",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "NEAR": "near",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "FTM": "fantom",
    "ALGO": "algorand",
    "VET": "vechain",
    "HBAR": "hedera-hashgraph",
    "FIL": "filecoin",
    "EGLD": "elrond-erd-2",
    "SAND": "the-sandbox",
    "MANA": "decentraland",
    "AXS": "axie-infinity",
    "AAVE": "aave",
    "MKR": "maker",
    "CRV": "curve-dao-token",
    "LDO": "lido-dao",
    "SNX": "havven",
    "COMP": "compound-governance-token",
    "SUSHI": "sushi",
    "YFI": "yearn-finance",
    "PEPE": "pepe",
    "WIF": "dogwifcoin",
    "BONK": "bonk",
}


def get_coingecko_id(ticker: str) -> str:
    """
    Convert a ticker symbol to CoinGecko ID.

    Args:
        ticker: Cryptocurrency ticker symbol (e.g., "BTC")

    Returns:
        CoinGecko ID (e.g., "bitcoin")

    Note:
        If ticker is not in the mapping, returns lowercase ticker
        which may or may not work with CoinGecko API.
    """
    return TICKER_TO_COINGECKO_ID.get(ticker.upper(), ticker.lower())
