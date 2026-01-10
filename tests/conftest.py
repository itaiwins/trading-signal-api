"""
Pytest configuration and fixtures.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings cache before each test."""
    from src.config import get_settings
    get_settings.cache_clear()
    yield
