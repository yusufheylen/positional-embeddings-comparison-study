"""Pytest configuration and fixtures for PE tests."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.parent))


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with common attributes."""
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    return tokenizer
