"""
Basic pytest configuration and shared fixtures for ultra-fast-backtester tests.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_market_data():
    """Provide sample market data for testing."""
    return pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [102, 103, 104, 105, 106],
        'Low': [99, 100, 101, 102, 103],
        'Close': [101, 102, 103, 104, 105],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range('2023-01-01', periods=5, freq='D'))


@pytest.fixture
def sample_strategy():
    """Provide a mock strategy for testing."""
    class MockStrategy:
        def __init__(self, symbol="TEST"):
            self.symbol = symbol
        
        def generate_signals(self, data, positions, cash):
            return []
    
    return MockStrategy()


@pytest.fixture
def initial_capital():
    """Provide initial capital for testing."""
    return 100000 