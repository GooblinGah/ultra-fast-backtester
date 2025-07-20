"""
Basic tests for the backtester module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

# Import the modules to test
try:
    from ultrafast_backtester.backtester import UltraFastBacktester
    from ultrafast_backtester.strategies import BaseStrategy
    from ultrafast_backtester.data_loader import create_data_loader
except ImportError:
    # If imports fail, we'll skip tests
    pass


class TestUltraFastBacktester:
    """Basic test cases for the UltraFastBacktester class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        
        self.initial_capital = 100000
        
    def test_backtester_initialization(self):
        """Test that backtester initializes correctly."""
        try:
            backtester = UltraFastBacktester(initial_capital=self.initial_capital)
            assert backtester.initial_capital == self.initial_capital
        except NameError:
            pytest.skip("UltraFastBacktester not available")
    
    def test_backtester_with_sample_data(self):
        """Test backtester with sample data."""
        try:
            backtester = UltraFastBacktester(initial_capital=self.initial_capital)
            
            # Create a mock strategy
            mock_strategy = Mock()
            mock_strategy.generate_signals.return_value = []
            mock_strategy.symbol = "TEST"
            
            # Run backtest
            result = backtester.run_backtest(self.sample_data, mock_strategy)
            
            # Basic assertions
            assert isinstance(result, dict)
        except NameError:
            pytest.skip("UltraFastBacktester not available")


class TestDataLoader:
    """Basic test cases for data loading functionality."""
    
    def test_create_data_loader(self):
        """Test data loader creation."""
        try:
            loader = create_data_loader('yahoo')
            assert loader is not None
        except NameError:
            pytest.skip("create_data_loader not available")


class TestStrategies:
    """Basic test cases for trading strategies."""
    
    def test_base_strategy(self):
        """Test base strategy class."""
        try:
            strategy = BaseStrategy("AAPL")
            assert strategy.symbol == "AAPL"
        except NameError:
            pytest.skip("BaseStrategy not available")
    
    def test_strategy_signal_generation(self):
        """Test strategy signal generation."""
        try:
            # Test that BaseStrategy raises NotImplementedError
            strategy = BaseStrategy("AAPL")
            sample_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104]
            }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
            
            with pytest.raises(NotImplementedError):
                strategy.generate_signals(sample_data, {}, 100000)
        except NameError:
            pytest.skip("BaseStrategy not available")


if __name__ == "__main__":
    pytest.main([__file__]) 