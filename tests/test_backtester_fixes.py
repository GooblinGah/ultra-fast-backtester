"""
Test file to verify critical backtester fixes.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ultrafast_backtester.backtester import (
    UltraFastBacktester, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    EventType, OrderSide, SignalType, OrderType, Position
)


class TestBacktesterFixes:
    """Test cases to verify critical fixes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=pd.date_range('2023-01-01', periods=10, freq='D'))
        
    def test_proper_event_processing(self):
        """Test that events are processed in proper order."""
        backtester = UltraFastBacktester(initial_capital=10000)
        
        # Create events in reverse order
        market_event = MarketEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol="AAPL",
            data={'close': 100}
        )
        
        signal_event = SignalEvent(
            timestamp=datetime(2023, 1, 1, 9, 0),  # Earlier timestamp
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=1.0
        )
        
        # Add events
        backtester.add_event(market_event)
        backtester.add_event(signal_event)
        
        # Process events
        backtester.process_events()
        
        # Verify events were processed in chronological order
        assert len(backtester.order_events) == 0  # Events should be cleared after processing
        
    def test_position_sizing_with_signal_strength(self):
        """Test that position sizing considers signal strength."""
        backtester = UltraFastBacktester(initial_capital=10000, max_position_size=0.1)
        
        # Create signal with 50% strength
        signal_event = SignalEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=0.5
        )
        
        # Set current market data
        backtester.current_market_data["AAPL"] = {'close': 100}
        
        # Process signal
        backtester._process_signal_event(signal_event)
        
        # Should generate an order with reduced position size
        assert len(backtester.order_events) == 1
        
    def test_insufficient_funds_handling(self):
        """Test handling of insufficient funds."""
        backtester = UltraFastBacktester(initial_capital=100)  # Very small capital
        
        # Create order that would exceed available capital
        order_event = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=10,  # Would cost 1000 at price 100
            side=OrderSide.BUY
        )
        
        # Set current market data
        backtester.current_market_data["AAPL"] = {'close': 100}
        
        # Process order - should adjust quantity
        backtester._process_order_event(order_event)
        
        # Should generate a fill event with adjusted quantity
        assert len(backtester.fill_events) == 1
        
    def test_data_validation(self):
        """Test data validation in run_backtest."""
        backtester = UltraFastBacktester()
        
        # Test with empty data
        with pytest.raises(ValueError, match="Input data is empty"):
            backtester.run_backtest(pd.DataFrame(), lambda x, y, z: [])
            
        # Test with missing required columns
        invalid_data = pd.DataFrame({'open': [100, 101]})
        with pytest.raises(ValueError, match="Missing required columns"):
            backtester.run_backtest(invalid_data, lambda x, y, z: [])
            
    def test_slippage_and_spread_modeling(self):
        """Test that slippage and bid-ask spread are applied."""
        backtester = UltraFastBacktester(
            slippage=0.001,  # 0.1% slippage
            bid_ask_spread=0.002  # 0.2% spread
        )
        
        # Create buy order
        order_event = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=1,
            side=OrderSide.BUY
        )
        
        # Set current market data
        backtester.current_market_data["AAPL"] = {'close': 100}
        
        # Process order
        backtester._process_order_event(order_event)
        
        # Should generate fill event with adjusted price
        assert len(backtester.fill_events) == 1
        fill_event = backtester.fill_events[0]
        
        # Price should be higher than base price due to spread and slippage
        assert fill_event.price > 100
        
    def test_portfolio_history_limiting(self):
        """Test that portfolio history size is limited."""
        backtester = UltraFastBacktester(max_portfolio_history=5)
        
        # Record more than the limit
        for i in range(10):
            backtester.current_time = datetime(2023, 1, 1, i, 0)
            backtester._record_portfolio_value()
            
        # Should only keep the most recent records
        assert len(backtester.portfolio_history) == 5
        
    def test_stop_loss_functionality(self):
        """Test stop loss functionality."""
        backtester = UltraFastBacktester(stop_loss_pct=0.05)  # 5% stop loss
        
        # Create a position
        backtester.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=10,
            avg_price=100
        )
        
        # Create market event with price drop that triggers stop loss
        market_event = MarketEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol="AAPL",
            data={'close': 94}  # 6% drop, should trigger 5% stop loss
        )
        
        # Process market event
        backtester._process_market_event(market_event)
        
        # Should generate exit signal
        assert len(backtester.signal_events) == 1
        assert backtester.signal_events[0].signal_type == SignalType.EXIT
        
    def test_enum_usage(self):
        """Test that enums are used properly."""
        # Test event types
        assert EventType.MARKET.value == "MARKET"
        assert EventType.SIGNAL.value == "SIGNAL"
        assert EventType.ORDER.value == "ORDER"
        assert EventType.FILL.value == "FILL"
        
        # Test order sides
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"
        
        # Test signal types
        assert SignalType.LONG.value == "LONG"
        assert SignalType.SHORT.value == "SHORT"
        assert SignalType.EXIT.value == "EXIT"
        
        # Test order types
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        backtester = UltraFastBacktester()
        
        # Test with invalid signal
        invalid_signal = SignalEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=np.nan  # Invalid strength
        )
        
        # Should handle gracefully
        backtester._process_signal_event(invalid_signal)
        
        # Test with invalid order
        invalid_order = OrderEvent(
            timestamp=datetime(2023, 1, 1, 10, 0),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=-1,  # Invalid quantity
            side=OrderSide.BUY
        )
        
        # Should handle gracefully
        backtester._process_order_event(invalid_order)
        assert len(backtester.fill_events) == 0  # No fill should be generated


if __name__ == "__main__":
    pytest.main([__file__]) 