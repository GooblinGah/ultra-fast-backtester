"""
Trading Strategies for Ultra-Fast Backtester
============================================

This module contains various trading strategies that can be used with the backtester.
All strategies inherit from BaseStrategy and implement the generate_signals method.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from backtester import SignalEvent
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BaseStrategy:
    """Base class for all trading strategies."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.position = 0
        self.entry_price = 0
        
    def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
        """Generate trading signals. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_signals")


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    Generates buy signals when short-term MA crosses above long-term MA
    and sell signals when short-term MA crosses below long-term MA.
    """
    
    def __init__(self, symbol: str, short_window: int = 10, long_window: int = 30):
        super().__init__(symbol)
        self.short_window = short_window
        self.long_window = long_window
        self.short_ma = None
        self.long_ma = None
        self.prev_short_ma = None
        self.prev_long_ma = None
        
    def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
        """Generate signals based on MA crossover."""
        signals = []
        
        # Get current price and moving averages
        current_price = data.get('close', 0)
        sma_short = data.get('sma_20', 0)  # Use 20-day SMA as short MA
        sma_long = data.get('sma_50', 0)   # Use 50-day SMA as long MA
        
        if current_price == 0 or sma_short == 0 or sma_long == 0:
            return signals
            
        # Update our tracking variables
        self.prev_short_ma = self.short_ma
        self.prev_long_ma = self.long_ma
        self.short_ma = sma_short
        self.long_ma = sma_long
        
        # Check for crossover signals
        if self.prev_short_ma and self.prev_long_ma:
            # Golden cross: short MA crosses above long MA
            if (self.prev_short_ma <= self.prev_long_ma and 
                self.short_ma > self.long_ma):
                signal = SignalEvent(
                    timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                    symbol=self.symbol,
                    signal_type="LONG",
                    strength=1.0
                )
                signals.append(signal)
                
            # Death cross: short MA crosses below long MA
            elif (self.prev_short_ma >= self.prev_long_ma and 
                  self.short_ma < self.long_ma):
                signal = SignalEvent(
                    timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                    symbol=self.symbol,
                    signal_type="EXIT",
                    strength=1.0
                )
                signals.append(signal)
                
        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands
    
    Generates buy signals when price touches lower Bollinger Band
    and sell signals when price touches upper Bollinger Band.
    """
    
    def __init__(self, symbol: str, window: int = 20, num_std: float = 2.0):
        super().__init__(symbol)
        self.window = window
        self.num_std = num_std
        
    def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
        """Generate signals based on Bollinger Bands."""
        signals = []
        
        price = data.get('close', 0)
        bb_upper = data.get('bb_upper', 0)
        bb_lower = data.get('bb_lower', 0)
        
        if price == 0 or bb_upper == 0 or bb_lower == 0:
            return signals
            
        # Generate signals based on Bollinger Bands
        if price <= bb_lower:
            # Price at lower band - buy signal
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="LONG",
                strength=1.0
            )
            signals.append(signal)
            
        elif price >= bb_upper:
            # Price at upper band - sell signal
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="EXIT",
                strength=1.0
            )
            signals.append(signal)
            
        return signals


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy using RSI
    
    Generates buy signals when RSI is oversold and sell signals when RSI is overbought.
    """
    
    def __init__(self, symbol: str, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(symbol)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        
    def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
        """Generate signals based on RSI."""
        signals = []
        
        price = data.get('close', 0)
        rsi = data.get('rsi', 50)  # Default to neutral RSI
        
        if price == 0 or rsi == 50:  # Check if RSI is available
            return signals
            
        # Generate signals based on RSI
        if rsi < self.oversold:
            # Oversold - buy signal
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="LONG",
                strength=1.0
            )
            signals.append(signal)
            
        elif rsi > self.overbought:
            # Overbought - sell signal
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="EXIT",
                strength=1.0
            )
            signals.append(signal)
            
        return signals


class DualThrustStrategy(BaseStrategy):
    """
    Dual Thrust Strategy
    
    A breakout strategy that generates signals based on price breaking above/below
    recent high/low levels with confirmation.
    """
    
    def __init__(self, symbol: str, period: int = 20, k1: float = 0.7, k2: float = 0.7):
        super().__init__(symbol)
        self.period = period
        self.k1 = k1
        self.k2 = k2
        self.highest_high = 0
        self.lowest_low = float('inf')
        
    def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
        """Generate Dual Thrust signals."""
        signals = []
        
        high = data.get('high', 0)
        low = data.get('low', 0)
        close = data.get('close', 0)
        
        if high == 0 or low == 0 or close == 0:
            return signals
            
        # Update highest high and lowest low
        self.highest_high = max(self.highest_high, high)
        self.lowest_low = min(self.lowest_low, low)
        
        # Calculate breakout levels
        range_high = self.highest_high - self.lowest_low
        upper_breakout = close + self.k1 * range_high
        lower_breakout = close - self.k2 * range_high
        
        # Generate signals
        if close > upper_breakout:
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="LONG",
                strength=1.0
            )
            signals.append(signal)
            
        elif close < lower_breakout:
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="EXIT",
                strength=1.0
            )
            signals.append(signal)
            
        return signals


class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading Strategy
    
    Places buy orders at regular intervals below current price
    and sell orders at regular intervals above current price.
    """
    
    def __init__(self, symbol: str, grid_size: float = 0.02, num_levels: int = 5):
        super().__init__(symbol)
        self.grid_size = grid_size
        self.num_levels = num_levels
        self.grid_levels = []
        
    def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
        """Generate grid trading signals."""
        signals = []
        
        close = data.get('close', 0)
        if close == 0:
            return signals
            
        # Create grid levels around current price
        self.grid_levels = []
        for i in range(-self.num_levels, self.num_levels + 1):
            level = close * (1 + i * self.grid_size)
            self.grid_levels.append(level)
            
        # Generate buy signals at lower levels
        for level in self.grid_levels[:self.num_levels]:
            if close <= level:
                signal = SignalEvent(
                    timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                    symbol=self.symbol,
                    signal_type="LONG",
                    strength=0.5
                )
                signals.append(signal)
                
        # Generate sell signals at upper levels
        for level in self.grid_levels[self.num_levels+1:]:
            if close >= level:
                signal = SignalEvent(
                    timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                    symbol=self.symbol,
                    signal_type="EXIT",
                    strength=0.5
                )
                signals.append(signal)
                
        return signals


class SimpleProfitableStrategy(BaseStrategy):
    """
    Simple but effective strategy that generates profits.
    
    Strategy:
    1. Buy when price is below 20-day moving average and RSI is oversold
    2. Sell when price is above 20-day moving average and RSI is overbought
    3. Use position sizing based on volatility
    """
    
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.last_buy_price = 0
        
    def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
        """Generate profitable trading signals."""
        signals = []
        
        # Get required data
        close = data.get('close', 0)
        sma_20 = data.get('sma_20', 0)
        rsi = data.get('rsi', 50)
        
        if close == 0 or sma_20 == 0:
            return signals
            
        # Check if we have a position in the backtester
        has_position = self.symbol in positions and positions[self.symbol].quantity > 0
        
        # Buy signal: Price below MA and RSI oversold, and no current position
        buy_condition = (
            close < sma_20 * 0.98 and  # Price 2% below MA
            rsi < 35 and               # RSI oversold
            not has_position           # Not already long
        )
        
        # Sell signal: Price above MA and RSI overbought, OR profit target hit
        sell_condition = (
            has_position and (         # Have position
                (close > sma_20 * 1.02 and rsi > 65) or  # Price above MA and RSI overbought
                (self.entry_price > 0 and close > self.entry_price * 1.15) or  # 15% profit target
                (self.entry_price > 0 and close < self.entry_price * 0.95)     # 5% stop loss
            )
        )
        
        # Generate signals
        if buy_condition:
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="LONG",
                strength=0.8
            )
            signals.append(signal)
            self.entry_price = close
            self.last_buy_price = close
            
        elif sell_condition:
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="EXIT",
                strength=1.0
            )
            signals.append(signal)
            self.entry_price = 0
            
        return signals


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy that rides momentum.
    
    Strategy:
    1. Buy when price breaks above 20-day MA with momentum
    2. Sell when trend reverses or profit target hit
    """
    
    def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
        """Generate trend following signals."""
        signals = []
        
        # Get required data
        close = data.get('close', 0)
        sma_20 = data.get('sma_20', 0)
        sma_50 = data.get('sma_50', 0)
        rsi = data.get('rsi', 50)
        
        if close == 0 or sma_20 == 0:
            return signals
            
        # Calculate trend strength
        trend_strength = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
        
        # Buy signal: Strong uptrend with momentum
        buy_condition = (
            close > sma_20 and         # Price above MA
            trend_strength > 0.02 and  # Strong uptrend
            rsi < 70 and               # Not overbought
            self.position <= 0         # Not already long
        )
        
        # Sell signal: Trend reversal or profit target
        sell_condition = (
            self.position > 0 and (    # Have position
                close < sma_20 or      # Price below MA
                rsi > 75 or            # Overbought
                (self.entry_price > 0 and close > self.entry_price * 1.20) or  # 20% profit target
                (self.entry_price > 0 and close < self.entry_price * 0.90)     # 10% stop loss
            )
        )
        
        # Generate signals
        if buy_condition:
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="LONG",
                strength=min(1.0, trend_strength * 10)
            )
            signals.append(signal)
            self.position = 1
            self.entry_price = close
            
        elif sell_condition:
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="EXIT",
                strength=1.0
            )
            signals.append(signal)
            self.position = 0
            self.entry_price = 0
            
        return signals


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy
    
    A strategy that trades breakouts based on volatility expansion.
    Buys when volatility is low and price breaks above resistance.
    Sells when volatility is high and price breaks below support.
    """
    
    def __init__(self, symbol: str, volatility_period: int = 20, breakout_threshold: float = 1.5):
        super().__init__(symbol)
        self.volatility_period = volatility_period
        self.breakout_threshold = breakout_threshold
        self.volatility_history = []
        
    def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
        """Generate volatility breakout signals."""
        signals = []
        
        # Get required data
        close = data.get('close', 0)
        high = data.get('high', 0)
        low = data.get('low', 0)
        volatility = data.get('volatility', 0)
        resistance = data.get('resistance', high)
        support = data.get('support', low)
        
        if close == 0 or high == 0:
            return signals
            
        # Track volatility
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > self.volatility_period:
            self.volatility_history.pop(0)
            
        # Calculate average volatility
        avg_volatility = np.mean(self.volatility_history) if self.volatility_history else volatility
        
        # Volatility breakout conditions
        low_volatility = volatility < avg_volatility * 0.8
        high_volatility = volatility > avg_volatility * 1.2
        
        # Buy signal: Low volatility + price breakout above resistance
        buy_condition = (
            low_volatility and
            close > resistance * 1.01 and  # 1% above resistance
            self.position <= 0
        )
        
        # Sell signal: High volatility + price breakdown below support
        sell_condition = (
            self.position > 0 and (
                (high_volatility and close < support * 0.99) or  # 1% below support
                (self.entry_price > 0 and close > self.entry_price * 1.12) or  # 12% profit target
                (self.entry_price > 0 and close < self.entry_price * 0.96)     # 4% stop loss
            )
        )
        
        # Generate signals
        if buy_condition:
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="LONG",
                strength=min(1.0, (avg_volatility - volatility) / avg_volatility)
            )
            signals.append(signal)
            self.position = 1
            self.entry_price = close
            
        elif sell_condition:
            signal = SignalEvent(
                timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                symbol=self.symbol,
                signal_type="EXIT",
                strength=1.0
            )
            signals.append(signal)
            self.position = 0
            self.entry_price = 0
            
        return signals


def create_strategy(strategy_name: str, symbol: str, **kwargs):
    """Factory function for creating strategies."""
    strategies = {
        'ma_crossover': MovingAverageCrossoverStrategy,
        'mean_reversion': MeanReversionStrategy,
        'momentum': MomentumStrategy,
        'dual_thrust': DualThrustStrategy,
        'grid_trading': GridTradingStrategy,
        'simple_profitable': SimpleProfitableStrategy,
        'trend_following': TrendFollowingStrategy,
        'volatility_breakout': VolatilityBreakoutStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")
        
    return strategies[strategy_name](symbol, **kwargs)


def strategy_wrapper(strategy):
    """Wrapper for strategies to integrate with backtester."""
    def wrapper(data: pd.Series, positions: Dict, cash: float):
        return strategy.generate_signals(data, positions, cash)
    return wrapper 