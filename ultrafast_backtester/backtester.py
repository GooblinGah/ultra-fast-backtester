"""
Ultra-Fast Event-Driven Backtester
==================================

A high-performance backtesting engine designed for speed and accuracy.
Uses vectorized operations and event-driven architecture for optimal performance.
"""

import numpy as np
import pandas as pd
from numba import jit, prange
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Trade:
    """Represents a single trade execution."""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float = 0.0
    
@dataclass
class Position:
    """Represents a position in a symbol."""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class Portfolio:
    """Represents the current portfolio state."""
    cash: float
    positions: Dict[str, Position]
    total_value: float
    total_pnl: float

class Event:
    """Base class for all events in the backtesting system."""
    def __init__(self, timestamp: datetime, event_type: str):
        self.timestamp = timestamp
        self.event_type = event_type

class MarketEvent(Event):
    """Market data event."""
    def __init__(self, timestamp: datetime, symbol: str, data: Dict):
        super().__init__(timestamp, "MARKET")
        self.symbol = symbol
        self.data = data

class SignalEvent(Event):
    """Trading signal event."""
    def __init__(self, timestamp: datetime, symbol: str, signal_type: str, strength: float = 1.0):
        super().__init__(timestamp, "SIGNAL")
        self.symbol = symbol
        self.signal_type = signal_type  # 'LONG', 'SHORT', 'EXIT'
        self.strength = strength

class OrderEvent(Event):
    """Order event."""
    def __init__(self, timestamp: datetime, symbol: str, order_type: str, 
                 quantity: float, side: str, price: Optional[float] = None):
        super().__init__(timestamp, "ORDER")
        self.symbol = symbol
        self.order_type = order_type  # 'MARKET', 'LIMIT'
        self.quantity = quantity
        self.side = side  # 'BUY', 'SELL'
        self.price = price

class FillEvent(Event):
    """Fill event representing executed trade."""
    def __init__(self, timestamp: datetime, symbol: str, quantity: float, 
                 side: str, price: float, commission: float = 0.0):
        super().__init__(timestamp, "FILL")
        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.price = price
        self.commission = commission

@jit(nopython=True, parallel=True)
def calculate_returns_vectorized(prices: np.ndarray) -> np.ndarray:
    """Calculate returns using vectorized operations for maximum speed."""
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return returns

@jit(nopython=True)
def calculate_sma_vectorized(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate Simple Moving Average using vectorized operations."""
    sma = np.zeros_like(prices)
    for i in range(window - 1, len(prices)):
        sma[i] = np.mean(prices[i - window + 1:i + 1])
    return sma

@jit(nopython=True)
def calculate_ema_vectorized(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate Exponential Moving Average using vectorized operations."""
    alpha = 2.0 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

@jit(nopython=True)
def calculate_bollinger_bands_vectorized(prices: np.ndarray, window: int, 
                                       num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands using vectorized operations."""
    sma = calculate_sma_vectorized(prices, window)
    upper_band = np.zeros_like(prices)
    lower_band = np.zeros_like(prices)
    
    for i in range(window - 1, len(prices)):
        std = np.std(prices[i - window + 1:i + 1])
        upper_band[i] = sma[i] + (num_std * std)
        lower_band[i] = sma[i] - (num_std * std)
    
    return upper_band, sma, lower_band

class UltraFastBacktester:
    """
    Ultra-fast event-driven backtesting engine.
    
    Features:
    - Vectorized operations for maximum speed
    - Event-driven architecture
    - Real-time portfolio tracking
    - Advanced risk management
    - Performance analytics
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.events = []
        self.portfolio_history = []
        self.current_time = None
        
        # Performance tracking
        self.returns = []
        self.drawdowns = []
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
    def reset(self):
        """Reset the backtester to initial state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.events = []
        self.portfolio_history = []
        self.current_time = None
        self.returns = []
        self.drawdowns = []
        
    def add_event(self, event: Event):
        """Add an event to the event queue."""
        self.events.append(event)
        
    def process_events(self):
        """Process all events in chronological order."""
        # Sort events by timestamp
        self.events.sort(key=lambda x: x.timestamp)
        
        for event in self.events:
            self.current_time = event.timestamp
            
            if event.event_type == "MARKET":
                self._process_market_event(event)
            elif event.event_type == "SIGNAL":
                self._process_signal_event(event)
            elif event.event_type == "ORDER":
                self._process_order_event(event)
            elif event.event_type == "FILL":
                self._process_fill_event(event)
                
        # Clear processed events
        self.events = []
        
    def _process_market_event(self, event: MarketEvent):
        """Process market data event."""
        # Update position values
        if event.symbol in self.positions:
            current_price = event.data.get('close', 0)
            if current_price > 0:
                position = self.positions[event.symbol]
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                
        # Record portfolio value
        self._record_portfolio_value()
        
    def _process_signal_event(self, event: SignalEvent):
        """Process trading signal event."""
        # Generate orders based on signals
        if event.signal_type == "LONG":
            # Buy signal - use 10% of available cash
            position_size = self.current_capital * 0.1
            current_price = self._get_current_price(event.symbol)
            if current_price > 0:
                quantity = position_size / current_price
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    order_type="MARKET",
                    quantity=quantity,
                    side="BUY"
                )
                self.add_event(order)
                
        elif event.signal_type == "EXIT":
            # Sell signal - close position if exists
            if event.symbol in self.positions:
                position = self.positions[event.symbol]
                if position.quantity > 0:
                    order = OrderEvent(
                        timestamp=event.timestamp,
                        symbol=event.symbol,
                        order_type="MARKET",
                        quantity=position.quantity,
                        side="SELL"
                    )
                    self.add_event(order)
        
    def _process_order_event(self, event: OrderEvent):
        """Process order event."""
        # Simulate order execution (immediate fill for market orders)
        if event.order_type == "MARKET":
            # Use the current market data price instead of _get_current_price
            if hasattr(self, 'current_market_data') and event.symbol in self.current_market_data:
                fill_price = self.current_market_data[event.symbol].get('close', 0)
            else:
                fill_price = self._get_current_price(event.symbol)
                
            if fill_price > 0:
                commission = abs(event.quantity * fill_price * self.commission)
                fill_event = FillEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    quantity=event.quantity,
                    side=event.side,
                    price=fill_price,
                    commission=commission
                )
                self.add_event(fill_event)
                
    def _process_fill_event(self, event: FillEvent):
        """Process fill event."""
        # Update positions
        if event.symbol not in self.positions:
            self.positions[event.symbol] = Position(
                symbol=event.symbol,
                quantity=0.0,
                avg_price=0.0
            )
            
        position = self.positions[event.symbol]
        
        if event.side == "BUY":
            # Calculate new average price
            total_cost = position.quantity * position.avg_price + event.quantity * event.price
            total_quantity = position.quantity + event.quantity
            if total_quantity > 0:
                position.avg_price = total_cost / total_quantity
            position.quantity += event.quantity
        else:  # SELL
            # Calculate realized P&L
            if position.quantity > 0:
                realized_pnl = (event.price - position.avg_price) * min(event.quantity, position.quantity)
                position.realized_pnl += realized_pnl
                
            position.quantity -= event.quantity
            
        # Update cash
        if event.side == "BUY":
            self.current_capital -= (event.quantity * event.price + event.commission)
        else:
            self.current_capital += (event.quantity * event.price - event.commission)
            
        # Remove position if quantity is zero
        if abs(position.quantity) < 1e-6:
            del self.positions[event.symbol]
            
        # Record trade
        trade = Trade(
            timestamp=event.timestamp,
            symbol=event.symbol,
            side=event.side,
            quantity=event.quantity,
            price=event.price,
            commission=event.commission
        )
        self.trades.append(trade)
        
        # Update portfolio value immediately after trade
        self._record_portfolio_value()
        
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        # Use the current market data if available
        if hasattr(self, 'current_market_data') and symbol in self.current_market_data:
            return self.current_market_data[symbol].get('close', 100.0)
        return 100.0  # Fallback
        
    def _record_portfolio_value(self):
        """Record current portfolio value."""
        total_value = self.current_capital
        positions_value = 0
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price > 0:
                position_value = position.quantity * current_price
                positions_value += position_value
                total_value += position_value
                
                # Update unrealized P&L
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                
        self.portfolio_history.append({
            'timestamp': self.current_time,
            'total_value': total_value,
            'cash': self.current_capital,
            'positions_value': positions_value
        })
        
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_history:
            return {}
            
        values = np.array([p['total_value'] for p in self.portfolio_history])
        returns = calculate_returns_vectorized(values)
        
        # Calculate metrics
        total_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
        annualized_return = total_return * 252 / len(values) if len(values) > 1 else 0
        
        # Calculate Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calculate other metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        win_rate = len([t for t in self.trades if t.side == "SELL" and 
                       self.positions.get(t.symbol, Position(t.symbol, 0, 0)).realized_pnl > 0]) / max(len(self.trades), 1)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_value': values[-1] if len(values) > 0 else self.initial_capital
        }
        
    def run_backtest(self, data: pd.DataFrame, strategy: Callable):
        """Run the backtest with given data and strategy."""
        self.reset()
        self.current_market_data = {}
        
        # Process data row by row to maintain correct timing
        for index, row in data.iterrows():
            # Update current time and market data
            self.current_time = index
            symbol = row.get('symbol', 'UNKNOWN')
            self.current_market_data[symbol] = row.to_dict()
            
            # Process market event immediately
            market_event = MarketEvent(
                timestamp=index,
                symbol=symbol,
                data=row.to_dict()
            )
            self._process_market_event(market_event)
            
            # Generate and process signals immediately
            signals = strategy(row, self.positions, self.current_capital)
            for signal in signals:
                self._process_signal_event(signal)
            
            # Process any pending events immediately to update positions
            self.process_events()
                
        # Process any remaining events (orders and fills)
        self.process_events()
        
        # Calculate final performance metrics
        return self.calculate_performance_metrics()
        
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
            
        return pd.DataFrame([
            {
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'value': trade.quantity * trade.price
            }
            for trade in self.trades
        ])
        
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as a DataFrame."""
        if not self.portfolio_history:
            return pd.DataFrame()
            
        return pd.DataFrame(self.portfolio_history) 