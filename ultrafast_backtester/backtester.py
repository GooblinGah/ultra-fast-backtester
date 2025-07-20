"""
Ultra Fast Event-Driven Backtester
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
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class EventType(Enum):
    """Event types for the backtesting system."""
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"

class OrderSide(Enum):
    """Order sides."""
    BUY = "BUY"
    SELL = "SELL"

class SignalType(Enum):
    """Signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"

class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"

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
    def __init__(self, timestamp: datetime, event_type: EventType):
        self.timestamp = timestamp
        self.event_type = event_type

class MarketEvent(Event):
    """Market data event."""
    def __init__(self, timestamp: datetime, symbol: str, data: Dict):
        super().__init__(timestamp, EventType.MARKET)
        self.symbol = symbol
        self.data = data

class SignalEvent(Event):
    """Trading signal event."""
    def __init__(self, timestamp: datetime, symbol: str, signal_type: SignalType, strength: float = 1.0):
        super().__init__(timestamp, EventType.SIGNAL)
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength

class OrderEvent(Event):
    """Order event."""
    def __init__(self, timestamp: datetime, symbol: str, order_type: OrderType, 
                 quantity: float, side: OrderSide, price: Optional[float] = None):
        super().__init__(timestamp, EventType.ORDER)
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.side = side
        self.price = price

class FillEvent(Event):
    """Fill event representing executed trade."""
    def __init__(self, timestamp: datetime, symbol: str, quantity: float, 
                 side: OrderSide, price: float, commission: float = 0.0):
        super().__init__(timestamp, EventType.FILL)
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
    Event-driven backtesting engine.
    
    Features:
    - Vectorized operations for maximum speed
    - Event-driven architecture
    - Real-time portfolio tracking
    - Advanced risk management
    - Performance analytics
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001, 
                 max_position_size: float = 0.1, max_portfolio_history: int = 10000,
                 slippage: float = 0.0001, bid_ask_spread: float = 0.0002,
                 max_drawdown_limit: float = 0.2, stop_loss_pct: float = 0.05):
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_position_size = max_position_size  # Maximum position size as fraction of capital
        self.max_portfolio_history = max_portfolio_history  # Limit portfolio history size
        self.slippage = slippage  # Slippage as fraction of price
        self.bid_ask_spread = bid_ask_spread  # Bid-ask spread as fraction of price
        self.max_drawdown_limit = max_drawdown_limit  # Maximum allowed drawdown
        self.stop_loss_pct = stop_loss_pct  # Stop loss percentage
        
        # Portfolio state
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        
        # Event queues
        self.market_events = []
        self.signal_events = []
        self.order_events = []
        self.fill_events = []
        
        # Performance tracking
        self.portfolio_history = []
        self.current_time = None
        self.current_market_data = {}
        
        # Performance metrics
        self.returns = []
        self.drawdowns = []
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
    def reset(self):
        """Reset the backtester to initial state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        
        # Clear all event queues
        self.market_events = []
        self.signal_events = []
        self.order_events = []
        self.fill_events = []
        
        self.portfolio_history = []
        self.current_time = None
        self.current_market_data = {}
        self.returns = []
        self.drawdowns = []
        
    def add_event(self, event: Event):
        """Add an event to the appropriate event queue."""
        if event.event_type == EventType.MARKET:
            self.market_events.append(event)
        elif event.event_type == EventType.SIGNAL:
            self.signal_events.append(event)
        elif event.event_type == EventType.ORDER:
            self.order_events.append(event)
        elif event.event_type == EventType.FILL:
            self.fill_events.append(event)
        
    def process_events(self):
        """Process all events in chronological order."""
        # Combine all events and sort by timestamp
        all_events = (self.market_events + self.signal_events + 
                     self.order_events + self.fill_events)
        all_events.sort(key=lambda x: x.timestamp)
        
        # Clear all event queues
        self.market_events = []
        self.signal_events = []
        self.order_events = []
        self.fill_events = []
        
        # Process events in order
        for event in all_events:
            self.current_time = event.timestamp
            
            if event.event_type == EventType.MARKET:
                self._process_market_event(event)
            elif event.event_type == EventType.SIGNAL:
                self._process_signal_event(event)
            elif event.event_type == EventType.ORDER:
                self._process_order_event(event)
            elif event.event_type == EventType.FILL:
                self._process_fill_event(event)
                
    def _process_market_event(self, event: MarketEvent):
        """Process market data event."""
        # Update current market data
        self.current_market_data[event.symbol] = event.data
        
        # Update position values and check stop losses
        if event.symbol in self.positions:
            current_price = event.data.get('close', 0)
            if current_price > 0:
                position = self.positions[event.symbol]
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                
                # Check stop loss
                if position.quantity > 0:  # Long position
                    loss_pct = (position.avg_price - current_price) / position.avg_price
                    if loss_pct >= self.stop_loss_pct:
                        # Trigger stop loss
                        stop_loss_signal = SignalEvent(
                            timestamp=event.timestamp,
                            symbol=event.symbol,
                            signal_type=SignalType.EXIT,
                            strength=1.0
                        )
                        self.signal_events.append(stop_loss_signal)
                        
        # Record portfolio value
        self._record_portfolio_value()
        
        # Check maximum drawdown limit
        if len(self.portfolio_history) > 1:
            current_value = self.portfolio_history[-1]['total_value']
            peak_value = max([p['total_value'] for p in self.portfolio_history])
            current_drawdown = (peak_value - current_value) / peak_value
            
            if current_drawdown >= self.max_drawdown_limit:
                print(f"Warning: Maximum drawdown limit reached ({current_drawdown:.2%})")
                # Could implement automatic position closing here
        
    def _process_signal_event(self, event: SignalEvent):
        """Process trading signal event."""
        try:
            if event.signal_type == SignalType.LONG:
                self._generate_buy_order(event)
            elif event.signal_type == SignalType.SHORT:
                self._generate_sell_order(event)
            elif event.signal_type == SignalType.EXIT:
                self._generate_exit_order(event)
        except Exception as e:
            print(f"Error processing signal event: {e}")
                
    def _generate_buy_order(self, event: SignalEvent):
        """Generate a buy order based on signal strength and available capital."""
        current_price = self._get_current_price(event.symbol)
        if current_price <= 0:
            return
            
        # Calculate position size based on signal strength and available capital
        position_value = self.current_capital * self.max_position_size * event.strength
        
        # Check if we have enough cash
        if position_value > self.current_capital:
            position_value = self.current_capital * 0.95  # Leave some cash for commissions
            
        if position_value <= 0:
            return
            
        quantity = position_value / current_price
        
        # Validate quantity
        if quantity <= 0 or np.isnan(quantity) or np.isinf(quantity):
            return
            
        order = OrderEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            order_type=OrderType.MARKET,
            quantity=quantity,
            side=OrderSide.BUY
        )
        self.order_events.append(order)
        
    def _generate_sell_order(self, event: SignalEvent):
        """Generate a sell order for short positions."""
        current_price = self._get_current_price(event.symbol)
        if current_price <= 0:
            return
            
        # For short positions, use the same logic as buy but with SELL side
        position_value = self.current_capital * self.max_position_size * event.strength
        
        if position_value <= 0:
            return
            
        quantity = position_value / current_price
        
        # Validate quantity
        if quantity <= 0 or np.isnan(quantity) or np.isinf(quantity):
            return
            
        order = OrderEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            order_type=OrderType.MARKET,
            quantity=quantity,
            side=OrderSide.SELL
        )
        self.order_events.append(order)
        
    def _generate_exit_order(self, event: SignalEvent):
        """Generate an exit order to close existing position."""
        if event.symbol not in self.positions:
            return
            
        position = self.positions[event.symbol]
        if abs(position.quantity) < 1e-6:
            return
            
        # Close the entire position
        order = OrderEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
            side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        )
        self.order_events.append(order)
        
    def _process_order_event(self, event: OrderEvent):
        """Process order event with proper validation and slippage modeling."""
        try:
            # Validate order
            if event.quantity <= 0 or np.isnan(event.quantity) or np.isinf(event.quantity):
                return
                
            if event.order_type == OrderType.MARKET:
                base_price = self._get_current_price(event.symbol)
                
                if base_price <= 0:
                    return
                    
                # Apply slippage and bid-ask spread
                if event.side == OrderSide.BUY:
                    # Buy at ask price (higher)
                    fill_price = base_price * (1 + self.bid_ask_spread / 2)
                    # Add slippage for large orders
                    slippage_impact = min(self.slippage * event.quantity / 1000, 0.001)  # Cap slippage
                    fill_price *= (1 + slippage_impact)
                else:  # SELL
                    # Sell at bid price (lower)
                    fill_price = base_price * (1 - self.bid_ask_spread / 2)
                    # Add slippage for large orders
                    slippage_impact = min(self.slippage * event.quantity / 1000, 0.001)  # Cap slippage
                    fill_price *= (1 - slippage_impact)
                    
                # Calculate commission
                commission = abs(event.quantity * fill_price * self.commission)
                
                # Check if we have enough cash for buy orders
                if event.side == OrderSide.BUY:
                    total_cost = event.quantity * fill_price + commission
                    if total_cost > self.current_capital:
                        # Adjust quantity to fit available cash
                        adjusted_quantity = (self.current_capital - commission) / fill_price
                        if adjusted_quantity <= 0:
                            return
                        event.quantity = adjusted_quantity
                        commission = abs(adjusted_quantity * fill_price * self.commission)
                
                # Create fill event
                fill_event = FillEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    quantity=event.quantity,
                    side=event.side,
                    price=fill_price,
                    commission=commission
                )
                self.fill_events.append(fill_event)
                
        except Exception as e:
            print(f"Error processing order event: {e}")
                
    def _process_fill_event(self, event: FillEvent):
        """Process fill event with proper error handling."""
        try:
            # Validate fill
            if event.quantity <= 0 or np.isnan(event.quantity) or np.isinf(event.quantity):
                return
                
            if event.price <= 0 or np.isnan(event.price) or np.isinf(event.price):
                return
                
            # Initialize position if it doesn't exist
            if event.symbol not in self.positions:
                self.positions[event.symbol] = Position(
                    symbol=event.symbol,
                    quantity=0.0,
                    avg_price=0.0
                )
                
            position = self.positions[event.symbol]
            
            if event.side == OrderSide.BUY:
                # Calculate new average price
                total_cost = position.quantity * position.avg_price + event.quantity * event.price
                total_quantity = position.quantity + event.quantity
                if total_quantity > 0:
                    position.avg_price = total_cost / total_quantity
                position.quantity += event.quantity
                
                # Update cash
                total_cost = event.quantity * event.price + event.commission
                if total_cost > self.current_capital:
                    raise ValueError(f"Insufficient funds: need ${total_cost:.2f}, have ${self.current_capital:.2f}")
                self.current_capital -= total_cost
                
            else:  # SELL
                # Calculate realized P&L
                if position.quantity > 0:
                    realized_pnl = (event.price - position.avg_price) * min(event.quantity, position.quantity)
                    position.realized_pnl += realized_pnl
                    
                position.quantity -= event.quantity
                
                # Update cash
                self.current_capital += (event.quantity * event.price - event.commission)
                
            # Remove position if quantity is zero
            if abs(position.quantity) < 1e-6:
                del self.positions[event.symbol]
                
            # Record trade
            trade = Trade(
                timestamp=event.timestamp,
                symbol=event.symbol,
                side=event.side.value,
                quantity=event.quantity,
                price=event.price,
                commission=event.commission
            )
            self.trades.append(trade)
            
            # Update portfolio value immediately after trade
            self._record_portfolio_value()
            
        except Exception as e:
            print(f"Error processing fill event: {e}")
        
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol with caching."""
        if symbol in self.current_market_data:
            return self.current_market_data[symbol].get('close', 0.0)
        return 0.0
        
    def _record_portfolio_value(self):
        """Record current portfolio value with size limiting."""
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
                
        portfolio_record = {
            'timestamp': self.current_time,
            'total_value': total_value,
            'cash': self.current_capital,
            'positions_value': positions_value
        }
        
        self.portfolio_history.append(portfolio_record)
        
        # Limit portfolio history size to prevent memory issues
        if len(self.portfolio_history) > self.max_portfolio_history:
            # Keep only the most recent records
            self.portfolio_history = self.portfolio_history[-self.max_portfolio_history:]
        
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
        
        # Calculate win rate from trades
        profitable_trades = 0
        total_trades = len(self.trades)
        for trade in self.trades:
            if trade.side == "SELL":
                # Find corresponding buy trade for P&L calculation
                symbol_trades = [t for t in self.trades if t.symbol == trade.symbol and t.side == "BUY"]
                if symbol_trades:
                    buy_price = symbol_trades[-1].price
                    if trade.price > buy_price:
                        profitable_trades += 1
                        
        win_rate = profitable_trades / max(total_trades, 1)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_value': values[-1] if len(values) > 0 else self.initial_capital
        }
        
    def run_backtest(self, data: pd.DataFrame, strategy: Callable):
        """Run the backtest with given data and strategy."""
        self.reset()
        
        # Validate input data
        if data.empty:
            raise ValueError("Input data is empty")
            
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for data quality issues
        if data['close'].isnull().any():
            print("Warning: Found null values in close prices, filling with forward fill")
            data = data.fillna(method='ffill')
            
        # Check for outliers (prices that are too extreme)
        price_changes = data['close'].pct_change().abs()
        outlier_threshold = 0.5  # 50% price change
        outliers = price_changes > outlier_threshold
        if outliers.any():
            print(f"Warning: Found {outliers.sum()} potential price outliers")
        
        # Process data row by row to maintain correct timing
        for index, row in data.iterrows():
            # Update current time and market data
            self.current_time = index
            symbol = row.get('symbol', 'UNKNOWN')
            
            # Validate row data
            if pd.isnull(row['close']) or row['close'] <= 0:
                print(f"Warning: Invalid close price at {index}: {row['close']}")
                continue
                
            # Create and process market event
            market_event = MarketEvent(
                timestamp=index,
                symbol=symbol,
                data=row.to_dict()
            )
            self._process_market_event(market_event)
            
            # Generate signals from strategy
            try:
                signals = strategy(row, self.positions, self.current_capital)
                if signals:
                    for signal in signals:
                        if isinstance(signal, dict):
                            # Convert dict to SignalEvent
                            signal_event = SignalEvent(
                                timestamp=index,
                                symbol=signal.get('symbol', symbol),
                                signal_type=SignalType(signal.get('signal_type', 'LONG')),
                                strength=signal.get('strength', 1.0)
                            )
                        else:
                            signal_event = signal
                        self.signal_events.append(signal_event)
            except Exception as e:
                print(f"Error generating signals at {index}: {e}")
                continue
            
            # Process all events at the end of each time step
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