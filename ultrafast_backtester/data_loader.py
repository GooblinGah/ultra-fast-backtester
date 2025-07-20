"""
Data Loader for Ultra-Fast Backtester
=====================================

This module handles data fetching, processing, and preparation for backtesting.
Supports multiple data sources and formats.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Base class for data loading operations."""
    
    def __init__(self):
        self.cache = {}
        
    def load_data(self, symbol: str, start_date: str, end_date: str, 
                  interval: str = '1d') -> pd.DataFrame:
        """Load data for a given symbol and date range."""
        raise NotImplementedError("Subclasses must implement load_data")
        
    def get_ohlcv(self, symbol: str, start_date: str, end_date: str, 
                  interval: str = '1d') -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        raise NotImplementedError("Subclasses must implement get_ohlcv")

class YahooFinanceLoader(DataLoader):
    """Data loader for Yahoo Finance."""
    
    def __init__(self):
        super().__init__()
        
    def load_data(self, symbol: str, start_date: str, end_date: str, 
                  interval: str = '1d') -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
            if cache_key in self.cache:
                return self.cache[cache_key].copy()
                
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
                
            # Clean and prepare data
            data = self._prepare_data(data, symbol)
            
            # Cache the data
            self.cache[cache_key] = data.copy()
            
            return data
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_ohlcv(self, symbol: str, start_date: str, end_date: str, 
                  interval: str = '1d') -> pd.DataFrame:
        """Get OHLCV data from Yahoo Finance."""
        return self.load_data(symbol, start_date, end_date, interval)
        
    def _prepare_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Prepare and clean the data."""
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Add symbol column
        data['symbol'] = symbol
        
        # Add technical indicators
        data = self._add_technical_indicators(data)
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        return data
        
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the data."""
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        
        # Calculate log returns
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Calculate volatility (rolling standard deviation of returns)
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Calculate moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Calculate MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Calculate RSI
        data['rsi'] = self._calculate_rsi(data['close'])
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'])
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        data['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Calculate ATR (Average True Range)
        data['atr'] = self._calculate_atr(data)
        
        # Calculate support and resistance levels
        data['support'] = data['low'].rolling(window=20).min()
        data['resistance'] = data['high'].rolling(window=20).max()
        
        return data
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                  std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
        
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr

class SyntheticDataLoader(DataLoader):
    """Generate synthetic market data for testing."""
    
    def __init__(self, seed: int = 42):
        super().__init__()
        np.random.seed(seed)
        
    def load_data(self, symbol: str, start_date: str, end_date: str, 
                  interval: str = '1d') -> pd.DataFrame:
        """Generate synthetic OHLCV data."""
        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate date range
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Generate synthetic price data using geometric Brownian motion
        n_days = len(dates)
        initial_price = 100.0
        mu = 0.0001  # Daily return mean
        sigma = 0.02  # Daily volatility
        
        # Generate price path
        returns = np.random.normal(mu, sigma, n_days)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame(index=dates)
        data['open'] = prices * (1 + np.random.normal(0, 0.001, n_days))
        data['high'] = np.maximum(data['open'], prices) * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
        data['low'] = np.minimum(data['open'], prices) * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
        data['close'] = prices
        data['volume'] = np.random.randint(1000000, 10000000, n_days)
        data['symbol'] = symbol
        
        # Add technical indicators
        data = self._add_technical_indicators(data)
        
        return data
        
    def get_ohlcv(self, symbol: str, start_date: str, end_date: str, 
                  interval: str = '1d') -> pd.DataFrame:
        """Get synthetic OHLCV data."""
        return self.load_data(symbol, start_date, end_date, interval)
        
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to synthetic data."""
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Calculate moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Calculate RSI
        data['rsi'] = self._calculate_rsi(data['close'])
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'])
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        
        return data
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                  std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band

def create_data_loader(source: str = 'yahoo') -> DataLoader:
    """Factory function to create data loader instances."""
    loaders = {
        'yahoo': YahooFinanceLoader,
        'synthetic': SyntheticDataLoader
    }
    
    if source not in loaders:
        raise ValueError(f"Unknown data source: {source}")
        
    return loaders[source]()

def load_multiple_symbols(symbols: List[str], start_date: str, end_date: str, 
                         source: str = 'yahoo') -> Dict[str, pd.DataFrame]:
    """Load data for multiple symbols."""
    loader = create_data_loader(source)
    data_dict = {}
    
    for symbol in symbols:
        try:
            data = loader.load_data(symbol, start_date, end_date)
            if not data.empty:
                data_dict[symbol] = data
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            
    return data_dict

def prepare_data_for_backtest(data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Prepare data specifically for backtesting."""
    if data.empty:
        return data
        
    # Ensure we have required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Add symbol if not present
    if 'symbol' not in data.columns and symbol:
        data['symbol'] = symbol
        
    # Sort by timestamp
    data = data.sort_index()
    
    # Remove any rows with NaN values in critical columns
    data = data.dropna(subset=['close', 'volume'])
    
    # Ensure all numeric columns are float
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
    return data 