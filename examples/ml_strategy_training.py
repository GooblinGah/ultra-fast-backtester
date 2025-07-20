#!/usr/bin/env python3
"""
Machine Learning Strategy Training Example
==========================================

This script demonstrates how to use machine learning with the backtester
to create predictive trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import backtester components
try:
    from ultrafast_backtester.backtester import UltraFastBacktester
    from ultrafast_backtester.strategies import create_strategy
    from ultrafast_backtester.data_loader import create_data_loader
    from ultrafast_backtester.ml_utils import FeatureEngineer, MLStrategyTrainer
    from ultrafast_backtester.visualization import BacktestVisualizer
except ImportError:
    print("Warning: Could not import ultrafast_backtester modules")
    print("This example requires the package to be installed in development mode")
    print("Run: pip install -e .")
    exit(1)


def create_sample_data():
    """Create sample market data for ML training."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Create realistic price movements
    np.random.seed(42)
    
    initial_price = 100.0
    returns = np.zeros(n_days)
    
    # Add trend and volatility
    trend = 0.0002
    volatility = 0.015 + 0.005 * np.sin(2 * np.pi * np.arange(n_days) / 90)
    returns = trend + np.random.normal(0, volatility, n_days)
    
    # Calculate prices
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['Open'] = prices * (1 + np.random.normal(0, 0.002, n_days))
    data['High'] = np.maximum(data['Open'], prices) * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
    data['Low'] = np.minimum(data['Open'], prices) * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
    data['Close'] = prices
    data['Volume'] = np.random.randint(1000000, 10000000, n_days)
    
    return data


def create_technical_features(data):
    """Create technical indicators as features."""
    features = data.copy()
    
    # Price-based features
    features['returns'] = data['Close'].pct_change()
    features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    features['price_change'] = data['Close'] - data['Close'].shift(1)
    
    # Moving averages
    features['sma_5'] = data['Close'].rolling(window=5).mean()
    features['sma_10'] = data['Close'].rolling(window=10).mean()
    features['sma_20'] = data['Close'].rolling(window=20).mean()
    features['ema_12'] = data['Close'].ewm(span=12).mean()
    features['ema_26'] = data['Close'].ewm(span=26).mean()
    
    # Price ratios
    features['price_sma5_ratio'] = data['Close'] / features['sma_5']
    features['price_sma20_ratio'] = data['Close'] / features['sma_20']
    features['ema_ratio'] = features['ema_12'] / features['ema_26']
    
    # Volatility features
    features['volatility_5'] = features['returns'].rolling(window=5).std()
    features['volatility_20'] = features['returns'].rolling(window=20).std()
    
    # RSI
    features['rsi'] = calculate_rsi(data['Close'])
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['Close'])
    features['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
    features['bb_width'] = bb_upper - bb_lower
    
    # MACD
    features['macd'] = features['ema_12'] - features['ema_26']
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    # Volume features
    features['volume_sma'] = data['Volume'].rolling(window=20).mean()
    features['volume_ratio'] = data['Volume'] / features['volume_sma']
    
    return features


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band


def create_target_variable(features, forward_period=5, threshold=0.02):
    """Create target variable for ML training."""
    # Calculate future returns
    future_returns = features['Close'].shift(-forward_period) / features['Close'] - 1
    
    # Create binary classification target
    target = np.where(future_returns > threshold, 1, 0)
    
    return pd.Series(target, index=features.index, name='target')


def train_ml_model(features, target):
    """Train a machine learning model."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Prepare data
        feature_cols = [col for col in features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        X = features[feature_cols].dropna()
        y = target.loc[X.index]
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print("Warning: No valid data for training")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return model, feature_cols
        
    except ImportError:
        print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
        return None


def create_ml_strategy(model, feature_cols, symbol='AAPL'):
    """Create a trading strategy based on ML predictions."""
    class MLStrategy:
        def __init__(self, symbol, model, feature_cols):
            self.symbol = symbol
            self.model = model
            self.feature_cols = feature_cols
            self.position = 0
        
        def generate_signals(self, data, positions, cash):
            """Generate trading signals based on ML predictions."""
            signals = []
            
            if self.model is None:
                return signals
            
            try:
                # Create features for current data
                features = create_technical_features(data)
                X = features[self.feature_cols].iloc[-1:].dropna()
                
                if len(X) == 0:
                    return signals
                
                # Get prediction
                prediction = self.model.predict(X)[0]
                probability = self.model.predict_proba(X)[0].max()
                
                # Generate signals based on prediction
                if prediction == 1 and probability > 0.6 and self.position == 0:
                    # Buy signal
                    signals.append({
                        'timestamp': data.index[-1],
                        'symbol': self.symbol,
                        'signal_type': 'LONG',
                        'strength': probability
                    })
                    self.position = 1
                
                elif prediction == 0 and self.position == 1:
                    # Sell signal
                    signals.append({
                        'timestamp': data.index[-1],
                        'symbol': self.symbol,
                        'signal_type': 'SHORT',
                        'strength': 1.0
                    })
                    self.position = 0
                
            except Exception as e:
                print(f"Error generating ML signals: {e}")
            
            return signals
    
    return MLStrategy(symbol, model, feature_cols)


def main():
    """Main function to run ML strategy training example."""
    print("Ultra-Fast Backtester - ML Strategy Training Example")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample market data...")
    data = create_sample_data()
    print(f"Generated {len(data)} data points")
    
    # Create features
    print("\nCreating technical features...")
    features = create_technical_features(data)
    
    # Create target variable
    print("Creating target variable...")
    target = create_target_variable(features, forward_period=5, threshold=0.02)
    
    # Train ML model
    print("\nTraining machine learning model...")
    model_result = train_ml_model(features, target)
    
    if model_result is None:
        print("Could not train model. Exiting.")
        return
    
    model, feature_cols = model_result
    
    # Create ML strategy
    print("\nCreating ML-based trading strategy...")
    ml_strategy = create_ml_strategy(model, feature_cols, 'AAPL')
    
    # Run backtest with ML strategy
    print("\nRunning backtest with ML strategy...")
    try:
        backtester = UltraFastBacktester(initial_capital=100000, commission=0.001)
        
        # Use last 20% of data for testing
        test_start = int(len(data) * 0.8)
        test_data = data.iloc[test_start:]
        
        start_time = datetime.now()
        metrics = backtester.run_backtest(test_data, ml_strategy)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        print(f"ML Backtest completed in {duration:.3f} seconds")
        
        # Display results
        print("\nML Strategy Backtest Results:")
        print("-" * 40)
        print(f"Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return', 0)*100:.2f}%)")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)")
        print(f"Final Value: ${metrics.get('final_value', 100000):,.2f}")
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        print("This might be due to missing backtester implementation")


if __name__ == "__main__":
    main() 