"""
Machine Learning Utilities for Ultra-Fast Backtester

This module provides utilities for:
- Feature engineering from market data
- Model training and validation
- Strategy optimization using ML
- Performance prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering utilities for trading strategies."""
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize feature engineer.
        
        Args:
            lookback_periods: List of periods for rolling features
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        self.scaler = StandardScaler()
        
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical analysis features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['close'].shift(1)
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Moving averages
        for period in self.lookback_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
            df[f'price_ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']
        
        # Volatility features
        for period in self.lookback_periods:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            df[f'volatility_annualized_{period}'] = df[f'volatility_{period}'] * np.sqrt(252)
        
        # Momentum features
        for period in self.lookback_periods:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
    
    def create_target_variable(self, data: pd.DataFrame, forward_period: int = 1, threshold: float = 0.01) -> pd.Series:
        """
        Create target variable for ML models.
        
        Args:
            data: DataFrame with price data
            forward_period: Number of periods to look forward
            threshold: Minimum return threshold for classification
            
        Returns:
            Series with target labels (1 for positive return, 0 for negative)
        """
        future_returns = data['close'].shift(-forward_period) / data['close'] - 1
        target = (future_returns > threshold).astype(int)
        return target


class MLStrategyTrainer:
    """Machine learning strategy trainer."""
    
    def __init__(self, feature_columns: List[str] = None):
        """
        Initialize ML strategy trainer.
        
        Args:
            feature_columns: List of feature column names to use
        """
        self.feature_columns = feature_columns
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_data(self, data: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for ML training.
        
        Args:
            data: Feature DataFrame
            target: Target variable Series
            
        Returns:
            Tuple of (features, target) with NaN values removed
        """
        # Combine features and target
        df = data.copy()
        df['target'] = target
        
        # Remove NaN values
        df = df.dropna()
        
        # Separate features and target
        if self.feature_columns:
            features = df[self.feature_columns]
        else:
            # Use all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = df[numeric_cols].drop('target', axis=1, errors='ignore')
        
        target_clean = df['target']
        
        return features, target_clean
    
    def train_model(self, features: pd.DataFrame, target: pd.Series, 
                   model_type: str = 'random_forest', **kwargs) -> Dict:
        """
        Train a machine learning model.
        
        Args:
            features: Feature DataFrame
            target: Target variable Series
            model_type: Type of model to train
            **kwargs: Additional model parameters
            
        Returns:
            Dictionary with model, scaler, and performance metrics
        """
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Select model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42, **kwargs)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(features_scaled, target)
        
        # Make predictions
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(target, predictions),
            'precision': precision_score(target, predictions),
            'recall': recall_score(target, predictions),
            'f1': f1_score(target, predictions)
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(features.columns, model.feature_importances_))
        else:
            importance = {}
        
        # Store results
        self.models[model_type] = model
        self.scalers[model_type] = scaler
        self.feature_importance[model_type] = importance
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'feature_importance': importance
        }
    
    def cross_validate(self, features: pd.DataFrame, target: pd.Series, 
                      model_type: str = 'random_forest', cv_splits: int = 5, **kwargs) -> Dict:
        """
        Perform time series cross-validation.
        
        Args:
            features: Feature DataFrame
            target: Target variable Series
            model_type: Type of model to train
            cv_splits: Number of cross-validation splits
            **kwargs: Additional model parameters
            
        Returns:
            Dictionary with cross-validation results
        """
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Select model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42, **kwargs)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, features_scaled, target, cv=tscv, scoring='accuracy')
        
        return {
            'cv_scores': cv_scores,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'model_type': model_type
        }
    
    def predict(self, features: pd.DataFrame, model_type: str = 'random_forest') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained model.
        
        Args:
            features: Feature DataFrame
            model_type: Type of model to use
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train the model first.")
        
        model = self.models[model_type]
        scaler = self.scalers[model_type]
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]
        
        return predictions, probabilities


class StrategyOptimizer:
    """Optimize strategy parameters using machine learning."""
    
    def __init__(self, backtester, strategy_class, data: pd.DataFrame):
        """
        Initialize strategy optimizer.
        
        Args:
            backtester: Backtester instance
            strategy_class: Strategy class to optimize
            data: Market data
        """
        self.backtester = backtester
        self.strategy_class = strategy_class
        self.data = data
        self.results = []
        
    def optimize_parameters(self, param_grid: Dict, metric: str = 'sharpe_ratio', 
                          n_trials: int = 100) -> Dict:
        """
        Optimize strategy parameters using grid search or random search.
        
        Args:
            param_grid: Dictionary of parameter ranges
            metric: Metric to optimize
            n_trials: Number of trials for random search
            
        Returns:
            Dictionary with best parameters and results
        """
        best_score = -np.inf
        best_params = None
        best_result = None
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid, n_trials)
        
        for params in param_combinations:
            try:
                # Create strategy with parameters
                strategy = self.strategy_class(**params)
                strategy_func = strategy_wrapper(strategy)
                
                # Run backtest
                result = self.backtester.run_backtest(self.data, strategy_func)
                
                # Get metric score
                score = result.get(metric, -np.inf)
                
                # Store result
                self.results.append({
                    'params': params,
                    'result': result,
                    'score': score
                })
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                    
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result,
            'all_results': self.results
        }
    
    def _generate_param_combinations(self, param_grid: Dict, n_trials: int) -> List[Dict]:
        """Generate parameter combinations for optimization."""
        import itertools
        
        # For small parameter grids, use full grid search
        total_combinations = 1
        for values in param_grid.values():
            if isinstance(values, (list, tuple)):
                total_combinations *= len(values)
        
        if total_combinations <= n_trials:
            # Full grid search
            keys = param_grid.keys()
            values = param_grid.values()
            combinations = list(itertools.product(*values))
            return [dict(zip(keys, combo)) for combo in combinations]
        else:
            # Random search
            combinations = []
            for _ in range(n_trials):
                params = {}
                for key, values in param_grid.items():
                    if isinstance(values, (list, tuple)):
                        params[key] = np.random.choice(values)
                    elif isinstance(values, tuple) and len(values) == 2:
                        # Range of values
                        params[key] = np.random.uniform(values[0], values[1])
                    else:
                        params[key] = values
                combinations.append(params)
            return combinations 