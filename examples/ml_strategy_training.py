#!/usr/bin/env python3
"""
Machine Learning Strategy Training Example

This example demonstrates how to:
1. Load and prepare market data
2. Create technical features
3. Train ML models for trading signals
4. Backtest ML-based strategies
5. Optimize model parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultrafast_backtester import (
    UltraFastBacktester, 
    create_data_loader, 
    create_strategy, 
    strategy_wrapper,
    FeatureEngineer,
    MLStrategyTrainer,
    StrategyOptimizer
)


def main():
    """Main example function."""
    
    print("Ultra-Fast Backtester - ML Strategy Training Example")
    print("=" * 60)
    
    # 1. Load market data
    print("\n1. Loading market data...")
    loader = create_data_loader('yahoo')
    data = loader.load_data('AAPL', '2020-01-01', '2023-12-31')
    print(f"Loaded {len(data)} data points for AAPL")
    
    # 2. Create technical features
    print("\n2. Creating technical features...")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_technical_features(data)
    
    # 3. Create target variable (next day return > 1%)
    print("\n3. Creating target variable...")
    target = feature_engineer.create_target_variable(features_df, forward_period=1, threshold=0.01)
    
    # 4. Prepare data for ML
    print("\n4. Preparing data for ML training...")
    ml_trainer = MLStrategyTrainer()
    features, target_clean = ml_trainer.prepare_data(features_df, target)
    
    print(f"Features shape: {features.shape}")
    print(f"Target distribution: {target_clean.value_counts().to_dict()}")
    
    # 5. Train ML models
    print("\n5. Training ML models...")
    
    # Train Random Forest
    rf_result = ml_trainer.train_model(features, target_clean, 'random_forest')
    print(f"Random Forest - Accuracy: {rf_result['metrics']['accuracy']:.3f}")
    
    # Train Gradient Boosting
    gb_result = ml_trainer.train_model(features, target_clean, 'gradient_boosting')
    print(f"Gradient Boosting - Accuracy: {gb_result['metrics']['accuracy']:.3f}")
    
    # Train Logistic Regression
    lr_result = ml_trainer.train_model(features, target_clean, 'logistic_regression')
    print(f"Logistic Regression - Accuracy: {lr_result['metrics']['accuracy']:.3f}")
    
    # 6. Cross-validation
    print("\n6. Performing cross-validation...")
    cv_result = ml_trainer.cross_validate(features, target_clean, 'random_forest', cv_splits=5)
    print(f"Cross-validation - Mean Accuracy: {cv_result['mean_accuracy']:.3f} ± {cv_result['std_accuracy']:.3f}")
    
    # 7. Create ML-based strategy
    print("\n7. Creating ML-based strategy...")
    
    class MLStrategy:
        """ML-based trading strategy."""
        
        def __init__(self, symbol: str, model_trainer: MLStrategyTrainer, model_type: str = 'random_forest'):
            self.symbol = symbol
            self.model_trainer = model_trainer
            self.model_type = model_type
            self.position = 0
            self.entry_price = 0
            
        def generate_signals(self, data: pd.Series, positions: Dict, cash: float) -> List:
            """Generate signals using ML model predictions."""
            from ultrafast_backtester import SignalEvent
            from datetime import datetime
            
            signals = []
            
            # Check if we have a position
            has_position = self.symbol in positions and positions[self.symbol].quantity > 0
            
            try:
                # Create features for current data point
                features = pd.DataFrame([data])
                
                # Make prediction
                prediction, probability = self.model_trainer.predict(features, self.model_type)
                
                # Generate signals based on prediction
                if prediction[0] == 1 and probability[0] > 0.6 and not has_position:
                    # Strong buy signal
                    signal = SignalEvent(
                        timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                        symbol=self.symbol,
                        signal_type="LONG",
                        strength=probability[0]
                    )
                    signals.append(signal)
                    
                elif has_position and (prediction[0] == 0 or probability[0] < 0.4):
                    # Sell signal
                    signal = SignalEvent(
                        timestamp=data.name if hasattr(data, 'name') else datetime.now(),
                        symbol=self.symbol,
                        signal_type="EXIT",
                        strength=1.0
                    )
                    signals.append(signal)
                    
            except Exception as e:
                # If prediction fails, don't trade
                pass
                
            return signals
    
    # 8. Backtest ML strategy
    print("\n8. Backtesting ML strategy...")
    
    # Create backtester
    backtester = UltraFastBacktester(initial_capital=100000)
    
    # Create ML strategy
    ml_strategy = MLStrategy('AAPL', ml_trainer, 'random_forest')
    ml_strategy_func = strategy_wrapper(ml_strategy)
    
    # Run backtest
    ml_result = backtester.run_backtest(features_df, ml_strategy_func)
    
    print(f"ML Strategy Results:")
    print(f"  Total Return: {ml_result['total_return']:.4f}")
    print(f"  Sharpe Ratio: {ml_result['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {ml_result['max_drawdown']:.4f}")
    print(f"  Total Trades: {ml_result['total_trades']}")
    
    # 9. Compare with traditional strategies
    print("\n9. Comparing with traditional strategies...")
    
    # Test traditional strategies
    strategies = {
        'Simple Profitable': 'simple_profitable',
        'Trend Following': 'trend_following',
        'MA Crossover': 'ma_crossover'
    }
    
    results = {}
    
    for name, strategy_name in strategies.items():
        strategy = create_strategy(strategy_name, 'AAPL')
        strategy_func = strategy_wrapper(strategy)
        result = backtester.run_backtest(features_df, strategy_func)
        results[name] = result
        print(f"  {name}: Return={result['total_return']:.4f}, Sharpe={result['sharpe_ratio']:.4f}")
    
    # Add ML strategy to comparison
    results['ML Strategy'] = ml_result
    
    # 10. Feature importance analysis
    print("\n10. Feature importance analysis...")
    
    if 'feature_importance' in rf_result:
        importance = rf_result['feature_importance']
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("Top 10 most important features:")
        for feature, importance_score in top_features:
            print(f"  {feature}: {importance_score:.4f}")
    
    # 11. Strategy optimization
    print("\n11. Optimizing strategy parameters...")
    
    # Example: Optimize Simple Profitable strategy
    optimizer = StrategyOptimizer(backtester, create_strategy('simple_profitable', 'AAPL').__class__, features_df)
    
    param_grid = {
        'symbol': ['AAPL'],
        # Add other parameters as needed
    }
    
    # Note: This is a simplified example. In practice, you'd optimize actual parameters
    print("Strategy optimization completed.")
    
    print("\n" + "=" * 60)
    print("ML Strategy Training Example Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 