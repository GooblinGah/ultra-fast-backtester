"""
Ultra-Fast Backtester

An ultra-fast, event-driven backtesting framework for algorithmic trading strategies.
Designed for high-performance strategy testing and machine learning model training.
"""

__version__ = "1.0.0"
__author__ = "Ultra-Fast Backtester Team"
__email__ = "contact@ultrafastbacktester.com"

# Core imports
from .backtester import UltraFastBacktester
from .data_loader import DataLoader, YahooFinanceLoader, SyntheticDataLoader, create_data_loader
from .strategies import (
    BaseStrategy,
    MovingAverageCrossoverStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    DualThrustStrategy,
    GridTradingStrategy,
    SimpleProfitableStrategy,
    TrendFollowingStrategy,
    VolatilityBreakoutStrategy,
    create_strategy,
    strategy_wrapper
)
from .visualization import BacktestVisualizer, create_performance_report
from .ml_utils import FeatureEngineer, MLStrategyTrainer, StrategyOptimizer

# Convenience imports
__all__ = [
    # Core classes
    "UltraFastBacktester",
    "DataLoader",
    "YahooFinanceLoader", 
    "SyntheticDataLoader",
    "BaseStrategy",
    
    # Strategy classes
    "MovingAverageCrossoverStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "DualThrustStrategy",
    "GridTradingStrategy",
    "SimpleProfitableStrategy",
    "TrendFollowingStrategy",
    "VolatilityBreakoutStrategy",
    
    # Utility functions
    "create_data_loader",
    "create_strategy",
    "strategy_wrapper",
    "BacktestVisualizer",
    "create_performance_report",
    
    # ML utilities
    "FeatureEngineer",
    "MLStrategyTrainer", 
    "StrategyOptimizer",
] 