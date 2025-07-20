"""
Ultra Fast Backtester

An event-driven backtesting framework for algorithmic trading strategies.
"""

__version__ = "1.0.0"
__author__ = "Adi Sivahuma"
__email__ = "adi.siv@berkeley.edu"

from .backtester import (
    UltraFastBacktester,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    EventType,
    OrderSide,
    SignalType,
    OrderType,
    Position
)

from .data_loader import create_data_loader
from .strategies import create_strategy, strategy_wrapper
from .visualization import BacktestVisualizer, create_performance_report
# ML utilities not yet implemented
# from .ml_utils import FeatureEngineer, MLStrategyTrainer

__all__ = [
    'UltraFastBacktester',
    'MarketEvent',
    'SignalEvent', 
    'OrderEvent',
    'FillEvent',
    'EventType',
    'OrderSide',
    'SignalType',
    'OrderType',
    'Position',
    'create_data_loader',
    'create_strategy',
    'strategy_wrapper',
    'BacktestVisualizer',
    'create_performance_report',
    # 'FeatureEngineer',
    # 'MLStrategyTrainer'
] 