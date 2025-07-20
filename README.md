# Ultra Fast Backtester

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ⚠️ Development Status

This project is currently in **active development**. While functional, it's not yet production-ready.
- Not yet published to PyPI
- API may change

Feel free to try it out and provide feedback!

An event-driven backtesting framework for algorithmic trading strategies with built-in machine learning capabilities.

## Features

- **Event-Driven Architecture**: Efficient processing with JIT compilation
- **Real Market Data**: Yahoo Finance integration for historical data
- **Multiple Strategies**: 8+ built-in trading strategies
- **Machine Learning Ready**: Feature engineering and ML model training utilities
- **Professional Metrics**: Sharpe ratio, drawdown, volatility, and more
- **Easy Extension**: Simple inheritance model for custom strategies
- **CLI Interface**: Command-line tools for quick testing
- **Visualization**: Comprehensive plotting and reporting

## Installation

**Note: This package is currently in development and not yet published to PyPI.**

### From Source
```bash
git clone https://github.com/GooblinGah/ultra-fast-backtester.git
cd ultra-fast-backtester
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev,ml,viz]"
```

## Quick Start

### Basic Backtest
```python
from ultrafast_backtester import UltraFastBacktester, create_strategy, create_data_loader

# Load data
loader = create_data_loader('yahoo')
data = loader.load_data('AAPL', '2020-01-01', '2023-12-31')

# Create backtester and strategy
backtester = UltraFastBacktester(initial_capital=100000)
strategy = create_strategy('trend_following', 'AAPL')

# Run backtest
result = backtester.run_backtest(data, strategy)
print(f"Total Return: {result['total_return']:.4f}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
```

### Command Line Interface
```bash
# Run single backtest
python -m cli backtest --symbol AAPL --strategy trend_following --data-source yahoo

# Compare multiple strategies
python -m cli compare --symbol TSLA --data-source yahoo

# Optimize strategy parameters
python -m cli optimize --strategy simple_profitable --symbol NVDA
```

## Machine Learning Integration

### Feature Engineering
```python
from ultrafast_backtester import FeatureEngineer, MLStrategyTrainer

# Create technical features
feature_engineer = FeatureEngineer()
features = feature_engineer.create_technical_features(data)

# Create target variable
target = feature_engineer.create_target_variable(features, forward_period=1, threshold=0.01)
```

### Model Training
```python
# Train ML models
ml_trainer = MLStrategyTrainer()
result = ml_trainer.train_model(features, target, 'random_forest')

print(f"Accuracy: {result['metrics']['accuracy']:.3f}")
print(f"Feature Importance: {result['feature_importance']}")
```

### ML-Based Strategy
```python
class MLStrategy:
    def __init__(self, symbol, model_trainer, model_type='random_forest'):
        self.symbol = symbol
        self.model_trainer = model_trainer
        self.model_type = model_type
    
    def generate_signals(self, data, positions, cash):
        # Use ML model to generate trading signals
        prediction, probability = self.model_trainer.predict(data, self.model_type)
        # Generate signals based on prediction
        return signals
```

## Built-in Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Simple Profitable** | RSI + Moving Average combination | Trend following with momentum |
| **Trend Following** | Momentum-based trend detection | Strong trending markets |
| **MA Crossover** | Golden/Death cross signals | Medium-term trends |
| **Mean Reversion** | Bollinger Bands mean reversion | Range-bound markets |
| **Momentum RSI** | RSI overbought/oversold signals | Short-term momentum |
| **Dual Thrust** | Breakout strategy | Volatile markets |
| **Grid Trading** | Multi-level grid system | Sideways markets |
| **Volatility Breakout** | Volatility expansion signals | Breakout opportunities |

## Performance Metrics

- **Total Return**: Overall portfolio performance
- **Annualized Return**: Yearly return rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Portfolio risk measure
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of executed trades

## Custom Strategies

Create your own strategy by inheriting from `BaseStrategy`:

```python
from ultrafast_backtester import BaseStrategy, SignalEvent

class MyCustomStrategy(BaseStrategy):
    def __init__(self, symbol, param1=10, param2=0.5):
        super().__init__(symbol)
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data, positions, cash):
        signals = []
        
        # Your strategy logic here
        if buy_condition:
            signals.append(SignalEvent(
                timestamp=data.name,
                symbol=self.symbol,
                signal_type="LONG",
                strength=1.0
            ))
        
        return signals
```

## Project Structure

```
ultra-fast-backtester/
├── ultrafast_backtester/          # Main package
│   ├── __init__.py               # Package initialization
│   ├── backtester.py             # Core backtesting engine
│   ├── data_loader.py            # Data loading utilities
│   ├── strategies.py             # Trading strategies
│   ├── visualization.py          # Plotting and visualization
│   └── ml_utils.py               # Machine learning utilities
├── examples/                     # Example scripts
│   └── ml_strategy_training.py   # ML training example
├── cli.py                        # Command line interface
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Examples

### Complete ML Training Example
```bash
python examples/ml_strategy_training.py
```

### Strategy Comparison
```python
from ultrafast_backtester import plot_strategy_comparison

# Compare multiple strategies
strategies = ['simple_profitable', 'trend_following', 'ma_crossover']
results = {}

for strategy in strategies:
    result = backtester.run_backtest(data, create_strategy(strategy, 'AAPL'))
    results[strategy] = result

plot_strategy_comparison(results)
```

## Performance

- **Memory**: Efficient memory usage with vectorized operations
- **Scalability**: Supports multiple symbols and strategies
- **Accuracy**: Real market data with proper event processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Support

- **Issues**: [GitHub Issues](https://github.com/GooblinGah/ultra-fast-backtester/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GooblinGah/ultra-fast-backtester/discussions)
- **Email**: adi.siv@berkeley.edu

## Acknowledgments

- Built with [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/)
- Performance optimized with [Numba](https://numba.pydata.org/)
- Data provided by [Yahoo Finance](https://finance.yahoo.com/)
- Machine learning powered by [Scikit-learn](https://scikit-learn.org/)

---

**Disclaimer**: This software is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research before making investment decisions. 