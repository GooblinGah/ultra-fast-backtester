# Ultra Fast Backtester - Production Guide

## Overview

This is a production-ready event-driven backtesting framework for algorithmic trading strategies. It has been extensively tested with real market data and synthetic datasets, achieving a 97.9% test success rate.

## Installation

```bash
# Install from source
git clone https://github.com/GooblinGah/ultra-fast-backtester.git
cd ultra-fast-backtester
pip install -e .

# Install with all dependencies
pip install -e ".[dev,ml,viz]"
```

## Core Features

- Event-driven backtesting architecture
- Real market data integration via Yahoo Finance
- Multiple built-in trading strategies
- Comprehensive performance metrics
- Extensible strategy framework
- Command-line interface
- Visualization and reporting tools

## Quick Start

```python
from ultrafast_backtester import UltraFastBacktester, create_strategy, create_data_loader

# Load data
loader = create_data_loader('yahoo')
data = loader.load_data('AAPL', '2023-01-01', '2023-12-31')

# Create backtester and strategy
backtester = UltraFastBacktester(initial_capital=100000)
strategy = create_strategy('trend_following', 'AAPL')

# Run backtest
result = backtester.run_backtest(data, strategy)
print(f"Total Return: {result['total_return']:.4f}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
```

## Command Line Usage

```bash
# Run single backtest
python -m cli backtest --symbol AAPL --strategy trend_following --data-source yahoo

# Compare multiple strategies
python -m cli compare --symbol TSLA --data-source yahoo

# Optimize strategy parameters
python -m cli optimize --strategy simple_profitable --symbol NVDA
```

## Testing

```bash
# Run validation tests
python -m pytest tests/test_validation.py -v

# Run comprehensive tests
python tests/test_comprehensive.py

# Run full test suite
python -m pytest tests/ -v
```

## Built-in Strategies

- Simple Profitable: RSI + Moving Average combination
- Trend Following: Momentum-based trend detection
- MA Crossover: Golden/Death cross signals
- Mean Reversion: Bollinger Bands mean reversion
- Momentum RSI: RSI overbought/oversold signals
- Dual Thrust: Breakout strategy
- Grid Trading: Multi-level grid system
- Volatility Breakout: Volatility expansion signals

## Performance Metrics

- Total Return: Overall portfolio performance
- Annualized Return: Yearly return rate
- Sharpe Ratio: Risk-adjusted returns
- Maximum Drawdown: Largest peak-to-trough decline
- Volatility: Portfolio risk measure
- Win Rate: Percentage of profitable trades
- Total Trades: Number of executed trades

## Custom Strategies

```python
from ultrafast_backtester import BaseStrategy, SignalEvent

class MyCustomStrategy(BaseStrategy):
    def __init__(self, symbol, param1=10, param2=0.5):
        super().__init__(symbol)
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data, positions, cash):
        signals = []
        # Strategy logic here
        if buy_condition:
            signals.append(SignalEvent(
                timestamp=data.name,
                symbol=self.symbol,
                signal_type="LONG",
                strength=1.0
            ))
        return signals
```

## Production Considerations

1. **Data Validation**: The backtester handles missing data and null values automatically
2. **Error Handling**: Comprehensive error handling for edge cases
3. **Performance**: Optimized for speed with vectorized operations
4. **Memory Usage**: Efficient memory management for large datasets
5. **Scalability**: Supports multiple symbols and strategies

## Testing Results

- **Validation Tests**: 100% pass rate
- **Comprehensive Tests**: 97.9% success rate (47/48 tests passed)
- **Edge Cases**: Properly handled empty data, null values, and single-row datasets
- **Real Data**: Successfully tested with 10 major S&P 500 stocks
- **Synthetic Data**: Tested with trending, volatile, and sideways market conditions

## Dependencies

- Core: NumPy, Pandas, Numba
- Data: yfinance, python-dateutil
- ML: scikit-learn, xgboost, lightgbm
- Visualization: matplotlib, seaborn, plotly
- CLI: click, rich, tqdm

## License

MIT License - see LICENSE file for details.

## Support

- Issues: GitHub Issues
- Email: adi.siv@berkeley.edu

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough research before making investment decisions. 