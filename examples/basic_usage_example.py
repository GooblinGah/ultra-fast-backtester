#!/usr/bin/env python3
"""
Example Usage of Ultra-Fast Backtester
======================================

This script demonstrates how to use the backtester with different strategies
and data sources. It shows both basic usage and advanced features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ultrafast_backtester.backtester import UltraFastBacktester
from ultrafast_backtester.strategies import create_strategy, strategy_wrapper
from ultrafast_backtester.data_loader import create_data_loader
from ultrafast_backtester.visualization import BacktestVisualizer, create_performance_report


def create_sample_data():
    """Create sample market data for demonstration."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Create realistic price movements
    np.random.seed(42)
    
    # Start with $100
    initial_price = 100.0
    
    # Generate returns with trend and volatility
    returns = np.zeros(n_days)
    
    # Add trend component
    trend = 0.0003  # Slight upward bias
    
    # Add cyclical patterns
    cycle = 0.015 * np.sin(2 * np.pi * np.arange(n_days) / 60)
    
    # Add momentum periods
    momentum = np.zeros(n_days)
    momentum[50:150] = 0.0008 * np.arange(100) / 100  # Strong uptrend
    momentum[200:300] = -0.0006 * np.arange(100) / 100  # Downtrend
    
    # Add volatility
    volatility = 0.012 + 0.008 * np.sin(2 * np.pi * np.arange(n_days) / 90)
    
    # Combine all components
    returns = trend + cycle + momentum + np.random.normal(0, volatility, n_days)
    
    # Add some extreme moves
    for i in range(0, n_days, 30):
        if i > 0:
            returns[i] += np.random.choice([0.05, -0.05])
    
    # Calculate prices
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['open'] = prices * (1 + np.random.normal(0, 0.002, n_days))
    data['high'] = np.maximum(data['open'], prices) * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
    data['low'] = np.minimum(data['open'], prices) * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
    data['close'] = prices
    data['volume'] = np.random.randint(1000000, 10000000, n_days)
    data['symbol'] = 'AAPL'
    
    # Add technical indicators
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=20).std()
    
    # Moving averages
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['ema_26'] = data['close'].ewm(span=26).mean()
    
    # RSI
    data['rsi'] = calculate_rsi(data['close'])
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['close'])
    data['bb_upper'] = bb_upper
    data['bb_middle'] = bb_middle
    data['bb_lower'] = bb_lower
    
    # Support and resistance
    data['support'] = data['low'].rolling(window=20).min()
    data['resistance'] = data['high'].rolling(window=20).max()
    
    # MACD
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    return data


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


def run_basic_example():
    """Run a basic backtest example."""
    print("Ultra-Fast Backtester - Basic Example")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample market data...")
    data = create_sample_data()
    print(f"Generated {len(data)} data points")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"Final price: ${data['close'].iloc[-1]:.2f}")
    
    # Create backtester
    backtester = UltraFastBacktester(initial_capital=100000, commission=0.001)
    
    # Create strategy
    strategy = create_strategy('simple_profitable', 'AAPL')
    strategy_func = strategy_wrapper(strategy)
    
    # Run backtest
    print("\nRunning backtest...")
    start_time = datetime.now()
    metrics = backtester.run_backtest(data, strategy_func)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"Backtest completed in {duration:.3f} seconds")
    
    # Display results
    print("\nBacktest Results:")
    print("-" * 30)
    print(f"Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return', 0)*100:.2f}%)")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)")
    print(f"Final Value: ${metrics.get('final_value', 100000):,.2f}")
    
    # Get trade details
    trade_history = backtester.get_trade_history()
    print(f"Total Trades: {len(trade_history)}")
    
    if not trade_history.empty:
        buy_trades = trade_history[trade_history['side'] == 'BUY']
        sell_trades = trade_history[trade_history['side'] == 'SELL']
        total_volume = trade_history['value'].sum()
        
        print(f"Buy Trades: {len(buy_trades)}")
        print(f"Sell Trades: {len(sell_trades)}")
        print(f"Total Volume: ${total_volume:,.2f}")
        
        # Calculate profit/loss
        profit_loss = metrics.get('final_value', 100000) - 100000
        if profit_loss > 0:
            print(f"Profit: +${profit_loss:,.2f}")
        elif profit_loss < 0:
            print(f"Loss: ${profit_loss:,.2f}")
        else:
            print("Breakeven: $0.00")
    
    return metrics, trade_history


def run_strategy_comparison():
    """Compare multiple strategies."""
    print("\n\nStrategy Comparison Example")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Define strategies to test
    strategies = {
        'Simple Profitable': 'simple_profitable',
        'Trend Following': 'trend_following',
        'Mean Reversion': 'mean_reversion',
        'Momentum RSI': 'momentum',
        'MA Crossover': 'ma_crossover',
        'Volatility Breakout': 'volatility_breakout'
    }
    
    results = {}
    
    # Test each strategy
    for strategy_name, strategy_type in strategies.items():
        print(f"\nTesting {strategy_name}...")
        
        backtester = UltraFastBacktester(initial_capital=100000, commission=0.001)
        strategy = create_strategy(strategy_type, 'AAPL')
        strategy_func = strategy_wrapper(strategy)
        
        start_time = datetime.now()
        metrics = backtester.run_backtest(data, strategy_func)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        print(f"  Completed in {duration:.3f} seconds")
        print(f"  Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return', 0)*100:.2f}%)")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)")
        
        results[strategy_name] = metrics
    
    # Find best strategy
    best_strategy = max(results.keys(), 
                       key=lambda x: results[x].get('sharpe_ratio', -999))
    
    print(f"\nStrategy Comparison Results:")
    print("-" * 40)
    print(f"Best Strategy by Sharpe Ratio: {best_strategy}")
    print(f"Best Sharpe Ratio: {results[best_strategy].get('sharpe_ratio', 0):.4f}")
    print(f"Best Total Return: {results[best_strategy].get('total_return', 0)*100:.2f}%")
    
    # Show all results in a table
    print(f"\nDetailed Results:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Return %':<10} {'Sharpe':<8} {'Drawdown %':<12} {'Final Value':<12}")
    print("-" * 80)
    
    for strategy_name, result in results.items():
        return_pct = result.get('total_return', 0) * 100
        sharpe = result.get('sharpe_ratio', 0)
        drawdown_pct = result.get('max_drawdown', 0) * 100
        final_value = result.get('final_value', 100000)
        
        if return_pct > 0:
            return_str = f"+{return_pct:>7.2f}%"
        else:
            return_str = f"{return_pct:>8.2f}%"
        
        print(f"{strategy_name:<20} {return_str:<10} {sharpe:>7.3f} {drawdown_pct:>10.2f}% ${final_value:>10,.0f}")
    
    return results


def run_parameter_optimization():
    """Demonstrate parameter optimization."""
    print("\n\nParameter Optimization Example")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Define parameter ranges to test
    rsi_periods = [10, 14, 20]
    oversold_levels = [25, 30, 35]
    overbought_levels = [65, 70, 75]
    
    best_result = None
    best_params = None
    best_sharpe = -999
    
    print("Testing parameter combinations...")
    
    for rsi_period in rsi_periods:
        for oversold in oversold_levels:
            for overbought in overbought_levels:
                # Create strategy with current parameters
                strategy = create_strategy('momentum', 'AAPL', 
                                         rsi_period=rsi_period,
                                         oversold=oversold,
                                         overbought=overbought)
                strategy_func = strategy_wrapper(strategy)
                
                # Run backtest
                backtester = UltraFastBacktester(initial_capital=100000, commission=0.001)
                metrics = backtester.run_backtest(data, strategy_func)
                
                sharpe = metrics.get('sharpe_ratio', 0)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = metrics
                    best_params = {
                        'rsi_period': rsi_period,
                        'oversold': oversold,
                        'overbought': overbought
                    }
    
    print(f"\nOptimization Results:")
    print("-" * 30)
    print(f"Best Parameters: {best_params}")
    print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
    print(f"Best Total Return: {best_result.get('total_return', 0)*100:.2f}%")
    print(f"Best Max Drawdown: {best_result.get('max_drawdown', 0)*100:.2f}%")
    
    return best_params, best_result


def main():
    """Main function to run all examples."""
    print("Ultra-Fast Event-Driven Backtester - Examples")
    print("=" * 60)
    
    # Run basic example
    metrics, trades = run_basic_example()
    
    # Run strategy comparison
    comparison_results = run_strategy_comparison()
    
    # Run parameter optimization
    best_params, best_result = run_parameter_optimization()
    
    print("\n\nAll examples completed successfully!")
    print("The backtester demonstrates:")
    print("- Fast execution with vectorized operations")
    print("- Multiple trading strategies")
    print("- Comprehensive performance metrics")
    print("- Parameter optimization capabilities")
    print("- Professional output formatting")


if __name__ == "__main__":
    main() 