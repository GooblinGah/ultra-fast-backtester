#!/usr/bin/env python3
"""
Command Line Interface for Ultra-Fast Backtester
===============================================

A simple CLI to run backtests from the command line.
"""

import click
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ultrafast_backtester import (
    UltraFastBacktester,
    create_strategy, 
    strategy_wrapper,
    create_data_loader
)
from ultrafast_backtester.visualization import BacktestVisualizer, create_performance_report

@click.group()
def cli():
    """Ultra-Fast Event-Driven Backtester CLI"""
    pass

@cli.command()
@click.option('--symbol', default='AAPL', help='Trading symbol')
@click.option('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2023-12-31', help='End date (YYYY-MM-DD)')
@click.option('--strategy', default='simple_profitable', 
              type=click.Choice(['ma_crossover', 'mean_reversion', 'momentum', 'dual_thrust', 'grid_trading', 'simple_profitable', 'trend_following', 'volatility_breakout']),
              help='Trading strategy')
@click.option('--initial-capital', default=100000, help='Initial capital')
@click.option('--commission', default=0.001, help='Commission rate')
@click.option('--data-source', default='synthetic', 
              type=click.Choice(['yahoo', 'synthetic']),
              help='Data source')
@click.option('--plot/--no-plot', default=True, help='Generate plots')
@click.option('--output', help='Output file for results')
def backtest(symbol, start_date, end_date, strategy, initial_capital, commission, data_source, plot, output):
    """Run a single backtest"""
    
    click.echo(f"Running backtest for {symbol} using {strategy} strategy")
    click.echo(f"Date range: {start_date} to {end_date}")
    click.echo(f"Initial capital: ${initial_capital:,.2f}")
    click.echo(f"Commission: {commission:.3f}")
    
    # Load data
    click.echo("Loading data...")
    loader = create_data_loader(data_source)
    data = loader.load_data(symbol, start_date, end_date)
    
    if data.empty:
        click.echo("Failed to load data")
        return
        
    click.echo(f"Loaded {len(data)} data points")
    
    # Create backtester
    backtester = UltraFastBacktester(initial_capital=initial_capital, commission=commission)
    
    # Create strategy
    strategy_instance = create_strategy(strategy, symbol)
    strategy_func = strategy_wrapper(strategy_instance)
    
    # Run backtest
    click.echo("Running backtest...")
    start_time = datetime.now()
    metrics = backtester.run_backtest(data, strategy_func)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    click.echo(f"Backtest completed in {duration:.2f} seconds")
    
    # Display results
    click.echo("\n" + "="*50)
    click.echo("BACKTEST RESULTS")
    click.echo("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            click.echo(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            click.echo(f"{key.replace('_', ' ').title()}: {value}")
    
    # Get detailed results
    trade_history = backtester.get_trade_history()
    portfolio_history = backtester.get_portfolio_history()
    
    click.echo(f"\nTotal trades: {len(trade_history)}")
    if not trade_history.empty:
        click.echo(f"Total volume: ${trade_history['value'].sum():,.2f}")
    
    # Generate plots if requested
    if plot:
        click.echo("\nGenerating plots...")
        visualizer = BacktestVisualizer()
        
        # Portfolio performance
        visualizer.plot_portfolio_performance(portfolio_history, 
                                            title=f"{strategy.title()} Strategy Performance")
        
        # Trade analysis
        if not trade_history.empty:
            visualizer.plot_trade_analysis(trade_history, portfolio_history)
        
        # Risk metrics
        visualizer.plot_risk_metrics(portfolio_history)
    
    # Save results if output file specified
    if output:
        click.echo(f"\nSaving results to {output}...")
        report = create_performance_report(metrics, trade_history, portfolio_history)
        with open(output, 'w') as f:
            f.write(report)
        click.echo(f"Results saved to {output}")
    
    click.echo("\nBacktest completed.")

@cli.command()
@click.option('--symbol', default='AAPL', help='Trading symbol')
@click.option('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2023-12-31', help='End date (YYYY-MM-DD)')
@click.option('--initial-capital', default=100000, help='Initial capital')
@click.option('--data-source', default='synthetic', 
              type=click.Choice(['yahoo', 'synthetic']),
              help='Data source')
def compare(symbol, start_date, end_date, initial_capital, data_source):
    """Compare multiple strategies"""
    
    click.echo(f"Comparing strategies for {symbol}")
    click.echo(f"Date range: {start_date} to {end_date}")
    
    # Load data
    click.echo("Loading data...")
    loader = create_data_loader(data_source)
    data = loader.load_data(symbol, start_date, end_date)
    
    if data.empty:
        click.echo("Failed Failed to load data")
        return
    
    # Define strategies to test
    strategies = {
        'Simple Profitable': 'simple_profitable',
        'Trend Following': 'trend_following',
        'MA Crossover': 'ma_crossover',
        'Mean Reversion': 'mean_reversion',
        'Momentum RSI': 'momentum',
        'Dual Thrust': 'dual_thrust',
        'Grid Trading': 'grid_trading',
        'Volatility Breakout': 'volatility_breakout'
    }
    
    results = {}
    
    # Test each strategy
    for strategy_name, strategy_type in strategies.items():
        click.echo(f"\nTesting {strategy_name}...")
        
        backtester = UltraFastBacktester(initial_capital=initial_capital)
        strategy = create_strategy(strategy_type, symbol)
        strategy_func = strategy_wrapper(strategy)
        
        start_time = datetime.now()
        metrics = backtester.run_backtest(data, strategy_func)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        click.echo(f"  Completed in {duration:.2f}s")
        click.echo(f"  Total Return: {metrics.get('total_return', 0):.4f}")
        click.echo(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        
        results[strategy_name] = metrics
    
    # Find best strategy
    best_strategy = max(results.keys(), 
                       key=lambda x: results[x].get('sharpe_ratio', -999))
    
    click.echo("\n" + "="*50)
    click.echo("STRATEGY COMPARISON RESULTS")
    click.echo("="*50)
    click.echo(f"Best strategy by Sharpe ratio: {best_strategy}")
    click.echo(f"Best Sharpe ratio: {results[best_strategy].get('sharpe_ratio', 0):.4f}")
    
    # Generate comparison plot
    click.echo("\nGenerating comparison plot...")
    visualizer = BacktestVisualizer()
    visualizer.plot_strategy_comparison(results)
    
    click.echo("\nStrategy comparison completed.")

@cli.command()
@click.option('--symbol', default='AAPL', help='Trading symbol')
@click.option('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2023-12-31', help='End date (YYYY-MM-DD)')
@click.option('--strategy', default='momentum', 
              type=click.Choice(['ma_crossover', 'mean_reversion', 'momentum', 'dual_thrust', 'simple_profitable', 'trend_following', 'volatility_breakout']),
              help='Strategy to optimize')
@click.option('--initial-capital', default=100000, help='Initial capital')
@click.option('--data-source', default='synthetic', 
              type=click.Choice(['yahoo', 'synthetic']),
              help='Data source')
def optimize(symbol, start_date, end_date, strategy, initial_capital, data_source):
    """Optimize strategy parameters"""
    
    click.echo(f"Optimizing {strategy} strategy for {symbol}")
    
    # Load data
    click.echo("Loading data...")
    loader = create_data_loader(data_source)
    data = loader.load_data(symbol, start_date, end_date)
    
    if data.empty:
        click.echo("Failed Failed to load data")
        return
    
    # Define parameter ranges based on strategy
    if strategy == 'ma_crossover':
        param_ranges = {
            'short_window': [10, 15, 20, 25, 30],
            'long_window': [40, 50, 60, 70, 80]
        }
    elif strategy == 'mean_reversion':
        param_ranges = {
            'window': [10, 15, 20, 25, 30],
            'num_std': [1.5, 2.0, 2.5, 3.0]
        }
    elif strategy == 'momentum':
        param_ranges = {
            'rsi_period': [10, 14, 20, 30],
            'oversold': [20, 25, 30, 35],
            'overbought': [65, 70, 75, 80]
        }
    elif strategy == 'dual_thrust':
        param_ranges = {
            'period': [10, 15, 20, 25, 30],
            'k1': [0.5, 0.7, 1.0, 1.2],
            'k2': [0.5, 0.7, 1.0, 1.2]
        }
    
    results = {}
    total_combinations = 1
    for param_list in param_ranges.values():
        total_combinations *= len(param_list)
    
    click.echo(f"Testing {total_combinations} parameter combinations...")
    
    # Generate parameter combinations
    import itertools
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))
        
        # Skip invalid combinations
        if strategy == 'ma_crossover' and params['short_window'] >= params['long_window']:
            continue
        
        param_key = f"{strategy}({','.join(f'{k}={v}' for k, v in params.items())})"
        click.echo(f"  Testing {param_key}...")
        
        backtester = UltraFastBacktester(initial_capital=initial_capital)
        strategy_instance = create_strategy(strategy, symbol, **params)
        strategy_func = strategy_wrapper(strategy_instance)
        
        metrics = backtester.run_backtest(data, strategy_func)
        results[param_key] = metrics
    
    # Find best parameters
    best_params = max(results.keys(), 
                     key=lambda x: results[x].get('sharpe_ratio', -999))
    
    click.echo("\n" + "="*50)
    click.echo("OPTIMIZATION RESULTS")
    click.echo("="*50)
    click.echo(f"Best parameters: Best parameters: {best_params}")
    click.echo(f"Best Sharpe ratio: Best Sharpe ratio: {results[best_params].get('sharpe_ratio', 0):.4f}")
    click.echo(f"Best total return: Best total return: {results[best_params].get('total_return', 0):.4f}")
    
    # Generate optimization plot
    click.echo("\nGenerating optimization results...")
    visualizer = BacktestVisualizer()
    visualizer.plot_strategy_comparison(results, 
                                      metrics=['sharpe_ratio', 'total_return', 'max_drawdown'])
    
    click.echo("\nParameter optimization completed! Parameter optimization completed!")

@cli.command()
def info():
    """Show system information and available features"""
    
    click.echo("Ultra-Fast Event-Driven Backtester")
    click.echo("=" * 50)
    click.echo()
    click.echo("Available Strategies:")
    click.echo("  • Moving Average Crossover")
    click.echo("  • Mean Reversion (Bollinger Bands)")
    click.echo("  • Momentum (RSI)")
    click.echo("  • Dual Thrust")
    click.echo("  • Grid Trading")
    click.echo()
    click.echo("Data Sources:")
    click.echo("  • Yahoo Finance (real market data)")
    click.echo("  • Synthetic data (for testing)")
    click.echo()
    click.echo("Visualization Features:")
    click.echo("  • Portfolio performance charts")
    click.echo("  • Trade analysis")
    click.echo("  • Risk metrics")
    click.echo("  • Strategy comparison")
    click.echo()
    click.echo("Performance Features:")
    click.echo("  • Vectorized operations with NumPy")
    click.echo("  • JIT compilation with Numba")
    click.echo("  • Event-driven architecture")
    click.echo()
    click.echo("Example Usage:")
    click.echo("  python cli.py backtest --symbol AAPL --strategy ma_crossover")
    click.echo("  python cli.py compare --symbol AAPL")
    click.echo("  python cli.py optimize --strategy ma_crossover")

if __name__ == '__main__':
    cli() 