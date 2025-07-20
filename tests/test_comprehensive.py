#!/usr/bin/env python3
"""
Focused Extensive Backtester Testing
====================================

This script performs focused but comprehensive testing of the backtester with:
- Real market data from major stocks
- Synthetic data with various patterns
- Different parameter combinations
- Edge cases and stress tests
- Performance benchmarks
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
import json
import os
from typing import Dict, List, Tuple, Any

from ultrafast_backtester.backtester import (
    UltraFastBacktester, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    EventType, OrderSide, SignalType, OrderType, Position
)

warnings.filterwarnings('ignore')

class FocusedBacktesterTester:
    """Focused comprehensive testing class for the backtester."""
    
    def __init__(self):
        self.results = []
        self.test_data = {}
        
    def create_synthetic_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create various synthetic datasets for testing."""
        datasets = {}
        
        # 1. Trending dataset (upward trend)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_days = len(dates)
        
        # Upward trend with noise
        trend = np.linspace(100, 150, n_days)
        noise = np.random.normal(0, 2, n_days)
        prices = trend + noise
        
        datasets['trending_up'] = pd.DataFrame({
            'close': prices,
            'open': prices * (1 + np.random.normal(0, 0.01, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
            'volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        # 2. Trending dataset (downward trend)
        trend_down = np.linspace(150, 100, n_days)
        noise = np.random.normal(0, 2, n_days)
        prices_down = trend_down + noise
        
        datasets['trending_down'] = pd.DataFrame({
            'close': prices_down,
            'open': prices_down * (1 + np.random.normal(0, 0.01, n_days)),
            'high': prices_down * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
            'low': prices_down * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
            'volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        # 3. Volatile dataset (high volatility)
        np.random.seed(42)
        returns = np.random.normal(0, 0.03, n_days)  # 3% daily volatility
        prices_vol = 100 * np.exp(np.cumsum(returns))
        
        datasets['volatile'] = pd.DataFrame({
            'close': prices_vol,
            'open': prices_vol * (1 + np.random.normal(0, 0.01, n_days)),
            'high': prices_vol * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
            'low': prices_vol * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
            'volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        # 4. Sideways dataset (range-bound)
        np.random.seed(123)
        base_price = 100
        prices_sideways = base_price + 10 * np.sin(np.linspace(0, 4*np.pi, n_days)) + np.random.normal(0, 1, n_days)
        
        datasets['sideways'] = pd.DataFrame({
            'close': prices_sideways,
            'open': prices_sideways * (1 + np.random.normal(0, 0.01, n_days)),
            'high': prices_sideways * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
            'low': prices_sideways * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
            'volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        print(f"Created {len(datasets)} synthetic datasets")
        return datasets
    
    def download_real_data(self, symbols: List[str], start_date: str = '2023-01-01', 
                          end_date: str = '2023-12-31') -> Dict[str, pd.DataFrame]:
        """Download real market data for given symbols."""
        real_data = {}
        
        print(f"Downloading data for {len(symbols)} symbols...")
        
        # Column mapping for yfinance data
        column_mapping = {
            'Close': 'close',
            'Open': 'open', 
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        }
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty and len(data) > 30:  # At least 30 days of data
                    # Rename columns to lowercase
                    data = data.rename(columns=column_mapping)
                    real_data[symbol] = data
                    print(f"Downloaded {symbol}: {len(data)} days")
                else:
                    print(f"Skipped {symbol}: insufficient data")
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
        
        print(f"Successfully downloaded data for {len(real_data)} symbols")
        return real_data
    
    def create_test_strategies(self) -> Dict[str, callable]:
        """Create various test strategies."""
        strategies = {}
        
        # 1. Buy and hold strategy
        def buy_and_hold(data, positions, cash):
            if not positions:  # No positions yet
                return [{
                    'timestamp': data.name,
                    'symbol': 'STOCK',
                    'signal_type': 'LONG',
                    'strength': 1.0
                }]
            return []
        strategies['buy_and_hold'] = buy_and_hold
        
        # 1b. Buy and hold strategy for real data
        def buy_and_hold_real(data, positions, cash):
            # Get symbol from data if available, otherwise use 'STOCK'
            symbol = getattr(data, 'name', 'STOCK')
            if isinstance(symbol, str) and symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'JPM', 'JNJ']:
                target_symbol = symbol
            else:
                target_symbol = 'STOCK'
                
            if target_symbol not in positions:  # No position in this symbol
                return [{
                    'timestamp': data.name,
                    'symbol': target_symbol,
                    'signal_type': 'LONG',
                    'strength': 1.0
                }]
            return []
        strategies['buy_and_hold_real'] = buy_and_hold_real
        
        # 1c. Generic buy and hold strategy that uses the data's symbol
        def buy_and_hold_generic(data, positions, cash):
            # Use the symbol from the data's name attribute
            symbol = getattr(data, 'name', 'STOCK')
            if isinstance(symbol, str):
                target_symbol = symbol
            else:
                target_symbol = 'STOCK'
                
            if target_symbol not in positions:  # No position in this symbol
                return [{
                    'timestamp': data.name,
                    'symbol': target_symbol,
                    'signal_type': 'LONG',
                    'strength': 1.0
                }]
            return []
        strategies['buy_and_hold_generic'] = buy_and_hold_generic
        
        # 2. Simple moving average crossover
        def ma_crossover(data, positions, cash):
            if len(data) < 20:  # Need enough data
                return []
            
            # Calculate moving averages
            sma_short = data['close'].rolling(window=5).mean().iloc[-1]
            sma_long = data['close'].rolling(window=20).mean().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            signals = []
            
            # Buy signal: short MA crosses above long MA
            if sma_short > sma_long and current_price > sma_short:
                signals.append({
                    'timestamp': data.name,
                    'symbol': 'STOCK',
                    'signal_type': 'LONG',
                    'strength': 0.8
                })
            
            # Sell signal: short MA crosses below long MA
            elif sma_short < sma_long and current_price < sma_short:
                signals.append({
                    'timestamp': data.name,
                    'symbol': 'STOCK',
                    'signal_type': 'EXIT',
                    'strength': 1.0
                })
            
            return signals
        strategies['ma_crossover'] = ma_crossover
        
        # 3. RSI strategy
        def rsi_strategy(data, positions, cash):
            if len(data) < 14:  # Need enough data for RSI
                return []
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            signals = []
            
            # Buy signal: RSI oversold
            if current_rsi < 30:
                signals.append({
                    'timestamp': data.name,
                    'symbol': 'STOCK',
                    'signal_type': 'LONG',
                    'strength': 0.9
                })
            
            # Sell signal: RSI overbought
            elif current_rsi > 70:
                signals.append({
                    'timestamp': data.name,
                    'symbol': 'STOCK',
                    'signal_type': 'EXIT',
                    'strength': 1.0
                })
            
            return signals
        strategies['rsi'] = rsi_strategy
        
        print(f"Created {len(strategies)} test strategies")
        return strategies
    
    def create_parameter_combinations(self) -> List[Dict]:
        """Create various parameter combinations for testing."""
        combinations = []
        
        # Base parameters
        base_params = {
            'initial_capital': 100000,
            'commission': 0.001,
            'max_position_size': 0.1,
            'max_portfolio_history': 10000,
            'slippage': 0.0001,
            'bid_ask_spread': 0.0002,
            'max_drawdown_limit': 0.2,
            'stop_loss_pct': 0.05
        }
        
        # Test different capital amounts
        for capital in [10000, 50000, 100000, 500000]:
            params = base_params.copy()
            params['initial_capital'] = capital
            combinations.append(params)
        
        # Test different commission rates
        for commission in [0.0005, 0.001, 0.002]:
            params = base_params.copy()
            params['commission'] = commission
            combinations.append(params)
        
        # Test different position sizes
        for pos_size in [0.05, 0.1, 0.2]:
            params = base_params.copy()
            params['max_position_size'] = pos_size
            combinations.append(params)
        
        # Test different slippage values
        for slippage in [0.00005, 0.0001, 0.0002]:
            params = base_params.copy()
            params['slippage'] = slippage
            combinations.append(params)
        
        # Test different stop loss percentages
        for stop_loss in [0.02, 0.05, 0.1]:
            params = base_params.copy()
            params['stop_loss_pct'] = stop_loss
            combinations.append(params)
        
        print(f"Created {len(combinations)} parameter combinations")
        return combinations
    
    def run_single_test(self, data: pd.DataFrame, strategy: callable, 
                       params: Dict, test_name: str) -> Dict:
        """Run a single backtest with given parameters."""
        try:
            start_time = time.time()
            
            # Create backtester with parameters
            backtester = UltraFastBacktester(**params)
            
            # Run backtest
            results = backtester.run_backtest(data, strategy)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Get additional metrics
            trade_history = backtester.get_trade_history()
            portfolio_history = backtester.get_portfolio_history()
            
            test_result = {
                'test_name': test_name,
                'parameters': params,
                'execution_time': execution_time,
                'data_points': len(data),
                'results': results,
                'total_trades': len(trade_history),
                'portfolio_records': len(portfolio_history),
                'success': True,
                'error': None
            }
            
            return test_result
            
        except Exception as e:
            return {
                'test_name': test_name,
                'parameters': params,
                'execution_time': 0,
                'data_points': len(data),
                'results': {},
                'total_trades': 0,
                'portfolio_records': 0,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_tests(self):
        """Run comprehensive testing suite."""
        print("Starting comprehensive backtester testing...")
        
        # Major stocks for real data testing
        major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'JPM', 'JNJ']
        
        # Create synthetic datasets
        synthetic_datasets = self.create_synthetic_datasets()
        
        # Download real data
        real_data = self.download_real_data(major_stocks)
        
        # Create strategies
        strategies = self.create_test_strategies()
        
        # Create parameter combinations
        param_combinations = self.create_parameter_combinations()
        
        # Store all test data
        self.test_data = {
            'synthetic': synthetic_datasets,
            'real': real_data,
            'strategies': strategies,
            'parameters': param_combinations
        }
        
        # Run tests
        all_results = []
        
        # Test 1: Synthetic data with all strategies and base parameters
        print("\n1. Testing synthetic data with all strategies...")
        base_params = {
            'initial_capital': 100000,
            'commission': 0.001,
            'max_position_size': 0.1,
            'max_portfolio_history': 10000,
            'slippage': 0.0001,
            'bid_ask_spread': 0.0002,
            'max_drawdown_limit': 0.2,
            'stop_loss_pct': 0.05
        }
        
        for dataset_name, dataset in synthetic_datasets.items():
            # Set the dataset name as the symbol
            dataset.name = dataset_name
            for strategy_name, strategy in strategies.items():
                test_name = f"synthetic_{dataset_name}_{strategy_name}"
                result = self.run_single_test(dataset, strategy, base_params, test_name)
                all_results.append(result)
                print(f"  {test_name}: {'SUCCESS' if result['success'] else 'FAILED'}")
                if result['success']:
                    print(f"    Return: {result['results'].get('total_return', 0)*100:.2f}%, "
                          f"Trades: {result['total_trades']}, Time: {result['execution_time']:.3f}s")
        
        # Test 2: Real data with buy and hold strategy
        print("\n2. Testing real data with buy and hold strategy...")
        for symbol, data in real_data.items():
            # Add symbol name to data for strategy to use
            data.name = symbol
            test_name = f"real_{symbol}_buy_and_hold"
            result = self.run_single_test(data, strategies['buy_and_hold_real'], base_params, test_name)
            all_results.append(result)
            print(f"  {test_name}: {'SUCCESS' if result['success'] else 'FAILED'}")
            if result['success']:
                print(f"    Return: {result['results'].get('total_return', 0)*100:.2f}%, "
                      f"Trades: {result['total_trades']}, Time: {result['execution_time']:.3f}s")
        
        # Test 3: Parameter sensitivity with trending dataset
        print("\n3. Testing parameter sensitivity...")
        trending_data = synthetic_datasets['trending_up']
        for i, params in enumerate(param_combinations[:15]):  # Limit to first 15 for speed
            test_name = f"param_test_{i}_ma_crossover"
            result = self.run_single_test(trending_data, strategies['ma_crossover'], params, test_name)
            all_results.append(result)
            print(f"  {test_name}: {'SUCCESS' if result['success'] else 'FAILED'}")
            if result['success']:
                print(f"    Return: {result['results'].get('total_return', 0)*100:.2f}%, "
                      f"Time: {result['execution_time']:.3f}s")
        
        # Test 4: Edge cases
        print("\n4. Testing edge cases...")
        
        # Empty dataset
        empty_data = pd.DataFrame()
        result = self.run_single_test(empty_data, strategies['buy_and_hold'], base_params, "edge_empty_data")
        all_results.append(result)
        print(f"  edge_empty_data: {'SUCCESS' if result['success'] else 'FAILED'}")
        
        # Single row dataset
        single_row = pd.DataFrame({'close': [100]}, index=[pd.Timestamp('2023-01-01')])
        result = self.run_single_test(single_row, strategies['buy_and_hold'], base_params, "edge_single_row")
        all_results.append(result)
        print(f"  edge_single_row: {'SUCCESS' if result['success'] else 'FAILED'}")
        
        # Dataset with null values
        null_data = synthetic_datasets['sideways'].copy()
        null_data.loc[null_data.index[10], 'close'] = np.nan
        result = self.run_single_test(null_data, strategies['buy_and_hold'], base_params, "edge_null_values")
        all_results.append(result)
        print(f"  edge_null_values: {'SUCCESS' if result['success'] else 'FAILED'}")
        
        # Store results
        self.results = all_results
        
        # Generate summary
        self.generate_test_summary()
        
        return all_results
    
    def generate_test_summary(self):
        """Generate comprehensive test summary."""
        if not self.results:
            print("No test results to summarize")
            return
        
        successful_tests = [r for r in self.results if r['success']]
        failed_tests = [r for r in self.results if not r['success']]
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {len(self.results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Success Rate: {len(successful_tests)/len(self.results)*100:.1f}%")
        
        if successful_tests:
            # Performance metrics
            execution_times = [r['execution_time'] for r in successful_tests]
            total_returns = [r['results'].get('total_return', 0) for r in successful_tests]
            sharpe_ratios = [r['results'].get('sharpe_ratio', 0) for r in successful_tests]
            max_drawdowns = [r['results'].get('max_drawdown', 0) for r in successful_tests]
            
            print(f"\nPerformance Metrics (Successful Tests):")
            print(f"Average Execution Time: {np.mean(execution_times):.3f}s")
            print(f"Min Execution Time: {np.min(execution_times):.3f}s")
            print(f"Max Execution Time: {np.max(execution_times):.3f}s")
            print(f"Average Total Return: {np.mean(total_returns)*100:.2f}%")
            print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.3f}")
            print(f"Average Max Drawdown: {np.mean(max_drawdowns)*100:.2f}%")
            
            # Strategy performance analysis
            print(f"\nStrategy Performance Analysis:")
            strategy_results = {}
            for result in successful_tests:
                if 'synthetic' in result['test_name'] or 'real' in result['test_name']:
                    strategy_name = result['test_name'].split('_')[-1]
                    if strategy_name not in strategy_results:
                        strategy_results[strategy_name] = []
                    strategy_results[strategy_name].append(result['results'].get('total_return', 0))
            
            for strategy, returns in strategy_results.items():
                avg_return = np.mean(returns) * 100
                print(f"  {strategy}: {avg_return:.2f}% average return")
        
        if failed_tests:
            print(f"\nFailed Tests:")
            for test in failed_tests:
                print(f"  {test['test_name']}: {test['error']}")
        
        # Save detailed results
        self.save_test_results()
    
    def save_test_results(self):
        """Save detailed test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_test_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {
                'test_name': result['test_name'],
                'parameters': result['parameters'],
                'execution_time': result['execution_time'],
                'data_points': result['data_points'],
                'results': result['results'],
                'total_trades': result['total_trades'],
                'portfolio_records': result['portfolio_records'],
                'success': result['success'],
                'error': result['error']
            }
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {filename}")


def run_comprehensive_tests():
    """Main function to run comprehensive tests."""
    tester = FocusedBacktesterTester()
    results = tester.run_comprehensive_tests()
    return results


if __name__ == "__main__":
    run_comprehensive_tests() 