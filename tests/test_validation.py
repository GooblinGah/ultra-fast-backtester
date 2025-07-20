#!/usr/bin/env python3
"""
Quick Validation Test for Backtester
====================================

This script runs a quick validation of the backtester to ensure it works
before running the full extensive testing suite.
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings

from ultrafast_backtester.backtester import (
    UltraFastBacktester, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    EventType, OrderSide, SignalType, OrderType, Position
)

warnings.filterwarnings('ignore')

def create_test_data():
    """Create simple test data."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Create realistic price movements
    np.random.seed(42)
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_days)  # Slight upward bias
    prices = initial_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.01, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
        'volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    return data

def create_test_strategies():
    """Create simple test strategies."""
    strategies = {}
    
    # Buy and hold
    def buy_and_hold(data, positions, cash):
        if not positions:
            return [{
                'timestamp': data.name,
                'symbol': 'STOCK',
                'signal_type': 'LONG',
                'strength': 1.0
            }]
        return []
    strategies['buy_and_hold'] = buy_and_hold
    
    # Simple moving average
    def ma_strategy(data, positions, cash):
        if len(data) < 20:
            return []
        
        sma_short = data['close'].rolling(window=5).mean().iloc[-1]
        sma_long = data['close'].rolling(window=20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if sma_short > sma_long and current_price > sma_short:
            return [{
                'timestamp': data.name,
                'symbol': 'STOCK',
                'signal_type': 'LONG',
                'strength': 0.8
            }]
        elif sma_short < sma_long and current_price < sma_short:
            return [{
                'timestamp': data.name,
                'symbol': 'STOCK',
                'signal_type': 'EXIT',
                'strength': 1.0
            }]
        return []
    strategies['ma_strategy'] = ma_strategy
    
    return strategies

def test_basic_functionality():
    """Test basic backtester functionality."""
    print("Testing basic functionality...")
    
    # Create test data
    data = create_test_data()
    strategies = create_test_strategies()
    
    # Test with different parameters
    test_cases = [
        {
            'name': 'Basic Test',
            'params': {
                'initial_capital': 100000,
                'commission': 0.001,
                'max_position_size': 0.1
            }
        },
        {
            'name': 'High Commission Test',
            'params': {
                'initial_capital': 100000,
                'commission': 0.01,
                'max_position_size': 0.1
            }
        },
        {
            'name': 'Small Capital Test',
            'params': {
                'initial_capital': 10000,
                'commission': 0.001,
                'max_position_size': 0.1
            }
        },
        {
            'name': 'Large Position Test',
            'params': {
                'initial_capital': 100000,
                'commission': 0.001,
                'max_position_size': 0.5
            }
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\nRunning {test_case['name']}...")
        
        for strategy_name, strategy in strategies.items():
            try:
                start_time = time.time()
                
                backtester = UltraFastBacktester(**test_case['params'])
                result = backtester.run_backtest(data, strategy)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                trade_history = backtester.get_trade_history()
                portfolio_history = backtester.get_portfolio_history()
                
                test_result = {
                    'test_name': f"{test_case['name']}_{strategy_name}",
                    'parameters': test_case['params'],
                    'execution_time': execution_time,
                    'data_points': len(data),
                    'results': result,
                    'total_trades': len(trade_history),
                    'portfolio_records': len(portfolio_history),
                    'success': True,
                    'error': None
                }
                
                results.append(test_result)
                
                print(f"  {strategy_name}: SUCCESS - {execution_time:.3f}s")
                print(f"    Total Return: {result.get('total_return', 0)*100:.2f}%")
                print(f"    Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
                print(f"    Total Trades: {len(trade_history)}")
                
            except Exception as e:
                test_result = {
                    'test_name': f"{test_case['name']}_{strategy_name}",
                    'parameters': test_case['params'],
                    'execution_time': 0,
                    'data_points': len(data),
                    'results': {},
                    'total_trades': 0,
                    'portfolio_records': 0,
                    'success': False,
                    'error': str(e)
                }
                
                results.append(test_result)
                print(f"  {strategy_name}: FAILED - {e}")
    
    return results

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    strategies = create_test_strategies()
    base_params = {
        'initial_capital': 100000,
        'commission': 0.001,
        'max_position_size': 0.1
    }
    
    edge_cases = [
        {
            'name': 'Empty Dataset',
            'data': pd.DataFrame(),
            'expected_error': True
        },
        {
            'name': 'Single Row Dataset',
            'data': pd.DataFrame({'close': [100]}, index=[pd.Timestamp('2023-01-01')]),
            'expected_error': False
        },
        {
            'name': 'Missing Close Column',
            'data': pd.DataFrame({'open': [100, 101], 'high': [102, 103]}),
            'expected_error': True
        },
        {
            'name': 'Null Values',
            'data': pd.DataFrame({'close': [100, np.nan, 102]}),
            'expected_error': False
        }
    ]
    
    results = []
    
    for case in edge_cases:
        print(f"\nTesting {case['name']}...")
        
        try:
            backtester = UltraFastBacktester(**base_params)
            result = backtester.run_backtest(case['data'], strategies['buy_and_hold'])
            
            if case['expected_error']:
                print(f"  FAILED: Expected error but got success")
                results.append({
                    'test_name': f"edge_{case['name']}",
                    'success': False,
                    'error': 'Expected error but got success'
                })
            else:
                print(f"  SUCCESS: Handled correctly")
                results.append({
                    'test_name': f"edge_{case['name']}",
                    'success': True,
                    'error': None
                })
                
        except Exception as e:
            if case['expected_error']:
                print(f"  SUCCESS: Correctly handled error - {e}")
                results.append({
                    'test_name': f"edge_{case['name']}",
                    'success': True,
                    'error': None
                })
            else:
                print(f"  FAILED: Unexpected error - {e}")
                results.append({
                    'test_name': f"edge_{case['name']}",
                    'success': False,
                    'error': str(e)
                })
    
    return results

def test_parameter_sensitivity():
    """Test parameter sensitivity."""
    print("\nTesting parameter sensitivity...")
    
    data = create_test_data()
    strategies = create_test_strategies()
    
    # Test different parameter combinations
    param_tests = [
        {'commission': 0.0005},
        {'commission': 0.002},
        {'max_position_size': 0.05},
        {'max_position_size': 0.2},
        {'slippage': 0.00005},
        {'slippage': 0.0002},
        {'stop_loss_pct': 0.02},
        {'stop_loss_pct': 0.1}
    ]
    
    results = []
    
    for i, param_test in enumerate(param_tests):
        print(f"\nParameter test {i+1}: {param_test}")
        
        base_params = {
            'initial_capital': 100000,
            'commission': 0.001,
            'max_position_size': 0.1,
            'slippage': 0.0001,
            'stop_loss_pct': 0.05
        }
        base_params.update(param_test)
        
        try:
            start_time = time.time()
            
            backtester = UltraFastBacktester(**base_params)
            result = backtester.run_backtest(data, strategies['ma_strategy'])
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            results.append({
                'test_name': f"param_test_{i+1}",
                'parameters': base_params,
                'execution_time': execution_time,
                'results': result,
                'success': True,
                'error': None
            })
            
            print(f"  SUCCESS: {execution_time:.3f}s")
            print(f"    Total Return: {result.get('total_return', 0)*100:.2f}%")
            
        except Exception as e:
            results.append({
                'test_name': f"param_test_{i+1}",
                'parameters': base_params,
                'execution_time': 0,
                'results': {},
                'success': False,
                'error': str(e)
            })
            
            print(f"  FAILED: {e}")
    
    return results

def run_validation():
    """Run the complete quick validation suite."""
    print("="*60)
    print("VALIDATION TEST SUITE")
    print("="*60)
    
    all_results = []
    
    # Test basic functionality
    basic_results = test_basic_functionality()
    all_results.extend(basic_results)
    
    # Test edge cases
    edge_results = test_edge_cases()
    all_results.extend(edge_results)
    
    # Test parameter sensitivity
    param_results = test_parameter_sensitivity()
    all_results.extend(param_results)
    
    # Generate summary
    successful_tests = [r for r in all_results if r['success']]
    failed_tests = [r for r in all_results if not r['success']]
    
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {len(all_results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {len(successful_tests)/len(all_results)*100:.1f}%")
    
    if successful_tests:
        execution_times = [r['execution_time'] for r in successful_tests if 'execution_time' in r]
        if execution_times:
            print(f"\nPerformance:")
            print(f"Average Execution Time: {np.mean(execution_times):.3f}s")
            print(f"Min Execution Time: {np.min(execution_times):.3f}s")
            print(f"Max Execution Time: {np.max(execution_times):.3f}s")
    
    if failed_tests:
        print(f"\nFailed Tests:")
        for test in failed_tests:
            print(f"  {test['test_name']}: {test['error']}")
    
    # Determine if ready for extensive testing
    success_rate = len(successful_tests) / len(all_results)
    if success_rate >= 0.9:  # 90% success rate
        print(f"\nREADY FOR EXTENSIVE TESTING")
        print(f"Success rate {success_rate*100:.1f}% meets threshold (90%)")
        return True
    else:
        print(f"\nNOT READY FOR EXTENSIVE TESTING")
        print(f"Success rate {success_rate*100:.1f}% below threshold (90%)")
        return False

if __name__ == "__main__":
    ready = run_validation()
    if ready:
        print("\nProceeding to extensive testing...")
        # You can call the extensive testing here
    else:
        print("\nPlease fix issues before running extensive tests.") 