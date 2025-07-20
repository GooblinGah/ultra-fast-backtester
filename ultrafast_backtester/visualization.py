"""
Visualization Module for Ultra-Fast Backtester
==============================================

This module provides comprehensive plotting and visualization capabilities
for backtesting results, performance metrics, and trading analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BacktestVisualizer:
    """Comprehensive visualization class for backtesting results."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        
    def plot_portfolio_performance(self, portfolio_history: pd.DataFrame, 
                                  benchmark_data: Optional[pd.DataFrame] = None,
                                  title: str = "Portfolio Performance") -> None:
        """Plot portfolio value over time with optional benchmark comparison."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])
        
        # Plot portfolio value
        if not portfolio_history.empty:
            portfolio_history['total_value'].plot(ax=ax1, label='Portfolio', linewidth=2)
            
            # Add benchmark if provided
            if benchmark_data is not None and not benchmark_data.empty:
                # Normalize benchmark to start at same value as portfolio
                benchmark_start = portfolio_history['total_value'].iloc[0]
                benchmark_normalized = benchmark_data['close'] / benchmark_data['close'].iloc[0] * benchmark_start
                benchmark_normalized.plot(ax=ax1, label='Benchmark', linewidth=2, alpha=0.7)
                
            ax1.set_title(title, fontsize=16, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Plot drawdown
            self._plot_drawdown(portfolio_history['total_value'], ax2)
            
        plt.tight_layout()
        plt.show()
        
    def plot_trade_analysis(self, trade_history: pd.DataFrame, 
                           portfolio_history: pd.DataFrame) -> None:
        """Plot detailed trade analysis."""
        if trade_history.empty:
            print("No trades to analyze")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Plot 1: Trade P&L distribution
        trade_pnl = []
        for _, trade in trade_history.iterrows():
            if trade['side'] == 'SELL':
                # Calculate P&L for sell trades
                # This is a simplified calculation
                pnl = trade['value'] * 0.01  # Placeholder
                trade_pnl.append(pnl)
                
        if trade_pnl:
            axes[0, 0].hist(trade_pnl, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Trade P&L Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('P&L ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
            
        # Plot 2: Trade volume over time
        trade_history['timestamp'] = pd.to_datetime(trade_history['timestamp'])
        trade_volume = trade_history.groupby(trade_history['timestamp'].dt.date)['value'].sum()
        trade_volume.plot(ax=axes[0, 1], kind='bar', alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Daily Trade Volume', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Trade Volume ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Buy vs Sell distribution
        side_counts = trade_history['side'].value_counts()
        axes[1, 0].pie(side_counts.values, labels=side_counts.index, autopct='%1.1f%%', 
                      colors=['lightblue', 'lightcoral'])
        axes[1, 0].set_title('Buy vs Sell Distribution', fontweight='bold')
        
        # Plot 4: Portfolio composition over time
        if not portfolio_history.empty:
            portfolio_history['positions_value'] = portfolio_history['total_value'] - portfolio_history['cash']
            portfolio_history[['cash', 'positions_value']].plot(ax=axes[1, 1], kind='area', 
                                                               stacked=True, alpha=0.7)
            axes[1, 1].set_title('Portfolio Composition', fontweight='bold')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Value ($)')
            axes[1, 1].legend(['Cash', 'Positions'])
            
        plt.tight_layout()
        plt.show()
        
    def plot_risk_metrics(self, portfolio_history: pd.DataFrame, 
                         returns: Optional[pd.Series] = None) -> None:
        """Plot risk and performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Calculate returns if not provided
        if returns is None and not portfolio_history.empty:
            returns = portfolio_history['total_value'].pct_change().dropna()
            
        if returns is not None and len(returns) > 0:
            # Plot 1: Returns distribution
            axes[0, 0].hist(returns, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0, 0].set_title('Returns Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('Returns')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {returns.mean():.4f}')
            axes[0, 0].legend()
            
            # Plot 2: Rolling volatility
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            rolling_vol.plot(ax=axes[0, 1], color='orange', linewidth=2)
            axes[0, 1].set_title('30-Day Rolling Volatility', fontweight='bold')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Annualized Volatility')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Rolling Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            rolling_sharpe = (returns.rolling(window=30).mean() * 252 - risk_free_rate) / rolling_vol
            rolling_sharpe.plot(ax=axes[1, 0], color='green', linewidth=2)
            axes[1, 0].set_title('30-Day Rolling Sharpe Ratio', fontweight='bold')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.7)
            
            # Plot 4: Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            drawdown.plot(ax=axes[1, 1], color='red', linewidth=2)
            axes[1, 1].set_title('Drawdown', fontweight='bold')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Drawdown')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            
        plt.tight_layout()
        plt.show()
        
    def plot_strategy_comparison(self, results: Dict[str, Dict], 
                               metrics: List[str] = None) -> None:
        """Plot comparison of multiple strategies."""
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
            
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= 4:  # Only plot first 4 metrics
                break
                
            values = [results[strategy].get(metric, 0) for strategy in results.keys()]
            strategies = list(results.keys())
            
            # Create bar plot
            bars = axes[i].bar(strategies, values, alpha=0.7, 
                              color=plt.cm.Set3(np.linspace(0, 1, len(strategies))))
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
                
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()
        
    def plot_price_with_signals(self, price_data: pd.DataFrame, 
                               signals: List, title: str = "Price Chart with Signals") -> None:
        """Plot price data with trading signals overlaid."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot price
        if 'close' in price_data.columns:
            price_data['close'].plot(ax=ax, linewidth=2, label='Price')
            
        # Add moving averages if available
        if 'sma_20' in price_data.columns:
            price_data['sma_20'].plot(ax=ax, alpha=0.7, label='SMA 20')
        if 'sma_50' in price_data.columns:
            price_data['sma_50'].plot(ax=ax, alpha=0.7, label='SMA 50')
            
        # Add Bollinger Bands if available
        if all(col in price_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            ax.fill_between(price_data.index, price_data['bb_upper'], price_data['bb_lower'], 
                           alpha=0.2, color='gray', label='Bollinger Bands')
            price_data['bb_middle'].plot(ax=ax, alpha=0.5, color='gray')
            
        # Add signals
        buy_signals = [s for s in signals if s.signal_type == "LONG"]
        sell_signals = [s for s in signals if s.signal_type in ["EXIT", "SHORT"]]
        
        if buy_signals:
            buy_times = [s.timestamp for s in buy_signals]
            buy_prices = [price_data.loc[s.timestamp, 'close'] if s.timestamp in price_data.index 
                         else price_data['close'].iloc[-1] for s in buy_signals]
            ax.scatter(buy_times, buy_prices, color='green', marker='^', s=100, 
                      label='Buy Signal', zorder=5)
            
        if sell_signals:
            sell_times = [s.timestamp for s in sell_signals]
            sell_prices = [price_data.loc[s.timestamp, 'close'] if s.timestamp in price_data.index 
                          else price_data['close'].iloc[-1] for s in sell_signals]
            ax.scatter(sell_times, sell_prices, color='red', marker='v', s=100, 
                      label='Sell Signal', zorder=5)
            
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_performance_summary(self, metrics: Dict, 
                               title: str = "Performance Summary") -> None:
        """Create a summary dashboard of key performance metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Create a summary table
        summary_data = []
        for key, value in metrics.items():
            if isinstance(value, float):
                summary_data.append([key.replace('_', ' ').title(), f'{value:.4f}'])
            else:
                summary_data.append([key.replace('_', ' ').title(), str(value)])
                
        # Plot summary table
        table = axes[0, 0].table(cellText=summary_data, colLabels=['Metric', 'Value'],
                                cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        axes[0, 0].set_title('Performance Metrics', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Plot key metrics as gauges or bars
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        colors = ['green', 'blue', 'red', 'orange']
        
        for i, (metric, color) in enumerate(zip(key_metrics, colors)):
            if metric in metrics:
                row = (i + 1) // 2
                col = (i + 1) % 2
                if row < 2 and col < 3:
                    value = metrics[metric]
                    axes[row, col].bar([metric.replace('_', ' ').title()], [value], 
                                     color=color, alpha=0.7)
                    axes[row, col].set_title(f'{metric.replace("_", " ").title()}', 
                                           fontweight='bold')
                    axes[row, col].tick_params(axis='x', rotation=45)
                    
        # Hide unused subplots
        axes[1, 2].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def _plot_drawdown(self, portfolio_values: pd.Series, ax) -> None:
        """Helper function to plot drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        drawdown.plot(ax=ax, color='red', linewidth=2)
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.set_title('Drawdown', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
    def save_plots(self, filename: str, dpi: int = 300) -> None:
        """Save the current figure to a file."""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {filename}")

def create_performance_report(metrics: Dict, trade_history: pd.DataFrame, 
                            portfolio_history: pd.DataFrame) -> str:
    """Create a comprehensive text report of backtest results."""
    report = []
    report.append("=" * 60)
    report.append("BACKTEST PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Performance Metrics
    report.append("PERFORMANCE METRICS:")
    report.append("-" * 30)
    for key, value in metrics.items():
        if isinstance(value, float):
            report.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            report.append(f"{key.replace('_', ' ').title()}: {value}")
    report.append("")
    
    # Trade Analysis
    if not trade_history.empty:
        report.append("TRADE ANALYSIS:")
        report.append("-" * 30)
        report.append(f"Total Trades: {len(trade_history)}")
        report.append(f"Buy Trades: {len(trade_history[trade_history['side'] == 'BUY'])}")
        report.append(f"Sell Trades: {len(trade_history[trade_history['side'] == 'SELL'])}")
        report.append(f"Total Volume: ${trade_history['value'].sum():,.2f}")
        report.append("")
        
    # Risk Metrics
    if not portfolio_history.empty:
        returns = portfolio_history['total_value'].pct_change().dropna()
        if len(returns) > 0:
            report.append("RISK METRICS:")
            report.append("-" * 30)
            report.append(f"Volatility: {returns.std() * np.sqrt(252):.4f}")
            report.append(f"Skewness: {returns.skew():.4f}")
            report.append(f"Kurtosis: {returns.kurtosis():.4f}")
            report.append(f"VaR (95%): {returns.quantile(0.05):.4f}")
            report.append("")
            
    return "\n".join(report) 