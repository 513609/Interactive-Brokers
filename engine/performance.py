import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_report(portfolio_history_df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
    """
    Calculates all performance metrics and generates a detailed summary plot.
    """
    # --- METRIC CALCULATIONS ---
    initial_value = portfolio_history_df['value'].iloc[0]
    final_value = portfolio_history_df['value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100

    running_max = portfolio_history_df['value'].cummax()
    drawdown = (portfolio_history_df['value'] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    daily_returns = portfolio_history_df['value'].pct_change().dropna()
    sharpe_ratio = 0
    if daily_returns.std() > 0:
        # Assuming 0% risk-free rate
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    # --- Print Report ---
    print("\n" + "="*50)
    print(" Backtest Performance Report ".center(50, "="))
    print("="*50)
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Total Return:            {total_return:.2f}%")
    print("-" * 50)
    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown:          {max_drawdown:.2f}%")
    print("="*50)
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(portfolio_history_df.index, portfolio_history_df['value'], label='Strategy Performance', color='royalblue')
    
    if benchmark_df is not None:
        benchmark_equity = (benchmark_df['close'] / benchmark_df['close'].iloc[0]) * initial_value
        ax1.plot(benchmark_equity.index, benchmark_equity, label='Benchmark (Buy & Hold)', color='gray', linestyle='--')
        
    ax1.set_title('Portfolio Equity Curve')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
    ax2.set_title('Portfolio Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
