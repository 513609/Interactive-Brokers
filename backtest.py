import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.moving_average_crossover import generate_signals
import os

def run_portfolio_backtest(initial_capital=10000.0, position_risk_percent=0.05):
    """
    Runs a realistic, time-based backtest on a single portfolio across all S&P 500 stocks.
    """
    print("Loading all S&P 500 data...")
    data_dir = "data/historical_data/"
    all_files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv') and 'SPY' not in f]
    
    all_data = {}
    for filename in all_files:
        symbol = filename.replace('_data.csv', '')
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        if len(df) > 50:
            signals_df = generate_signals(df)
            df['positions'] = signals_df['positions']
            all_data[symbol] = df

    master_index = pd.DatetimeIndex([])
    for df in all_data.values():
        master_index = master_index.union(df.index)
    all_dates = master_index.sort_values()

    cash = initial_capital
    portfolio_value_history = []
    positions = {}
    trades = []
    
    print("Starting day-by-day portfolio simulation...")
    for date in all_dates:
        for symbol in list(positions.keys()):
            if date in all_data[symbol].index:
                signal = all_data[symbol].loc[date, 'positions']
                if signal == -1.0:
                    current_price = all_data[symbol].loc[date, 'close']
                    position_info = positions.pop(symbol)
                    exit_price = current_price
                    pnl = (exit_price - position_info['entry_price']) * position_info['shares']
                    cash += position_info['shares'] * exit_price
                    trades.append({
                        'symbol': symbol, 'pnl': pnl, 
                        'pnl_percent': (exit_price / position_info['entry_price'] - 1) * 100
                    })

        current_portfolio_value = cash + sum(pos['shares'] * all_data[s].loc[date, 'close'] 
                                             for s, pos in positions.items() if date in all_data[s].index)
        
        for symbol, df in all_data.items():
            if date in df.index and symbol not in positions:
                signal = df.loc[date, 'positions']
                if signal == 1.0:
                    position_size_usd = current_portfolio_value * position_risk_percent
                    if cash >= position_size_usd:
                        current_price = df.loc[date, 'close']
                        shares_to_buy = position_size_usd // current_price
                        if shares_to_buy > 0:
                            cash -= shares_to_buy * current_price
                            positions[symbol] = {'shares': shares_to_buy, 'entry_price': current_price}

        current_portfolio_value = cash + sum(pos['shares'] * all_data[s].loc[date, 'close'] 
                                             for s, pos in positions.items() if date in all_data[s].index)
        portfolio_value_history.append({'date': date, 'value': current_portfolio_value})

    return pd.DataFrame(portfolio_value_history).set_index('date'), trades

def analyze_results(portfolio_df, trades):
    """Calculates and displays detailed performance metrics and a plot."""
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    profit_factor = winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) if abs(losing_trades['pnl'].sum()) > 0 else float('inf')
    
    running_max = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    print("\n--- Realistic Portfolio Backtest Summary ---")
    print(f"Initial Capital: ${portfolio_df['value'].iloc[0]:,.2f}")
    print(f"Final Portfolio Value: ${portfolio_df['value'].iloc[-1]:,.2f}")
    print(f"Total Return: {(portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0] - 1) * 100:.2f}%")
    print("-" * 20)
    print(f"Total Trades Executed: {len(trades_df)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # --- PLOTTING SECTION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot Strategy Performance
    ax1.plot(portfolio_df.index, portfolio_df['value'], label='SMA Crossover Strategy', color='royalblue')
    
    # --- NEW CODE TO PLOT SPY BENCHMARK ---
    try:
        spy_data = pd.read_csv('data/historical_data/SPY_data.csv', index_col='date', parse_dates=True)
        spy_data = spy_data.reindex(portfolio_df.index, method='ffill')
        spy_equity = (spy_data['close'] / spy_data['close'].iloc[0]) * portfolio_df['value'].iloc[0]
        ax1.plot(spy_equity.index, spy_equity, label='SPY (Buy & Hold)', color='gray', linestyle='--')
    except FileNotFoundError:
        print("\nSPY_data.csv not found. Skipping benchmark plot.")
    # --- END OF NEW CODE ---
    
    ax1.set_title('Portfolio Performance with Dynamic Capital Allocation')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Drawdown
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
    ax2.set_title('Portfolio Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    final_portfolio, all_trades = run_portfolio_backtest(
        initial_capital=10000.0, 
        position_risk_percent=0.05
    )
    if not final_portfolio.empty:
        analyze_results(final_portfolio, all_trades)