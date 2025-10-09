import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.moving_average_crossover import generate_signals
import os
import json
from collections import defaultdict

def create_symbol_industry_dataframe(data_dir="data/historical_data/", fundamentals_dir="data/fundamental_data/"):
    """
    Scans data folders and creates a DataFrame mapping each stock symbol to its industry.
    """
    # --- List all files in the data directories for verification ---
    print(f"\n--- Files in Historical Data Directory ({data_dir}) ---")
    try:
        historical_files = sorted(os.listdir(data_dir))
        if historical_files:
            for f_name in historical_files:
                print(f"- {f_name}")
        else:
            print("No files found.")
    except FileNotFoundError:
        print(f"Directory not found: {data_dir}")

    print(f"\n--- Files in Fundamental Data Directory ({fundamentals_dir}) ---")
    try:
        fundamental_files = sorted(os.listdir(fundamentals_dir))
        if fundamental_files:
            for f_name in fundamental_files:
                print(f"- {f_name}")
        else:
            print("No files found.")
    except FileNotFoundError:
        print(f"Directory not found: {fundamentals_dir}")

    print("\nGrouping stocks by industry...")
    all_files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv') and 'SPY' not in f]
    industry_mapping = []
 
    for filename in all_files:
        symbol = filename.replace('_data.csv', '')
        row_data = {'symbol': symbol, 'industry': 'Unknown', 'fundamentals_file': '', 'file_preview': ''}
        # Check for both .json and .JSON extensions to be more robust
        base_filename = f"{symbol}_fundamentals"
        fundamentals_path_lower = os.path.join(fundamentals_dir, f"{base_filename}.json")
        fundamentals_path_upper = os.path.join(fundamentals_dir, f"{base_filename}.JSON")

        fundamentals_path = None
        if os.path.exists(fundamentals_path_lower):
            fundamentals_path = fundamentals_path_lower
        elif os.path.exists(fundamentals_path_upper):
            fundamentals_path = fundamentals_path_upper
        
        row_data['fundamentals_file'] = os.path.basename(fundamentals_path) if fundamentals_path else f"{base_filename}.json (Not Found)"
        
        if fundamentals_path:
            try:
                # Use utf-8 encoding, which matches the writer script (fundamental_data_fetcher.py)
                with open(fundamentals_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Add a check for empty file content before trying to parse
                if not file_content.strip():
                    print(f"Warning: Fundamentals file for {symbol} is empty. Assigning to 'Unknown'.")
                    row_data['file_preview'] = '(Empty File)'
                else:
                    row_data['file_preview'] = file_content[:10].replace('\n', '\\n') # Show first 10 chars
                    fundamentals = json.loads(file_content)
                    
                    info_data = fundamentals.get('info', {}) # Safely get the info dictionary
                    if 'industry' in info_data and info_data['industry']:
                        # Directly extract the industry from the 'info' dictionary
                        row_data['industry'] = info_data['industry']
                    else:
                        # --- Enhanced Diagnostic Logging ---
                        # This handles cases where 'industry' is missing, empty, or null.
                        print(f"Warning: 'industry' key not found or is empty in fundamentals for {symbol}. Assigning to 'Unknown'.")
                        print(f"  -> Info data received: {info_data}") # Print the entire info dict for inspection
                        row_data['industry'] = 'Unknown'

            except FileNotFoundError:
                # This case is already handled by the fundamentals_path check, but as a safeguard:
                print(f"Warning: File disappeared before it could be read: {fundamentals_path}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read or find industry for {symbol}. Error: {e}. Assigning to 'Unknown'.")
        else:
            # --- Enhanced Logging ---
            # Be more explicit about which files were checked to help with debugging.
            print(f"Warning: Fundamentals file not found for '{symbol}'. Assigning to 'Unknown'.")
            print(f"  - Looked for: {fundamentals_path_lower}")
            print(f"  - Looked for: {fundamentals_path_upper}")
        
        # --- Unconditional Final Check ---
        # This will print the final assigned industry for EVERY symbol, making it clear what was decided.
        print(f"  -> Final assignment for {symbol}: Industry='{row_data['industry']}'")
        industry_mapping.append(row_data)
        
    df = pd.DataFrame(industry_mapping)
    return df

def run_portfolio_backtest(symbols, initial_capital=10000.0, position_risk_percent=0.05, data_dir="data/historical_data/"):
    """
    Runs a realistic, time-based backtest on a portfolio of stocks for a specific industry.
    """
    all_data = {}
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_data.csv")
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        if len(df) > 50:
            signals_df = generate_signals(df)
            df['positions'] = signals_df['positions']
            all_data[symbol] = df

    if not all_data:
        return pd.DataFrame(), []

    master_index = pd.DatetimeIndex([])
    for df in all_data.values():
        master_index = master_index.union(df.index)
    all_dates = master_index.sort_values()

    cash = initial_capital
    portfolio_value_history = []
    positions = {}
    trades = []
    
    for date in all_dates:
        # Process exits first
        for symbol in list(positions.keys()):
            if date in all_data[symbol].index:
                signal = all_data[symbol].loc[date, 'positions']
                if signal == -1.0: # Exit signal
                    current_price = all_data[symbol].loc[date, 'close']
                    position_info = positions.pop(symbol)
                    exit_price = current_price
                    pnl = (exit_price - position_info['entry_price']) * position_info['shares']
                    cash += position_info['shares'] * exit_price
                    trades.append({
                        'symbol': symbol, 'pnl': pnl, 
                        'pnl_percent': (exit_price / position_info['entry_price'] - 1) * 100
                    })

        # Calculate current portfolio value before considering new entries
        current_portfolio_value = cash + sum(pos['shares'] * all_data[s].loc[date, 'close'] 
                                             for s, pos in positions.items() if date in all_data[s].index)
        
        # Process entries
        for symbol, df in all_data.items():
            if date in df.index and symbol not in positions:
                signal = df.loc[date, 'positions']
                if signal == 1.0: # Entry signal
                    position_size_usd = current_portfolio_value * position_risk_percent
                    if cash >= position_size_usd:
                        current_price = df.loc[date, 'close']
                        shares_to_buy = position_size_usd // current_price
                        if shares_to_buy > 0:
                            cash -= shares_to_buy * current_price
                            positions[symbol] = {'shares': shares_to_buy, 'entry_price': current_price}

        # Update portfolio value at end of day
        current_portfolio_value = cash + sum(pos['shares'] * all_data[s].loc[date, 'close'] 
                                             for s, pos in positions.items() if date in all_data[s].index)
        portfolio_value_history.append({'date': date, 'value': current_portfolio_value})

    return pd.DataFrame(portfolio_value_history).set_index('date'), trades

def print_industry_summary(portfolio_df, trades, industry_name):
    """Calculates and prints detailed performance metrics for an industry."""
    if portfolio_df.empty or not trades:
        print(f"\n--- No trades executed for {industry_name} ---")
        return

    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['pnl'] > 0] 
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    profit_factor = winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) if abs(losing_trades['pnl'].sum()) > 0 else float('inf')
    
    running_max = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    print(f"\n--- Backtest Summary for: {industry_name} ---")
    print(f"Initial Capital: ${portfolio_df['value'].iloc[0]:,.2f}")
    print(f"Final Portfolio Value: ${portfolio_df['value'].iloc[-1]:,.2f}")
    print(f"Total Return: {(portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0] - 1) * 100:.2f}%")
    print("-" * 20)
    print(f"Total Trades Executed: {len(trades_df)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")

def plot_all_results(all_portfolios, initial_capital):
    """
    Plots all industry strategy results and a SPY benchmark on a single graph.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(16, 9))

    master_index = pd.DatetimeIndex([])
    for df in all_portfolios.values():
        master_index = master_index.union(df.index)
    master_index = master_index.sort_values()

    # --- Plot SPY Benchmark ---
    try:
        spy_data = pd.read_csv('data/historical_data/SPY_data.csv', index_col='date', parse_dates=True)
        spy_data = spy_data.reindex(master_index, method='ffill')
        
        # Ensure SPY data is not all NaN after reindexing
        if not spy_data['close'].dropna().empty:
            spy_equity = (spy_data['close'] / spy_data['close'].dropna().iloc[0]) * initial_capital
            plt.plot(spy_equity.index, spy_equity, label='SPY (Buy & Hold)', color='black', linestyle='--', linewidth=2)
        else:
            print("\nWarning: SPY data could not be aligned with portfolio dates. Skipping benchmark plot.")
            
    except FileNotFoundError:
        print("\nSPY_data.csv not found. Skipping benchmark plot.")

    # --- Plot Each Industry Strategy ---
    for industry, portfolio_df in all_portfolios.items():
        if not portfolio_df.empty:
            plt.plot(portfolio_df.index, portfolio_df['value'], label=industry, alpha=0.8)

    plt.title('Backtest Performance: All Industries vs. SPY', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_top_performers(top_portfolios, initial_capital):
    """
    Plots the top 10 performing industry strategies and a SPY benchmark.
    """
    if not top_portfolios:
        print("No top performers to plot.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(16, 9))

    # --- Plot SPY Benchmark ---
    try:
        spy_data = pd.read_csv('data/historical_data/SPY_data.csv', index_col='date', parse_dates=True)
        master_index = pd.DatetimeIndex([])
        for df in top_portfolios.values():
            master_index = master_index.union(df.index)

        spy_data = spy_data.reindex(master_index.sort_values(), method='ffill').dropna()
        
        # Ensure SPY data is not all NaN after reindexing and dropping NaNs
        if not spy_data.empty:
            spy_equity = (spy_data['close'] / spy_data['close'].iloc[0]) * initial_capital
            plt.plot(spy_equity.index, spy_equity, label='SPY (Buy & Hold)', color='black', linestyle='--', linewidth=2)
        else:
            print("\nWarning: SPY data could not be aligned with top performer dates. Skipping benchmark plot.")

    except FileNotFoundError:
        print("\nSPY_data.csv not found. Skipping benchmark plot for top performers graph.")

    # --- Plot Each Top Industry Strategy ---
    for industry, portfolio_df in top_portfolios.items():
        plt.plot(portfolio_df.index, portfolio_df['value'], label=industry, alpha=0.8)

    plt.title('Backtest Performance: Top 10 Industries vs. SPY', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    INITIAL_CAPITAL = 10000.0
    POSITION_RISK = 0.05

    # Step 1: Create a DataFrame with all symbols and their industries
    industry_df = create_symbol_industry_dataframe()

    if industry_df.empty:
        print("\nNo stock data found to create industry dataframe. Exiting.")
    else:
        print("\n--- Stock to Industry Mapping ---")
        print(industry_df.to_string(index=False))
    
    # Step 2: Group by industry and run a backtest for each group
    industry_groups = industry_df.groupby('industry')
    
    # Count unique industries, excluding 'Unknown'
    unique_industry_count = industry_df[industry_df['industry'] != 'Unknown']['industry'].nunique()
    print(f"\nFound {unique_industry_count} unique industries to backtest (excluding 'Unknown').")

    all_portfolio_results = {}
    for industry, group in industry_groups:
        symbols = group['symbol'].tolist()
        if industry == 'Unknown' or len(symbols) < 2: # Skip small or unknown groups
            print(f"\nSkipping industry '{industry}' with {len(symbols)} stock(s).")
            continue
        
        print(f"\n--- Running backtest for industry: {industry} ({len(symbols)} stocks) ---")
        
        final_portfolio, all_trades = run_portfolio_backtest(
            symbols=symbols,
            initial_capital=INITIAL_CAPITAL, 
            position_risk_percent=POSITION_RISK
        )
        if not final_portfolio.empty:
            print_industry_summary(final_portfolio, all_trades, industry)
            all_portfolio_results[industry] = final_portfolio

    # Step 3: Plot all results on a single graph
    if all_portfolio_results:
        print("\n--- Generating combined performance plot ---")
        plot_all_results(all_portfolio_results, INITIAL_CAPITAL)

        # Step 4: Identify top 10 performers and plot them
        print("\n--- Identifying top 10 performing industries ---")
        
        # Calculate final value for each industry
        performance_metrics = []
        for industry, portfolio_df in all_portfolio_results.items():
            final_value = portfolio_df['value'].iloc[-1]
            performance_metrics.append((industry, final_value))
        
        # Sort by final value (descending) and get the top 10
        sorted_performance = sorted(performance_metrics, key=lambda item: item[1], reverse=True)
        top_10_industries = [item[0] for item in sorted_performance[:10]]
        top_10_portfolios = {industry: all_portfolio_results[industry] for industry in top_10_industries}
        
        plot_top_performers(top_10_portfolios, INITIAL_CAPITAL)