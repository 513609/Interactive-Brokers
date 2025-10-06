import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# --- Helper and Plotting Functions (Copied from backtest_per_industry.py) ---
# These functions are reused for loading data and visualizing results.

def create_symbol_industry_dataframe(data_dir="data/historical_data/", fundamentals_dir="data/fundamental_data/"):
    """Scans data folders and creates a DataFrame mapping each stock symbol to its industry."""
    all_files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv') and 'SPY' not in f]
    industry_mapping = []
 
    for filename in all_files:
        symbol = filename.replace('_data.csv', '')
        industry = 'Unknown'
        fundamentals_path = os.path.join(fundamentals_dir, f"{symbol}_fundamentals.json")
        
        if os.path.exists(fundamentals_path):
            try:
                with open(fundamentals_path, 'r', encoding='utf-8') as f:
                    fundamentals = json.load(f)
                    info_data = fundamentals.get('info', {})
                    if 'industry' in info_data and info_data['industry']:
                        industry = info_data['industry']
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read industry for {symbol}. Error: {e}. Assigning to 'Unknown'.")
        
        industry_mapping.append({'symbol': symbol, 'industry': industry})
        
    return pd.DataFrame(industry_mapping)

def print_industry_summary(portfolio_df, trades, industry_name):
    """Calculates and prints detailed performance metrics for an industry."""
    if portfolio_df.empty:
        print(f"\n--- No results for {industry_name} ---")
        return

    trades_df = pd.DataFrame(trades)
    win_rate = 0
    profit_factor = float('inf')

    if not trades_df.empty:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        if len(trades_df) > 0:
            win_rate = len(winning_trades) / len(trades_df) * 100
        if abs(losing_trades['pnl'].sum()) > 0:
            profit_factor = winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())

    running_max = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    print(f"\n--- Momentum Backtest Summary for: {industry_name} ---")
    print(f"Initial Capital: ${portfolio_df['value'].iloc[0]:,.2f}")
    print(f"Final Portfolio Value: ${portfolio_df['value'].iloc[-1]:,.2f}")
    print(f"Total Return: {(portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0] - 1) * 100:.2f}%")
    print("-" * 20)
    print(f"Total Rebalances (Trades): {len(trades_df)}")
    print(f"Win Rate (profitable months): {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")

def calculate_performance_metrics(portfolio_df, initial_capital):
    """
    Calculates key performance metrics for a given portfolio history.
    Returns a dictionary with the calculated metrics.
    """
    if portfolio_df.empty or len(portfolio_df) < 2:
        return {
            'total_return': 0, 'max_drawdown': 0, 
            'sharpe_ratio': 0, 'calmar_ratio': 0
        }

    # Total Return
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # Maximum Drawdown
    running_max = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - running_max) / running_max
    max_drawdown = abs(drawdown.min()) * 100

    # Annualized Return (for Calmar Ratio)
    days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    years = days / 365.25
    annualized_return = 0
    if years > 0:
        annualized_return = ((1 + (total_return / 100)) ** (1 / years)) - 1

    # Calmar Ratio
    calmar_ratio = annualized_return / (max_drawdown / 100) if max_drawdown > 0 else 0

    # Sharpe Ratio (Annualized, assuming 0 risk-free rate)
    daily_returns = portfolio_df['value'].pct_change().dropna()
    sharpe_ratio = 0
    if daily_returns.std() > 0 and not daily_returns.empty:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio
    }

def plot_all_results(all_portfolios, initial_capital, momentum_lookback=126):
    """Plots all industry strategy results and a SPY benchmark on a single graph."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(16, 9))
    
    try:
        spy_data = pd.read_csv('data/historical_data/SPY_data.csv', index_col='date', parse_dates=True)
        
        # Determine the common start date for the benchmark
        master_index = pd.DatetimeIndex([])
        for df in all_portfolios.values():
            master_index = master_index.union(df.index)
        
        # The strategy starts after the lookback period. Find the first valid date.
        valid_dates = master_index.sort_values()
        strategy_start_date = valid_dates[momentum_lookback] if len(valid_dates) > momentum_lookback else valid_dates[0]
        
        spy_data = spy_data.reindex(valid_dates, method='ffill')
        # Normalize SPY from the strategy's actual start date
        spy_data = spy_data.loc[strategy_start_date:] # Filter SPY data to start from the same date
        spy_start_price = spy_data.loc[strategy_start_date, 'close']
        spy_equity = (spy_data['close'] / spy_start_price) * initial_capital
        plt.plot(spy_equity.index, spy_equity, label='SPY (Buy & Hold)', color='black', linestyle='--', linewidth=2)
    except FileNotFoundError:
        print("\nSPY_data.csv not found. Skipping benchmark plot.")

    for industry, portfolio_df in all_portfolios.items():
        if not portfolio_df.empty:
            plt.plot(portfolio_df.index, portfolio_df['value'], label=industry, alpha=0.8)

    plt.title('Momentum Strategy Performance: All Industries vs. SPY', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_top_performers(top_portfolios, performance_data, spy_metrics, initial_capital, momentum_lookback=126):
    """
    Plots the top 10 performing industry strategies and a SPY benchmark.
    """
    if not top_portfolios or not performance_data:
        print("No top performers to plot.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(16, 9))

    # --- Plot SPY Benchmark ---
    try:
        spy_data = pd.read_csv('data/historical_data/SPY_data.csv', index_col='date', parse_dates=True)
        
        # Determine the common start date for the benchmark
        master_index = pd.DatetimeIndex([])
        for df in top_portfolios.values():
            master_index = master_index.union(df.index)

        # The strategy starts after the lookback period. Find the first valid date.
        valid_dates = master_index.sort_values()
        strategy_start_date = valid_dates[momentum_lookback] if len(valid_dates) > momentum_lookback else valid_dates[0]

        spy_data = spy_data.reindex(valid_dates, method='ffill')
        spy_start_price = spy_data.loc[strategy_start_date, 'close']
        spy_data = spy_data.loc[strategy_start_date:] # Filter SPY data to start from the same date
        spy_equity = (spy_data['close'] / spy_start_price) * initial_capital
        
        spy_label = "SPY (Buy & Hold)"
        if spy_metrics:
            spy_label = (f"SPY (Return: {spy_metrics.get('total_return', 0):.1f}%, "
                         f"Sharpe: {spy_metrics.get('sharpe_ratio', 0):.2f}, "
                         f"Calmar: {spy_metrics.get('calmar_ratio', 0):.2f})")

        plt.plot(spy_equity.index, spy_equity, label=spy_label, color='black', linestyle='--', linewidth=2)
    except FileNotFoundError:
        print("\nSPY_data.csv not found. Skipping benchmark plot for top performers graph.")

    # --- Plot Each Top Industry Strategy ---
    for industry, portfolio_df in sorted(top_portfolios.items(), key=lambda item: item[1]['value'].iloc[-1], reverse=True):
        metrics = performance_data.get(industry, {})
        label = (
            f"{industry} "
            f"(Return: {metrics.get('total_return', 0):.1f}%, "
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
            f"Calmar: {metrics.get('calmar_ratio', 0):.2f})"
        )
        plt.plot(portfolio_df.index, portfolio_df['value'], label=label, alpha=0.8)

    plt.title('Momentum Strategy Performance: Top 10 Industries vs. SPY', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=9, title='Industries (Total Return, Sharpe Ratio, Calmar Ratio)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def print_top_performer_trade_log(industry_name, trades):
    """
    Prints a detailed log of rebalancing trades for the top-performing industry.
    """
    print("\n" + "="*60)
    print(f" Trade Log for Top Performing Industry: {industry_name} ".center(60, "="))
    print("="*60)

    if not trades:
        print("No trades were logged for this industry.")
        return

    for trade in trades:
        date = trade['date'].strftime('%Y-%m-%d')
        pnl = trade['pnl']
        value = trade['portfolio_value']
        
        print(f"\n--- Rebalance Date: {date} ---")
        print(f"Portfolio Value: ${value:,.2f} | Month P&L: ${pnl:,.2f}")
        print(f"  Positions Sold: {', '.join(trade['stocks_sold']) if trade['stocks_sold'] else 'None'}")
        print(f"  New Positions Bought: {', '.join(trade['stocks_bought']) if trade['stocks_bought'] else 'None'}")

def plot_stop_loss_comparison_graph(industry_name, with_sl_df, without_sl_df, with_sl_metrics, without_sl_metrics):
    """
    Plots a side-by-side comparison of the equity curves for a strategy with and without a stop-loss.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(16, 9))

    # Plotting the 'With Stop-Loss' strategy
    label_with_sl = (
        f"With 5% Stop-Loss (Return: {with_sl_metrics.get('total_return', 0):.1f}%, "
        f"Sharpe: {with_sl_metrics.get('sharpe_ratio', 0):.2f}, "
        f"Max DD: {with_sl_metrics.get('max_drawdown', 0):.1f}%)"
    )
    plt.plot(with_sl_df.index, with_sl_df['value'], label=label_with_sl, color='royalblue', linewidth=2)

    # Plotting the 'Without Stop-Loss' strategy
    label_without_sl = (
        f"Without Stop-Loss (Return: {without_sl_metrics.get('total_return', 0):.1f}%, "
        f"Sharpe: {without_sl_metrics.get('sharpe_ratio', 0):.2f}, "
        f"Max DD: {without_sl_metrics.get('max_drawdown', 0):.1f}%)"
    )
    plt.plot(without_sl_df.index, without_sl_df['value'], label=label_without_sl, color='darkorange', linestyle='--', linewidth=2)

    plt.title(f'Stop-Loss Comparison for Industry: {industry_name}', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def print_stop_loss_comparison_table(comparison_data):
    """
    Prints a formatted table comparing performance metrics with and without a stop-loss.
    """
    if not comparison_data:
        print("No data available for stop-loss comparison.")
        return

    df = pd.DataFrame(comparison_data)
    df = df.set_index('Industry')

    print("\n" + "="*80)
    print(" Stop-Loss Performance Comparison ".center(80, "="))
    print("="*80)
    print(df.to_string(float_format="%.2f"))
    print("-" * 80)


# --- New Momentum Backtest Engine ---

def run_momentum_portfolio_backtest(
    symbols, 
    initial_capital=10000.0, 
    data_dir="data/historical_data/",
    momentum_lookback=126,  # Approx. 6 months
    stop_loss_percent=0.05, # 5% stop-loss
    top_quantile=0.2 # Invest in top 20%
):
    """
    Runs a cross-sectional momentum backtest on a portfolio of stocks for a specific industry.
    """
    all_data = {}
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_data.csv")
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        
        # Calculate the momentum score (rate of change over the lookback period)
        df['momentum'] = df['close'].pct_change(periods=momentum_lookback)
        
        if len(df) > momentum_lookback:
            all_data[symbol] = df

    if not all_data or len(all_data) < 2:
        return pd.DataFrame(), []

    # Create a master index of all trading days
    master_index = pd.DatetimeIndex([])
    for df in all_data.values():
        master_index = master_index.union(df.index)
    all_dates = master_index.sort_values()

    # Determine rebalancing dates (first trading day of each month)
    rebalance_dates = all_dates.to_series().resample('MS').first().dropna()

    cash = initial_capital
    portfolio_value_history = []
    positions = {} # {symbol: {'shares': float, 'entry_price': float}}
    trades = []
    
    for date in all_dates:
        # --- Daily Stop-Loss Check (runs every day) ---
        if stop_loss_percent is not None and stop_loss_percent > 0:
            # This logic is separate from the monthly rebalancing.
            for symbol in list(positions.keys()):
                if date in all_data[symbol].index:
                    current_price = all_data[symbol].loc[date, 'close']
                    entry_price = positions[symbol]['entry_price']
                    
                    # Check if the price has dropped below the stop-loss level
                    if current_price < entry_price * (1 - stop_loss_percent):
                        # Sell the position
                        position_info = positions.pop(symbol)
                        cash += position_info['shares'] * current_price
                        # Note: This exit is not logged in the monthly 'trades' log to keep it simple,
                        # but the portfolio value will reflect the change.


        # --- Rebalancing Logic ---
        if date in rebalance_dates:
            # 1. Calculate portfolio value before rebalancing and sell everything
            current_portfolio_value = cash + sum(positions[s]['shares'] * all_data[s].loc[date, 'close'] 
                                                 for s in positions if date in all_data[s].index)
            
            stocks_sold = list(positions.keys())
            # Liquidate all positions
            for symbol, position_info in positions.items():
                if date in all_data[symbol].index:
                    cash += position_info['shares'] * all_data[symbol].loc[date, 'close']
            
            positions.clear()
            
            # 2. Rank stocks by momentum
            momentum_scores = []
            for symbol, df in all_data.items():
                if date in df.index and not pd.isna(df.loc[date, 'momentum']):
                    momentum_scores.append((symbol, df.loc[date, 'momentum']))
            
            # Sort by momentum score, descending
            momentum_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 3. Identify new stocks to buy
            num_to_buy = max(1, int(len(momentum_scores) * top_quantile))
            stocks_to_buy = [item[0] for item in momentum_scores[:num_to_buy]]
            
            # 4. Allocate capital and buy new positions
            if stocks_to_buy:
                capital_per_stock = current_portfolio_value / len(stocks_to_buy)
                for symbol in stocks_to_buy:
                    current_price = all_data[symbol].loc[date, 'close']
                    shares_to_buy = capital_per_stock // current_price
                    if shares_to_buy > 0:
                        cash -= shares_to_buy * current_price
                        positions[symbol] = {
                            'shares': shares_to_buy,
                            'entry_price': current_price
                        }
            
            # Log the rebalance as a "trade" for monthly P&L calculation
            if portfolio_value_history:
                pnl = current_portfolio_value - portfolio_value_history[-1]['value']
                trades.append({
                    'date': date, 
                    'pnl': pnl,
                    'portfolio_value': current_portfolio_value,
                    'stocks_sold': stocks_sold,
                    'stocks_bought': stocks_to_buy
                })

        # --- Daily Portfolio Value Update ---
        current_portfolio_value = cash + sum(positions[s]['shares'] * all_data[s].loc[date, 'close'] 
                                             for s in positions if date in all_data[s].index)
        portfolio_value_history.append({'date': date, 'value': current_portfolio_value})

    return pd.DataFrame(portfolio_value_history).set_index('date'), trades


if __name__ == "__main__":
    INITIAL_CAPITAL = 10000.0

    # Step 1: Create a DataFrame with all symbols and their industries
    print("--- Grouping stocks by industry ---")
    industry_df = create_symbol_industry_dataframe()

    if industry_df.empty:
        print("\nNo stock data found to create industry dataframe. Exiting.")
        exit() # Exit if no data
    
    # --- New Industry Summary Printout ---
    print("\n" + "="*50)
    print(" Industry & Stock Count Summary ".center(50, "="))
    print("="*50)
    industry_counts = industry_df['industry'].value_counts()
    print(f"Total Unique Industries: {len(industry_counts)}")
    print(f"Total Stocks with Industry Data: {len(industry_df)}")
    print("-" * 50)
    # Create and print the summary as a DataFrame
    summary_df = industry_counts.reset_index()
    summary_df.columns = ['Industry', 'Number of Tickers']
    print(summary_df.to_string(index=False))
    print("="*50)

    # Step 2: Group by industry and run a backtest for each group
    industry_groups = industry_df.groupby('industry')
    
    # --- Modified to store results for both strategies ---
    all_portfolio_results_with_sl = {}
    all_portfolio_results_without_sl = {}
    all_trade_logs = {} # Store trade logs for the primary (with SL) strategy
    stop_loss_comparison_data = []

    for industry, group in industry_groups:
        symbols = group['symbol'].tolist()
        # Only include industries with more than 10 stocks for statistical significance.
        if industry == 'Unknown' or len(symbols) <= 10:
            print(f"\nSkipping industry '{industry}' with {len(symbols)} stock(s).")
            continue
        
        print(f"\n--- Running Backtests for industry: {industry} ({len(symbols)} stocks) ---")

        # Run 1: With Stop-Loss
        print("  -> Running with 5% Stop-Loss...")
        final_portfolio, all_trades = run_momentum_portfolio_backtest(symbols=symbols, initial_capital=INITIAL_CAPITAL, stop_loss_percent=0.05)
        if not final_portfolio.empty:
            all_portfolio_results_with_sl[industry] = final_portfolio
            all_trade_logs[industry] = all_trades
            metrics_with_sl = calculate_performance_metrics(final_portfolio, INITIAL_CAPITAL)

        # Run 2: Without Stop-Loss
        print("  -> Running without Stop-Loss...")
        final_portfolio_no_sl, _ = run_momentum_portfolio_backtest(symbols=symbols, initial_capital=INITIAL_CAPITAL, stop_loss_percent=None)
        if not final_portfolio_no_sl.empty:
            all_portfolio_results_without_sl[industry] = final_portfolio_no_sl
            metrics_without_sl = calculate_performance_metrics(final_portfolio_no_sl, INITIAL_CAPITAL)

        # Store comparison data if both runs were successful
        if industry in all_portfolio_results_with_sl and industry in all_portfolio_results_without_sl:
            stop_loss_comparison_data.append({
                'Industry': industry,
                'Return (w/ SL)': metrics_with_sl['total_return'], 'Return (no SL)': metrics_without_sl['total_return'],
                'Sharpe (w/ SL)': metrics_with_sl['sharpe_ratio'], 'Sharpe (no SL)': metrics_without_sl['sharpe_ratio'],
                'Max DD (w/ SL)': metrics_with_sl['max_drawdown'], 'Max DD (no SL)': metrics_without_sl['max_drawdown'],
            })

    # Step 3: Plot all results on a single graph
    MOMENTUM_LOOKBACK = 126 # Must match the parameter in the backtest function
    if all_portfolio_results_with_sl:
        print("\n--- Generating combined performance plot for Momentum Strategy ---")
        # We plot the primary strategy (with stop-loss)
        plot_all_results(all_portfolio_results_with_sl, INITIAL_CAPITAL, momentum_lookback=MOMENTUM_LOOKBACK)

        # Step 4: Identify top performers, print metrics, plot them, and show top trade log
        print("\n--- Identifying top 10 performing industries ---")
        
        all_performance_metrics = {}
        for industry, portfolio_df in all_portfolio_results_with_sl.items():
            # Calculate full metrics for each industry
            metrics = calculate_performance_metrics(portfolio_df, INITIAL_CAPITAL)
            metrics['final_value'] = portfolio_df['value'].iloc[-1]
            all_performance_metrics[industry] = metrics

        # Sort by final value (descending) and get the top 10
        sorted_industries = sorted(all_performance_metrics.items(), key=lambda item: item[1]['final_value'], reverse=True)
        top_10_industries = [item[0] for item in sorted_industries[:10]]
        top_10_portfolios = {industry: all_portfolio_results_with_sl[industry] for industry in top_10_industries}
        
        # --- Calculate and Print SPY Benchmark Metrics ---
        spy_metrics = {}
        try:
            spy_data = pd.read_csv('data/historical_data/SPY_data.csv', index_col='date', parse_dates=True)
            # Use the index from the top performer to align dates
            if sorted_industries:
                top_performer_df = all_portfolio_results_with_sl[sorted_industries[0][0]]
                spy_data_aligned = spy_data.reindex(top_performer_df.index, method='ffill').dropna()
                spy_portfolio_df = pd.DataFrame(
                    (spy_data_aligned['close'] / spy_data_aligned['close'].iloc[0]) * INITIAL_CAPITAL
                )
                spy_portfolio_df.columns = ['value']
                spy_metrics = calculate_performance_metrics(spy_portfolio_df, INITIAL_CAPITAL)
                spy_metrics['final_value'] = spy_portfolio_df['value'].iloc[-1]
        except FileNotFoundError:
            print("SPY data not found, cannot calculate benchmark metrics.")

        print("\n--- Performance Summary (by final value) ---")
        if spy_metrics:
            print(f"SPY Benchmark: ${spy_metrics['final_value']:,.2f} (Return: {spy_metrics['total_return']:.1f}%, Sharpe: {spy_metrics['sharpe_ratio']:.2f}, Calmar: {spy_metrics['calmar_ratio']:.2f})")
            print("-" * 80)

        for i, (industry, metrics) in enumerate(sorted_industries[:10]):
            print(f"{i+1}. {industry}: ${metrics['final_value']:,.2f} (Return: {metrics['total_return']:.1f}%, Sharpe: {metrics['sharpe_ratio']:.2f}, Calmar: {metrics['calmar_ratio']:.2f})")

        plot_top_performers(top_10_portfolios, all_performance_metrics, spy_metrics, INITIAL_CAPITAL, momentum_lookback=MOMENTUM_LOOKBACK)

        # Step 5: Print the detailed trade log for the #1 performing industry
        if sorted_industries:
            top_industry_name = sorted_industries[0][0]
            top_industry_trades = all_trade_logs.get(top_industry_name, [])
            print_top_performer_trade_log(top_industry_name, top_industry_trades)

        # Step 6: Show the stop-loss comparison table and graph
        print_stop_loss_comparison_table(stop_loss_comparison_data)

        if sorted_industries:
            top_industry_name = sorted_industries[0][0]
            if top_industry_name in all_portfolio_results_with_sl and top_industry_name in all_portfolio_results_without_sl:
                print(f"\n--- Generating Stop-Loss comparison graph for top industry: {top_industry_name} ---")
                plot_stop_loss_comparison_graph(
                    top_industry_name,
                    all_portfolio_results_with_sl[top_industry_name],
                    all_portfolio_results_without_sl[top_industry_name],
                    all_performance_metrics[top_industry_name],
                    calculate_performance_metrics(all_portfolio_results_without_sl[top_industry_name], INITIAL_CAPITAL)
                )