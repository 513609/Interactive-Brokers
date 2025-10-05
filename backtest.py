import pandas as pd
import matplotlib.pyplot as plt
from strategies.moving_average_crossover import generate_signals

def run_backtest(symbol='AAPL', initial_capital=100000.0):
    """
    Runs a backtest for the SMA crossover strategy and prints the results.
    """
    # 1. LOAD DATA
    file_path = f"data/historical_data/{symbol}_data.csv"
    data = pd.read_csv(file_path, index_col='date', parse_dates=True)

    # 2. GENERATE SIGNALS
    # Use the function from your strategies folder
    signals_df = generate_signals(data)
    
    # Merge signals with the main dataframe
    data = data.join(signals_df['positions'])
    data['positions'].fillna(0.0, inplace=True)

    # 3. SIMULATE TRADING
    portfolio = pd.DataFrame(index=data.index)
    portfolio['holdings'] = 0.0  # Total value of stock held
    portfolio['cash'] = initial_capital
    portfolio['total'] = initial_capital
    
    positions = 0  # Number of shares held

    for i in range(len(data)):
        signal = data['positions'].iloc[i]
        close_price = data['close'].iloc[i]

        if signal == 1.0: # Buy Signal
            # "Buy" as many shares as possible
            shares_to_buy = portfolio['cash'].iloc[i] // close_price
            positions += shares_to_buy
            portfolio.loc[data.index[i]:, 'cash'] -= shares_to_buy * close_price
            print(f"{data.index[i].date()}: BUY {shares_to_buy} shares at ${close_price:.2f}")

        elif signal == -1.0: # Sell Signal
            # Sell all shares
            portfolio.loc[data.index[i]:, 'cash'] += positions * close_price
            print(f"{data.index[i].date()}: SELL {positions} shares at ${close_price:.2f}")
            positions = 0

        # Update portfolio total value for the day
        portfolio.loc[data.index[i]:, 'holdings'] = positions * close_price
        portfolio.loc[data.index[i]:, 'total'] = portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i]

    return portfolio, data

def analyze_results(portfolio, data):
    """Calculates and prints performance metrics."""
    final_portfolio_value = portfolio['total'].iloc[-1]
    total_return = (final_portfolio_value / portfolio['total'].iloc[0] - 1) * 100
    
    # Calculate Buy & Hold return
    buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100

    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Strategy Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print("------------------------")

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(portfolio['total'], label='Strategy Portfolio Value')
    # Plot Buy & Hold performance
    plt.plot(data['close'] / data['close'].iloc[0] * portfolio['total'].iloc[0], label='Buy & Hold (AAPL)')
    plt.title('SMA Crossover Strategy vs. Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Run the backtest and analyze the results
    portfolio_results, market_data = run_backtest(symbol='AAPL')
    analyze_results(portfolio_results, market_data)