import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt

# --- Step 1: Setup and Parameter Definition ---

# Define global parameters for the strategy
REGRESSION_WINDOW = 36
FORMATION_PERIOD_MONTHS = 12
EXCLUDE_MONTHS = 2
DECILE_SIZE = 0.10

# Derived parameter
MOMENTUM_WINDOW = FORMATION_PERIOD_MONTHS - EXCLUDE_MONTHS

def calculate_residuals(df: pd.DataFrame, stock_cols: list, ff_cols: list) -> pd.DataFrame:
    """
    Calculates the residuals for each stock based on a rolling 3-factor Fama-French model.

    Args:
        df (pd.DataFrame): DataFrame containing stock returns and FF factors.
        stock_cols (list): List of stock ticker column names.
        ff_cols (list): List of Fama-French factor column names.

    Returns:
        pd.DataFrame: A DataFrame of residuals (alphas) for each stock.
    """
    print(f"Calculating residuals using a {REGRESSION_WINDOW}-month rolling window...")
    
    # Prepare independent variables (factors) and add a constant for the intercept
    X = sm.add_constant(df[ff_cols])
    
    all_residuals = {}

    for i, stock in enumerate(stock_cols):
        # Prepare dependent variable (excess return)
        y = df[stock] - df['RF_y']
        # *** FIX: Explicitly name the series to ensure it becomes the correct column name after concatenation.
        # The original error was a KeyError because this series was being added to the dataframe with a default integer column name (0) instead of the stock ticker.
        y.name = stock
        
        # Combine for rolling regression, dropping NaNs for the specific stock
        rolling_df = pd.concat([y, X], axis=1).dropna()
        
        # Ensure there's enough data to run the regression
        if len(rolling_df) < REGRESSION_WINDOW:
            print(f"  Skipping {stock}: Not enough data ({len(rolling_df)} months) for regression window.")
            continue

        # Perform rolling OLS regression
        rols = RollingOLS(
            endog=rolling_df[stock], 
            exog=rolling_df[X.columns], 
            window=REGRESSION_WINDOW
        )
        rres = rols.fit()
        
        # The residual is the alpha (intercept) of the regression
        # It represents the part of the return not explained by the factors.
        residuals = rres.params['const']
        all_residuals[stock] = residuals
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(stock_cols)} stocks...")

    residuals_df = pd.DataFrame(all_residuals)
    print("✅ Residual calculation complete.")
    return residuals_df

def calculate_momentum_scores(residuals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the residual momentum score for each stock.
    The score is the sum of residuals over the formation period divided by their standard deviation.

    Args:
        residuals_df (pd.DataFrame): DataFrame of monthly residuals.

    Returns:
        pd.DataFrame: DataFrame of momentum scores, shifted to align for trading.
    """
    print(f"Calculating momentum scores over a {MOMENTUM_WINDOW}-month formation period (t-11 to t-2)...")
    
    # Calculate rolling sum and standard deviation over the 10-month formation window
    rolling_sum = residuals_df.rolling(window=MOMENTUM_WINDOW, min_periods=MOMENTUM_WINDOW).sum()
    rolling_std = residuals_df.rolling(window=MOMENTUM_WINDOW, min_periods=MOMENTUM_WINDOW).std()

    # Calculate the raw momentum score
    # Replace division by zero with NaN, then fill NaN with 0
    momentum_scores = (rolling_sum / rolling_std).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Shift the scores forward by 2 months to implement the t-2 lag.
    # This ensures that the score used for trading at the end of month 't'
    # is based on data that was available at the end of month 't-2'.
    shifted_scores = momentum_scores.shift(EXCLUDE_MONTHS)
    
    print("✅ Momentum score calculation complete.")
    return shifted_scores

def backtest_strategy(df: pd.DataFrame, scores_df: pd.DataFrame, stock_cols: list) -> pd.Series:
    """
    Backtests the long-short residual momentum strategy.

    Args:
        df (pd.DataFrame): The original DataFrame with returns.
        scores_df (pd.DataFrame): DataFrame with calculated momentum scores.
        stock_cols (list): List of stock ticker columns.

    Returns:
        pd.Series: A time series of the strategy's monthly returns.
    """
    print("Backtesting the long-short strategy...")
    
    strategy_returns = {}
    
    # Align the main dataframe with the scores dataframe to handle NaNs and different start dates
    aligned_df, aligned_scores = df.align(scores_df, join='inner', axis=0)

    # Iterate through each month in the scores dataframe to form portfolios
    for t in aligned_scores.index[:-1]: # Go up to the second to last month
        # Get scores for the current month and drop any stocks with NaN scores
        scores_t = aligned_scores.loc[t].dropna()
        
        if scores_t.empty:
            continue

        # Rank stocks based on their momentum score
        ranked_scores = scores_t.sort_values(ascending=False)
        
        # Determine portfolio size based on the decile
        n = int(len(ranked_scores) * DECILE_SIZE)
        if n == 0:
            continue # Skip if portfolios would be empty

        # Identify the long and short portfolios
        long_portfolio = ranked_scores.head(n).index
        short_portfolio = ranked_scores.tail(n).index
        
        # The trade happens at the end of month t, and we hold for one month.
        # So, we calculate the return using data from the next month (t+1).
        # We use .get(t + pd.DateOffset(months=1)) to safely get the next month's index
        next_month = t + pd.DateOffset(months=1)
        if next_month not in aligned_df.index:
            continue

        # Calculate equally-weighted returns for the portfolios for the next month
        long_return = aligned_df.loc[next_month, long_portfolio].mean()
        short_return = aligned_df.loc[next_month, short_portfolio].mean()
        
        # The strategy return is the difference between the long and short portfolios
        strategy_returns[next_month] = long_return - short_return

    print("✅ Backtest complete.")
    return pd.Series(strategy_returns, name="ResidualMomentum")

def analyze_performance(strategy_returns: pd.Series, benchmark_returns: pd.Series):
    """
    Calculates and prints performance metrics and plots the cumulative returns.
    """
    print("\n--- Strategy Performance Analysis ---")
    
    # --- Performance Metrics ---
    # Compound Annual Growth Rate (CAGR)
    total_return = (1 + strategy_returns).prod() - 1
    num_years = len(strategy_returns) / 12
    cagr = (1 + total_return) ** (1 / num_years) - 1

    # Annualized Sharpe Ratio
    sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(12)

    # Maximum Drawdown
    cumulative = (1 + strategy_returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    print(f"Compound Annual Growth Rate (CAGR): {cagr:.2%}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    
    # Plot strategy returns
    (1 + strategy_returns).cumprod().plot(label='Residual Momentum Strategy', legend=True)
    
    # Plot benchmark returns
    (1 + benchmark_returns).cumprod().plot(label='Equal-Weighted S&P 500 Benchmark', legend=True, linestyle='--')
    
    plt.title('Cumulative Performance: Residual Momentum vs. Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.yscale('log') # Use a log scale for better visualization over long periods
    plt.show()


if __name__ == "__main__":
    # --- Load and Prepare Data ---
    file_path = "data/monthly_returns_with_factors.csv"
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Identify stock and factor columns
    ff_cols = ['Mkt-RF', 'SMB', 'HML']
    # All other columns except the risk-free rate are assumed to be stocks
    stock_cols = [col for col in df.columns if col not in ff_cols + ['RF_y']]

    # --- Execute Strategy Steps ---
    # Step 2: Calculate monthly residuals
    residuals_df = calculate_residuals(df, stock_cols, ff_cols)

    # Step 3: Calculate the residual momentum score
    scores_df = calculate_momentum_scores(residuals_df)

    # Step 4: Backtest the strategy
    strategy_returns = backtest_strategy(df, scores_df, stock_cols)

    # --- Analyze and Visualize ---
    # Create an equal-weighted benchmark of all stocks
    benchmark_returns = df[stock_cols].mean(axis=1)
    
    # Align benchmark and strategy returns for comparison
    aligned_benchmark, aligned_strategy = benchmark_returns.align(strategy_returns, join='inner')

    # Step 5: Analyze and visualize performance
    analyze_performance(aligned_strategy, aligned_benchmark)