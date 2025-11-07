import pandas as pd
from pathlib import Path
import sys

def calculate_monthly_returns(data_dir: Path) -> pd.DataFrame:
    """
    Scans a directory of daily historical stock data CSVs, calculates
    monthly returns for each stock, and returns a combined DataFrame.

    Args:
        data_dir (Path): The directory containing the historical data files.

    Returns:
        pd.DataFrame: A DataFrame where each column is a stock's monthly
                      return and the index is the month-end date.
    """
    print(f"Calculating monthly returns from files in: {data_dir}")
    all_monthly_returns = []
    
    # Find all stock data files, excluding the SPY benchmark
    stock_files = [f for f in data_dir.glob('*_data.csv') if 'SPY' not in f.name]

    if not stock_files:
        print(f"Error: No stock data files found in '{data_dir}'.")
        print("Please run the data_fetcher.py script first.")
        sys.exit(1)

    for file_path in stock_files:
        symbol = file_path.stem.replace('_data', '')
        try:
            # Read daily data, parsing 'date' column as datetime objects
            daily_df = pd.read_csv(file_path, index_col='date', parse_dates=True)

            if daily_df.empty or 'close' not in daily_df.columns:
                print(f"Warning: Skipping {symbol} due to empty or invalid data.")
                continue

            # Resample to get the last closing price of each month
            monthly_close = daily_df['close'].resample('M').last()

            # Calculate monthly percentage returns
            monthly_returns = monthly_close.pct_change()

            # Rename the series to the stock's symbol
            monthly_returns.name = symbol
            all_monthly_returns.append(monthly_returns)
        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")

    # Combine all individual stock return Series into a single DataFrame
    if not all_monthly_returns:
        print("Error: Failed to calculate monthly returns for any stock.")
        sys.exit(1)

    returns_df = pd.concat(all_monthly_returns, axis=1)
    print(f"Successfully calculated monthly returns for {len(returns_df.columns)} stocks.")
    return returns_df

if __name__ == "__main__":
    # Define file paths
    DATA_DIR = Path("data/historical_data")
    FF_FACTORS_PATH = Path("data/FamaFrench/Fama_French_3_Factors.csv")
    OUTPUT_PATH = Path("data/monthly_returns_with_factors.csv")

    # --- Step 1: Load and prepare Fama-French data ---
    print(f"Loading Fama-French factors from: {FF_FACTORS_PATH}")
    ff_df = pd.read_csv(FF_FACTORS_PATH, index_col='Date', parse_dates=True)
    # The FF data is already monthly, with the date as the first of the month.
    # We will align it to month-end to match our resampled stock returns.
    ff_df.index = ff_df.index + pd.offsets.MonthEnd(0)

    # --- Step 2: Calculate monthly returns for all stocks ---
    stock_returns_df = calculate_monthly_returns(DATA_DIR)

    # --- Step 3: Merge the two DataFrames ---
    print("Merging stock returns with Fama-French factors...")
    # Use an inner join to ensure we only have dates where both datasets are available
    merged_df = pd.merge(stock_returns_df, ff_df, left_index=True, right_index=True, how='inner')

    # --- Step 4: Save the final DataFrame ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # FIX: Add index_label='Date' to name the index column in the CSV file.
    merged_df.to_csv(OUTPUT_PATH, index_label='Date')

    print("\n--- Process Complete ---")
    print(f"✅ Merged data successfully saved to: {OUTPUT_PATH}")
    print(f"Final DataFrame shape: {merged_df.shape}")
    print("Final DataFrame head:")
    print(merged_df.head().to_string())