import configparser
from ib_insync import IB, Stock
import pandas as pd
from utils.sp500_tickers import get_sp500_tickers
import os

# This is the function to get data for a single stock
def fetch_and_save_data(symbol, duration='2 Y', bar_size='1 day'):
    # ... (code for this function) ...
    pass

# This is the function we need to run now
def fetch_all_sp500_data(duration='10 Y', bar_size='1 day'):
    """
    Fetches historical data for all S&P 500 stocks and saves them to CSVs.
    Checks if data already exists and updates it if it's less than 10 years old.
    """
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    ibkr_config = config['ibkr']

    tickers = ['SPY'] + get_sp500_tickers()
    tickers = sorted(list(set(tickers)))

    data_dir = "data/historical_data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ib = IB()
    try:
        ib.connect(
            ibkr_config.get('host'),
            ibkr_config.getint('port'),
            clientId=102
        )
        print("Connection successful.")

        for i, symbol in enumerate(tickers):
            file_path = f"{data_dir}{symbol}_data.csv"
            print(f"Processing {symbol} ({i + 1}/{len(tickers)})...")

            try:
                existing_df = None
                if os.path.exists(file_path):
                    try:
                        existing_df = pd.read_csv(file_path)
                        if not existing_df.empty:
                            existing_df['date'] = pd.to_datetime(existing_df['date'])
                            
                            # Check if we have at least 10 years of data
                            years_of_data = (existing_df['date'].max() - existing_df['date'].min()).days / 365.25
                            if years_of_data >= 9.9: # Using 9.9 to account for weekends/holidays
                                print(f"--> {symbol} already has {years_of_data:.1f} years of data. Skipping.")
                                continue
                    except pd.errors.EmptyDataError:
                        print(f"--> {symbol} data file is empty. Will fetch new data.")


                # If file doesn't exist or has less than 10 years of data, fetch it.
                print(f"--> Fetching 10 years of data for {symbol}.")
                contract = Stock(symbol, 'SMART', 'USD')
                bars = ib.reqHistoricalData(
                    contract, endDateTime='', durationStr=duration,
                    barSizeSetting=bar_size, whatToShow='TRADES', useRTH=True
                )

                if bars:
                    new_df = pd.DataFrame(bars)
                    new_df['date'] = pd.to_datetime(new_df['date'])
                    
                    if existing_df is not None and not existing_df.empty:
                        # Append new data and remove duplicates
                        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date'], keep='last')
                    else:
                        combined_df = new_df

                    combined_df = combined_df.sort_values(by='date')
                    combined_df['date'] = combined_df['date'].dt.date
                    combined_df.to_csv(file_path, index=False)
                    print(f"--> Data for {symbol} saved to {file_path}")
                else:
                    print(f"--> No new data returned for {symbol}.")

                ib.sleep(2)  # Pacing request
            except Exception as e:
                print(f"--> Error processing {symbol}: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()

if __name__ == "__main__":
    # Ensure this calls the function to get ALL stocks
    fetch_all_sp500_data()