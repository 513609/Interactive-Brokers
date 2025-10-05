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
def fetch_all_sp500_data(duration='2 Y', bar_size='1 day'):
    """
    Fetches historical data for all S&P 500 stocks and saves them to CSVs.
    """
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    ibkr_config = config['ibkr']
    tickers = get_sp500_tickers()

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
            # Check if file already exists to avoid re-downloading
            if os.path.exists(f"{data_dir}{symbol}_data.csv"):
                print(f"Data for {symbol} already exists. Skipping.")
                continue

            print(f"Fetching data for {symbol} ({i + 1}/{len(tickers)})...")
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                bars = ib.reqHistoricalData(
                    contract, endDateTime='', durationStr=duration,
                    barSizeSetting=bar_size, whatToShow='TRADES', useRTH=True
                )
                if bars:
                    df = pd.DataFrame(bars)
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    df.to_csv(f"{data_dir}{symbol}_data.csv", index=False)
                else:
                    print(f"--> No data returned for {symbol}.")

                ib.sleep(2) # Pacing request
            except Exception as e:
                print(f"--> Error fetching data for {symbol}: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()

if __name__ == "__main__":
    # Ensure this calls the function to get ALL stocks
    fetch_all_sp500_data()