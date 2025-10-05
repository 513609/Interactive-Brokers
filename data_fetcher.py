import configparser
from ib_insync import IB, Stock
import pandas as pd

def fetch_and_save_data(symbol='AAPL', duration='2 Y', bar_size='1 day'):
    """Connects to IBKR, fetches historical data, and saves it to a CSV."""
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    ibkr_config = config['ibkr']

    ib = IB()
    try:
        ib.connect(
            ibkr_config.get('host'),
            ibkr_config.getint('port'),
            clientId=101  # Use a different client ID from your main bot
        )
        print("Connection successful.")

        # Define the contract for the stock
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Request historical data
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )

        if not bars:
            print(f"No data returned for {symbol}.")
            return

        # Convert to pandas DataFrame and save
        df = pd.DataFrame(bars)
        # Ensure the date column is just the date part (not datetime)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        file_path = f"data/historical_data/{symbol}_data.csv"
        df.to_csv(file_path, index=False)
        print(f"Data for {symbol} saved successfully to {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()

if __name__ == "__main__":
    # Run this script once to get your data
    fetch_and_save_data(symbol='AAPL')