import pandas as pd
import os
from typing import List, Dict, Generator, Tuple

class DataHandler:
    """
    Handles loading, preparing, and streaming historical market data for the backtest.
    """
    def __init__(self, symbols: List[str], data_dir: str = "data/"):
        self.symbols = symbols
        self.data_dir = data_dir
        self.all_data: Dict[str, pd.DataFrame] = self._load_all_data()
        self.master_timeline: pd.DatetimeIndex = self._create_master_timeline()

    def _load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Loads all CSV files for the given symbols into a dictionary."""
        data = {}
        for symbol in self.symbols:
            file_path = os.path.join(self.data_dir, f"{symbol}_data.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col='date', parse_dates=True)
                data[symbol] = df
            else:
                raise FileNotFoundError(f"Data file for {symbol} not found at {file_path}")
        return data

    def _create_master_timeline(self) -> pd.DatetimeIndex:
        """Creates a sorted, unique list of all dates available across all data files."""
        master_index = pd.DatetimeIndex([])
        for df in self.all_data.values():
            master_index = master_index.union(df.index)
        return master_index.sort_values()

    def stream_bars(self, start_date_str: str, end_date_str: str) -> Generator[Tuple[pd.Timestamp, Dict[str, pd.DataFrame]], None, None]:
        """
        A generator that yields data for one day at a time within a specified date range.
        This simulates the market moving forward one bar at a time.
        """
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        for date in self.master_timeline:
            if start_date <= date <= end_date:
                # Provides all historical data UP TO the current date for calculations
                current_data_slice = {
                    symbol: df.loc[df.index <= date]
                    for symbol, df in self.all_data.items()
                    if not df.loc[df.index <= date].empty
                }
                yield date, current_data_slice
