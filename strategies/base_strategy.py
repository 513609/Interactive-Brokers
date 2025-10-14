from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies. It defines the common
    interface that the backtesting engine uses to interact with any strategy.
    """
    @abstractmethod
    def generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        The core method of any strategy. It must be implemented by subclasses.

        Args:
            date (pd.Timestamp): The current date of the backtest simulation.
            data (Dict[str, pd.DataFrame]): A dictionary of historical data up to 'date'.

        Returns:
            Dict[str, str]: A dictionary of signals, e.g., {'AAPL': 'BUY', 'MSFT': 'SELL'}.
                            Valid signals are 'BUY', 'SELL', or 'HOLD'.
        """
        raise NotImplementedError("Each strategy must implement the generate_signals method.")
