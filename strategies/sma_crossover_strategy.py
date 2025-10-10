import pandas as pd
from typing import Dict, List
from strategies.base_strategy import BaseStrategy

class SmaCrossoverStrategy(BaseStrategy):
    """
    A concrete implementation of a strategy based on Simple Moving Average crossovers.
    """
    def __init__(self, symbols: List[str], short_window: int = 20, long_window: int = 50):
        self.symbols = symbols
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Generates BUY/SELL signals based on SMA crossovers."""
        signals = {}
        for symbol in self.symbols:
            if symbol in data and len(data[symbol]) >= self.long_window:
                df = data[symbol]
                short_sma = df['close'].rolling(window=self.short_window).mean()
                long_sma = df['close'].rolling(window=self.long_window).mean()
                
                if short_sma.iloc[-1] > long_sma.iloc[-1] and short_sma.iloc[-2] <= long_sma.iloc[-2]:
                    signals[symbol] = 'BUY'
                elif short_sma.iloc[-1] < long_sma.iloc[-1] and short_sma.iloc[-2] >= long_sma.iloc[-2]:
                    signals[symbol] = 'SELL'
                else:
                    signals[symbol] = 'HOLD'
            else:
                signals[symbol] = 'HOLD' # Not enough data to generate a signal
        return signals
