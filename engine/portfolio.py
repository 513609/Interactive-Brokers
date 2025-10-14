import pandas as pd
from typing import Dict
from engine.execution_handler import ExecutionHandler

class Portfolio:
    """
    Manages the portfolio's state, including cash, holdings, and value history.
    It uses the ExecutionHandler to process trades realistically.
    """
    def __init__(self, initial_capital: float, execution_handler: ExecutionHandler):
        self.initial_capital = initial_capital
        self.execution_handler = execution_handler
        self.cash: float = initial_capital
        self.holdings: Dict[str, float] = {}  # {symbol: shares}
        self.history: list = []

    def update_value(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]):
        """Calculates the total market value of the portfolio for the current day."""
        total_value = self.cash
        for symbol, shares in self.holdings.items():
            if shares > 0 and symbol in data and date in data[symbol].index:
                total_value += shares * data[symbol].loc[date, 'close']
        self.history.append({'date': date, 'value': total_value})

    def execute_signal(self, date: pd.Timestamp, symbol: str, signal: str, data: Dict[str, pd.DataFrame], capital_per_trade_percent: float):
        """Processes a trading signal from a strategy."""
        if symbol not in data or date not in data[symbol].index:
            return
        
        current_price = data[symbol].loc[date, 'close']
        
        # --- SELL LOGIC ---
        if signal == 'SELL' and self.holdings.get(symbol, 0) > 0:
            shares_to_sell = self.holdings[symbol]
            execution = self.execution_handler.execute_order(current_price, shares_to_sell, 'SELL')
            self.cash += execution['proceeds']
            self.holdings.pop(symbol) # Remove from holdings

        # --- BUY LOGIC ---
        elif signal == 'BUY' and self.holdings.get(symbol, 0) == 0:
            portfolio_value = self.history[-1]['value'] if self.history else self.initial_capital
            investment_amount = portfolio_value * capital_per_trade_percent
            
            if self.cash >= investment_amount:
                shares_to_buy = investment_amount / current_price
                execution = self.execution_handler.execute_order(current_price, shares_to_buy, 'BUY')
                if self.cash >= execution['cost']:
                    self.cash -= execution['cost']
                    self.holdings[symbol] = execution['shares']

    def get_history_df(self) -> pd.DataFrame:
        """Returns the portfolio value history as a DataFrame."""
        return pd.DataFrame(self.history).set_index('date')
