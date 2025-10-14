import random
from typing import Dict

class ExecutionHandler:
    """
    Simulates the execution of trades, applying realistic costs like
    commission, bid-ask spread, and slippage.
    """
    def __init__(self, commission_per_trade: float, spread_percent: float, slippage_percent: float):
        self.commission = commission_per_trade
        self.spread = spread_percent
        self.slippage = slippage_percent

    def _simulate_execution_price(self, nominal_price: float, action: str) -> float:
        """Calculates a realistic execution price."""
        if action == 'BUY':
            # When buying, you cross the spread (pay slightly more) and may have positive slippage
            spread_adjustment = nominal_price * (self.spread / 2)
            slippage_adjustment = nominal_price * self.slippage * random.uniform(0, 1)
            return nominal_price + spread_adjustment + slippage_adjustment
        elif action == 'SELL':
            # When selling, you cross the spread (receive slightly less) and may have negative slippage
            spread_adjustment = nominal_price * (self.spread / 2)
            slippage_adjustment = nominal_price * self.slippage * random.uniform(0, 1)
            return nominal_price - spread_adjustment - slippage_adjustment
        return nominal_price

    def execute_order(self, current_price: float, shares: float, action: str) -> Dict:
        """Executes an order and returns the execution details."""
        execution_price = self._simulate_execution_price(current_price, action)
        total_cost = (execution_price * shares) + self.commission
        total_proceeds = (execution_price * shares) - self.commission
        
        if action == 'BUY':
            return {'price': execution_price, 'cost': total_cost, 'shares': shares}
        elif action == 'SELL':
            return {'price': execution_price, 'proceeds': total_proceeds, 'shares': shares}
        return {}
