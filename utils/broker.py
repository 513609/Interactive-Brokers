# In utils/broker.py
from ib_insync import Stock, Order

def place_order(ib_connection, symbol, quantity, order_type='MKT', action='BUY'):
    """
    Creates and places an order with IBKR.
    'ib_connection' is the active IB() object.
    """
    contract = Stock(symbol, 'SMART', 'USD')
    order = Order(action=action, orderType=order_type, totalQuantity=quantity)
    
    trade = ib_connection.placeOrder(contract, order)
    
    print(f"Placed {action} order for {quantity} shares of {symbol}.")
    return trade