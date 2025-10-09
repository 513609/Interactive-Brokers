import configparser
import sys
from ib_insync import IB, Stock, Order, util
from utils.logger import log

def print_portfolio(ib: IB, account_number: str):
    """Fetches and prints the current portfolio and key account values."""
    log.info("--- Fetching Account and Portfolio Details ---")
    
    # Fetch account summary
    account_summary = ib.accountSummary(account_number)
    summary_data = {item.tag: item.value for item in account_summary}
    
    net_liquidation = float(summary_data.get('NetLiquidation', '0.0'))
    available_funds = float(summary_data.get('AvailableFunds', '0.0'))
    
    log.info(f"Account Net Liquidation: ${net_liquidation:,.2f}")
    log.info(f"Available Funds for Trading: ${available_funds:,.2f}")

    # Fetch and print portfolio positions
    portfolio = ib.portfolio()
    if not portfolio:
        log.info("Portfolio is currently empty.")
    else:
        log.info("Current Portfolio:")
        for position in portfolio:
            log.info(
                f"  - {position.contract.symbol}: {position.position} shares "
                f"@ avg cost ${position.averageCost:,.2f} "
                f"(Market Value: ${position.marketValue:,.2f})"
            )
    log.info("-" * 44)

def run_trade_test():
    """
    Connects to IBKR, prints the portfolio, places a test trade,
    and prints the portfolio again.
    """
    log.info("--- Starting IBKR Portfolio and Trade Test ---")

    # 1. Load Configuration
    try:
        config = configparser.ConfigParser()
        config.read('config/config.ini')
        ibkr_config = config['ibkr']
        account_number = ibkr_config.get('account_number')
        host = ibkr_config.get('host')
        port = ibkr_config.getint('port')
        client_id = ibkr_config.getint('client_id') + 200 # Use a unique client_id
    except Exception as e:
        log.error(f"Failed to read configuration: {e}")
        sys.exit(1)

    ib = IB()
    try:
        # 2. Connect to IBKR
        log.info(f"Connecting to IBKR at {host}:{port}...")
        ib.connect(host, port, clientId=client_id, timeout=10)
        log.info("Connection successful.")

        # 3. Print Initial Portfolio
        print_portfolio(ib, account_number)

        # 4. Define the Trade (1 share of Google)
        contract = Stock('GOOGL', 'SMART', 'USD')
        order = Order(action='BUY', totalQuantity=1, orderType='MKT')

        # 5. Check if the trade is feasible ('What-If' order)
        log.info("Performing 'What-If' check for buying 1 share of GOOGL...")
        what_if_order = ib.whatIfOrder(contract, order)
        
        # Correctly fetch AvailableFunds from the account summary list
        account_summary = ib.accountSummary(account_number)
        summary_data = {item.tag: item.value for item in account_summary}
        equity_with_loan = float(what_if_order.equityWithLoanChange)
        available_funds = float(summary_data.get('AvailableFunds', '0.0'))

        # The 'equityWithLoanChange' shows the cash impact. For a BUY, it's negative.
        # We check if our available funds can cover this cost.
        if available_funds >= abs(equity_with_loan):
            log.info(f"--> [SUCCESS] Check passed. Sufficient funds available.")
            log.info(f"    - Estimated Cost: ${abs(equity_with_loan):,.2f}")
            log.info(f"    - Estimated Commission: {what_if_order.maxCommission}")
            
            # 6. Place the actual trade
            log.info("Placing live market order for 1 GOOGL share...")
            trade = ib.placeOrder(contract, order)
            
            # 7. Wait for the order to complete
            log.info("Waiting for order to fill...")
            ib.sleep(5) # Initial wait
            
            while not trade.isDone():
                ib.sleep(1)
            
            log.info(f"--> [SUCCESS] Trade completed. Status: {trade.orderStatus.status}")

        else:
            log.error("--> [FAIL] Insufficient funds to place the trade.")
            log.error(f"    - Required Funds: ${abs(equity_with_loan):,.2f}")
            log.error(f"    - Available Funds: ${available_funds:,.2f}")

        # 8. Print Final Portfolio
        log.info("\nWaiting a few seconds for portfolio to update...")
        ib.sleep(5) # Give IB's systems a moment to update the portfolio view
        print_portfolio(ib, account_number)

    except ConnectionRefusedError:
        log.error(f"Connection refused. Is TWS or IB Gateway running on port {port}?")
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}")
    finally:
        if ib.isConnected():
            log.info("Disconnecting from IBKR...")
            ib.disconnect()
        log.info("--- Test Finished ---")

if __name__ == "__main__":
    # Make sure the logger from ib_insync prints to the console
    util.logToConsole()
    run_trade_test()