import configparser
import sys
from ib_insync import IB, Stock, Order
from utils.logger import log

def run_health_check():
    """
    Connects to IBKR and performs a series of checks to ensure
    the environment is ready for algorithmic and order trading.
    """
    log.info("--- Starting IBKR Health Check ---")
    all_checks_passed = True

    # 1. Configuration Check
    try:
        log.info("1. Checking configuration file (config/config.ini)...")
        config = configparser.ConfigParser()
        config_path = 'config/config.ini'
        if not config.read(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        ibkr_config = config['ibkr']
        account_number = ibkr_config.get('account_number')
        host = ibkr_config.get('host')
        port = ibkr_config.getint('port')
        # Use a unique client_id for the health check to avoid conflicts
        client_id = ibkr_config.getint('client_id') + 100

        if not all([account_number, host, port, client_id]):
            raise ValueError("One or more required config values are missing in [ibkr] section.")

        log.info("--> [SUCCESS] Configuration loaded and valid.")
    except Exception as e:
        log.error(f"--> [FAIL] Failed to read or parse configuration: {e}")
        log.error("--- Health Check Incomplete ---")
        sys.exit(1) # Exit if config is bad, no point in continuing

    ib = IB()
    try:
        # 2. Connection Check
        log.info(f"2. Attempting to connect to IBKR at {host}:{port}...")
        ib.connect(host, port, clientId=client_id, timeout=10)
        log.info("--> [SUCCESS] Connection to IBKR established.")
        log.info(f"    - TWS/Gateway Version: {ib.client.serverVersion()}")
        log.info(f"    - Connection Time: {ib.reqCurrentTime().isoformat()}")

        # 3. Account Check
        log.info(f"3. Verifying account number '{account_number}'...")
        managed_accounts = ib.managedAccounts()
        if account_number in managed_accounts:
            log.info(f"--> [SUCCESS] Account '{account_number}' is available.")
        else:
            all_checks_passed = False
            log.error(f"--> [FAIL] Account '{account_number}' not found in managed accounts: {managed_accounts}")

        # 4. Market Data Check (Algo Trading Prerequisite)
        log.info("4. Checking market data subscription for a common stock (SPY)...")
        spy_contract = Stock('SPY', 'SMART', 'USD')
        ib.reqMktData(spy_contract, '', False, False)
        ib.sleep(2)  # Allow time for data to arrive
        ticker = ib.ticker(spy_contract)
        ib.cancelMktData(spy_contract)

        if ticker and ticker.last > 0:
            log.info(f"--> [SUCCESS] Market data received for SPY. Last price: {ticker.last}")
        else:
            all_checks_passed = False
            log.error("--> [FAIL] Did not receive market data for SPY. Check market data subscriptions.")

        # 5. Order Trading Check
        log.info("5. Performing a 'What-If' order check to verify trading setup...")
        what_if_contract = Stock('AAPL', 'SMART', 'USD')
        what_if_order = Order(action='BUY', totalQuantity=1, orderType='MKT')

        # Use whatIfOrder to check margin and commissions without placing the order
        order_state = ib.whatIfOrder(what_if_contract, what_if_order)

        # A successful 'what-if' will have a status of 'PreSubmitted'. Any other status
        # (like 'Inactive' or 'Cancelled' due to errors like insufficient funds) is a failure.
        if hasattr(order_state, 'status') and order_state.status == 'PreSubmitted' and 'error' not in order_state.warningText.lower():
            log.info("--> [SUCCESS] 'What-If' order was accepted. Trading setup appears correct.")
            log.info(f"    - Estimated Initial Margin: {order_state.initMarginChange}")
            log.info(f"    - Estimated Commission: {order_state.maxCommission}")
        else:
            all_checks_passed = False
            log.error("--> [FAIL] 'What-If' order was rejected or failed.")
            warning = getattr(order_state, 'warningText', 'N/A')
            log.error(f"    - Status: {getattr(order_state, 'status', 'N/A')}, Warning/Error: {warning}")
            log.error("    - Check trading permissions and account status in Account Management.")

    except ConnectionRefusedError:
        all_checks_passed = False
        log.error(f"--> [FAIL] Connection refused. Is TWS or IB Gateway running and configured for API connections on port {port}?")
    except Exception as e:
        all_checks_passed = False
        log.error(f"--> [FAIL] An unexpected error occurred: {e}")
    finally:
        if ib.isConnected():
            log.info("Disconnecting from IBKR...")
            ib.disconnect()

        log.info("--- Health Check Finished ---")
        if all_checks_passed:
            log.info("✅ All checks passed. System is ready for trading.")
        else:
            log.error("❌ One or more checks failed. Please review the logs and correct the issues.")
            sys.exit(1)

if __name__ == "__main__":
    run_health_check()