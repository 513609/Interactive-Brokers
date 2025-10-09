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
        log.info("4. Checking market data subscription for a common stock (AAPL)...")
        aapl_contract = Stock('AAPL', 'SMART', 'USD')
        ib.reqMktData(aapl_contract, '', False, False)
        ib.sleep(3)  # Allow a bit more time for data to arrive
        ticker = ib.ticker(aapl_contract)
        is_delayed_attempt = False

        # If live data fails (common with paper accounts), try requesting delayed data
        if not (ticker and ticker.last > 0):
            is_delayed_attempt = True
            log.warning("--> [INFO] Live market data request failed. Trying again with delayed data setting...")
            ib.cancelMktData(aapl_contract) # Clean up the previous request
            ib.reqMarketDataType(3) # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen
            ib.reqMktData(aapl_contract, '', False, False)
            ib.sleep(3)
            ticker = ib.ticker(aapl_contract)

        # Final check after potentially trying both live and delayed
        if ticker and ticker.last > 0:
            data_type = "Delayed" if is_delayed_attempt else "Live"
            log.info(f"--> [SUCCESS] {data_type} market data received for AAPL. Last price: {ticker.last}")
        else:
            all_checks_passed = False
            log.error("--> [FAIL] Did not receive market data for AAPL. Check market data subscriptions.")
            log.error("    - This usually means you lack market data subscriptions for US Equities.")
            log.error("    - For paper accounts, you may need to subscribe in Account Management.")

        ib.cancelMktData(aapl_contract)
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