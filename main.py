import configparser
from ib_insync import IB
from utils.logger import log

def main():
    """Main function to run the trading bot."""
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    ibkr_config = config['ibkr']
    account_number = ibkr_config.get('account_number')
    host = ibkr_config.get('host')
    port = ibkr_config.getint('port')
    client_id = ibkr_config.getint('client_id')

    ib = IB()
    try:
        log.info("Connecting to IBKR TWS/Gateway...")
        ib.connect(host, port, clientId=client_id)
        
        log.info("Connection successful.")
        log.info(f"Connected to account: {ib.managedAccounts()}")

        # Your trading logic will go here
        log.info("Holding connection for 5 seconds...")
        ib.sleep(5)

    except Exception as e:
        log.error(f"An error occurred: {e}")
    finally:
        if ib.isConnected():
            log.info("Disconnecting from IBKR...")
            ib.disconnect()

if __name__ == "__main__":
    main()