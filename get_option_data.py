import configparser
import sys
import logging
import pandas as pd
from ib_insync import IB, Stock, Option, util
from datetime import date
from io import StringIO

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_option_chain(ib: IB, symbol: str, exchange: str, currency: str):
    """
    Fetches the option chain for a given symbol using an existing IB connection.

    Args:
        ib (IB): An active and connected ib_insync IB instance.
        symbol (str): The stock ticker symbol (e.g., 'ASML').
        exchange (str): The primary exchange of the stock (e.g., 'AEB').
        currency (str): The currency of the stock (e.g., 'EUR').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing (calls_df, puts_df).
    """
    logging.info(f"Fetching option chain for {symbol}...")
    
    # 1. Define and qualify the underlying stock contract
    stock_contract = Stock(symbol, exchange, currency)
    contract_details = ib.reqContractDetails(stock_contract)
    if not contract_details:
        logging.error(f"Could not find contract details for {symbol}.")
        return pd.DataFrame(), pd.DataFrame()
    
    stock_conId = contract_details[0].contract.conId
    logging.info(f"Found conId for {symbol}: {stock_conId}")

    # 2. Request option chain parameters
    chains = ib.reqSecDefOptParams(
        underlyingSymbol=stock_contract.symbol,
        futFopExchange='',
        underlyingSecType=stock_contract.secType,
        underlyingConId=stock_conId
    )
    if not chains:
        logging.warning(f"No option chains found for {symbol}.")
        return pd.DataFrame(), pd.DataFrame()

    # 3. Build individual Option contracts
    option_contracts = []
    for chain in chains:
        for expiration in chain.expirations:
            for strike in chain.strikes:
                for right in ['C', 'P']:
                    contract = Option(
                        symbol=symbol,
                        lastTradeDateOrContractMonth=expiration,
                        strike=strike, right=right, exchange=chain.exchange,
                        currency=currency, tradingClass=chain.tradingClass,
                        multiplier=chain.multiplier
                    )
                    option_contracts.append(contract)
    
    logging.info(f"Generated {len(option_contracts)} total option contracts.")

    # 4. Create and separate DataFrame
    options_df = pd.DataFrame([opt.dict() for opt in option_contracts])
    calls_df = options_df[options_df['right'] == 'C'].copy()
    puts_df = options_df[options_df['right'] == 'P'].copy()
    
    logging.info(f"✅ Separated into {len(calls_df)} Calls and {len(puts_df)} Puts.")
    return calls_df, puts_df

def get_option_prices(ib: IB, contracts: list[Option]) -> pd.DataFrame:
    """
    Fetches real-time market data for a list of option contracts.

    Args:
        ib (IB): An active and connected ib_insync IB instance.
        contracts (list[Option]): A list of ib_insync Option contract objects.

    Returns:
        pd.DataFrame: A DataFrame with the price data for the requested contracts.
    """
    if not contracts:
        logging.warning("No contracts provided to fetch prices for.")
        return pd.DataFrame()

    logging.info(f"Requesting market data for {len(contracts)} contracts...")
    
    # 1. Request market data for all contracts. ib_insync handles the batching.
    for contract in contracts:
        ib.reqMktData(contract, '', False, False)
    
    # Give the server time to return all the data
    ib.sleep(2) # Adjust sleep time if data seems incomplete

    price_data = []
    # 2. Retrieve the ticker data for each contract
    for contract in contracts:
        ticker = ib.ticker(contract)
        price_data.append({
            'conId': contract.conId,
            'symbol': contract.symbol,
            'expiration': contract.lastTradeDateOrContractMonth,
            'strike': contract.strike,
            'right': contract.right,
            'bid': ticker.bid,
            'ask': ticker.ask,
            'last': ticker.last,
            'close': ticker.close,
            'volume': ticker.volume
        })
    
    # 3. Clean up by canceling the market data subscriptions
    for contract in contracts:
        ib.cancelMktData(contract)
        
    logging.info("✅ Price data retrieved.")
    return pd.DataFrame(price_data)

if __name__ == "__main__":
    # Ensure ib_insync's logs also go to the console for detailed debugging
    util.logToConsole()
    
    # It's best practice to wrap the connection in a single block
    ib = IB()
    try:
        # --- Load Configuration ---
        config = configparser.ConfigParser()
        config.read('config/config.ini')
        host = config['ibkr']['host']
        port = int(config['ibkr']['port'])
        client_id = int(config['ibkr']['client_id']) + 400

        # --- Connect to IBKR ---
        ib.connect(host, port, clientId=client_id)

        # ========================================================================
        # STEP 1: Get all available option contracts
        # ========================================================================
        calls_df, puts_df = get_option_chain(ib, symbol='ASML', exchange='AEB', currency='EUR')

        if calls_df.empty or puts_df.empty:
            raise RuntimeError("Failed to retrieve option chain. Exiting.")

        # ========================================================================
        # STEP 2: Filter the chain to get a small list of contracts you want prices for
        # --- This is the CRITICAL step to avoid hitting API limits ---
        # ========================================================================
        
        # Example: Let's get prices for the nearest expiration date
        # And for a few strike prices (e.g., 5 strikes above and 5 below a price)
        
        # Convert string dates to actual dates for sorting
        calls_df['expiration_date'] = pd.to_datetime(calls_df['lastTradeDateOrContractMonth'])
        
        # Find the closest expiration date from today
        closest_expiration = calls_df[calls_df['expiration_date'] > pd.Timestamp.now()]['expiration_date'].min()
        closest_expiration_str = closest_expiration.strftime('%Y%m%d')
        
        logging.info(f"Filtering for closest expiration: {closest_expiration_str}")

        # Filter the DataFrame for this specific expiration
        filtered_calls = calls_df[calls_df['lastTradeDateOrContractMonth'] == closest_expiration_str].copy()
        
        # Example: Assume the current ASML stock price is around 950 EUR
        # Let's pick a few strikes around this price.
        assumed_stock_price = 950
        strike_range = 50
        
        filtered_calls = filtered_calls[
            (filtered_calls['strike'] >= assumed_stock_price - strike_range) &
            (filtered_calls['strike'] <= assumed_stock_price + strike_range)
        ]
        
        logging.info(f"Found {len(filtered_calls)} call contracts matching filter criteria.")

        # ========================================================================
        # STEP 3: Convert the filtered DataFrame rows back into Contract objects
        # ========================================================================
        contracts_to_price = [
            Option(**row) for index, row in filtered_calls[['symbol', 'lastTradeDateOrContractMonth', 'strike', 'right', 'exchange', 'currency', 'tradingClass']].iterrows()
        ]
        
        # ========================================================================
        # STEP 4: Pass the small list of contracts to the pricing function
        # ========================================================================
        prices_df = get_option_prices(ib, contracts_to_price)

        if not prices_df.empty:
            print("\n" + "="*60)
            print("📈 Real-Time Prices for Filtered ASML Call Options")
            print(f"Expiration: {closest_expiration_str}")
            print("="*60)
            print(prices_df.to_string())
        else:
            print("\nCould not retrieve price data for the filtered options.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if ib.isConnected():
            logging.info("Disconnecting from IBKR...")
            ib.disconnect()