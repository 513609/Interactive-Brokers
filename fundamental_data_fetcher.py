import yfinance as yf
from utils.sp500_tickers import get_sp500_tickers
import os
import json
import pandas as pd
import time

def fetch_all_sp500_fundamentals_yfinance():
    """
    Fetches fundamental data for all S&P 500 stocks using yfinance
    and saves them to JSON files.
    """
    # --- 1. Configuration and Setup ---
    tickers = get_sp500_tickers()
    if not tickers:
        print("Could not retrieve ticker list. Exiting.")
        return

    data_dir = "data/fundamental_data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # --- 2. Data Fetching Loop ---
    for i, symbol in enumerate(sorted(tickers)):
        file_path = f"{data_dir}{symbol}_fundamentals.json"

        # Check if file already exists to avoid re-downloading
        if os.path.exists(file_path):
            print(f"Data for {symbol} already exists. Skipping.")
            continue

        print(f"Fetching fundamental data for {symbol} ({i + 1}/{len(tickers)})...")
        
        try:
            # Create a Ticker object
            ticker = yf.Ticker(symbol)
            
            # --- Fetch a wide range of data ---
            # .info contains a dictionary of company summary data
            company_info = ticker.info
            
            # --- Fetch Annual Data ---
            income_statement_df = ticker.financials
            balance_sheet_df = ticker.balance_sheet
            cash_flow_df = ticker.cashflow
            
            # --- Fetch Quarterly Data ---
            quarterly_income_statement_df = ticker.quarterly_financials
            quarterly_balance_sheet_df = ticker.quarterly_balance_sheet
            quarterly_cash_flow_df = ticker.quarterly_cashflow

            # --- Fetch Other Useful Data ---
            recommendations_df = ticker.recommendations
            major_holders_df = ticker.major_holders

            # --- 3. Data Cleaning ---
            # Create a list of all DataFrame objects to clean them in a loop
            dataframes_to_clean = [
                income_statement_df, balance_sheet_df, cash_flow_df,
                quarterly_income_statement_df, quarterly_balance_sheet_df, quarterly_cash_flow_df,
                recommendations_df, major_holders_df
            ]

            cleaned_dataframes = []
            for df in dataframes_to_clean:
                # FIX 1: Convert DataFrame columns (Timestamps) to strings for JSON compatibility.
                if not df.empty and isinstance(df.columns, pd.DatetimeIndex):
                    df.columns = df.columns.astype(str)
                
                # FIX 2: Replace NaN with None for valid JSON output (NaN is not valid JSON).
                df = df.where(pd.notnull(df), None)
                cleaned_dataframes.append(df)

            # Unpack the cleaned dataframes back to their original variables
            (income_statement_df, balance_sheet_df, cash_flow_df,
             quarterly_income_statement_df, quarterly_balance_sheet_df, quarterly_cash_flow_df,
             recommendations_df, major_holders_df) = cleaned_dataframes

            # Check if we got any useful data
            if not company_info.get('symbol'):
                print(f"--> No data returned for {symbol}. It may be delisted or invalid on Yahoo Finance.")
                continue

            # --- Combine all data into a single dictionary ---
            all_fundamentals = {
                'info': company_info,
                'income_statement': income_statement_df.to_dict(orient='index'),
                'balance_sheet': balance_sheet_df.to_dict(orient='index'),
                'cash_flow': cash_flow_df.to_dict(orient='index')
            }

            # Save the dictionary to a JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(all_fundamentals, f, indent=4)
            print(f"--> Successfully saved fundamental data for {symbol}.")

            # Be respectful to the data source by adding a small delay
            time.sleep(0.5)

        except Exception as e:
            print(f"--> An error occurred for {symbol}: {e}")

if __name__ == "__main__":
    fetch_all_sp500_fundamentals_yfinance()