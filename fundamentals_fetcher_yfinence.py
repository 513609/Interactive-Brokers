import yfinance as yf
from utils.sp500_tickers import get_sp500_tickers
import os
import pandas as pd
import time

def create_quarterly_earnings_dataset():
    """
    Fetches a historical dataset of actual earnings report dates, EPS,
    analyst estimates, and revenue for all S&P 500 stocks.
    """
    # --- 1. Setup ---
    tickers = get_sp500_tickers() 
    if not tickers:
        print("Could not retrieve ticker list. Exiting.")
        return

    output_dir = "data/eps_data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- 2. Data Fetching and Processing Loop ---
    for i, symbol in enumerate(sorted(tickers)):
        output_path = os.path.join(output_dir, f"{symbol}_EPS.csv")
        if os.path.exists(output_path):
            print(f"Data for {symbol} already exists. Skipping.")
            continue

        print(f"Processing {symbol} ({i + 1}/{len(tickers)})...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # --- Get Earnings Dates and EPS data ---
            earnings_dates = ticker.get_earnings_dates(limit=40)
            if earnings_dates is None or earnings_dates.empty:
                print(f"--> No earnings dates found for {symbol}. Skipping.")
                continue

            # Clean up the earnings data
            earnings_dates.reset_index(inplace=True)
            earnings_dates.rename(columns={
                'Earnings Date': 'EarningsDate',
                'Reported EPS': 'ReportedEPS',
                'EPS Estimate': 'EstimateEPS'
            }, inplace=True)
            
            earnings_dates['EarningsDate'] = pd.to_datetime(earnings_dates['EarningsDate'].dt.date)

            # --- Get Long-Term Financials for Revenue ---
            quarterly_financials = ticker.quarterly_financials
            if quarterly_financials is None or quarterly_financials.empty:
                print(f"--> No quarterly financials found for {symbol}. Skipping revenue.")
                revenue_df = pd.DataFrame() # Create empty df to avoid errors
            else:
                financials_df = quarterly_financials.T.reset_index()
                financials_df.rename(columns={'index': 'QuarterEndDate'}, inplace=True)
                financials_df['QuarterEndDate'] = pd.to_datetime(financials_df['QuarterEndDate'])
                
                revenue_col = 'Total Revenue' if 'Total Revenue' in financials_df.columns else 'Revenues'
                if revenue_col not in financials_df.columns:
                     print(f"--> Could not find revenue column for {symbol}. Revenue will be blank.")
                     revenue_df = pd.DataFrame()
                else:
                    revenue_df = financials_df[['QuarterEndDate', revenue_col]].rename(columns={revenue_col: 'ReportedRevenue'})

            # --- Map Revenue to the Correct Earnings Date ---
            earnings_dates['QuarterEndDate'] = earnings_dates['EarningsDate'].apply(
                lambda date: date - pd.tseries.offsets.QuarterBegin(1, startingMonth=date.month) + pd.tseries.offsets.QuarterEnd()
            )

            if not revenue_df.empty:
                final_df = pd.merge(earnings_dates, revenue_df, on='QuarterEndDate', how='left')
            else:
                final_df = earnings_dates
                final_df['ReportedRevenue'] = None

            # --- MODIFIED: Data Cleaning Step ---
            # Drop any rows where ReportedEPS or EstimateEPS is missing (NaN)
            final_df.dropna(subset=['ReportedEPS', 'EstimateEPS'], inplace=True)

            if final_df.empty:
                print(f"--> No complete EPS/Estimate rows found for {symbol} after cleaning. Skipping.")
                continue
            # --- END OF MODIFICATION ---

            # --- Finalize and Save ---
            final_df['Symbol'] = symbol
            final_df['EPSBeat'] = (final_df['ReportedEPS'] > final_df['EstimateEPS']).astype(int)
            
            final_cols = ['Symbol', 'EarningsDate', 'ReportedEPS', 'EstimateEPS', 'EPSBeat', 'ReportedRevenue']
            final_df = final_df.reindex(columns=final_cols)
            
            final_df.sort_values(by=['EarningsDate'], inplace=True)
            
            # CORRECTED: Use the correct format string for the date
            final_df['EarningsDate'] = final_df['EarningsDate'].dt.strftime('%Y-%m-%d')
            
            final_df.to_csv(output_path, index=False)
            print(f"--> Successfully created dataset for {symbol} at: {output_path}")

            time.sleep(0.5)

        except Exception as e:
            print(f"--> An error occurred for {symbol}: {e}")

if __name__ == "__main__":
    create_quarterly_earnings_dataset()

