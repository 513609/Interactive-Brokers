import pandas_datareader.data as web
import pandas as pd
from pathlib import Path
import datetime

def download_fama_french_factors(
    start_date: datetime.date = datetime.date(1926, 7, 1)
) -> pd.DataFrame:
    """
    Downloads the Fama-French 3-Factor model data (monthly) from Ken French's data library.

    Args:
        start_date (datetime.date): The start date for fetching the data. 
                                     Defaults to the earliest available date.

    Returns:
        pd.DataFrame: A DataFrame containing the Mkt-RF, SMB, HML, and RF factors,
                      with a proper DatetimeIndex. Returns are in decimal form (e.g., 0.01 for 1%).
    """
    print("Downloading Fama-French 3-Factor data...")
    # The data reader returns a dictionary of DataFrames. The monthly data is the first item.
    ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date)
    
    # The desired DataFrame is at index 0 for monthly factors
    df_ff = ff_data[0]
    
    # The data is in percentage, so we convert it to decimal form
    df_ff = df_ff / 100
    
    # The index is a PeriodIndex, convert it to a DatetimeIndex for easier merging
    df_ff.index = df_ff.index.to_timestamp()
    
    print("Download complete.")
    return df_ff

if __name__ == "__main__":
    # Define the output directory and create it if it doesn't exist
    output_dir = Path("data/FamaFrench")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "Fama_French_3_Factors.csv"

    # Download the data
    fama_french_df = download_fama_french_factors()

    # Save the data to a CSV file
    fama_french_df.to_csv(output_file)
    print(f"✅ Fama-French data successfully saved to: {output_file}")
