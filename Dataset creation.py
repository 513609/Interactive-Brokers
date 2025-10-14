import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Define the main data directory
data_path = Path('data')
output_dir = data_path / 'sector_datasets'

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory created at: {output_dir}")

# --- Step 1: Load Static Fundamental Data ---
print("\n1. Loading fundamental data...")
fundamentals_list = []
# MODIFIED: Point directly to the correct folder and use the yfinance JSON structure.
fundamentals_path = data_path / 'fundamental_data'

print(f"Loading fundamental data from: {fundamentals_path}")
for f in tqdm(list(fundamentals_path.glob('*_fundamentals.json'))):
    with open(f, 'r', encoding='utf-8') as file:
        try:
            # The structure is confirmed to be {'info': {...}}
            data = json.load(file).get('info', {})
            
            # Use the 'symbol' from the file content if available, otherwise parse filename
            ticker = data.get('symbol', f.stem.split('_')[0])
            sector = data.get('sector', 'N/A')
            industry = data.get('industry', 'N/A')
            
            fundamentals_list.append({'symbol': ticker, 'sector': sector, 'industry': industry})
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON for file {f.name}")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while processing {f.name}: {e}")

fundamentals_df = pd.DataFrame(fundamentals_list)
print(f"Loaded fundamental data for {len(fundamentals_df)} symbols.")


# --- Step 2: Process All Stocks to Get Raw Features ---
print("\n2. Processing earnings and historical price data...")
all_periods_data = []
eps_files = list((data_path / 'eps_data').glob('*_EPS.csv'))

for eps_file in tqdm(eps_files):
    ticker = eps_file.stem.split('_')[0]
    historical_file = data_path / 'historical_data' / f'{ticker}_data.csv'

    if not historical_file.exists():
        continue

    eps_df = pd.read_csv(eps_file)
    prices_df = pd.read_csv(historical_file)

    eps_df['EarningsDate'] = pd.to_datetime(eps_df['EarningsDate'])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df.set_index('date', inplace=True)
    eps_df.sort_values('EarningsDate', inplace=True, ignore_index=True)

    for i in range(len(eps_df) - 1):
        current_earnings_date = eps_df.loc[i, 'EarningsDate']
        next_earnings_date = eps_df.loc[i+1, 'EarningsDate']

        try:
            start_price_date = prices_df[prices_df.index > current_earnings_date].index[0]
            end_price_date = prices_df[prices_df.index > next_earnings_date].index[0]
        except IndexError:
            continue

        start_price = prices_df.loc[start_price_date, 'close']
        end_price = prices_df.loc[end_price_date, 'close']

        pct_change = (end_price - start_price) / start_price
        
        # MODIFIED: Change target to be numerical (1, -1, 0)
        threshold = 0.025
        if pct_change > threshold:
            target = 1
        elif pct_change < -threshold:
            target = -1
        else:
            target = 0

        historical_subset = prices_df[prices_df.index < current_earnings_date]
        if len(historical_subset) < 90:
            continue

        returns_90d = historical_subset['close'].tail(90).pct_change().dropna()
        volatility_90d = returns_90d.std()
        momentum_30d = (historical_subset['close'][-1] / historical_subset['close'][-30]) - 1

        period_features = {
            'symbol': ticker,
            'earnings_date': current_earnings_date,
            'ReportedEPS': eps_df.loc[i, 'ReportedEPS'],
            'EPSBeat': eps_df.loc[i, 'EPSBeat'],
            'volatility_90d': volatility_90d,
            'momentum_30d': momentum_30d,
            'target': target,
            'pct_change': pct_change  # ADDED: Include the raw percentage change
        }
        all_periods_data.append(period_features)

# --- Step 3: Create Full DataFrame Before Splitting ---
print("\n3. Assembling the complete dataset before splitting...")
if not all_periods_data:
    print("No data was processed. Check file paths. Exiting.")
else:
    # Create the main dataframe
    full_dataset = pd.DataFrame(all_periods_data)
    # Merge with fundamentals to get sector and industry info
    full_dataset = pd.merge(full_dataset, fundamentals_df, on='symbol', how='left')

    # --- Step 4: Split, Process, and Save Dataset for Each Sector ---
    print("\n4. Splitting dataset by sector and saving files...")
    
    # Get a list of unique sectors with valid data
    sectors = full_dataset['sector'].dropna().unique()

    for sector in sectors:
        print(f"  -> Processing sector: {sector}")
        
        # Filter the dataset for the current sector
        sector_df = full_dataset[full_dataset['sector'] == sector].copy()
        
        # Apply quantile binning to the sector-specific data
        sector_df['volatility_bin'] = pd.qcut(sector_df['volatility_90d'], 10, labels=False, duplicates='drop')
        sector_df['momentum_bin'] = pd.qcut(sector_df['momentum_30d'], 10, labels=False, duplicates='drop')

        # Drop the original raw columns
        sector_df.drop(['volatility_90d', 'momentum_30d'], axis=1, inplace=True)

        # Final cleaning
        sector_df.ffill(inplace=True)
        sector_df.dropna(inplace=True)
        
        if sector_df.empty:
            print(f"    --> Skipping {sector} as no valid data remained after cleaning.")
            continue
        
        # Convert bin columns to integer type
        sector_df['volatility_bin'] = sector_df['volatility_bin'].astype(int)
        sector_df['momentum_bin'] = sector_df['momentum_bin'].astype(int)

        # Drop the industry and sector columns.
        sector_df.drop(columns=['industry', 'sector'], inplace=True, errors='ignore')

        # MODIFIED: Sort by date and then drop the symbol column
        sector_df.sort_values(by='earnings_date', inplace=True)
        sector_df.drop(columns=['symbol'], inplace=True, errors='ignore')

        # Define a clean filename and save the CSV
        clean_sector_name = sector.replace(' ', '_').replace('&', 'and')
        output_filename = f"{clean_sector_name}_ml_dataset.csv"
        output_path = output_dir / output_filename
        
        sector_df.to_csv(output_path, index=False)
        print(f"    --> ✅ Saved {sector} dataset with shape {sector_df.shape} to {output_filename}")

