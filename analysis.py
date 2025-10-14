import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import re

def analyze_dataset(df, title_prefix, output_dir):
    """
    Analyzes a dataframe by calculating statistics and creating plots
    for the 'target' and 'EPSBeat' columns.
    """
    print(f"\n--- Analysis for: {title_prefix} ---")

    if 'target' not in df.columns or 'EPSBeat' not in df.columns:
        print("  -> Skipping: 'target' or 'EPSBeat' column not found.")
        return

    # --- 1. Data-wise Analysis: Contingency Table (Crosstab) ---
    print("\n  1. Relationship between EPSBeat and subsequent Target performance:")
    
    # Create a crosstab to see the counts
    crosstab = pd.crosstab(df['EPSBeat'], df['target'])
    print("\n  Counts (Rows: EPSBeat, Columns: Target):")
    print(crosstab.to_string())

    # Create a normalized crosstab to see percentages
    crosstab_norm = pd.crosstab(df['EPSBeat'], df['target'], normalize='index')
    print("\n  Percentages (what % of beats/misses fall into each target category):")
    print((crosstab_norm * 100).round(2).to_string())

    # --- 2. Visual Analysis ---
    print("\n  2. Generating and saving plot...")

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Count Plot (Bar Chart)
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='EPSBeat', hue='target', data=df, palette='viridis')
    plt.title(f'Target Performance after Earnings Beat vs. Miss\n({title_prefix})', fontsize=16)
    plt.xlabel('Earnings Result (0 = Miss, 1 = Beat)', fontsize=12)
    plt.ylabel('Number of Occurrences', fontsize=12)
    ax.set_xticklabels(['Missed Estimate', 'Beat Estimate'])
    
    plot_filename = f"{title_prefix}_epsbeat_vs_target.png"
    plt.savefig(output_dir / plot_filename)
    plt.close()
    print(f"    -> ✅ Saved plot to {plot_filename}")


def main():
    # Define paths
    base_path = Path(__file__).parent
    sector_datasets_path = base_path / 'data' / 'sector_datasets'
    analysis_output_path = base_path / 'data' / 'analysis_results'

    # Create the output directory if it doesn't exist
    analysis_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Analysis results will be saved to: {analysis_output_path}\n")

    # Find all sector dataset CSVs
    dataset_files = list(sector_datasets_path.glob('*_ml_dataset.csv'))
    if not dataset_files:
        print("No sector datasets found. Please run 'Dataset creation.py' first.")
        return

    all_sector_data = []
    for f in tqdm(dataset_files, desc="Analyzing individual sectors"):
        sector_name_match = re.match(r'(.+)_ml_dataset\.csv', f.name)
        if sector_name_match:
            sector_name = sector_name_match.group(1).replace('_', ' ').title()
            df = pd.read_csv(f)
            df['sector'] = sector_name # Add sector column for combined analysis
            all_sector_data.append(df)
            analyze_dataset(df, sector_name, analysis_output_path)

    # Perform a combined analysis on all data
    if all_sector_data:
        combined_df = pd.concat(all_sector_data, ignore_index=True)
        analyze_dataset(combined_df, "All Sectors Combined", analysis_output_path)

if __name__ == '__main__':
    main()