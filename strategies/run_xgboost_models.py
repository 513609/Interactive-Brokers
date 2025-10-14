import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import quantstats as qs
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress future warnings from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_sector_model(dataset_path: Path, results_dir: Path):
    """
    Trains an XGBoost model for a given sector, evaluates it, and saves metrics and plots.

    Args:
        dataset_path (Path): Path to the sector's ML dataset CSV file.
        results_dir (Path): Directory to save the output plots and metrics.
    """
    try:
        # --- 1. Load and Prepare Data ---
        sector_name = dataset_path.stem.replace('_ml_dataset', '')
        print(f"\n--- Processing Sector: {sector_name.replace('_', ' ').title()} ---")

        df = pd.read_csv(dataset_path)
        df['earnings_date'] = pd.to_datetime(df['earnings_date'])
        df.set_index('earnings_date', inplace=True)

        if df.empty:
            print(f"Skipping {sector_name} as the dataset is empty.")
            return None

        # Define features (X) and target (y)
        features = ['ReportedEPS', 'EPSBeat', 'volatility_bin', 'momentum_bin']
        target_col = 'target'

        X = df[features]
        y = df[target_col]

        # XGBoost requires target labels to be 0, 1, 2, etc.
        # We map {-1, 0, 1} to {0, 1, 2} and keep the mapping for later.
        label_mapping = {-1: 0, 0: 1, 1: 2}
        inverse_label_mapping = {v: k for k, v in label_mapping.items()}
        y_mapped = y.map(label_mapping)

        # --- 2. Time-Series Split (80% train, 20% test) ---
        split_index = int(len(df) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y_mapped[:split_index], y_mapped[split_index:]
        test_dates = df.index[split_index:]
        test_pct_change = df['pct_change'][split_index:]

        if X_test.empty:
            print(f"Skipping {sector_name} due to insufficient data for a test set.")
            return None

        # --- 3. Train XGBoost Model ---
        print("Training XGBoost model...")
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)

        # --- 4. Make Predictions and Evaluate ---
        print("Evaluating model and calculating returns...")
        predictions_mapped = model.predict(X_test)
        # Convert predictions back to original labels (-1, 0, 1)
        predictions = pd.Series(predictions_mapped).map(inverse_label_mapping).values
        y_test_original = y_test.map(inverse_label_mapping)

        # --- 5. Calculate ML Metrics ---
        print("\n--- ML Performance Metrics ---")
        accuracy = accuracy_score(y_test_original, predictions)
        report = classification_report(y_test_original, predictions, zero_division=0)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        # --- 6. Calculate Trading Strategy Returns ---
        strategy_returns = pd.Series(np.zeros(len(predictions)), index=test_dates)
        # Long signal
        strategy_returns[predictions == 1] = test_pct_change[predictions == 1]
        # Short signal
        strategy_returns[predictions == -1] = -test_pct_change[predictions == -1]

        # --- Aggregate returns for the same day ---
        # If multiple companies report on the same day, average their returns.
        # This ensures the index is unique (one entry per day).
        daily_strategy_returns = strategy_returns.groupby(strategy_returns.index).mean()

        cumulative_returns = (1 + daily_strategy_returns).cumprod().rename(sector_name)

        # --- 7. Calculate Trading and Risk Metrics using quantstats ---
        print("\n--- Trading & Risk-Adjusted Metrics ---")
        qs.extend_pandas()
        
        # Temporarily suppress warnings from quantstats for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use .squeeze() to convert DataFrame to Series for quantstats
            metrics_df = qs.reports.metrics(daily_strategy_returns.squeeze(), display=False)

        # Print key metrics
        print(f"Cumulative Return: {metrics_df.loc['Cumulative Return'][0]:.2%}")
        print(f"Sharpe Ratio:      {metrics_df.loc['Sharpe'][0]:.2f}")
        print(f"Sortino Ratio:     {metrics_df.loc['Sortino'][0]:.2f}")
        print(f"Max Drawdown:      {metrics_df.loc['Max Drawdown'][0]:.2%}")
        print("-" * 40)

        # --- 8. Save Plots ---
        # Plot 1: Cumulative Returns
        plt.figure(figsize=(12, 6))
        cumulative_returns.plot(title=f'{sector_name.replace("_", " ")} Strategy Cumulative Returns')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f'{sector_name}_returns_plot.png')
        plt.close()

        # Plot 2: Confusion Matrix
        cm = confusion_matrix(y_test_original, predictions, labels=[-1, 0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 0, 1])
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'{sector_name.replace("_", " ")} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(results_dir / f'{sector_name}_confusion_matrix.png')
        plt.close()

        return cumulative_returns

    except Exception as e:
        print(f"An error occurred while processing {dataset_path.name}: {e}")
        return None

def main():
    """
    Main function to find all sector datasets, run the model for each,
    and generate a final comparison plot.
    """
    # Define paths
    base_dir = Path('.')
    sector_datasets_dir = base_dir / 'data' / 'sector_datasets'
    results_dir = base_dir / 'model_results'

    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # Find all sector ML dataset files
    sector_files = list(sector_datasets_dir.glob('*_ml_dataset.csv'))

    if not sector_files:
        print("No sector ML dataset files found. Please run 'Dataset creation.py' first.")
        return

    all_sector_returns = []

    # Loop through each file and run the model
    for file_path in tqdm(sector_files, desc="Processing all sectors"):
        sector_returns = run_sector_model(file_path, results_dir)
        if sector_returns is not None:
            all_sector_returns.append(sector_returns)

    # --- Final Comparison Plot ---
    if not all_sector_returns:
        print("\nNo models were successfully trained. Cannot generate comparison plot.")
        return

    print("\nGenerating final comparison plot for all sectors...")
    
    # Combine all series into a single DataFrame, forward-filling missing values
    comparison_df = pd.concat(all_sector_returns, axis=1).ffill()
    
    # Re-base all returns to start at 1
    comparison_df = comparison_df / comparison_df.iloc[0]

    # Create the plot using an explicit Axes object for clarity and correctness
    fig, ax = plt.subplots(figsize=(15, 8))
    comparison_df.plot(ax=ax)
    ax.set_title('All Sector Strategies - Comparative Performance')
    ax.set_ylabel('Cumulative Returns (Re-based to 1)')
    ax.set_xlabel('Date')
    ax.grid(True)
    ax.legend(title='Sectors', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    
    final_plot_path = results_dir / 'ALL_SECTORS_comparison_plot.png'
    plt.savefig(final_plot_path)
    plt.close()

    print(f"\n✅ All sectors processed. Final comparison plot saved to {final_plot_path}")

if __name__ == '__main__':
    # Install quantstats if you haven't already: pip install quantstats
    qs.extend_pandas()
    main()