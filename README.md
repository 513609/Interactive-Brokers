Robust Backtesting Engine for Quantitative Trading
This project provides a professional-grade, event-driven backtesting engine written in Python. It is designed to be modular, scalable, and realistic, allowing for the rigorous testing of quantitative trading strategies.

Features
Event-Driven Architecture: Simulates trading day-by-day for a realistic time-series analysis.

Realistic Cost Simulation: Models commissions, bid-ask spreads, and slippage to provide a more accurate picture of performance.

Scalable Strategy Framework: An abstract base class (BaseStrategy) allows for easy "plug-and-play" of new trading strategies.

Object-Oriented Design: Separates concerns into distinct components (Data, Portfolio, Execution, Performance) for maintainability.

Detailed Performance Analytics: Automatically calculates key metrics like Sharpe Ratio and Maximum Drawdown, and generates publication-quality plots.

YAML Configuration: Centralized and easy-to-understand configuration for all backtest parameters.

Project Structure
/
├── data/                 # Stores historical market data (CSV files)
├── engine/               # Core backtesting components
├── strategies/           # Trading strategy implementations
├── utils/                # Utility modules (e.g., logging)
├── .gitignore            # Files to be ignored by Git
├── config.yaml           # Main configuration file for backtests
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── run_backtest.py       # Main script to execute a backtest

Setup Instructions
Clone the Repository:

git clone <your-repository-url>
cd <repository-name>

Create a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

Add Data:
Place your historical price data in the data/ folder. Each file should be a CSV named {SYMBOL}_data.csv (e.g., AAPL_data.csv). The CSV must contain at least date and close columns.

How to Run a Backtest
Configure the Backtest:
Open config.yaml and adjust the parameters:

initial_capital, start_date, end_date

symbols to trade and the benchmark_symbol

execution costs (commission, spread, slippage)

strategy name and its specific parameters

Execute the Backtest:
Run the main script from your terminal:

python run_backtest.py

The script will print a detailed performance report to the console and display a plot of the equity curve and drawdown.

How to Create a New Strategy
Create a new Python file in the strategies/ folder (e.g., my_new_strategy.py).

In the new file, create a class that inherits from BaseStrategy.

Implement the required generate_signals(self, date, data) method. This method must return a dictionary of signals (e.g., {'AAPL': 'BUY'}).

In run_backtest.py, import your new strategy and update the "Strategy Selection" section to use it.