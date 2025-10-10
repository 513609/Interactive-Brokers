import yaml
import pandas as pd
from engine.data_handler import DataHandler
from engine.execution_handler import ExecutionHandler
from engine.portfolio import Portfolio
from engine.performance import generate_report
from strategies.sma_crossover_strategy import SmaCrossoverStrategy
# Import other strategies here as you create them

def run_backtest(config: dict):
    """
    Orchestrates the entire backtesting process.
    """
    # --- Initialization ---
    print("Initializing backtesting engine...")
    symbols_to_trade = config['backtest']['symbols']
    benchmark_symbol = config['backtest']['benchmark_symbol']
    all_symbols = list(set(symbols_to_trade + [benchmark_symbol]))

    data_handler = DataHandler(symbols=all_symbols)
    execution_handler = ExecutionHandler(
        commission_per_trade=config['execution']['commission_per_trade'],
        spread_percent=config['execution']['bid_ask_spread_percent'],
        slippage_percent=config['execution']['slippage_percent']
    )
    portfolio = Portfolio(
        initial_capital=config['backtest']['initial_capital'],
        execution_handler=execution_handler
    )
    
    # --- Strategy Selection (Plug-and-Play) ---
    # This is the only section you change to test a different strategy
    if config['strategy']['name'] == 'SmaCrossoverStrategy':
        strategy = SmaCrossoverStrategy(
            symbols=symbols_to_trade,
            short_window=config['strategy']['parameters']['short_window'],
            long_window=config['strategy']['parameters']['long_window']
        )
    else:
        raise ValueError(f"Strategy '{config['strategy']['name']}' not recognized.")
    
    # --- Main Backtesting Loop ---
    print("Starting backtest simulation...")
    data_stream = data_handler.stream_bars(
        config['backtest']['start_date'],
        config['backtest']['end_date']
    )
    
    for date, current_market_data in data_stream:
        # 1. Generate signals from the strategy
        signals = strategy.generate_signals(date, current_market_data)
        
        # 2. Execute signals in the portfolio
        for symbol, signal in signals.items():
            portfolio.execute_signal(
                date, symbol, signal, current_market_data,
                config['strategy']['parameters']['capital_per_trade_percent']
            )
            
        # 3. Update total portfolio value for the day
        portfolio.update_value(date, current_market_data)
        
    print("Backtest simulation complete.")
    
    # --- Performance Analysis ---
    portfolio_history = portfolio.get_history_df()
    benchmark_data = data_handler.all_data[benchmark_symbol]
    benchmark_data = benchmark_data.loc[portfolio_history.index]
    
    generate_report(portfolio_history, benchmark_df=benchmark_data)

if __name__ == "__main__":
    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    run_backtest(config)
