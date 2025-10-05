import pandas as pd

def generate_signals(data):
    """
    Generates a DataFrame of trading signals based on a moving average crossover.
    'data' is a pandas DataFrame with a 'close' price column.
    """
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Create short and long simple moving averages
    short_sma = data['close'].rolling(window=20, min_periods=1).mean()
    long_sma = data['close'].rolling(window=50, min_periods=1).mean()

    # 1. CORRECTED ASSIGNMENT USING .loc to fix the warning
    # Create signal when short SMA crosses above long SMA
    signals.loc[short_sma.index[20:], 'signal'] = (short_sma[20:] > long_sma[20:]).astype(float)

    # The 'positions' column will show 1.0 on a buy signal, -1.0 on a sell signal
    signals['positions'] = signals['signal'].diff()
    
    # 2. CORRECTED RETURN VALUE
    # Return the entire DataFrame of signals for backtesting
    return signals