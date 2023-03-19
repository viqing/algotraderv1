import pandas as pd
import numpy as np

def simple_moving_average_strategy(prices_df, ma_short = 50, ma_long = 200):

    ma_short_df = prices_df.rolling(window = ma_short).mean()
    ma_long_df = prices_df.rolling(window = ma_long).mean()

    signal_df = np.where(ma_short_df > ma_long_df, 1.0, 0.0)
    signal_df = np.where(ma_short_df < ma_long_df, -1.0, signal_df)

    signal_df = pd.DataFrame(signal_df, columns=prices_df.columns+'_sma_signal', index=prices_df.index)

    return signal_df