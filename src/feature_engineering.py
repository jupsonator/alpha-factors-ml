import pandas as pd
import os
import numpy as np
import ta

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Daily returns
    df['return_1d'] = df['Close'].pct_change()

    # 5-day and 21-day returns
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_21d'] = df['Close'].pct_change(21)

    # Rolling volatility
    df['vol_5d'] = df['return_1d'].rolling(window=5).std()

    # Price rank over 5 days
    df['rank_5d'] = df['Close'].rolling(window=5).apply(lambda x: x.argsort().argsort()[-1] / 4 if len(x) == 5 else np.nan)


    # RSI
    df['rsi_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # Moving averages
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_21'] = df['Close'].rolling(window=21).mean()

    # Price/MA
    df['price_over_ma10'] = df['Close'] / df['ma_10']

    # Drop rows with NaNs (early periods)
    df = df.dropna()

    return df