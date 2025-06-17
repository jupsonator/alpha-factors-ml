import pandas as pd
import os
import numpy as np
import ta

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Daily returns
    df['return_1d'] = df['Close'].pct_change()
    df['log_return_1d'] = np.log(df['Close'] / df['Close'].shift(1))

    # 5-day and 21-day returns
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_21d'] = df['Close'].pct_change(21)
    df['log_return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
    df['log_return_21d'] = np.log(df['Close'] / df['Close'].shift(21))

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

    # Momentum over 5 and 21 days
    df['momentum_5d'] = df['Close'] / df['Close'].shift(5)
    df['momentum_21d'] = df['Close'] / df['Close'].shift(21)

    # Rolling 21-day volatility
    df['vol_21d'] = df['return_1d'].rolling(window=21).std()

    # Price relative to 52-week high and low
    df['price_52w_high'] = df['Close'] / df['Close'].rolling(window=252).max()
    df['price_52w_low'] = df['Close'] / df['Close'].rolling(window=252).min()

    #Moving Average Convergence Divergence
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    #Bollinger Bands
    boll = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_bbm'] = boll.bollinger_mavg()
    df['bb_bbh'] = boll.bollinger_hband()
    df['bb_bbl'] = boll.bollinger_lband()
    df['bb_width'] = df['bb_bbh'] - df['bb_bbl']




    # Drop rows with NaNs (early periods)
    df = df.dropna()

    return df