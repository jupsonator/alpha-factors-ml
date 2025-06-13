import pandas as pd
import os
from feature_engineering import compute_features
from data_loader import TICKERS


def build_dataset(tickers, raw_path='data/raw', out_path='data/features'):
    os.makedirs(out_path, exist_ok=True)
    for ticker in tickers:
        print(f"Processing {ticker}...")
        df = pd.read_csv(f"{raw_path}/{ticker}.csv", skiprows=2)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        print(df.columns)
        features = compute_features(df)
        features.to_csv(f"{out_path}/{ticker}_features.csv")

if __name__ == "__main__":
    build_dataset(TICKERS)