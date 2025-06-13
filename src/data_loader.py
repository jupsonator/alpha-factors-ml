import yfinance as yf
import pandas as pd
import os

print("Current working directory:", os.getcwd())
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'JNJ', 'PG']

def fetch_and_save_data(tickers, start='2015-01-01', end='2024-06-01', path='data/raw'):
    os.makedirs(path, exist_ok=True)
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end)
        df.to_csv(f"{path}/{ticker}.csv")

if __name__ == "__main__":
    fetch_and_save_data(TICKERS)