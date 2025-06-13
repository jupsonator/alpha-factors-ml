import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

LABELLED_DIR = "data/labelled"

def load_and_stack_labelled_data(path=LABELLED_DIR):
    dfs = []
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            ticker = filename.split("_")[0]
            df = pd.read_csv(os.path.join(path, filename), parse_dates=["Date"])
            df["ticker"] = ticker
            dfs.append(df)
    return pd.concat(dfs)

def main():
    df = load_and_stack_labelled_data()
    df = df.dropna()

    # Define features
    feature_cols = [col for col in df.columns if col not in ['Date', 'ticker', 'Close', 'forward_return']]

    # Time-based split
    df = df.sort_values("Date")
    split_date = "2022-01-01"
    train = df[df["Date"] < split_date]
    test = df[df["Date"] >= split_date]

    X_train = train[feature_cols]
    y_train = train["forward_return"]
    X_test = test[feature_cols]
    y_test = test["forward_return"]

    # Train model
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)

    print("R^2:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

    # Optional: plot predictions
    plt.scatter(y_test, y_pred, alpha=0.2)
    plt.xlabel("Actual Return (5D)")
    plt.ylabel("Predicted Return (5D)")
    plt.title("Predicted vs Actual 5-Day Return")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
