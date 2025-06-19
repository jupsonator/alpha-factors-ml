import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import yfinance as yf
from collections import Counter

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

def remove_highly_correlated_features(df, features, threshold=0.9):
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [f for f in features if f not in to_drop]

def get_risk_free_rate(start, end):
    """Fetches the 13-week T-Bill rate (^IRX) from Yahoo and calculates 21-day compounded RFR"""
    rfr_df = yf.download("^IRX", start=start, end=end, auto_adjust=False)
    if isinstance(rfr_df.index, pd.MultiIndex):
        rfr_df.index = rfr_df.index.get_level_values(0)
    rfr_df = rfr_df[['Close']].rename(columns={'Close': 'irx'})
    rfr_df['daily_rfr'] = rfr_df['irx'] / 100 / 252
    rfr_df['rfr_21d'] = (1 + rfr_df['daily_rfr']) ** 21 - 1
    rfr_df = rfr_df[['rfr_21d']]
    rfr_df.index = rfr_df.index.get_level_values(0)
    return rfr_df

def main():
    df = load_and_stack_labelled_data().dropna()
    df = df.sort_values("Date")

    # Get range for RFR download
    start_date, end_date = df["Date"].min(), df["Date"].max()
    rfr = get_risk_free_rate(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    # Merge risk-free rate into the main DataFrame
    rfr = rfr.reset_index()  
    rfr.columns = ['Date', 'rfr_21d']  
    df = df.merge(rfr, on="Date", how="left")
    df = df.dropna(subset=["forward_return", "rfr_21d"])

     # ----- Feature Signal Strength Diagnostic -----
    plt.figure(figsize=(10, 4))
    corrs = df.corr(numeric_only=True)['forward_return'].drop('forward_return')
    corrs.sort_values(inplace=True)
    sns.barplot(x=corrs.index, y=corrs.values)
    plt.xticks(rotation=45)
    plt.title("Correlation of Features with Forward Return")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Remove low-signal features based on correlation threshold
    signal_threshold = 0.05
    initial_features = corrs[abs(corrs) > signal_threshold].index.tolist()
    # Classification target: outperforming risk-free rate over 21 days
    df["target_class"] = (df["forward_return"] > df["rfr_21d"]).astype(int)

    # Correlation filtering
    corrs = df.corr(numeric_only=True)['forward_return'].drop('forward_return')
    signal_threshold = 0.05
    initial_features = corrs[abs(corrs) > signal_threshold].index.tolist()
    selected_features = remove_highly_correlated_features(df, initial_features)
    for col in ['target_class', 'forward_return']:
        if col in selected_features:
            selected_features.remove(col)

    # Filter out non-predictive raw price features
    non_predictive = ['Close', 'Volume']
    selected_features = [f for f in selected_features if f not in non_predictive]

    print("Final selected features after dropping correlated ones:", selected_features)

    # Train/test split
    split_date = "2022-01-01"
    train = df[df["Date"] < split_date]
    test = df[df["Date"] >= split_date]

    X_train = train[selected_features]
    y_train = train["target_class"]
    X_test = test[selected_features]
    y_test = test["target_class"]

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Compute scale_pos_weight to handle class imbalance
    class_counts = Counter(y_train)
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"Class distribution: {class_counts}, scale_pos_weight={scale_pos_weight:.2f}")
    
    model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)
    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=tscv,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print("Best hyperparameters:", search.best_params_)

    # Feature importance
    best_model = search.best_estimator_
    feature_importances = best_model.feature_importances_
    top_indices = np.argsort(feature_importances)[::-1][:10]
    top_features = [selected_features[i] for i in top_indices]
    top_importances = feature_importances[top_indices]
    print("Top features by importance:", top_features)

    # Final model training
    best_model.fit(X_train[top_features], y_train)
    y_pred = best_model.predict(X_test[top_features])

    # Evaluation
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Plot Top Feature Importances
    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_features, y=top_importances)
    plt.xticks(rotation=45)
    plt.title("Top Feature Importances (XGBoost Classifier)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()