import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

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

def main():
    df = load_and_stack_labelled_data().dropna()
    df = df.sort_values("Date")

    # Binary classification target
    df["target_class"] = (df["forward_return"] > 0).astype(int)

    # Correlation filtering
    corrs = df.corr(numeric_only=True)['forward_return'].drop('forward_return')
    initial_features = corrs[abs(corrs) > 0.02].index.tolist()
    selected_features = remove_highly_correlated_features(df, initial_features)
    for col in ['target_class', 'forward_return']:
        if col in selected_features:
            selected_features.remove(col)
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

    model = XGBClassifier(random_state=42, eval_metric='logloss')
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