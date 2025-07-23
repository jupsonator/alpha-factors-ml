\# Quantitative ML Pipeline



This project explores whether simple technical indicators can help predict which stocks will outperform the short-term risk-free rate. It pulls historical data from Yahoo Finance, engineers signals, labels outcomes based on future returns, and trains a basic XGBoost model to see if there's any predictive edge.



I built this as a learning project to dive deeper into alpha signal generation and ML pipeline design. It's not meant to be production-grade, but it's a functional prototype that can serve as a base for more serious research or experimentation.



---



\## Overview



This project aims to:



\* Download historical stock and treasury data

\* Engineer alpha signals using technical indicators

\* Label data based on forward returns vs. risk-free rate

\* Train a classifier to predict outperformance

\* Analyse feature importance and model metrics



---



\## Tech Stack



\*\*Data \& Engineering\*\*



\* `yfinance` – for fetching stock and treasury data

\* `pandas`, `numpy` – for data manipulation

\* `ta` – for technical analysis



\*\*Modelling\*\*



\* `XGBoost` – for gradient boosted decision trees

\* `scikit-learn` – for metrics, model selection, cross-validation



\*\*Visualisation\*\*



\* `matplotlib`, `seaborn` – for feature and model diagnostics



---



\## Project Structure



alpha-factors-ml/

├── notebooks/

│ └── explore\_data.ipynb # EDA and visualisation notebook

├── reports/ # Folder for generated plots/reports

├── src/

│ ├── build\_features\_dataset.py # Builds feature datasets from raw prices

│ ├── compute\_labels.py # Labels data based on forward returns vs. RFR

│ ├── data\_loader.py # Downloads stock data from Yahoo Finance

│ ├── feature\_engineering.py # Computes technical indicators and signals

│ └── model\_train.py # Trains an XGBoost classifier and evaluates

├── data/

│ ├── raw/ # Raw Yahoo Finance CSVs

│ ├── features/ # Feature-rich datasets

│ └── labelled/ # Labelled datasets for ML

├── requirements.txt # Python dependencies

└── README.md # This file



---



\## Getting Started



\### Prerequisites



\* Python 3.8+

\* Virtual environment (recommended)



\### Installation



```bash

git clone https://github.com/jupsonator/alpha-factors-ml.git

cd alpha-factors-ml



python -m venv venv

source venv/bin/activate  # or venv\\Scripts\\activate on Windows



pip install -r requirements.txt



---



\## Workflow



\### Step 1: Download Raw Data

```bash

python src/data\_loader.py



\### Step 2: Compute Features

python src/build\_features\_dataset.py



\### Step 3: Compute Labels

python src/compute\_labels.py



\### Step 4: Train Model and Evaluate

python src/model\_train.py



---



\## Features Engineered



\*\*Returns\*\*



\* 1-day, 5-day, and 21-day percentage returns

\* 1-day, 5-day, and 21-day log returns



\*\*Volatility\*\*



\* Rolling standard deviation over 5-day and 21-day windows



\*\*Momentum \& Trend\*\*



\* Momentum ratios (price / price N days ago)

\* Price over moving averages (MA10, MA21)

\* MACD, MACD signal, MACD diff



\*\*Relative Strength\*\*



\* RSI (14-day)

\* Price rank within a rolling 5-day window

\* Price relative to 52-week high and 52-week low



\*\*Bollinger Bands\*\*



\* Upper, lower, and mid Bollinger Bands (20-day window)

\* Bollinger Band width (high - low)



---



\## Labelling Strategy



\*\*Forward Return\*\*



\* Calculated as the 21-day ahead return of the closing price



\*\*Binary Classification Target\*\*



\* `1` → forward return > 21-day compounded risk-free rate (`^IRX`)

\* `0` → otherwise



---



\## Model \& Evaluation



\*\*Model\*\*



\* XGBoost Classifier (`XGBClassifier`)



\*\*Techniques Used\*\*



\* Dropping low correlation features (`|corr| > 0.05`)

\* Removing high-correlation covariates (`ρ > 0.9`)

\* `TimeSeriesSplit` cross-validation (5-fold)

\* `RandomizedSearchCV` for hyperparameter tuning

\* Class imbalance handling using `scale\_pos\_weight`



\*\*Metrics Evaluated\*\*



\* Accuracy score

\* Confusion matrix

\* Precision, recall, F1-score 

\* Feature importance (Top 10 most important are visualised)



---



\## Example Outputs



\* Bar plot: Correlation of features with forward return

\* Feature importance plot (XGBoost)

\* Classification report and confusion matrix

\* Model accuracy score on test set



---



\## Potential Improvements



\* Rolling backtest engine with capital simulation

\* Model persistence (`joblib` save/load)

\* CLI support using `argparse`

\* SHAP-based model explainability

\* Docker support for full environment reproducibility

\* Unit testing with `pytest`



---



\## License



\[MIT License](LICENSE)



---





