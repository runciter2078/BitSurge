# BitSurge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicting Bitcoin's short-term movement using advanced technical indicators and ensemble machine learning models.

## Overview

BitSurge is a state-of-the-art Python project that predicts Bitcoin's short-term price movement using advanced technical indicators, ensemble machine learning models, and deep learning techniques. The script downloads hourly BTC-USDT data from Binance for the past 24 months, computes an extensive set of technical indicators and synthetic features, and trains multiple models (XGBoost, RandomForest, and LSTM) using rigorous time series validation to avoid data leakage. The final ensemble model combines the predictions of the individual models based on their performance.

## Features

- **Data Collection:**  
  - Robust downloading of historical BTC-USDT hourly data from Binance with error handling and retries.

- **Feature Engineering:**  
  - Calculation of a wide range of technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, CCI, etc.) and synthetic features.
  - Elimination of highly correlated features and dimensionality reduction using PCA (with logging of explained variance).

- **Target Variable:**  
  - Creation of a binary target indicating whether Bitcoin's price will increase by at least 0.25% in the next hour.
  - Management of class imbalance using computed class weights.

- **Modeling:**  
  - **XGBoost & RandomForest:**  
    - Hyperparameter optimization using Optuna with TimeSeriesSplit validation, optimizing based on F1-score.
  - **LSTM:**  
    - Unidirectional LSTM trained with TimeseriesGenerator to ensure temporal validation.
    - Example code is provided (commented) for further LSTM architecture optimization with Optuna.

- **Ensemble Learning:**  
  - Intelligent ensemble that combines model predictions by weighting them according to their F1-scores.
  - Threshold optimization using precision-recall curves.

- **Validation & Monitoring:**  
  - Walk-forward validation (backtesting) skeleton for realistic performance evaluation.
  - Data drift monitoring using Maximum Mean Discrepancy (MMD).

- **Interpretability:**  
  - Permutation importance analysis on the original feature set (without PCA) to identify key predictors.

- **Visualization & Logging:**  
  - Visualization of confusion matrices for individual models and the ensemble.
  - Detailed logging at every step to facilitate debugging and future maintenance.

## Requirements

- Python 3.7+
- Python Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `pandas_ta`
  - `requests`
  - `optuna`
  - `scikit-learn`
  - `xgboost`
  - `tensorflow` (with Keras)

You can install the required packages with:

```bash
pip install numpy pandas matplotlib seaborn pandas_ta requests optuna scikit-learn xgboost tensorflow
```

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/runciter2078/BitSurge.git
   cd BitSurge
   ```

2. **Run the Script:**

   Execute the script by running:

   ```bash
   python bit_surge.py
   ```

   The script will:
   - Download historical BTC-USDT data from Binance.
   - Compute an extensive set of technical indicators and synthetic features.
   - Create the binary target and handle class imbalance.
   - Preprocess data using a pipeline (StandardScaler + PCA) while logging the explained variance.
   - Train and optimize XGBoost and RandomForest models with temporal validation using TimeSeriesSplit.
   - Train an LSTM model using TimeseriesGenerator to ensure proper time-based validation.
   - Optimize the classification threshold using precision-recall curves.
   - Combine model predictions into an ensemble weighted by F1-score.
   - Provide examples for walk-forward validation and data drift (MMD) monitoring.
   - Evaluate each model and the ensemble with detailed metrics and confusion matrices.
   - Analyze feature importance using permutation importance on the raw feature set.

## Backtesting and Monitoring

- **Walk-Forward Validation:**  
  The script includes a skeleton function for walk-forward validation, enabling backtesting over moving windows.

- **Data Drift Monitoring:**  
  A function to compute Maximum Mean Discrepancy (MMD) is provided to monitor shifts in data distributions over time.

## Further Improvements

- **LSTM Architecture Optimization:**  
  Use Optuna to further optimize LSTM hyperparameters (e.g., number of units, dropout rates) for even better performance.

- **Real-Time Deployment:**  
  Integrate with an API framework (e.g., FastAPI) for real-time predictions.

- **Monitoring:**  
  Implement real-time performance monitoring and data drift tracking using tools like Prometheus and Grafana.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [Binance API](https://www.binance.com/) for providing access to historical market data.
- [pandas_ta](https://github.com/twopirllc/pandas-ta) for the technical indicator library.
- [Optuna](https://optuna.org/) for hyperparameter optimization.
- The open-source community for the many great tools and libraries that made this project possible.

## Disclaimer

This project is intended for educational and research purposes only and should not be considered financial advice. Cryptocurrency trading involves significant risk, and you should perform your own research before making any investment decisions.
```
