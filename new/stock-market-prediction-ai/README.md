# Stock Market Prediction AI Model

This project aims to evaluate various AI/ML models for stock market predictions using Microsoft stock data sourced from Yahoo Finance. The models implemented include:

- **Linear Regression**: A simple regression model to predict stock prices based on historical data.
- **XGBoost**: An optimized gradient boosting framework that is efficient and flexible.
- **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees.
- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network suitable for sequence prediction problems.
- **Transformer-based Time Series Model**: A model leveraging transformer architecture for time series forecasting.

## Project Structure

The project is organized into several directories and files:

- **src/**: Contains the source code for data handling, model training, evaluation, and utilities.
  - **data/**: Functions for loading, preprocessing, and feature engineering of stock data.
  - **models/**: Implementations of various machine learning models.
  - **evaluation/**: Functions for evaluating model performance and visualizing results.
  - **training/**: Scripts for training models on different time periods.
  - **utils/**: Utility functions and configuration settings.

- **notebooks/**: Jupyter notebooks for data exploration, model comparison, and results analysis.

- **data/**: Directory for storing raw data files.

- **models/**: Directory for saving trained models.

- **results/**: Directory for storing evaluation metrics and visualizations.

- **requirements.txt**: Lists the dependencies required for the project.

- **config.yaml**: Configuration settings in YAML format.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stock-market-prediction-ai
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the project settings in `config.yaml` as needed.

4. Run the Jupyter notebooks in the `notebooks/` directory for data exploration and model evaluation.

## Usage

- Load the stock data using the functions in `src/data/data_loader.py`.
- Preprocess the data with `src/data/data_preprocessor.py`.
- Perform feature engineering using `src/data/feature_engineering.py`.
- Train models using the scripts in `src/training/train_models.py` and `src/training/time_period_trainer.py`.
- Evaluate model performance with functions in `src/evaluation/metrics.py` and visualize results using `src/evaluation/visualizations.py`.

## Conclusion

This project provides a comprehensive framework for predicting stock prices using various machine learning models. Each model's performance can be compared across different time periods, allowing for a thorough analysis of their effectiveness in stock market prediction.