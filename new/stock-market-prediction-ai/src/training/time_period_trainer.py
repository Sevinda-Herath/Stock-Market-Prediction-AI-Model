# Contents of /stock-market-prediction-ai/stock-market-prediction-ai/src/training/time_period_trainer.py

import pandas as pd
from src.data.data_loader import load_stock_data
from src.models.linear_regression import LinearRegressionModel
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest import RandomForestModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel

def train_models_on_time_periods(stock_symbol, time_periods):
    """
    Trains various models on specified time periods using stock data.

    Parameters:
    stock_symbol (str): The stock symbol to load data for.
    time_periods (list): A list of time periods (in months) to train the models on.
    """
    # Load stock data for the specified stock symbol
    stock_data = load_stock_data(stock_symbol)

    # Iterate over each time period
    for months in time_periods:
        # Prepare the data for the specified time period
        train_data = prepare_data_for_time_period(stock_data, months)

        # Train and evaluate each model
        print(f"Training models for {months} months...")
        
        # Linear Regression
        lr_model = LinearRegressionModel()
        lr_model.train(train_data)
        lr_model.evaluate()

        # XGBoost
        xgb_model = XGBoostModel()
        xgb_model.train(train_data)
        xgb_model.evaluate()

        # Random Forest
        rf_model = RandomForestModel()
        rf_model.train(train_data)
        rf_model.evaluate()

        # LSTM
        lstm_model = LSTMModel()
        lstm_model.train(train_data)
        lstm_model.evaluate()

        # Transformer Model
        transformer_model = TransformerModel()
        transformer_model.train(train_data)
        transformer_model.evaluate()

def prepare_data_for_time_period(stock_data, months):
    """
    Prepares the stock data for training based on the specified time period.

    Parameters:
    stock_data (DataFrame): The stock data to prepare.
    months (int): The time period in months.

    Returns:
    DataFrame: The prepared training data.
    """
    # Calculate the number of days in the specified months
    days = months * 30  # Approximation
    # Select the last 'days' worth of data
    return stock_data.tail(days)

if __name__ == "__main__":
    # Define the stock symbol and time periods to evaluate
    stock_symbol = 'MSFT'  # Microsoft stock symbol
    time_periods = [3, 6, 12, 24, 36]  # Time periods in months

    # Train models on the specified time periods
    train_models_on_time_periods(stock_symbol, time_periods)