# Contents of /stock-market-prediction-ai/stock-market-prediction-ai/src/training/train_models.py

import pandas as pd
from src.data.data_loader import load_stock_data
from src.data.data_preprocessor import preprocess_data
from src.data.feature_engineering import create_features
from src.models.linear_regression import LinearRegressionModel
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest import RandomForestModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.evaluation.metrics import calculate_metrics
from src.evaluation.visualizations import plot_results

def train_models(time_periods):
    # Load Microsoft stock data from Yahoo Finance
    stock_data = load_stock_data('MSFT')
    
    # Iterate over each time period for training
    for period in time_periods:
        # Preprocess the data for the specified time period
        processed_data = preprocess_data(stock_data, period)
        
        # Create features for the model
        features, target = create_features(processed_data)
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegressionModel(),
            'XGBoost': XGBoostModel(),
            'Random Forest': RandomForestModel(),
            'LSTM': LSTMModel(),
            'Transformer': TransformerModel()
        }
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f'Training {model_name} on {period} months of data...')
            model.fit(features, target)  # Train the model
            
            # Make predictions
            predictions = model.predict(features)
            
            # Calculate evaluation metrics
            metrics = calculate_metrics(target, predictions)
            print(f'{model_name} Metrics: {metrics}')
            
            # Plot results
            plot_results(target, predictions, model_name, period)

if __name__ == "__main__":
    # Define the time periods for training
    time_periods = [3, 6, 12, 24, 36]  # in months
    train_models(time_periods)  # Start the training process