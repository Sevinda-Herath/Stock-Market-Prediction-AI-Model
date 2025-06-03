# feature_engineering.py

import pandas as pd

def create_features(data):
    """
    Create new features from the existing stock data to improve model performance.
    
    Parameters:
    data (DataFrame): The input stock data containing 'Close', 'Volume', etc.
    
    Returns:
    DataFrame: The input data with new features added.
    """
    
    # Ensure the data is sorted by date
    data = data.sort_index()
    
    # Create a new feature for daily returns
    data['Daily Return'] = data['Close'].pct_change()
    
    # Create a new feature for moving averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()  # 10-day simple moving average
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day simple moving average
    
    # Create a new feature for volatility (standard deviation of returns)
    data['Volatility'] = data['Daily Return'].rolling(window=10).std()
    
    # Create lag features
    for lag in range(1, 6):  # Create lag features for the last 5 days
        data[f'Lag_{lag}'] = data['Close'].shift(lag)
    
    # Drop rows with NaN values created by rolling functions
    data = data.dropna()
    
    return data

def feature_selection(data):
    """
    Select relevant features for the model training.
    
    Parameters:
    data (DataFrame): The input data with engineered features.
    
    Returns:
    DataFrame: The data with selected features.
    """
    
    # Select features for the model
    features = ['Close', 'SMA_10', 'SMA_50', 'Volatility'] + [f'Lag_{lag}' for lag in range(1, 6)]
    
    return data[features]