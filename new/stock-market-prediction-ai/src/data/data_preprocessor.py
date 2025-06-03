# Contents of /stock-market-prediction-ai/stock-market-prediction-ai/src/data/data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    
    Parameters:
    data (DataFrame): The input DataFrame containing stock data.
    
    Returns:
    DataFrame: DataFrame with missing values handled.
    """
    # Fill missing values with the previous value (forward fill)
    data.fillna(method='ffill', inplace=True)
    # Fill any remaining missing values with the mean of the column
    data.fillna(data.mean(), inplace=True)
    return data

def normalize_data(data):
    """
    Normalize the dataset using Min-Max scaling.
    
    Parameters:
    data (DataFrame): The input DataFrame containing stock data.
    
    Returns:
    DataFrame: Normalized DataFrame.
    """
    scaler = MinMaxScaler()
    # Normalize the data, excluding the date column if present
    if 'Date' in data.columns:
        date_column = data['Date']
        normalized_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=['Date'])), columns=data.columns[1:])
        normalized_data['Date'] = date_column
    else:
        normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    return normalized_data

def preprocess_data(file_path):
    """
    Load and preprocess the stock data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file containing stock data.
    
    Returns:
    DataFrame: Preprocessed DataFrame ready for modeling.
    """
    # Load the data
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Normalize the data
    data = normalize_data(data)
    
    return data