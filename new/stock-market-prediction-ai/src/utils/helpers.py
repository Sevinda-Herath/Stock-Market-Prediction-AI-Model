# helpers.py

# This file includes helper functions that can be used throughout the project, such as data formatting and logging.

import pandas as pd
import numpy as np
import logging

# Set up logging configuration
def setup_logging(log_file='app.log'):
    """
    Set up logging configuration.
    
    Parameters:
    log_file (str): The name of the log file where logs will be saved.
    """
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

# Function to format date columns in a DataFrame
def format_date_column(df, date_column):
    """
    Convert a date column to datetime format.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the date column.
    date_column (str): The name of the date column to format.
    
    Returns:
    pd.DataFrame: DataFrame with the formatted date column.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    return df

# Function to handle missing values in a DataFrame
def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the DataFrame based on the specified strategy.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    strategy (str): The strategy to handle missing values ('drop' or 'fill').
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(method='ffill')  # Forward fill as an example
    else:
        raise ValueError("Invalid strategy. Use 'drop' or 'fill'.")

# Function to normalize a DataFrame
def normalize_dataframe(df):
    """
    Normalize the DataFrame to have values between 0 and 1.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to normalize.
    
    Returns:
    pd.DataFrame: Normalized DataFrame.
    """
    return (df - df.min()) / (df.max() - df.min())

# Function to log model performance metrics
def log_metrics(metrics):
    """
    Log the performance metrics of the model.
    
    Parameters:
    metrics (dict): A dictionary containing model performance metrics.
    """
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")