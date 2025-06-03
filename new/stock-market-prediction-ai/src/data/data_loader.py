# Contents of /stock-market-prediction-ai/stock-market-prediction-ai/src/data/data_loader.py

import pandas as pd
import yfinance as yf

def load_stock_data(ticker, start_date, end_date):
    """
    Load stock data from Yahoo Finance for a given ticker symbol and date range.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'MSFT' for Microsoft).
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing the stock data with date as the index.
    """
    # Fetch the stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Check if data is returned
    if stock_data.empty:
        raise ValueError("No data found for the given ticker and date range.")
    
    # Return the stock data
    return stock_data

def save_stock_data(data, filename):
    """
    Save the stock data to a CSV file.

    Parameters:
    data (pd.DataFrame): The stock data to save.
    filename (str): The filename for the saved CSV file.
    """
    # Save the DataFrame to a CSV file
    data.to_csv(filename)
    print(f"Data saved to {filename}")