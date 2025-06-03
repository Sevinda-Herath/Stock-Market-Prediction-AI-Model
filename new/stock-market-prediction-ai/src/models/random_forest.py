# Contents of /stock-market-prediction-ai/stock-market-prediction-ai/src/models/random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    """
    Load the stock data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file containing stock data.
    
    Returns:
    pd.DataFrame: A DataFrame containing the stock data.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the stock data by handling missing values and normalizing features.
    
    Parameters:
    data (pd.DataFrame): The raw stock data.
    
    Returns:
    pd.DataFrame: The preprocessed stock data.
    """
    # Handle missing values by filling them with the mean of the column
    data.fillna(data.mean(), inplace=True)
    
    # Normalize the features (example: using Min-Max scaling)
    data = (data - data.min()) / (data.max() - data.min())
    
    return data

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model on the training data.
    
    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target variable.
    
    Returns:
    RandomForestRegressor: The trained Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using the test data.
    
    Parameters:
    model (RandomForestRegressor): The trained Random Forest model.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The test target variable.
    
    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        'Mean Squared Error': mse,
        'R-squared': r2
    }

def main(file_path):
    """
    Main function to load data, preprocess it, train the Random Forest model, and evaluate it.
    
    Parameters:
    file_path (str): The path to the CSV file containing stock data.
    """
    # Load the data
    data = load_data(file_path)
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Split the data into features and target variable
    X = processed_data.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
    y = processed_data['target_column']  # Replace 'target_column' with the actual target column name
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

# Example usage
# main('path_to_your_data.csv')  # Uncomment and replace with the actual path to your data file