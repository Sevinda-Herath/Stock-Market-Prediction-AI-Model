# Contents of /stock-market-prediction-ai/stock-market-prediction-ai/src/models/linear_regression.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionModel:
    def __init__(self):
        # Initialize the Linear Regression model
        self.model = LinearRegression()

    def train(self, X, y):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the model on the training data
        self.model.fit(X_train, y_train)
        
        # Make predictions on the test data
        predictions = self.model.predict(X_test)
        
        # Calculate and return evaluation metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

    def predict(self, X):
        # Make predictions on new data
        return self.model.predict(X)

    def save_model(self, filename):
        # Save the trained model to a file
        import joblib
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        # Load a trained model from a file
        import joblib
        self.model = joblib.load(filename)