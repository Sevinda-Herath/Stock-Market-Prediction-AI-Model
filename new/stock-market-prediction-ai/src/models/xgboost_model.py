# xgboost_model.py

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class XGBoostModel:
    def __init__(self, params=None):
        # Initialize the model with default parameters if none are provided
        self.params = params if params is not None else {
            'objective': 'reg:squarederror',
            'colsample_bytree': 0.3,
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10,
            'n_estimators': 100
        }
        self.model = None

    def load_data(self, data, target_column):
        # Split the data into features and target variable
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    def train(self, data, target_column):
        # Load the data
        X, y = self.load_data(data, target_column)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create the DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train the model
        self.model = xgb.train(self.params, dtrain)

        # Make predictions
        predictions = self.model.predict(dtest)

        # Calculate and return the mean squared error
        mse = mean_squared_error(y_test, predictions)
        return mse

    def predict(self, new_data):
        # Ensure the model is trained before making predictions
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        
        # Create DMatrix for new data
        dnew = xgb.DMatrix(new_data)
        
        # Make predictions
        return self.model.predict(dnew)