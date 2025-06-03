# Configuration settings for the stock market prediction project

# Data paths
DATA_PATH = '../data/raw/microsoft_stock_data.csv'  # Path to the raw Microsoft stock data

# Model parameters
LINEAR_REGRESSION_PARAMS = {
    'fit_intercept': True,
    'normalize': False
}

XGBOOST_PARAMS = {
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror'
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

LSTM_PARAMS = {
    'units': 50,
    'dropout': 0.2,
    'batch_size': 32,
    'epochs': 100
}

TRANSFORMER_PARAMS = {
    'num_heads': 8,
    'num_layers': 4,
    'd_model': 128,
    'dropout_rate': 0.1,
    'batch_size': 32,
    'epochs': 100
}

# Time periods for training
TIME_PERIODS = {
    '3_months': '90D',
    '6_months': '180D',
    '12_months': '365D',
    '24_months': '730D',
    '36_months': '1095D'
}