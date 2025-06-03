# visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_true, y_pred, model_name):
    """
    Plots the true values against the predicted values for a given model.

    Parameters:
    y_true (array-like): The actual stock prices.
    y_pred (array-like): The predicted stock prices from the model.
    model_name (str): The name of the model used for predictions.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='orange')
    plt.title(f'{model_name} Predictions vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()
    plt.show()

def plot_model_performance(metrics_dict):
    """
    Plots the performance metrics of different models for comparison.

    Parameters:
    metrics_dict (dict): A dictionary containing model names as keys and their corresponding metrics as values.
    """
    models = list(metrics_dict.keys())
    scores = list(metrics_dict.values())

    plt.figure(figsize=(10, 5))
    sns.barplot(x=models, y=scores, palette='viridis')
    plt.title('Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Score (e.g., RMSE)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()