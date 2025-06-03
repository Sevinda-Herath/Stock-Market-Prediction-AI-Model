def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values from the model.
    
    Returns:
    float: The RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values from the model.
    
    Returns:
    float: The MAE value.
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_r2(y_true, y_pred):
    """
    Calculate the R-squared (R2) score between true and predicted values.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values from the model.
    
    Returns:
    float: The R2 score.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a model using various metrics.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values from the model.
    
    Returns:
    dict: A dictionary containing RMSE, MAE, and R2 score.
    """
    metrics = {
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred)
    }
    return metrics