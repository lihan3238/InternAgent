import numpy as np
import torch
import torch.nn as nn

def MAE(predictions, targets):
    """
    Mean Absolute Error
    """
    return torch.mean(torch.abs(predictions - targets)).item()

def RMSE(predictions, targets):
    """
    Root Mean Square Error
    """
    mse = nn.MSELoss()
    return torch.sqrt(mse(predictions, targets)).item()

def MAPE(predictions, targets, epsilon=1e-8):
    """
    Mean Absolute Percentage Error
    """
    targets = torch.clamp(targets, min=epsilon)  # Avoid division by zero
    return torch.mean(torch.abs((predictions - targets) / targets) * 100).item()

def SMAPE(predictions, targets):
    """
    Symmetric Mean Absolute Percentage Error
    """
    denominator = torch.abs(predictions) + torch.abs(targets)
    denominator = torch.clamp(denominator, min=1e-8)  # Avoid division by zero
    return torch.mean(torch.abs(predictions - targets) / denominator * 200).item()

def R2(predictions, targets):
    """
    Coefficient of Determination (R²)
    """
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return (1 - ss_res / ss_tot).item()

def calculate_energy_metrics(y_true, y_pred):
    """
    Calculate comprehensive energy forecasting metrics

    Args:
        y_true: Ground truth values (torch.Tensor or numpy.ndarray)
        y_pred: Predicted values (torch.Tensor or numpy.ndarray)

    Returns:
        dict: Dictionary containing all metrics
    """
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    metrics = {
        'MAE': MAE(y_pred, y_true),
        'RMSE': RMSE(y_pred, y_true),
        'MAPE': MAPE(y_pred, y_true),
        'SMAPE': SMAPE(y_pred, y_true),
        'R2': R2(y_pred, y_true)
    }

    return metrics

def print_metrics(metrics_dict, prefix=""):
    """
    Print formatted metrics

    Args:
        metrics_dict: Dictionary of metrics
        prefix: Optional prefix for metric names
    """
    print(f"\n{prefix}Metrics:")
    print("-" * 30)
    for metric_name, value in metrics_dict.items():
        if metric_name in ['MAPE', 'SMAPE']:
            print(f"{metric_name}: {value:.2f}")
        else:
            print(f"{metric_name}: {value:.4f}")
    print("-" * 30)