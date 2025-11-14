import numpy as np

def calculate_mae(y_true, y_pred):
    """
    计算标量的平均绝对误差 (MAE)。
    参数:
        y_true (numpy.ndarray): 真实值。
        y_pred (numpy.ndarray): 预测值。
    返回:
        float: 标量 MAE。
    """
    mae = np.abs(y_true - y_pred).mean()
    return mae
