"""Model evaluation functions"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, dataset_name=""):
    """Calculate evaluation metrics: MSE, RMSE, MAE, R2"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
    }
    
    if dataset_name:
        logger.info(f"{dataset_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    return metrics


def get_residuals(y_true, y_pred):
    """Calculate absolute residuals (prediction errors)"""
    return np.abs(y_true - y_pred)
