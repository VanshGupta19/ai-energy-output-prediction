"""
Evaluation metrics suite for energy forecasting and interval reliability.
Includes robust handling for zero-values in percentage error calculations.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.logger import get_logger

logger = get_logger("metrics")


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(mean_squared_error(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1.0,
    mask_threshold: Optional[float] = 0.05,
) -> float:
    """
    Robust Mean Absolute Percentage Error (MAPE).
    
    Renewable solar energy has zero generation during nighttime. Direct division by zero
    causes division by zero / Inf. We provide a masked/epsilon-stabilized MAPE:
    MAPE = 100 * mean(|y_true - y_pred| / max(|y_true|, epsilon)) for values where y_true > threshold.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        epsilon: Small positive constant added to denominator.
        mask_threshold: Minimum actual value to include in percentage calculation.

    Returns:
        MAPE percentage (e.g. 8.45 for 8.45%).
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()

    if mask_threshold is not None:
        valid_mask = y_t > mask_threshold
        if np.sum(valid_mask) == 0:
            return 0.0
        denom = y_t[valid_mask]
        num = np.abs(y_t[valid_mask] - y_p[valid_mask])
    else:
        denom = np.maximum(np.abs(y_t), epsilon)
        num = np.abs(y_t - y_p)

    mape = np.mean(num / denom) * 100.0
    return float(mape)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R^2 score)."""
    return float(r2_score(y_true, y_pred))


def calculate_picp(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> float:
    """
    Prediction Interval Coverage Probability (PICP).
    PICP = (Count of actual values within [lower_bound, upper_bound]) / (Total predictions)

    Args:
        y_true: Ground truth target values.
        lower_bound: Lower prediction interval boundary.
        upper_bound: Upper prediction interval boundary.

    Returns:
        PICP as percentage (0.0 to 100.0).
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    lb = np.asarray(lower_bound, dtype=np.float64).flatten()
    ub = np.asarray(upper_bound, dtype=np.float64).flatten()

    inside = (y_t >= lb) & (y_t <= ub)
    picp = (np.sum(inside) / len(y_t)) * 100.0
    return float(picp)


def calculate_mpiw(lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """
    Mean Prediction Interval Width (MPIW).
    MPIW = mean(upper_bound - lower_bound)
    """
    lb = np.asarray(lower_bound, dtype=np.float64).flatten()
    ub = np.asarray(upper_bound, dtype=np.float64).flatten()
    return float(np.mean(ub - lb))


def evaluate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation dictionary across point and interval metrics.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()

    metrics = {
        "MAE": round(calculate_mae(y_t, y_p), 4),
        "MSE": round(calculate_mse(y_t, y_p), 4),
        "RMSE": round(calculate_rmse(y_t, y_p), 4),
        "MAPE": round(calculate_mape(y_t, y_p), 2),
        "R2": round(calculate_r2(y_t, y_p), 4),
    }

    if lower_bound is not None and upper_bound is not None:
        metrics["PICP"] = round(calculate_picp(y_t, lower_bound, upper_bound), 2)
        metrics["MPIW"] = round(calculate_mpiw(lower_bound, upper_bound), 4)

    return metrics
