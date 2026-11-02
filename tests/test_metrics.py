"""
Unit tests for evaluation metrics: MAE, MSE, RMSE, zero-safe MAPE, R2, and PICP.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    calculate_mae,
    calculate_mse,
    calculate_rmse,
    calculate_mape,
    calculate_r2,
    calculate_picp,
    calculate_mpiw,
    evaluate_all_metrics,
)


def test_standard_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 39.0])

    mae = calculate_mae(y_true, y_pred)
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)

    assert np.isclose(mae, (2 + 2 + 3 + 1) / 4.0)
    assert np.isclose(mse, (4 + 4 + 9 + 1) / 4.0)
    assert np.isclose(rmse, np.sqrt(mse))
    assert 0.9 < r2 <= 1.0


def test_zero_safe_mape():
    # In solar energy, night values are 0.0
    y_true = np.array([0.0, 0.0, 100.0, 200.0])
    y_pred = np.array([0.0, 5.0, 90.0, 210.0])

    # Must NOT divide by zero or raise warning/inf
    mape = calculate_mape(y_true, y_pred, mask_threshold=0.05)
    assert not np.isnan(mape)
    assert not np.isinf(mape)
    # Expected on positive values: mean(|100-90|/100, |200-210|/200) = (10% + 5%) / 2 = 7.5%
    assert np.isclose(mape, 7.5, atol=1e-3)


def test_picp_calculation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lower = np.array([8.0, 18.0, 35.0, 38.0, 45.0]) # Row 3 is outside interval (30 < 35)
    upper = np.array([12.0, 22.0, 40.0, 42.0, 55.0])

    picp = calculate_picp(y_true, lower, upper)
    # 4 out of 5 inside -> 80%
    assert np.isclose(picp, 80.0)

    mpiw = calculate_mpiw(lower, upper)
    # Average width = (4 + 4 + 5 + 4 + 10) / 5 = 5.4
    assert np.isclose(mpiw, 5.4)
