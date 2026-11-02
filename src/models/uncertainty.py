"""
Conformal Prediction and Empirical Uncertainty Intervals for Energy Forecasting.
Computes calibrated prediction intervals and evaluates Prediction Interval Coverage Probability (PICP).
"""

import os
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
from src.evaluation.metrics import calculate_picp, calculate_mpiw
from src.utils.logger import get_logger
from src.utils.io import save_model, load_model

logger = get_logger("uncertainty")


class ConformalPredictionInterval:
    r"""
    Split Conformal Prediction for distribution-free, finite-sample calibrated prediction intervals.
    
    Calibration:
    Computes absolute residual non-conformity scores: S_i = |y_i - \hat{y}_i| on held-out calibration set.
    Given target confidence 1 - alpha (e.g. 0.90 for 90%), computes conformal quantile:
    q = Quantile(S, ceil((n + 1) * (1 - alpha)) / n).
    
    Inference:
    Lower Bound = max(0, \hat{y} - q)
    Upper Bound = \hat{y} + q
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.quantile_q_: Optional[float] = None
        self.n_calib_samples_: int = 0

    def calibrate(self, y_calib_true: np.ndarray, y_calib_pred: np.ndarray) -> "ConformalPredictionInterval":
        """
        Calibrate prediction interval quantile using a held-out calibration partition.
        """
        y_t = np.asarray(y_calib_true).flatten()
        y_p = np.asarray(y_calib_pred).flatten()

        n = len(y_t)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Compute absolute non-conformity residuals
        scores = np.abs(y_t - y_p)
        # Conformal adjusted quantile level
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile_q_ = float(np.quantile(scores, q_level, method="higher"))
        self.n_calib_samples_ = n

        logger.info(
            f"Calibrated Conformal Interval ({self.confidence_level * 100:.0f}% confidence, "
            f"n={n}): Half-width q = {self.quantile_q_:.4f}"
        )
        return self

    def predict_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute lower and upper prediction interval bounds for new predictions.

        Returns:
            Tuple of (lower_bounds, upper_bounds).
        """
        if self.quantile_q_ is None:
            raise RuntimeError("ConformalPredictionInterval must be calibrated before generating intervals.")

        preds = np.asarray(y_pred).flatten()
        lower_bounds = np.maximum(0.0, preds - self.quantile_q_)
        upper_bounds = preds + self.quantile_q_

        return lower_bounds, upper_bounds

    def evaluate_coverage(
        self,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Generate intervals for test predictions and compute PICP, MPIW, and coverage gap.
        """
        lb, ub = self.predict_intervals(y_test_pred)
        picp = calculate_picp(y_test_true, lb, ub)
        mpiw = calculate_mpiw(lb, ub)
        target_coverage = self.confidence_level * 100.0

        results = {
            "target_coverage_pct": target_coverage,
            "observed_picp_pct": round(picp, 2),
            "coverage_gap_pct": round(picp - target_coverage, 2),
            "mpiw_width": round(mpiw, 4),
            "conformal_q": round(self.quantile_q_, 4),
            "calibration_samples": self.n_calib_samples_,
        }

        logger.info(
            f"Uncertainty Evaluation: Target={target_coverage:.1f}%, PICP={picp:.2f}%, "
            f"Width={mpiw:.4f}, Gap={results['coverage_gap_pct']:.2f}%"
        )
        return results

    def save(self, file_path: str) -> None:
        save_model(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> "ConformalPredictionInterval":
        return load_model(file_path)
