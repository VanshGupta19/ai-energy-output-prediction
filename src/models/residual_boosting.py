"""
Gradient Boosting Residual Correction Model.
Learns residual errors of the MLP-LSTM base model: residual = actual - mlp_lstm_prediction.
Final Prediction = mlp_lstm_prediction + predicted_residual.
"""

import os
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from src.utils.logger import get_logger
from src.utils.io import save_model, load_model

logger = get_logger("residual_boosting")


class ResidualBoostingModel:
    """
    Second-stage gradient boosting estimator for residual error correction.
    Supports XGBoost with automatic fallback to LightGBM or HistGradientBoosting.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        early_stopping_rounds: Optional[int] = None,
    ):
        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds

        self.model_ = None
        self.feature_names_: List[str] = []
        self._init_booster()

    def _init_booster(self):
        """Initialize the booster instance with fallbacks."""
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                kwargs = {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "subsample": self.subsample,
                    "colsample_bytree": self.colsample_bytree,
                    "random_state": self.random_state,
                    "n_jobs": -1,
                }
                if self.early_stopping_rounds is not None:
                    kwargs["early_stopping_rounds"] = self.early_stopping_rounds
                self.model_ = xgb.XGBRegressor(**kwargs)
                logger.info("Initialized XGBoostRegressor for residual correction.")
                return
            except ImportError:
                logger.warning("XGBoost not available. Falling back to LightGBM.")
                self.model_type = "lightgbm"

        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                self.model_ = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
                logger.info("Initialized LightGBMRegressor for residual correction.")
                return
            except ImportError:
                logger.warning("LightGBM not available. Falling back to HistGradientBoostingRegressor.")
                self.model_type = "hist_gradient_boosting"

        # Final Fallback
        self.model_ = HistGradientBoostingRegressor(
            max_iter=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        logger.info("Initialized HistGradientBoostingRegressor fallback.")

    def _build_residual_features(
        self,
        X_tabular: np.ndarray,
        base_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Assemble features available at prediction time:
        [Tabular Features at timestep t, Base MLP-LSTM Prediction].
        Does NOT include ground-truth target.
        """
        base_pred_col = np.asarray(base_predictions).reshape(-1, 1)
        return np.hstack([X_tabular, base_pred_col])

    def fit(
        self,
        X_train_tabular: np.ndarray,
        base_train_preds: np.ndarray,
        y_train_actual: np.ndarray,
        X_val_tabular: Optional[np.ndarray] = None,
        base_val_preds: Optional[np.ndarray] = None,
        y_val_actual: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Train Gradient Boosting model to predict residuals: (actual - base_prediction).
        """
        # Calculate residual errors
        train_residuals = np.asarray(y_train_actual) - np.asarray(base_train_preds)
        X_boost_train = self._build_residual_features(X_train_tabular, base_train_preds)

        # Filter any NaNs
        valid_mask_tr = ~np.isnan(train_residuals) & ~np.isnan(X_boost_train).any(axis=1)
        X_tr_clean = X_boost_train[valid_mask_tr]
        y_res_clean = train_residuals[valid_mask_tr]

        if feature_names:
            self.feature_names_ = list(feature_names) + ["base_mlp_lstm_prediction"]
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X_train_tabular.shape[1])] + ["base_mlp_lstm_prediction"]

        logger.info(f"Training {self.model_type.upper()} on residual target ({len(X_tr_clean)} samples, mean={np.mean(y_res_clean):.4f}, std={np.std(y_res_clean):.4f})...")

        if X_val_tabular is not None and base_val_preds is not None and y_val_actual is not None:
            val_residuals = np.asarray(y_val_actual) - np.asarray(base_val_preds)
            X_boost_val = self._build_residual_features(X_val_tabular, base_val_preds)
            valid_mask_v = ~np.isnan(val_residuals) & ~np.isnan(X_boost_val).any(axis=1)
            X_v_clean = X_boost_val[valid_mask_v]
            y_v_clean = val_residuals[valid_mask_v]

            if len(X_v_clean) > 0 and self.model_type == "xgboost" and hasattr(self.model_, "early_stopping_rounds") and self.model_.early_stopping_rounds:
                self.model_.fit(
                    X_tr_clean,
                    y_res_clean,
                    eval_set=[(X_v_clean, y_v_clean)],
                    verbose=False,
                )
            else:
                self.model_.fit(X_tr_clean, y_res_clean)
        else:
            self.model_.fit(X_tr_clean, y_res_clean)

        logger.info("Residual correction booster training complete.")
        return self

    def predict_residual(
        self,
        X_tabular: np.ndarray,
        base_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the residual error component.
        """
        X_boost = self._build_residual_features(X_tabular, base_predictions)
        return self.model_.predict(X_boost)

    def predict_final(
        self,
        X_tabular: np.ndarray,
        base_predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute final hybrid forecast:
        Final Prediction = Base Prediction + Predicted Residual.

        Returns:
            Tuple of (final_prediction, predicted_residual).
        """
        pred_residual = self.predict_residual(X_tabular, base_predictions)
        final_pred = base_predictions + pred_residual
        # Physically bound energy output to non-negative
        final_pred = np.maximum(0.0, final_pred)
        return final_pred, pred_residual

    def get_booster(self):
        """Return the underlying booster instance."""
        return self.model_

    def save(self, file_path: str) -> None:
        """Save booster model to disk."""
        save_model(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> "ResidualBoostingModel":
        """Load booster model from disk."""
        return load_model(file_path)
