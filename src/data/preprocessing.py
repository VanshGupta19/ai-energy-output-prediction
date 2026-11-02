"""
Sklearn-style Preprocessing Pipeline for time-series meteorological and energy data.
Enforces strict chronological boundaries and leak-free scaling.
"""

import os
from typing import Dict, Any, Tuple, Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from src.utils.logger import get_logger
from src.utils.io import save_model, load_model

logger = get_logger("preprocessing")


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Leak-free time-series preprocessing pipeline:
    1. Replaces known placeholder values (-999) with NaN.
    2. Performs time-aware interpolation for small gaps.
    3. Handles outliers with configurable bounds.
    4. Fits Scaler strictly on training partition and transforms validation/test partitions.
    """

    PLACEHOLDERS = [-999.0, -999, -9999.0, -9999, 999.0, 999]

    def __init__(
        self,
        scaler_type: str = "minmax",
        feature_range: Tuple[float, float] = (0.0, 1.0),
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        max_interp_gap: int = 6,
        target_column: str = "energy_output_kwh",
    ):
        self.scaler_type = scaler_type
        self.feature_range = feature_range
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.max_interp_gap = max_interp_gap
        self.target_column = target_column

        self.feature_columns_: List[str] = []
        self.feature_scaler_ = None
        self.target_scaler_ = None
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.is_fitted_ = False

    def _clean_missing_and_outliers(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Clean placeholders, interpolate missing values, and handle extreme outliers.
        """
        df_clean = df.copy()

        # Step 1: Replace placeholders with NaN
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[num_cols] = df_clean[num_cols].replace(self.PLACEHOLDERS, np.nan)

        # Step 2: Time-aware / Linear interpolation for short gaps
        if "timestamp" in df_clean.columns:
            df_clean = df_clean.set_index("timestamp")
            df_clean[num_cols] = df_clean[num_cols].interpolate(
                method="time", limit=self.max_interp_gap, limit_direction="forward"
            )
            df_clean = df_clean.reset_index()
        else:
            df_clean[num_cols] = df_clean[num_cols].interpolate(
                method="linear", limit=self.max_interp_gap, limit_direction="forward"
            )

        # Forward fill any remaining edge cases
        df_clean[num_cols] = df_clean[num_cols].ffill().bfill()

        # Step 3: Outlier Handling (learn bounds only during training)
        if is_training:
            self.outlier_bounds_ = {}
            for col in num_cols:
                vals = df_clean[col].dropna()
                if self.outlier_method == "iqr":
                    q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
                    iqr = q75 - q25
                    lower = q25 - (self.outlier_threshold * iqr)
                    upper = q75 + (self.outlier_threshold * iqr)
                else: # zscore
                    mean, std = vals.mean(), vals.std()
                    lower = mean - (self.outlier_threshold * std)
                    upper = mean + (self.outlier_threshold * std)
                self.outlier_bounds_[col] = (lower, upper)

        # Clip values to bounds
        for col, (lower, upper) in self.outlier_bounds_.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

        return df_clean

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray, pd.DataFrame]] = None):
        """
        Fit scaler and outlier parameters strictly on the training partition.
        """
        logger.info("Fitting PreprocessingPipeline on training partition...")
        df_clean = self._clean_missing_and_outliers(X, is_training=True)

        # Identify numeric feature columns excluding timestamp and target
        exclude_cols = ["timestamp", "date", "datetime", "unit", "site_id"]
        if self.target_column in df_clean.columns:
            exclude_cols.append(self.target_column)

        self.feature_columns_ = [
            c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in exclude_cols
        ]

        # Initialize Scalers
        if self.scaler_type == "minmax":
            self.feature_scaler_ = MinMaxScaler(feature_range=self.feature_range)
            self.target_scaler_ = MinMaxScaler(feature_range=self.feature_range)
        elif self.scaler_type == "standard":
            self.feature_scaler_ = StandardScaler()
            self.target_scaler_ = StandardScaler()
        elif self.scaler_type == "robust":
            self.feature_scaler_ = RobustScaler()
            self.target_scaler_ = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        # Fit feature scaler
        if self.feature_columns_:
            self.feature_scaler_.fit(df_clean[self.feature_columns_].values)

        # Fit target scaler if target provided
        if self.target_column in df_clean.columns:
            target_data = df_clean[[self.target_column]].values
            self.target_scaler_.fit(target_data)
        elif y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                target_data = y.values.reshape(-1, 1)
            else:
                target_data = np.array(y).reshape(-1, 1)
            self.target_scaler_.fit(target_data)

        self.is_fitted_ = True
        logger.info(f"Fitted scaler on {len(self.feature_columns_)} numerical features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted parameters.
        """
        if not self.is_fitted_:
            raise RuntimeError("PreprocessingPipeline must be fitted before calling transform.")

        df_clean = self._clean_missing_and_outliers(X, is_training=False)
        df_transformed = df_clean.copy()

        # Scale features
        if self.feature_columns_:
            scaled_features = self.feature_scaler_.transform(df_clean[self.feature_columns_].values)
            df_transformed[self.feature_columns_] = scaled_features

        # Scale target if present
        if self.target_column in df_clean.columns and self.target_scaler_ is not None:
            scaled_target = self.target_scaler_.transform(df_clean[[self.target_column]].values)
            df_transformed[self.target_column] = scaled_target.flatten()

        return df_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """Fit to training data, then transform."""
        return self.fit(X, y).transform(X)

    def inverse_transform_target(self, y_scaled: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original physical energy units (kWh / kW / MW).
        """
        if self.target_scaler_ is None or not self.is_fitted_:
            raise RuntimeError("Target scaler is not fitted.")

        y_arr = np.array(y_scaled).reshape(-1, 1)
        return self.target_scaler_.inverse_transform(y_arr).flatten()

    def save(self, file_path: str) -> None:
        """Save fitted pipeline to disk."""
        save_model(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> "PreprocessingPipeline":
        """Load fitted pipeline from disk."""
        return load_model(file_path)
