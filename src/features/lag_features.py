"""
Lag and historical rolling statistical feature generation.
Strictly ensures zero data leakage by conditioning all operations only on past observations.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("lag_features")


class LagFeatureGenerator:
    """
    Generates historical lag features and backward-only rolling statistics.
    """

    def __init__(
        self,
        energy_lags: Optional[List[int]] = None,
        weather_lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        rolling_stats: Optional[List[str]] = None,
        target_column: str = "energy_output_kwh",
    ):
        self.energy_lags = energy_lags or [1, 24, 168]
        self.weather_lags = weather_lags or [1, 24]
        self.rolling_windows = rolling_windows or [6, 24]
        self.rolling_stats = rolling_stats or ["mean", "std"]
        self.target_column = target_column

    def create_lags(self, df: pd.DataFrame, weather_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create lagged features for target and weather variables.

        Args:
            df: DataFrame sorted chronologically.
            weather_columns: Optional list of weather column names to lag.

        Returns:
            DataFrame with lag columns appended.
        """
        df_lags = df.copy()

        # 1. Target Lags (e.g. t-1, t-24, t-168)
        if self.target_column in df_lags.columns:
            for lag in self.energy_lags:
                col_name = f"{self.target_column}_lag_{lag}"
                df_lags[col_name] = df_lags[self.target_column].shift(lag)
                logger.debug(f"Created target lag feature: {col_name}")

        # 2. Weather Lags (e.g. t-1, t-24)
        if weather_columns:
            for col in weather_columns:
                if col in df_lags.columns:
                    for lag in self.weather_lags:
                        col_name = f"{col}_lag_{lag}"
                        df_lags[col_name] = df_lags[col].shift(lag)

        return df_lags

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling statistics on historical target observations.
        CRITICAL: Uses shift(1) so that current time t or future values never leak into the rolling window.

        Args:
            df: DataFrame sorted chronologically.

        Returns:
            DataFrame with rolling statistical features appended.
        """
        df_rolling = df.copy()

        if self.target_column in df_rolling.columns:
            # Shift by 1 first to guarantee that observation at time t is excluded from rolling window for t
            shifted_series = df_rolling[self.target_column].shift(1)

            for window in self.rolling_windows:
                for stat in self.rolling_stats:
                    col_name = f"{self.target_column}_rolling_{window}h_{stat}"
                    if stat == "mean":
                        df_rolling[col_name] = shifted_series.rolling(window=window, min_periods=1).mean()
                    elif stat == "std":
                        df_rolling[col_name] = shifted_series.rolling(window=window, min_periods=1).std().fillna(0.0)
                    elif stat == "min":
                        df_rolling[col_name] = shifted_series.rolling(window=window, min_periods=1).min()
                    elif stat == "max":
                        df_rolling[col_name] = shifted_series.rolling(window=window, min_periods=1).max()
                    logger.debug(f"Created rolling feature: {col_name}")

        return df_rolling
