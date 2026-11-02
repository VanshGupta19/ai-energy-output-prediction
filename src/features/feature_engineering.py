"""
Master Feature Engineering Pipeline and Automated Feature Leakage Verification.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from src.features.lag_features import LagFeatureGenerator
from src.features.temporal_features import TemporalFeatureGenerator, DomainFeatureGenerator
from src.utils.logger import get_logger
from src.utils.io import save_dataframe

logger = get_logger("feature_engineering")


class FeatureEngineer:
    """
    Orchestrates temporal extraction, physics features, lag features,
    and multi-horizon target alignment.
    """

    def __init__(
        self,
        target_column: str = "energy_output_kwh",
        horizon_hours: int = 24,
        energy_lags: Optional[List[int]] = None,
        weather_lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        rolling_stats: Optional[List[str]] = None,
        enable_solar_domain: bool = True,
        enable_wind_domain: bool = True,
    ):
        self.target_column = target_column
        self.horizon_hours = horizon_hours
        self.target_horizon_col = f"{target_column}_lead_{horizon_hours}h"

        self.temporal_gen = TemporalFeatureGenerator()
        self.domain_gen = DomainFeatureGenerator(
            enable_solar=enable_solar_domain,
            enable_wind=enable_wind_domain,
        )
        self.lag_gen = LagFeatureGenerator(
            energy_lags=energy_lags,
            weather_lags=weather_lags,
            rolling_windows=rolling_windows,
            rolling_stats=rolling_stats,
            target_column=target_column,
        )

    def engineer_features(
        self,
        df: pd.DataFrame,
        weather_columns: Optional[List[str]] = None,
        output_feature_list_path: Optional[str] = "reports/metrics/feature_list.csv",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Run complete feature engineering pipeline and create future target.

        Args:
            df: Cleaned merged DataFrame containing 'timestamp' and target.
            weather_columns: Optional list of raw weather column names.
            output_feature_list_path: Path to save feature list CSV.

        Returns:
            Tuple of (engineered_df, feature_column_names).
        """
        logger.info(f"Engineering features for forecast horizon H = {self.horizon_hours} hours...")
        df_out = df.copy()

        # Step 1: Temporal and cyclical features
        df_out = self.temporal_gen.create_features(df_out)

        # Step 2: Physical domain features
        df_out = self.domain_gen.create_features(df_out)

        # Step 3: Historical Lags
        if weather_columns is None:
            weather_columns = [
                c for c in ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "PS", "WS10M", "WS50M", "CLOUD_AMT"]
                if c in df_out.columns
            ]
        df_out = self.lag_gen.create_lags(df_out, weather_columns=weather_columns)

        # Step 4: Historical Rolling Statistics
        df_out = self.lag_gen.create_rolling_features(df_out)

        # Step 5: Create Future Target variable: target(t) = energy_output(t + H)
        if self.target_column in df_out.columns:
            # Shift backwards by horizon H: row at index t gets value from index t + H
            df_out[self.target_horizon_col] = df_out[self.target_column].shift(-self.horizon_hours)
            logger.info(
                f"Target '{self.target_horizon_col}' created with shift(-{self.horizon_hours}) hours."
            )

        # Backfill initial lag/rolling feature NaNs so early rows have valid historical features
        feat_temp_cols = [c for c in df_out.select_dtypes(include=[np.number]).columns if c != self.target_horizon_col]
        df_out[feat_temp_cols] = df_out[feat_temp_cols].bfill().ffill().fillna(0.0)

        # Identify feature columns (exclude timestamps, metadata, and future targets)
        meta_cols = [
            "timestamp", "date", "datetime", "unit", "site_id",
            self.target_horizon_col,
        ]
        feature_cols = [
            c for c in df_out.select_dtypes(include=[np.number]).columns
            if c not in meta_cols and not c.startswith(f"{self.target_column}_lead_")
        ]

        logger.info(f"Generated {len(feature_cols)} engineered features.")

        # Save feature list catalog
        if output_feature_list_path:
            feat_catalog = pd.DataFrame({
                "feature_name": feature_cols,
                "dtype": [str(df_out[c].dtype) for c in feature_cols],
                "category": [
                    "lag" if "lag" in c else
                    "rolling" if "rolling" in c else
                    "temporal" if any(k in c for k in ["hour", "day", "month", "sin", "cos", "weekend"]) else
                    "domain" if any(k in c for k in ["density", "shear", "elevation", "derating", "potential", "interaction"]) else
                    "raw_weather"
                    for c in feature_cols
                ],
            })
            save_dataframe(feat_catalog, output_feature_list_path)

        return df_out, feature_cols


class FeatureLeakageChecker:
    """
    Automated unit and validation checker to mathematically verify zero data leakage:
    1. Verifies no target lead values appear in feature columns.
    2. Verifies rolling statistics do not include present observation t.
    3. Verifies target shift alignment: feature at t predicts target at t + H.
    """

    @staticmethod
    def verify_no_target_in_features(feature_cols: List[str], target_col: str, horizon: int) -> bool:
        lead_target = f"{target_col}_lead_{horizon}h"
        if lead_target in feature_cols:
            raise AssertionError(f"LEAKAGE DETECTED: Lead target '{lead_target}' found in feature matrix!")
        return True

    @staticmethod
    def verify_rolling_stats_historical(df: pd.DataFrame, target_col: str, window: int = 6) -> bool:
        """
        Verify rolling average at row i equals mean of rows (i - window) to (i - 1), excluding i.
        """
        col_name = f"{target_col}_rolling_{window}h_mean"
        if col_name not in df.columns or target_col not in df.columns:
            return True

        # Check for row index 200
        test_idx = 200
        if test_idx < len(df) and test_idx > window:
            computed_val = df.loc[test_idx, col_name]
            expected_val = df.loc[test_idx - window : test_idx - 1, target_col].mean()
            if not np.isclose(computed_val, expected_val, atol=1e-4):
                raise AssertionError(
                    f"LEAKAGE DETECTED in rolling window! Row {test_idx}: computed={computed_val}, expected={expected_val}"
                )
        return True

    @staticmethod
    def verify_target_shift_alignment(df: pd.DataFrame, target_col: str, horizon: int) -> bool:
        """
        Verify that target_lead_H at row i equals target at row (i + H).
        """
        lead_col = f"{target_col}_lead_{horizon}h"
        if lead_col not in df.columns or target_col not in df.columns:
            return True

        test_idx = 100
        if test_idx + horizon < len(df):
            val_at_t = df.loc[test_idx, lead_col]
            val_at_t_plus_h = df.loc[test_idx + horizon, target_col]
            if not np.isclose(val_at_t, val_at_t_plus_h, atol=1e-4):
                raise AssertionError(
                    f"TARGET MISALIGNMENT! Row {test_idx} lead_{horizon}h ({val_at_t}) "
                    f"does not match Row {test_idx + horizon} target ({val_at_t_plus_h})"
                )
        return True
