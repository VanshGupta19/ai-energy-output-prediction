"""
Temporal feature extraction, cyclical trigonometry encodings, and physics-informed domain features.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("temporal_features")


class TemporalFeatureGenerator:
    """
    Extracts chronological calendar indicators and smooth cyclical trigonometric features.
    """

    def __init__(self, cyclical_vars: Optional[list] = None):
        self.cyclical_vars = cyclical_vars or ["hour", "dayofyear"]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract date/time and cyclical features.

        Args:
            df: DataFrame containing 'timestamp' datetime column.

        Returns:
            DataFrame with temporal and cyclical features.
        """
        if "timestamp" not in df.columns:
            raise KeyError("DataFrame must contain 'timestamp' column.")

        df_feat = df.copy()
        ts = pd.to_datetime(df_feat["timestamp"])

        # Calendar features
        df_feat["hour"] = ts.dt.hour
        df_feat["dayofweek"] = ts.dt.dayofweek
        df_feat["dayofmonth"] = ts.dt.day
        df_feat["month"] = ts.dt.month
        df_feat["dayofyear"] = ts.dt.dayofyear
        df_feat["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        # Cyclical Encodings (sin / cos)
        # Hour of day (24-hour cycle)
        df_feat["hour_sin"] = np.sin(2.0 * np.pi * df_feat["hour"] / 24.0)
        df_feat["hour_cos"] = np.cos(2.0 * np.pi * df_feat["hour"] / 24.0)

        # Day of year (365.25-day annual cycle)
        df_feat["dayofyear_sin"] = np.sin(2.0 * np.pi * df_feat["dayofyear"] / 365.25)
        df_feat["dayofyear_cos"] = np.cos(2.0 * np.pi * df_feat["dayofyear"] / 365.25)

        # Month of year (12-month cycle)
        df_feat["month_sin"] = np.sin(2.0 * np.pi * df_feat["month"] / 12.0)
        df_feat["month_cos"] = np.cos(2.0 * np.pi * df_feat["month"] / 12.0)

        return df_feat


class DomainFeatureGenerator:
    """
    Physics-grounded domain feature engineering for renewable energy output.
    Computes solar interactions, temperature derating coefficients, and wind power density.
    """

    def __init__(self, enable_solar: bool = True, enable_wind: bool = True):
        self.enable_solar = enable_solar
        self.enable_wind = enable_wind

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate physics-based domain features where variables are available.
        """
        df_dom = df.copy()

        # 1. Solar Physics Features
        if self.enable_solar:
            # Solar angle approximation based on hour
            if "hour" in df_dom.columns:
                # Geometric solar elevation proxy
                df_dom["solar_elevation_proxy"] = np.maximum(
                    0.0, np.sin(np.pi * (df_dom["hour"] - 6) / 12.0)
                )

            # Cloud-Irradiance Interaction
            if "ALLSKY_SFC_SW_DWN" in df_dom.columns and "CLOUD_AMT" in df_dom.columns:
                df_dom["clearness_cloud_interaction"] = (
                    df_dom["ALLSKY_SFC_SW_DWN"] * (1.0 - (df_dom["CLOUD_AMT"] / 100.0))
                )

            # Temperature Derating Factor: Photovoltaic cell efficiency drops with temperature (approx -0.4%/degC above 25C)
            if "T2M" in df_dom.columns and "ALLSKY_SFC_SW_DWN" in df_dom.columns:
                df_dom["pv_temp_derating_factor"] = (
                    1.0 - 0.004 * np.maximum(0.0, df_dom["T2M"] - 25.0)
                )
                df_dom["effective_solar_potential"] = (
                    df_dom["ALLSKY_SFC_SW_DWN"] * df_dom["pv_temp_derating_factor"]
                )

        # 2. Wind Physics Features: Wind Power Density P = 0.5 * rho * v^3
        if self.enable_wind:
            # 50m Wind Power Density
            if "RHOA" in df_dom.columns and "WS50M" in df_dom.columns:
                df_dom["wind_power_density_50m"] = 0.5 * df_dom["RHOA"] * (df_dom["WS50M"] ** 3)
            # 10m Wind Power Density fallback
            elif "RHOA" in df_dom.columns and "WS10M" in df_dom.columns:
                df_dom["wind_power_density_10m"] = 0.5 * df_dom["RHOA"] * (df_dom["WS10M"] ** 3)

            # Wind Shear Exponent if both 10m and 50m are available: alpha = ln(v50 / v10) / ln(50 / 10)
            if "WS50M" in df_dom.columns and "WS10M" in df_dom.columns:
                safe_ws10 = np.maximum(df_dom["WS10M"], 0.1)
                safe_ws50 = np.maximum(df_dom["WS50M"], 0.1)
                df_dom["wind_shear_alpha"] = np.log(safe_ws50 / safe_ws10) / np.log(50.0 / 10.0)

        return df_dom
