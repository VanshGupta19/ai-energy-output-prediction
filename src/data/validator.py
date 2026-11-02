"""
Data validation, quality inspection, and target integrity checker.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.io import save_dataframe

logger = get_logger("data_validator")


class DataValidator:
    """
    Validates meteorological and energy datasets, identifies missing/placeholder values,
    detects anomalies, and verifies target integrity.
    """

    KNOWN_PLACEHOLDERS = [-999.0, -999, -9999.0, -9999, 999.0, 999]

    PHYSICAL_BOUNDS = {
        "ALLSKY_SFC_SW_DWN": (0.0, 1.5), # kW/m^2
        "T2M": (-60.0, 65.0), # Celsius
        "RH2M": (0.0, 100.0), # %
        "PS": (30.0, 110.0), # kPa
        "WS10M": (0.0, 75.0), # m/s
        "WD10M": (0.0, 360.0), # deg
        "WS50M": (0.0, 90.0), # m/s
        "WD50M": (0.0, 360.0), # deg
        "RHOA": (0.5, 2.0), # kg/m^3
        "QV10M": (0.0, 50.0), # g/kg
        "CLOUD_AMT": (0.0, 100.0), # %
    }

    def __init__(self, target_column: str = "energy_output_kwh", allow_synthetic_demo: bool = False):
        self.target_column = target_column
        self.allow_synthetic_demo = allow_synthetic_demo

    def validate(
        self,
        weather_df: pd.DataFrame,
        energy_df: Optional[pd.DataFrame] = None,
        output_report_path: Optional[str] = "reports/data_quality/data_quality_report.csv",
    ) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """
        Run complete validation on weather and energy datasets.

        Returns:
            Tuple of (is_valid, quality_report_df, summary_metrics).
        """
        logger.info("Executing comprehensive data validation and quality audit...")
        report_rows = []

        # 1. Weather Data Inspection
        w_df = weather_df.copy()
        for col in w_df.select_dtypes(include=[np.number]).columns:
            # Detect placeholders
            placeholder_count = w_df[col].isin(self.KNOWN_PLACEHOLDERS).sum()
            null_count = w_df[col].isnull().sum()
            total_missing = placeholder_count + null_count
            missing_pct = (total_missing / len(w_df)) * 100.0

            # Calculate stats on valid values
            valid_vals = w_df[col][~w_df[col].isin(self.KNOWN_PLACEHOLDERS)].dropna()
            val_min = valid_vals.min() if not valid_vals.empty else np.nan
            val_max = valid_vals.max() if not valid_vals.empty else np.nan
            val_mean = valid_vals.mean() if not valid_vals.empty else np.nan
            val_std = valid_vals.std() if not valid_vals.empty else np.nan

            # Check physical bounds
            out_of_bounds = 0
            if col in self.PHYSICAL_BOUNDS and not valid_vals.empty:
                low_b, high_b = self.PHYSICAL_BOUNDS[col]
                out_of_bounds = int(((valid_vals < low_b) | (valid_vals > high_b)).sum())

            # Check constant
            is_constant = (val_std == 0.0) or (len(valid_vals.unique()) <= 1)

            report_rows.append({
                "dataset": "weather",
                "column": col,
                "dtype": str(w_df[col].dtype),
                "total_records": len(w_df),
                "placeholder_count": int(placeholder_count),
                "null_count": int(null_count),
                "missing_pct": round(missing_pct, 2),
                "min": round(val_min, 4) if pd.notnull(val_min) else np.nan,
                "max": round(val_max, 4) if pd.notnull(val_max) else np.nan,
                "mean": round(val_mean, 4) if pd.notnull(val_mean) else np.nan,
                "std": round(val_std, 4) if pd.notnull(val_std) else np.nan,
                "out_of_bounds_count": out_of_bounds,
                "is_constant": is_constant,
            })

        # 2. Energy Data Inspection
        has_valid_target = False
        target_unit_found = "unknown"

        if energy_df is not None and not energy_df.empty:
            e_df = energy_df.copy()
            if self.target_column in e_df.columns:
                target_col = self.target_column
                t_vals = e_df[target_col].dropna()
                has_valid_target = True

                # Determine unit
                if "unit" in e_df.columns:
                    target_unit_found = str(e_df["unit"].iloc[0])
                elif "kwh" in target_col.lower():
                    target_unit_found = "kWh"
                elif "mw" in target_col.lower():
                    target_unit_found = "MW"
                elif "kw" in target_col.lower():
                    target_unit_found = "kW"

                report_rows.append({
                    "dataset": "energy",
                    "column": target_col,
                    "dtype": str(e_df[target_col].dtype),
                    "total_records": len(e_df),
                    "placeholder_count": int(e_df[target_col].isin(self.KNOWN_PLACEHOLDERS).sum()),
                    "null_count": int(e_df[target_col].isnull().sum()),
                    "missing_pct": round((e_df[target_col].isnull().sum() / len(e_df)) * 100.0, 2),
                    "min": round(t_vals.min(), 4) if not t_vals.empty else np.nan,
                    "max": round(t_vals.max(), 4) if not t_vals.empty else np.nan,
                    "mean": round(t_vals.mean(), 4) if not t_vals.empty else np.nan,
                    "std": round(t_vals.std(), 4) if not t_vals.empty else np.nan,
                    "out_of_bounds_count": int((t_vals < 0).sum()), # Negative energy output is physically invalid
                    "is_constant": (t_vals.std() == 0) if not t_vals.empty else True,
                })
        elif self.target_column in w_df.columns:
            has_valid_target = True

        # Target Validation Assertion
        if not has_valid_target:
            msg = (
                f"Target column '{self.target_column}' NOT found in energy dataset! "
                "CRITICAL: Solar irradiance (ALLSKY_SFC_SW_DWN) CANNOT be used as energy output."
            )
            if not self.allow_synthetic_demo:
                logger.error(msg)
                raise ValueError(f"{msg} Set allow_synthetic_demo=True only for software testing demonstration.")
            else:
                logger.warning(f"{msg} Synthetic demonstration mode is enabled.")

        report_df = pd.DataFrame(report_rows)

        if output_report_path:
            save_dataframe(report_df, output_report_path)

        summary = {
            "total_columns_inspected": len(report_rows),
            "weather_rows": len(weather_df),
            "energy_rows": len(energy_df) if energy_df is not None else 0,
            "has_valid_target": has_valid_target,
            "target_unit": target_unit_found,
            "placeholders_detected": int(report_df["placeholder_count"].sum()),
            "constant_columns": report_df[report_df["is_constant"]]["column"].tolist(),
        }

        logger.info(f"Data validation complete: {summary['placeholders_detected']} placeholders found. Valid target: {has_valid_target}")
        return True, report_df, summary


def validate_dataset(
    weather_df: pd.DataFrame,
    energy_df: Optional[pd.DataFrame] = None,
    target_column: str = "energy_output_kwh",
    allow_synthetic_demo: bool = False,
) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
    """Convenience function to run validation."""
    validator = DataValidator(target_column=target_column, allow_synthetic_demo=allow_synthetic_demo)
    return validator.validate(weather_df, energy_df)
