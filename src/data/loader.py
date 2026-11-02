"""
Reusable data loaders for weather and energy time-series datasets.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("data_loader")


class DataLoader:
    """
    Robust dataset loader for tabular meteorological and energy generation datasets.
    Handles timestamp unification, chronological ordering, and integrity checks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def load_weather(self, file_path: str) -> pd.DataFrame:
        """
        Load NASA POWER or standard weather CSV dataset.
        Generates unified timestamp from YEAR, MO, DY, HR if present.

        Args:
            file_path: Path to weather CSV.

        Returns:
            Chronologically sorted DataFrame with 'timestamp' column.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Weather dataset file not found at: {file_path}")

        logger.info(f"Loading weather dataset from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded raw weather dataset with shape: {df.shape}")

        df = self._unify_timestamp(df, is_weather=True)
        df = self._validate_and_sort(df, dataset_name="weather")
        return df

    def load_energy(self, file_path: str) -> pd.DataFrame:
        """
        Load energy generation CSV dataset.

        Args:
            file_path: Path to energy CSV.

        Returns:
            Chronologically sorted DataFrame with 'timestamp' column.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Energy dataset file not found at: {file_path}")

        logger.info(f"Loading energy dataset from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded raw energy dataset with shape: {df.shape}")

        df = self._unify_timestamp(df, is_weather=False)
        df = self._validate_and_sort(df, dataset_name="energy")
        return df

    def _unify_timestamp(self, df: pd.DataFrame, is_weather: bool = True) -> pd.DataFrame:
        """
        Extract or construct a unified ISO datetime timestamp.
        """
        df = df.copy()
        date_cols = [c.upper() for c in df.columns]

        # Case 1: NASA POWER format (YEAR, MO, DY, HR)
        if all(k in date_cols for k in ["YEAR", "MO", "DY", "HR"]):
            # Match actual column case
            year_col = [c for c in df.columns if c.upper() == "YEAR"][0]
            mo_col = [c for c in df.columns if c.upper() == "MO"][0]
            dy_col = [c for c in df.columns if c.upper() == "DY"][0]
            hr_col = [c for c in df.columns if c.upper() == "HR"][0]

            logger.info("Constructing unified timestamp from (YEAR, MO, DY, HR)...")
            df["timestamp"] = pd.to_datetime({
                "year": df[year_col].astype(int),
                "month": df[mo_col].astype(int),
                "day": df[dy_col].astype(int),
                "hour": df[hr_col].astype(int),
            })
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "datetime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["datetime"])
        elif "Date" in df.columns or "date" in df.columns:
            d_col = "Date" if "Date" in df.columns else "date"
            if "Time" in df.columns or "time" in df.columns:
                t_col = "Time" if "Time" in df.columns else "time"
                df["timestamp"] = pd.to_datetime(df[d_col].astype(str) + " " + df[t_col].astype(str))
            else:
                df["timestamp"] = pd.to_datetime(df[d_col])
        else:
            raise ValueError(
                "Unable to construct timestamp. Expected columns ('YEAR', 'MO', 'DY', 'HR') "
                "or ('timestamp' / 'datetime' / 'date')."
            )

        return df

    def _validate_and_sort(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Ensure chronological order and detect duplicates or discontinuities.
        """
        # Detect duplicate timestamps
        n_duplicates = df.duplicated(subset=["timestamp"]).sum()
        if n_duplicates > 0:
            logger.warning(f"Detected {n_duplicates} duplicate timestamps in {dataset_name} dataset. Aggregating duplicates by mean.")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = df.groupby("timestamp")[num_cols].mean().reset_index()

        # Sort chronologically
        df = df.sort_values(by="timestamp").reset_index(drop=True)

        # Inspect continuity
        time_diffs = df["timestamp"].diff().dropna()
        if not time_diffs.empty:
            median_step = time_diffs.median()
            gaps = (time_diffs > median_step).sum()
            logger.info(
                f"{dataset_name.capitalize()} dataset: {len(df)} rows spanning {df['timestamp'].min()} "
                f"to {df['timestamp'].max()} with median step {median_step}. Gaps detected: {gaps}"
            )

        return df


def load_weather_data(file_path: str) -> pd.DataFrame:
    """Convenience helper to load weather data."""
    loader = DataLoader()
    return loader.load_weather(file_path)


def load_energy_data(file_path: str) -> pd.DataFrame:
    """Convenience helper to load energy data."""
    loader = DataLoader()
    return loader.load_energy(file_path)
