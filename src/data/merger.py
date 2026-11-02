"""
Temporal alignment and dataset merger for weather and energy time-series.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("data_merger")


class DataMerger:
    """
    Handles temporal alignment, frequency harmonization, and strict timestamp-based merging
    between meteorological records and energy generation outputs.
    """

    def __init__(self, frequency: str = "h"):
        self.frequency = frequency

    def merge(
        self,
        weather_df: pd.DataFrame,
        energy_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Merge weather and energy datasets on exact timestamps.

        Args:
            weather_df: Processed weather DataFrame with 'timestamp'.
            energy_df: Processed energy DataFrame with 'timestamp'.

        Returns:
            Tuple of (merged_df, merge_summary_dict).
        """
        if "timestamp" not in weather_df.columns:
            raise KeyError("weather_df must contain 'timestamp' column.")
        if "timestamp" not in energy_df.columns:
            raise KeyError("energy_df must contain 'timestamp' column.")

        # Ensure datetime type
        w_df = weather_df.copy()
        e_df = energy_df.copy()

        w_df["timestamp"] = pd.to_datetime(w_df["timestamp"])
        e_df["timestamp"] = pd.to_datetime(e_df["timestamp"])

        # Check and normalize timezones if present
        if w_df["timestamp"].dt.tz is not None:
            w_df["timestamp"] = w_df["timestamp"].dt.tz_convert(None)
        if e_df["timestamp"].dt.tz is not None:
            e_df["timestamp"] = e_df["timestamp"].dt.tz_convert(None)

        n_weather_before = len(w_df)
        n_energy_before = len(e_df)

        logger.info(f"Initiating dataset merge: Weather ({n_weather_before} rows), Energy ({n_energy_before} rows)")

        # Inner merge on timestamp
        merged_df = pd.merge(w_df, e_df, on="timestamp", how="inner", suffixes=("", "_energy_dup"))

        # Drop any duplicated columns from energy
        dup_cols = [c for c in merged_df.columns if c.endswith("_energy_dup")]
        if dup_cols:
            merged_df = merged_df.drop(columns=dup_cols)

        # Sort chronologically
        merged_df = merged_df.sort_values(by="timestamp").reset_index(drop=True)

        n_matched = len(merged_df)
        n_unmatched_weather = n_weather_before - n_matched
        n_unmatched_energy = n_energy_before - n_matched

        if n_matched == 0:
            raise ValueError(
                f"No matching timestamps found between weather [{w_df['timestamp'].min()} to {w_df['timestamp'].max()}] "
                f"and energy [{e_df['timestamp'].min()} to {e_df['timestamp'].max()}]."
            )

        summary = {
            "weather_rows_before": n_weather_before,
            "energy_rows_before": n_energy_before,
            "matched_rows": n_matched,
            "unmatched_weather_rows": n_unmatched_weather,
            "unmatched_energy_rows": n_unmatched_energy,
            "match_rate_pct": round((n_matched / max(n_weather_before, n_energy_before)) * 100.0, 2),
            "merged_start_date": str(merged_df["timestamp"].min()),
            "merged_end_date": str(merged_df["timestamp"].max()),
        }

        logger.info(
            f"Merge completed: {n_matched} matched rows ({summary['match_rate_pct']}%), "
            f"range: {summary['merged_start_date']} to {summary['merged_end_date']}"
        )

        return merged_df, summary


def merge_weather_energy(weather_df: pd.DataFrame, energy_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience helper to merge datasets."""
    merger = DataMerger()
    return merger.merge(weather_df, energy_df)
