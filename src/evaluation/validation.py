"""
Chronological dataset splitting and LSTM temporal sequence generation.
Ensures zero boundary leakage across train, validation, and test partitions.
"""

from typing import Dict, Any, Tuple, Optional, List, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.utils.logger import get_logger
from src.utils.io import save_dataframe

logger = get_logger("validation")


class ChronologicalSplitter:
    """
    Splits time-series data chronologically without random shuffling.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self,
        df: pd.DataFrame,
        output_summary_path: Optional[str] = "reports/metrics/train_validation_test_split_summary.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partition DataFrame into Train, Validation, and Test subsets chronologically.

        Returns:
            Tuple of (train_df, val_df, test_df, summary_df).
        """
        n_total = len(df)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_df = df.iloc[:n_train].copy().reset_index(drop=True)
        val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
        test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

        summary_rows = [
            {
                "split": "Train",
                "records": len(train_df),
                "ratio_pct": round(len(train_df) / n_total * 100.0, 2),
                "start_timestamp": str(train_df["timestamp"].min()) if "timestamp" in train_df.columns else "N/A",
                "end_timestamp": str(train_df["timestamp"].max()) if "timestamp" in train_df.columns else "N/A",
            },
            {
                "split": "Validation",
                "records": len(val_df),
                "ratio_pct": round(len(val_df) / n_total * 100.0, 2),
                "start_timestamp": str(val_df["timestamp"].min()) if "timestamp" in val_df.columns else "N/A",
                "end_timestamp": str(val_df["timestamp"].max()) if "timestamp" in val_df.columns else "N/A",
            },
            {
                "split": "Test",
                "records": len(test_df),
                "ratio_pct": round(len(test_df) / n_total * 100.0, 2),
                "start_timestamp": str(test_df["timestamp"].min()) if "timestamp" in test_df.columns else "N/A",
                "end_timestamp": str(test_df["timestamp"].max()) if "timestamp" in test_df.columns else "N/A",
            },
        ]
        summary_df = pd.DataFrame(summary_rows)

        if output_summary_path:
            save_dataframe(summary_df, output_summary_path)

        logger.info(
            f"Chronological split complete: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})"
        )
        return train_df, val_df, test_df, summary_df


class SequenceGenerator:
    """
    Transforms tabular 2D arrays into 3D sequential sliding windows for LSTM networks:
    Inputs: shape (N, sequence_length, n_features)
    Target: shape (N,)
    Flat Tabular Inputs at timestep t: shape (N, n_features)
    """

    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length

    def create_sequences(
        self,
        feature_df: pd.DataFrame,
        feature_columns: List[str],
        target_series: Union[pd.Series, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sliding lookback sequences.

        Args:
            feature_df: DataFrame with scaled feature columns.
            feature_columns: Names of feature columns.
            target_series: Aligned target values.

        Returns:
            Tuple of:
            - X_seq: 3D array of shape (N, sequence_length, n_features)
            - y_seq: 1D array of shape (N,)
            - X_flat: 2D array of shape (N, n_features) corresponding to feature state at timestep t (end of window).
        """
        features_mat = feature_df[feature_columns].values
        target_vec = np.asarray(target_series).flatten()

        n_samples = len(feature_df)
        if n_samples <= self.sequence_length:
            raise ValueError(
                f"Sample size ({n_samples}) must be strictly greater than sequence length ({self.sequence_length})."
            )

        X_seq_list = []
        y_seq_list = []
        X_flat_list = []

        for i in range(self.sequence_length, n_samples):
            # Sequence from i - sequence_length to i (length = sequence_length)
            seq = features_mat[i - self.sequence_length : i, :]
            target = target_vec[i]
            flat_state = features_mat[i - 1, :] # Feature vector at time t (last step of window)

            if not np.isnan(target) and not np.isnan(seq).any() and not np.isnan(flat_state).any():
                X_seq_list.append(seq)
                y_seq_list.append(target)
                X_flat_list.append(flat_state)

        X_seq = np.array(X_seq_list, dtype=np.float32)
        y_seq = np.array(y_seq_list, dtype=np.float32)
        X_flat = np.array(X_flat_list, dtype=np.float32)

        return X_seq, y_seq, X_flat

    def create_torch_dataloader(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        Wrap sequences into a PyTorch DataLoader.
        """
        tensor_x = torch.tensor(X_seq, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
