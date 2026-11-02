"""
Unit tests for LSTM sequence generation and boundary integrity.
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation.validation import SequenceGenerator, ChronologicalSplitter


def test_sequence_dimensions_and_alignment():
    n_samples = 100
    n_features = 8
    seq_len = 24

    features = np.random.randn(n_samples, n_features)
    feat_cols = [f"f_{i}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=feat_cols)
    target = np.arange(n_samples, dtype=float)

    seq_gen = SequenceGenerator(sequence_length=seq_len)
    X_seq, y_seq, X_flat = seq_gen.create_sequences(df, feat_cols, target)

    expected_n = n_samples - seq_len
    assert X_seq.shape == (expected_n, seq_len, n_features)
    assert len(y_seq) == expected_n
    assert X_flat.shape == (expected_n, n_features)

    # First target in y_seq must correspond to target at index seq_len
    assert y_seq[0] == target[seq_len]
    # First sequence in X_seq must match features from index 0 to seq_len - 1
    np.testing.assert_allclose(X_seq[0], features[0:seq_len], atol=1e-5)
    # First flat vector in X_flat must match feature at timestep seq_len - 1
    np.testing.assert_allclose(X_flat[0], features[seq_len - 1], atol=1e-5)


def test_chronological_splitter_no_overlap():
    n_total = 1000
    dates = pd.date_range("2023-01-01", periods=n_total, freq="h")
    df = pd.DataFrame({"timestamp": dates, "val": np.arange(n_total)})

    splitter = ChronologicalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df, _ = splitter.split(df, output_summary_path=None)

    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150

    # Ensure strictly zero temporal overlap
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()
