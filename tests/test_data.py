"""
Unit tests for data loading, NASA format parsing, validation, merging, and preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.merger import DataMerger
from src.data.preprocessing import PreprocessingPipeline


@pytest.fixture
def sample_nasa_weather_df():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    return pd.DataFrame({
        "YEAR": dates.year,
        "MO": dates.month,
        "DY": dates.day,
        "HR": dates.hour,
        "ALLSKY_SFC_SW_DWN": [0.5 if 6 <= h <= 18 else 0.0 for h in dates.hour],
        "T2M": 22.0 + np.random.randn(100),
        "RH2M": 55.0 + np.random.randn(100),
        "PS": 101.3 + np.random.randn(100) * 0.1,
        "WS10M": 4.0 + np.random.randn(100) * 0.2,
        "WS50M": 5.5 + np.random.randn(100) * 0.3,
        "RHOA": 1.18 + np.random.randn(100) * 0.01,
        "CLOUD_AMT": 20.0 + np.random.randn(100) * 2.0,
    })


@pytest.fixture
def sample_energy_df():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    gen = [200.0 if 6 <= h <= 18 else 0.0 for h in dates.hour]
    return pd.DataFrame({
        "timestamp": dates,
        "energy_output_kwh": gen,
        "unit": "kWh",
    })


def test_nasa_timestamp_unification(sample_nasa_weather_df, tmp_path):
    csv_path = tmp_path / "nasa_test.csv"
    sample_nasa_weather_df.to_csv(csv_path, index=False)

    loader = DataLoader()
    loaded_df = loader.load_weather(str(csv_path))

    assert "timestamp" in loaded_df.columns
    assert loaded_df["timestamp"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00")
    assert loaded_df["timestamp"].iloc[-1] == pd.Timestamp("2023-01-05 03:00:00")
    # Strictly chronologically sorted
    assert loaded_df["timestamp"].is_monotonic_increasing


def test_missing_value_placeholder_detection(sample_nasa_weather_df):
    # Insert NASA -999 placeholder
    sample_nasa_weather_df.loc[5, "T2M"] = -999.0
    sample_nasa_weather_df.loc[10, "ALLSKY_SFC_SW_DWN"] = -999.0

    validator = DataValidator(target_column="energy_output_kwh", allow_synthetic_demo=True)
    is_valid, q_df, summary = validator.validate(sample_nasa_weather_df, output_report_path=None)

    assert is_valid
    assert summary["placeholders_detected"] >= 2


def test_missing_target_validation_rejection(sample_nasa_weather_df):
    # If no energy data and allow_synthetic_demo=False, validator must raise ValueError
    validator = DataValidator(target_column="energy_output_kwh", allow_synthetic_demo=False)
    with pytest.raises(ValueError, match="CRITICAL"):
        validator.validate(sample_nasa_weather_df, energy_df=None, output_report_path=None)


def test_temporal_merger(sample_nasa_weather_df, sample_energy_df, tmp_path):
    loader = DataLoader()
    w_path = tmp_path / "w.csv"
    e_path = tmp_path / "e.csv"
    sample_nasa_weather_df.to_csv(w_path, index=False)
    sample_energy_df.to_csv(e_path, index=False)

    w_df = loader.load_weather(str(w_path))
    e_df = loader.load_energy(str(e_path))

    merger = DataMerger()
    merged_df, summary = merger.merge(w_df, e_df)

    assert len(merged_df) == 100
    assert summary["matched_rows"] == 100
    assert "energy_output_kwh" in merged_df.columns
    assert "ALLSKY_SFC_SW_DWN" in merged_df.columns


def test_train_only_scaling_integrity(sample_energy_df):
    # Ensure scaler fitted only on train does not leak test statistics
    n_train = 70
    train_df = sample_energy_df.iloc[:n_train].copy()
    test_df = sample_energy_df.iloc[n_train:].copy()

    pipeline = PreprocessingPipeline(target_column="energy_output_kwh")
    pipeline.fit(train_df)

    # Transform
    train_scaled = pipeline.transform(train_df)
    test_scaled = pipeline.transform(test_df)

    # Scaled values in train should be in [0, 1]
    assert train_scaled["energy_output_kwh"].min() >= 0.0
    assert train_scaled["energy_output_kwh"].max() <= 1.0

    # Inverse transform
    inv_train = pipeline.inverse_transform_target(train_scaled["energy_output_kwh"].values)
    np.testing.assert_allclose(inv_train, train_df["energy_output_kwh"].values, atol=1e-4)
