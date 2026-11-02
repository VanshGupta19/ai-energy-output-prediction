"""
Unit tests for feature engineering, lags, rolling statistics, and feature leakage prevention.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.lag_features import LagFeatureGenerator
from src.features.temporal_features import TemporalFeatureGenerator, DomainFeatureGenerator
from src.features.feature_engineering import FeatureEngineer, FeatureLeakageChecker


@pytest.fixture
def dummy_series_df():
    dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
    # Monotonically increasing energy values for easy mathematical verification
    return pd.DataFrame({
        "timestamp": dates,
        "energy_output_kwh": np.arange(200, dtype=float),
        "ALLSKY_SFC_SW_DWN": np.random.uniform(0, 1, 200),
        "T2M": np.random.uniform(10, 30, 200),
        "PS": 101.3 + np.random.randn(200) * 0.1,
        "WS10M": 3.5 + np.random.randn(200) * 0.5,
        "WS50M": 5.0 + np.random.randn(200) * 0.5,
        "RHOA": 1.20 + np.random.randn(200) * 0.01,
        "CLOUD_AMT": np.random.uniform(0, 100, 200),
    })


def test_lag_feature_shifts(dummy_series_df):
    lag_gen = LagFeatureGenerator(energy_lags=[1, 24], target_column="energy_output_kwh")
    df_lags = lag_gen.create_lags(dummy_series_df)

    # For lag 1: row 10 must have value from row 9
    assert df_lags.loc[10, "energy_output_kwh_lag_1"] == dummy_series_df.loc[9, "energy_output_kwh"]
    # For lag 24: row 50 must have value from row 26
    assert df_lags.loc[50, "energy_output_kwh_lag_24"] == dummy_series_df.loc[26, "energy_output_kwh"]


def test_rolling_statistics_historical_only(dummy_series_df):
    # Verify rolling stats never include current timestep t
    lag_gen = LagFeatureGenerator(rolling_windows=[6], rolling_stats=["mean"], target_column="energy_output_kwh")
    df_roll = lag_gen.create_rolling_features(dummy_series_df)

    # For row index 20, 6-hour mean must equal mean(values at index 14..19), NOT including index 20
    expected_mean = dummy_series_df.loc[14:19, "energy_output_kwh"].mean()
    computed_mean = df_roll.loc[20, "energy_output_kwh_rolling_6h_mean"]
    assert np.isclose(computed_mean, expected_mean, atol=1e-5)


def test_cyclical_encodings(dummy_series_df):
    temp_gen = TemporalFeatureGenerator()
    df_temp = temp_gen.create_features(dummy_series_df)

    assert "hour_sin" in df_temp.columns and "hour_cos" in df_temp.columns
    # sin^2 + cos^2 = 1
    trig_identity = (df_temp["hour_sin"] ** 2) + (df_temp["hour_cos"] ** 2)
    np.testing.assert_allclose(trig_identity, 1.0, atol=1e-5)


def test_domain_physics_features(dummy_series_df):
    dom_gen = DomainFeatureGenerator(enable_solar=True, enable_wind=True)
    df_dom = dom_gen.create_features(dummy_series_df)

    assert "wind_power_density_50m" in df_dom.columns
    # P = 0.5 * rho * v^3
    expected_wpd = 0.5 * dummy_series_df["RHOA"] * (dummy_series_df["WS50M"] ** 3)
    np.testing.assert_allclose(df_dom["wind_power_density_50m"].values, expected_wpd.values, atol=1e-4)


def test_target_horizon_shift_alignment(dummy_series_df):
    horizon = 24
    feat_eng = FeatureEngineer(target_column="energy_output_kwh", horizon_hours=horizon)
    df_eng, feat_cols = feat_eng.engineer_features(dummy_series_df, output_feature_list_path=None)

    lead_col = f"energy_output_kwh_lead_{horizon}h"
    assert lead_col in df_eng.columns
    # Value at index 10 in lead_col must equal energy_output_kwh at index 10 + 24 = 34
    assert df_eng.loc[10, lead_col] == dummy_series_df.loc[10 + horizon, "energy_output_kwh"]

    # Automated Leakage Checker must pass
    assert FeatureLeakageChecker.verify_no_target_in_features(feat_cols, "energy_output_kwh", horizon)
    assert FeatureLeakageChecker.verify_rolling_stats_historical(df_eng, "energy_output_kwh", 6)
    assert FeatureLeakageChecker.verify_target_shift_alignment(df_eng, "energy_output_kwh", horizon)
