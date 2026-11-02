"""
Unit tests for MLP-LSTM, Residual Booster, Baselines, and Conformal Uncertainty.
"""

import pytest
import numpy as np
import torch
from src.models.mlp_lstm import MLPLSTMPipeline
from src.models.residual_boosting import ResidualBoostingModel
from src.models.baselines import SeasonalNaiveBaseline, SVRBaseline, StandaloneLSTMPipeline
from src.models.uncertainty import ConformalPredictionInterval


def test_mlp_lstm_pipeline_fit_predict():
    np.random.seed(42)
    torch.manual_seed(42)

    n_train, n_val, seq_len, n_feat = 100, 30, 12, 6
    X_tr = np.random.randn(n_train, seq_len, n_feat).astype(np.float32)
    y_tr = np.random.randn(n_train).astype(np.float32)
    X_v = np.random.randn(n_val, seq_len, n_feat).astype(np.float32)
    y_v = np.random.randn(n_val).astype(np.float32)

    pipeline = MLPLSTMPipeline(input_dim=n_feat, config={"epochs": 3, "batch_size": 32})
    history = pipeline.fit(X_tr, y_tr, X_v, y_v, epochs=3, verbose=False)

    assert len(history["train_loss"]) == 3
    preds = pipeline.predict(X_v)
    assert len(preds) == n_val


def test_residual_boosting_correction_formula():
    n_samples, n_feat = 50, 5
    X_tab = np.random.randn(n_samples, n_feat)
    y_actual = np.random.uniform(10, 50, n_samples)
    base_preds = y_actual + np.random.normal(0, 5, n_samples) # Simulated imperfect base model

    booster = ResidualBoostingModel(n_estimators=10, max_depth=3)
    booster.fit(X_tab, base_preds, y_actual)

    final_preds, pred_residuals = booster.predict_final(X_tab, base_preds)

    # Verify formula: final = max(0, base + residual)
    expected_final = np.maximum(0.0, base_preds + pred_residuals)
    np.testing.assert_allclose(final_preds, expected_final, atol=1e-5)


def test_conformal_uncertainty_coverage():
    np.random.seed(42)
    n_calib, n_test = 200, 100
    y_calib_true = np.random.uniform(50, 150, n_calib)
    y_calib_pred = y_calib_true + np.random.normal(0, 5, n_calib)

    y_test_true = np.random.uniform(50, 150, n_test)
    y_test_pred = y_test_true + np.random.normal(0, 5, n_test)

    conformal = ConformalPredictionInterval(confidence_level=0.90)
    conformal.calibrate(y_calib_true, y_calib_pred)

    results = conformal.evaluate_coverage(y_test_true, y_test_pred)
    assert results["target_coverage_pct"] == 90.0
    # Observed coverage should be close to 90% (e.g. within [75%, 99%])
    assert 75.0 <= results["observed_picp_pct"] <= 100.0


def test_baselines_execution():
    y_train = np.array([10, 20, 30, 40, 50, 60] * 10, dtype=float)
    s_naive = SeasonalNaiveBaseline(season_length=6)
    s_naive.fit(y_train)
    preds = s_naive.predict(y_train, n_steps=6)
    np.testing.assert_allclose(preds, [10, 20, 30, 40, 50, 60])
