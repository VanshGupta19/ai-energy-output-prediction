"""
End-to-End Smoke Test Script.
Verifies the complete pipeline workflow on a compact data slice in under 30 seconds.
Usage: python scripts/run_smoke_test.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.merger import DataMerger
from src.data.preprocessing import PreprocessingPipeline
from src.features.feature_engineering import FeatureEngineer, FeatureLeakageChecker
from src.evaluation.validation import ChronologicalSplitter, SequenceGenerator
from src.evaluation.metrics import evaluate_all_metrics
from src.models.mlp_lstm import MLPLSTMPipeline
from src.models.residual_boosting import ResidualBoostingModel
from src.models.uncertainty import ConformalPredictionInterval
from src.explainability.shap_analysis import SHAPExplainer

logger = setup_logger("smoke_test")


def run_smoke_test():
    logger.info(">>> INITIATING FAST END-TO-END SMOKE TEST <<<")
    set_seed(42)

    # 1. Create small synthetic test dataframe (500 hourly records)
    date_range = pd.date_range(start="2023-01-01", periods=500, freq="h")
    hours = date_range.hour.values
    dayofyear = date_range.dayofyear.values
    solar_angle = np.maximum(0.0, np.sin(np.pi * (hours - 6) / 12.0))
    irradiance = solar_angle * 0.9 + np.random.normal(0, 0.02, size=500)
    irradiance = np.clip(irradiance, 0, 1.2)
    irradiance[solar_angle <= 0] = 0.0

    weather_df = pd.DataFrame({
        "timestamp": date_range,
        "ALLSKY_SFC_SW_DWN": irradiance,
        "T2M": 20.0 + 5.0 * np.sin(2 * np.pi * hours / 24.0),
        "RH2M": 50.0 + 10.0 * np.random.randn(500),
        "PS": 101.3 + np.random.randn(500) * 0.2,
        "WS10M": 3.0 + np.random.randn(500) * 0.5,
        "WS50M": 4.5 + np.random.randn(500) * 0.6,
        "RHOA": 1.20 + np.random.randn(500) * 0.01,
        "CLOUD_AMT": 30.0 + np.random.randn(500) * 5.0,
    })
    # Add one placeholder
    weather_df.loc[10, "T2M"] = -999.0

    energy_output = irradiance * 400.0 * (1.0 + np.random.normal(0, 0.02, size=500))
    energy_df = pd.DataFrame({
        "timestamp": date_range,
        "energy_output_kwh": np.maximum(0.0, energy_output),
        "unit": "kWh",
    })

    # 2. Validation
    validator = DataValidator(target_column="energy_output_kwh")
    is_valid, q_df, summary = validator.validate(weather_df, energy_df, output_report_path=None)
    assert is_valid, "Validation failed!"
    assert summary["has_valid_target"], "Target verification failed!"

    # 3. Merger
    merger = DataMerger()
    merged_df, _ = merger.merge(weather_df, energy_df)
    assert len(merged_df) == 500, "Merge row mismatch!"

    # 4. Splitter
    splitter = ChronologicalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df, _ = splitter.split(merged_df, output_summary_path=None)

    # 5. Preprocessing (Train-only fit)
    prep = PreprocessingPipeline(target_column="energy_output_kwh")
    tr_prep = prep.fit_transform(train_df)
    v_prep = prep.transform(val_df)
    te_prep = prep.transform(test_df)
    full_prep = pd.concat([tr_prep, v_prep, te_prep], ignore_index=True)

    # 6. Feature Engineering & Target creation (H=6 hours)
    feat_eng = FeatureEngineer(target_column="energy_output_kwh", horizon_hours=6)
    df_eng, feat_cols = feat_eng.engineer_features(full_prep, output_feature_list_path=None)

    # 7. Leakage Checker
    FeatureLeakageChecker.verify_no_target_in_features(feat_cols, "energy_output_kwh", 6)
    FeatureLeakageChecker.verify_rolling_stats_historical(df_eng, "energy_output_kwh", 6)
    FeatureLeakageChecker.verify_target_shift_alignment(df_eng, "energy_output_kwh", 6)

    # 8. Sequences
    seq_gen = SequenceGenerator(sequence_length=12)
    n_tr, n_v = len(train_df), len(val_df)
    train_eng = df_eng.iloc[:n_tr].copy().reset_index(drop=True)
    val_eng = df_eng.iloc[n_tr : n_tr + n_v].copy().reset_index(drop=True)
    test_eng = df_eng.iloc[n_tr + n_v :].copy().reset_index(drop=True)

    lead_col = "energy_output_kwh_lead_6h"
    X_tr_seq, y_tr_seq, X_tr_flat = seq_gen.create_sequences(train_eng, feat_cols, train_eng[lead_col])
    X_v_seq, y_v_seq, X_v_flat = seq_gen.create_sequences(val_eng, feat_cols, val_eng[lead_col])
    X_te_seq, y_te_seq, X_te_flat = seq_gen.create_sequences(test_eng, feat_cols, test_eng[lead_col])

    # 9. Train MLP-LSTM (5 epochs)
    mlp_lstm = MLPLSTMPipeline(input_dim=len(feat_cols), config={"epochs": 5, "batch_size": 32})
    mlp_lstm.fit(X_tr_seq, y_tr_seq, X_v_seq, y_v_seq, epochs=5, verbose=False)

    base_tr_s = mlp_lstm.predict(X_tr_seq)
    base_v_s = mlp_lstm.predict(X_v_seq)
    base_te_s = mlp_lstm.predict(X_te_seq)

    # 10. Train Residual Booster (10 trees)
    booster = ResidualBoostingModel(n_estimators=10, max_depth=3)
    booster.fit(X_tr_flat, base_tr_s, y_tr_seq, X_v_flat, base_v_s, y_v_seq, feature_names=feat_cols)
    final_te_s, _ = booster.predict_final(X_te_flat, base_te_s)

    # 11. Uncertainty Calibration & PICP
    y_v_act = prep.inverse_transform_target(y_v_seq)
    v_final_s, _ = booster.predict_final(X_v_flat, base_v_s)
    v_final_act = prep.inverse_transform_target(v_final_s)

    conformal = ConformalPredictionInterval(confidence_level=0.90)
    conformal.calibrate(y_v_act, v_final_act)

    y_te_act = prep.inverse_transform_target(y_te_seq)
    final_te_act = prep.inverse_transform_target(final_te_s)
    lb, ub = conformal.predict_intervals(final_te_act)

    # 12. Evaluate Metrics
    metrics = evaluate_all_metrics(y_te_act, final_te_act, lb, ub)
    assert "RMSE" in metrics and "PICP" in metrics, "Missing evaluation metrics!"

    # 13. Fast SHAP run
    explainer = SHAPExplainer(feature_names=feat_cols, top_n=5)
    explainer.explain(booster, X_te_flat, base_te_s, max_samples=30)

    logger.info(">>> SMOKE TEST PASSED WITH ZERO ERRORS! <<<")
    logger.info(f"Test Metrics: RMSE={metrics['RMSE']}, MAE={metrics['MAE']}, R2={metrics['R2']}, PICP={metrics['PICP']}%")
    return True


if __name__ == "__main__":
    run_smoke_test()
