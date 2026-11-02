"""
Hybrid Model Training and Evaluation Script.
Trains: MLP-LSTM Hybrid -> Gradient Boosting Residual Correction -> Conformal Uncertainty.
Usage: python scripts/run_training.py --horizon 24
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.io import load_yaml, save_dataframe, save_json
from src.data.loader import DataLoader
from src.data.merger import DataMerger
from src.data.preprocessing import PreprocessingPipeline
from src.features.feature_engineering import FeatureEngineer, FeatureLeakageChecker
from src.evaluation.validation import ChronologicalSplitter, SequenceGenerator
from src.evaluation.metrics import evaluate_all_metrics
from src.models.mlp_lstm import MLPLSTMPipeline
from src.models.residual_boosting import ResidualBoostingModel
from src.models.uncertainty import ConformalPredictionInterval
from src.visualization.plots import Visualizer

logger = setup_logger("run_training")


def train_and_evaluate(
    config_path: str = "config/config.yaml",
    model_config_path: str = "config/model_config.yaml",
    horizon: int = 24,
    epochs: Optional[int] = None,
    output_dir: Optional[str] = None,
):
    config = load_yaml(config_path)
    m_config = load_yaml(model_config_path)
    set_seed(config.get("random_seed", 42))

    target_col = config["data"]["target_column"]
    unit = config["data"].get("target_unit", "kWh")
    res_dir = output_dir or f"results/horizon_{horizon}"
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Load Data
    loader = DataLoader(config)
    weather_df = loader.load_weather(config["data"]["weather_path"])
    energy_df = loader.load_energy(config["data"]["energy_path"])
    merger = DataMerger()
    merged_df, _ = merger.merge(weather_df, energy_df)

    # 2. Chronological Split (70% Train, 15% Val, 15% Test)
    splitter = ChronologicalSplitter(
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
    )
    train_df, val_df, test_df, split_summary = splitter.split(merged_df)

    # 3. Fit Preprocessing Pipeline ONLY on Training Split
    pipeline = PreprocessingPipeline(
        scaler_type=config["preprocessing"]["scaler_type"],
        target_column=target_col,
    )
    train_df_prep = pipeline.fit_transform(train_df)
    val_df_prep = pipeline.transform(val_df)
    test_df_prep = pipeline.transform(test_df)
    pipeline.save(f"models/preprocessing_pipeline_h{horizon}.pkl")

    # Combine preprocessed partitions for contiguous lag/rolling calculation
    full_prep_df = pd.concat([train_df_prep, val_df_prep, test_df_prep], ignore_index=True)

    # 4. Feature Engineering
    feat_eng = FeatureEngineer(
        target_column=target_col,
        horizon_hours=horizon,
        energy_lags=config["features"]["energy_lags"],
        weather_lags=config["features"]["weather_lags"],
        rolling_windows=config["features"]["rolling_windows"],
        rolling_stats=config["features"]["rolling_stats"],
    )
    df_engineered, feature_cols = feat_eng.engineer_features(full_prep_df)

    # 5. Automated Feature Leakage Check
    FeatureLeakageChecker.verify_no_target_in_features(feature_cols, target_col, horizon)
    FeatureLeakageChecker.verify_rolling_stats_historical(df_engineered, target_col, window=6)
    FeatureLeakageChecker.verify_target_shift_alignment(df_engineered, target_col, horizon)

    # Re-slice partitions based on original chronological indices
    n_tr, n_v = len(train_df), len(val_df)
    train_eng = df_engineered.iloc[:n_tr].copy().reset_index(drop=True)
    val_eng = df_engineered.iloc[n_tr : n_tr + n_v].copy().reset_index(drop=True)
    test_eng = df_engineered.iloc[n_tr + n_v :].copy().reset_index(drop=True)

    # 6. Generate Lookback Sequences for LSTM and 2D arrays for Tabular Booster
    seq_len = m_config["mlp_lstm"].get("sequence_length", 24)
    seq_gen = SequenceGenerator(sequence_length=seq_len)
    lead_target_col = f"{target_col}_lead_{horizon}h"

    X_train_seq, y_train_seq, X_train_flat = seq_gen.create_sequences(
        train_eng, feature_cols, train_eng[lead_target_col]
    )
    X_val_seq, y_val_seq, X_val_flat = seq_gen.create_sequences(
        val_eng, feature_cols, val_eng[lead_target_col]
    )
    X_test_seq, y_test_seq, X_test_flat = seq_gen.create_sequences(
        test_eng, feature_cols, test_eng[lead_target_col]
    )

    logger.info(
        f"Generated Sequences: Train {X_train_seq.shape}, Val {X_val_seq.shape}, Test {X_test_seq.shape}"
    )

    # 7. Train Base MLP-LSTM Hybrid Network
    mlp_lstm_epochs = epochs or m_config["mlp_lstm"].get("epochs", 100)
    mlp_lstm_trainer = MLPLSTMPipeline(
        input_dim=len(feature_cols),
        config=m_config["mlp_lstm"],
        random_seed=config.get("random_seed", 42),
    )
    mlp_lstm_trainer.fit(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=mlp_lstm_epochs,
    )
    mlp_lstm_trainer.save(f"models/mlp_lstm_model_h{horizon}.pt")

    # Generate Base Predictions (scaled)
    base_train_preds_s = mlp_lstm_trainer.predict(X_train_seq)
    base_val_preds_s = mlp_lstm_trainer.predict(X_val_seq)
    base_test_preds_s = mlp_lstm_trainer.predict(X_test_seq)

    # 8. Train Gradient Boosting Residual Correction Model
    booster = ResidualBoostingModel(
        model_type=m_config["residual_boosting"].get("model_type", "xgboost"),
        n_estimators=m_config["residual_boosting"].get("n_estimators", 150),
        max_depth=m_config["residual_boosting"].get("max_depth", 5),
        learning_rate=m_config["residual_boosting"].get("learning_rate", 0.05),
        random_state=config.get("random_seed", 42),
    )
    booster.fit(
        X_train_flat, base_train_preds_s, y_train_seq,
        X_val_flat, base_val_preds_s, y_val_seq,
        feature_names=feature_cols,
    )
    booster.save(f"models/residual_boosting_model_h{horizon}.pkl")

    # Compute Final Predictions (scaled)
    final_test_preds_s, pred_residuals_s = booster.predict_final(X_test_flat, base_test_preds_s)
    final_val_preds_s, _ = booster.predict_final(X_val_flat, base_val_preds_s)

    # 9. Inverse Scaling to Physical kWh Units
    y_test_actual = pipeline.inverse_transform_target(y_test_seq)
    base_test_preds = pipeline.inverse_transform_target(base_test_preds_s)
    final_test_preds = pipeline.inverse_transform_target(final_test_preds_s)

    y_val_actual = pipeline.inverse_transform_target(y_val_seq)
    final_val_preds = pipeline.inverse_transform_target(final_val_preds_s)

    # 10. Conformal Prediction Intervals & PICP Uncertainty
    conformal = ConformalPredictionInterval(confidence_level=0.90)
    conformal.calibrate(y_val_actual, final_val_preds)
    conformal.save(f"models/conformal_interval_h{horizon}.pkl")

    lower_bounds, upper_bounds = conformal.predict_intervals(final_test_preds)
    uncertainty_results = conformal.evaluate_coverage(y_test_actual, final_test_preds)
    save_json(uncertainty_results, f"{res_dir}/uncertainty_metrics.json")
    save_dataframe(
        pd.DataFrame([uncertainty_results]), "reports/metrics/prediction_interval_results.csv"
    )

    # 11. Compute Overall Metrics
    base_metrics = evaluate_all_metrics(y_test_actual, base_test_preds)
    final_metrics = evaluate_all_metrics(y_test_actual, final_test_preds, lower_bounds, upper_bounds)

    logger.info(f"--- TEST METRICS (Horizon = {horizon}h) ---")
    logger.info(f"Base MLP-LSTM:  RMSE={base_metrics['RMSE']:.4f}, MAE={base_metrics['MAE']:.4f}, MAPE={base_metrics['MAPE']}%, R2={base_metrics['R2']:.4f}")
    logger.info(f"Final Proposed: RMSE={final_metrics['RMSE']:.4f}, MAE={final_metrics['MAE']:.4f}, MAPE={final_metrics['MAPE']}%, R2={final_metrics['R2']:.4f}, PICP={final_metrics.get('PICP')}%")

    # Save detailed prediction CSV
    test_timestamps = test_eng["timestamp"].iloc[seq_len:].reset_index(drop=True)
    pred_df = pd.DataFrame({
        "timestamp": test_timestamps,
        "actual_energy": y_test_actual,
        "base_mlp_lstm_prediction": base_test_preds,
        "final_hybrid_prediction": final_test_preds,
        "prediction_interval_lower_90": lower_bounds,
        "prediction_interval_upper_90": upper_bounds,
    })
    save_dataframe(pred_df, f"{res_dir}/test_predictions.csv")

    # Save model results summary
    summary_df = pd.DataFrame([
        {"model": "MLP-LSTM (Base)", "horizon_hours": horizon, **base_metrics},
        {"model": "MLP-LSTM + Residual Booster (Proposed)", "horizon_hours": horizon, **final_metrics},
    ])
    save_dataframe(summary_df, f"{res_dir}/metrics_summary.csv")
    save_dataframe(summary_df, "reports/metrics/hybrid_model_results.csv")

    # 12. Visualizations
    viz = Visualizer()
    viz.plot_actual_vs_predicted(y_test_actual, final_test_preds, unit=unit)
    viz.plot_forecast_comparison(y_test_actual, base_test_preds, final_test_preds, timestamps=test_timestamps, unit=unit)
    viz.plot_residual_distribution(y_test_actual, final_test_preds)
    viz.plot_residuals_over_time(y_test_actual, final_test_preds, timestamps=test_timestamps)
    viz.plot_prediction_intervals(y_test_actual, final_test_preds, lower_bounds, upper_bounds, timestamps=test_timestamps, unit=unit)
    viz.plot_correlation_heatmap(df_engineered, feature_cols)

    return {
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "uncertainty": uncertainty_results,
        "booster": booster,
        "X_test_flat": X_test_flat,
        "base_test_preds_s": base_test_preds_s,
        "feature_cols": feature_cols,
        "y_test_actual": y_test_actual,
        "base_test_preds": base_test_preds,
        "final_test_preds": final_test_preds,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate hybrid forecasting model.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_and_evaluate(
        config_path=args.config,
        model_config_path=args.model_config,
        horizon=args.horizon,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
