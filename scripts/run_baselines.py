"""
Baseline Models Benchmark Experiment Script.
Evaluates Seasonal Naive/ARIMA, SVR, Standalone LSTM, Transformer, and Proposed Hybrid model.
Usage: python scripts/run_baselines.py --horizon 24
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
from src.utils.io import load_yaml, save_dataframe
from src.data.loader import DataLoader
from src.data.merger import DataMerger
from src.data.preprocessing import PreprocessingPipeline
from src.features.feature_engineering import FeatureEngineer
from src.evaluation.validation import ChronologicalSplitter, SequenceGenerator
from src.evaluation.metrics import evaluate_all_metrics
from src.models.baselines import (
    SeasonalNaiveBaseline,
    ARIMABaseline,
    SVRBaseline,
    StandaloneLSTMPipeline,
    TransformerPipeline,
)
from src.models.mlp_lstm import MLPLSTMPipeline
from src.models.residual_boosting import ResidualBoostingModel
from src.visualization.plots import Visualizer

logger = setup_logger("run_baselines")


def run_baselines_benchmark(horizon: int = 24, epochs: int = 40):
    config = load_yaml("config/config.yaml")
    m_config = load_yaml("config/model_config.yaml")
    set_seed(config.get("random_seed", 42))

    target_col = config["data"]["target_column"]

    # 1. Load and Merge
    loader = DataLoader(config)
    weather_df = loader.load_weather(config["data"]["weather_path"])
    energy_df = loader.load_energy(config["data"]["energy_path"])
    merger = DataMerger()
    merged_df, _ = merger.merge(weather_df, energy_df)

    # 2. Chronological Split
    splitter = ChronologicalSplitter(
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
    )
    train_df, val_df, test_df, _ = splitter.split(merged_df)

    # 3. Preprocessing (Train-only fit)
    pipeline = PreprocessingPipeline(target_column=target_col)
    train_df_prep = pipeline.fit_transform(train_df)
    val_df_prep = pipeline.transform(val_df)
    test_df_prep = pipeline.transform(test_df)
    full_prep = pd.concat([train_df_prep, val_df_prep, test_df_prep], ignore_index=True)

    # 4. Feature Engineering
    feat_eng = FeatureEngineer(target_column=target_col, horizon_hours=horizon)
    df_eng, feature_cols = feat_eng.engineer_features(full_prep)

    n_tr, n_v = len(train_df), len(val_df)
    train_eng = df_eng.iloc[:n_tr].copy().reset_index(drop=True)
    val_eng = df_eng.iloc[n_tr : n_tr + n_v].copy().reset_index(drop=True)
    test_eng = df_eng.iloc[n_tr + n_v :].copy().reset_index(drop=True)

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

    y_test_actual = pipeline.inverse_transform_target(y_test_seq)
    results_list = []

    # ---------------------------------------------
    # Baseline 1: Seasonal Naive
    # ---------------------------------------------
    logger.info("Evaluating Baseline 1: Seasonal Naive (24h)...")
    s_naive = SeasonalNaiveBaseline(season_length=24)
    s_naive.fit(pipeline.inverse_transform_target(y_train_seq))
    naive_preds = s_naive.predict(y_train_seq, len(y_test_seq))
    m_naive = evaluate_all_metrics(y_test_actual, naive_preds)
    results_list.append({"model": "Seasonal Naive (24h)", "horizon_hours": horizon, **m_naive})

    # ---------------------------------------------
    # Baseline 2: SVR
    # ---------------------------------------------
    logger.info("Evaluating Baseline 2: Support Vector Regression (SVR)...")
    svr = SVRBaseline(C=10.0, epsilon=0.1)
    svr.fit(X_train_flat, y_train_seq)
    svr_preds_s = svr.predict(X_test_flat)
    svr_preds = pipeline.inverse_transform_target(svr_preds_s)
    m_svr = evaluate_all_metrics(y_test_actual, svr_preds)
    results_list.append({"model": "Support Vector Regression (SVR)", "horizon_hours": horizon, **m_svr})

    # ---------------------------------------------
    # Baseline 3: Standalone LSTM
    # ---------------------------------------------
    logger.info("Evaluating Baseline 3: Standalone LSTM...")
    lstm_base = StandaloneLSTMPipeline(input_dim=len(feature_cols), lstm_units=64, lstm_layers=2)
    lstm_base.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs)
    lstm_preds_s = lstm_base.predict(X_test_seq)
    lstm_preds = pipeline.inverse_transform_target(lstm_preds_s)
    m_lstm = evaluate_all_metrics(y_test_actual, lstm_preds)
    results_list.append({"model": "Standalone LSTM", "horizon_hours": horizon, **m_lstm})

    # ---------------------------------------------
    # Baseline 4: Transformer
    # ---------------------------------------------
    logger.info("Evaluating Baseline 4: Temporal Transformer...")
    trans = TransformerPipeline(input_dim=len(feature_cols), d_model=32, nhead=4, num_layers=2)
    trans.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs)
    trans_preds_s = trans.predict(X_test_seq)
    trans_preds = pipeline.inverse_transform_target(trans_preds_s)
    m_trans = evaluate_all_metrics(y_test_actual, trans_preds)
    results_list.append({"model": "Temporal Transformer", "horizon_hours": horizon, **m_trans})

    # ---------------------------------------------
    # Proposed Architecture: MLP-LSTM + Residual Booster
    # ---------------------------------------------
    logger.info("Evaluating Proposed Architecture: MLP-LSTM + Residual Booster...")
    mlp_lstm = MLPLSTMPipeline(input_dim=len(feature_cols), config={"epochs": epochs, "batch_size": 64})
    mlp_lstm.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs)
    base_tr_preds_s = mlp_lstm.predict(X_train_seq)
    base_val_preds_s = mlp_lstm.predict(X_val_seq)
    base_test_preds_s = mlp_lstm.predict(X_test_seq)

    base_preds = pipeline.inverse_transform_target(base_test_preds_s)
    m_base = evaluate_all_metrics(y_test_actual, base_preds)
    results_list.append({"model": "MLP-LSTM (Base Hybrid)", "horizon_hours": horizon, **m_base})

    booster = ResidualBoostingModel(n_estimators=100, max_depth=5)
    booster.fit(X_train_flat, base_tr_preds_s, y_train_seq, X_val_flat, base_val_preds_s, y_val_seq, feature_names=feature_cols)
    final_test_preds_s, _ = booster.predict_final(X_test_flat, base_test_preds_s)
    final_preds = pipeline.inverse_transform_target(final_test_preds_s)
    m_final = evaluate_all_metrics(y_test_actual, final_preds)
    results_list.append({"model": "MLP-LSTM + Residual Boosting (Proposed)", "horizon_hours": horizon, **m_final})

    # Save Results
    results_df = pd.DataFrame(results_list)
    out_path = "reports/metrics/baseline_results.csv"
    save_dataframe(results_df, out_path)
    save_dataframe(results_df, f"results/horizon_{horizon}/baseline_comparison.csv")

    logger.info("=== BASELINE EXPERIMENT COMPARISON ===")
    for _, r in results_df.iterrows():
        logger.info(f"{r['model']:<45} | RMSE: {r['RMSE']:<8.4f} | MAE: {r['MAE']:<8.4f} | MAPE: {r['MAPE']:<6.2f}% | R2: {r['R2']:<6.4f}")

    # Plot Model Comparison Bar Chart
    viz = Visualizer()
    viz.plot_model_comparison(results_df)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run baseline benchmark experiments.")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    run_baselines_benchmark(horizon=args.horizon, epochs=args.epochs)


if __name__ == "__main__":
    main()
