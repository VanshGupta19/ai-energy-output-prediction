"""
SHAP Explainability Execution Script.
Computes TreeSHAP explanations for the Gradient Boosting Residual Correction model.
Usage: python scripts/run_shap.py --horizon 24
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.io import load_yaml, load_model
from src.data.loader import DataLoader
from src.data.merger import DataMerger
from src.data.preprocessing import PreprocessingPipeline
from src.features.feature_engineering import FeatureEngineer
from src.evaluation.validation import ChronologicalSplitter, SequenceGenerator
from src.models.mlp_lstm import MLPLSTMPipeline
from src.models.residual_boosting import ResidualBoostingModel
from src.explainability.shap_analysis import SHAPExplainer

logger = setup_logger("run_shap")


def run_shap_analysis(horizon: int = 24):
    config = load_yaml("config/config.yaml")
    target_col = config["data"]["target_column"]

    # 1. Load Data
    loader = DataLoader(config)
    weather_df = loader.load_weather(config["data"]["weather_path"])
    energy_df = loader.load_energy(config["data"]["energy_path"])
    merger = DataMerger()
    merged_df, _ = merger.merge(weather_df, energy_df)

    splitter = ChronologicalSplitter(
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
    )
    train_df, val_df, test_df, _ = splitter.split(merged_df)

    # 2. Preprocess & Engineer
    pipeline = PreprocessingPipeline(target_column=target_col)
    train_df_prep = pipeline.fit_transform(train_df)
    val_df_prep = pipeline.transform(val_df)
    test_df_prep = pipeline.transform(test_df)
    full_prep = pd.concat([train_df_prep, val_df_prep, test_df_prep], ignore_index=True)

    feat_eng = FeatureEngineer(target_column=target_col, horizon_hours=horizon)
    df_eng, feature_cols = feat_eng.engineer_features(full_prep)

    n_tr, n_v = len(train_df), len(val_df)
    test_eng = df_eng.iloc[n_tr + n_v :].copy().reset_index(drop=True)

    seq_gen = SequenceGenerator(sequence_length=24)
    lead_target_col = f"{target_col}_lead_{horizon}h"
    X_test_seq, y_test_seq, X_test_flat = seq_gen.create_sequences(
        test_eng, feature_cols, test_eng[lead_target_col]
    )

    # Load trained models
    booster_path = f"models/residual_boosting_model_h{horizon}.pkl"
    mlp_lstm_path = f"models/mlp_lstm_model_h{horizon}.pt"

    if os.path.exists(booster_path) and os.path.exists(mlp_lstm_path):
        logger.info(f"Loading trained models from {mlp_lstm_path} and {booster_path}...")
        mlp_lstm = MLPLSTMPipeline.load(mlp_lstm_path)
        booster = ResidualBoostingModel.load(booster_path)
    else:
        logger.info("Trained model checkpoint not found. Training quick baseline for SHAP...")
        train_eng = df_eng.iloc[:n_tr].copy().reset_index(drop=True)
        val_eng = df_eng.iloc[n_tr : n_tr + n_v].copy().reset_index(drop=True)
        X_tr_seq, y_tr_seq, X_tr_flat = seq_gen.create_sequences(train_eng, feature_cols, train_eng[lead_target_col])
        X_v_seq, y_v_seq, X_v_flat = seq_gen.create_sequences(val_eng, feature_cols, val_eng[lead_target_col])

        mlp_lstm = MLPLSTMPipeline(input_dim=len(feature_cols), config={"epochs": 20, "batch_size": 64})
        mlp_lstm.fit(X_tr_seq, y_tr_seq, X_v_seq, y_v_seq, epochs=20, verbose=False)
        base_tr_s = mlp_lstm.predict(X_tr_seq)
        base_v_s = mlp_lstm.predict(X_v_seq)

        booster = ResidualBoostingModel(n_estimators=80, max_depth=4)
        booster.fit(X_tr_flat, base_tr_s, y_tr_seq, X_v_flat, base_v_s, y_v_seq, feature_names=feature_cols)

    base_test_preds_s = mlp_lstm.predict(X_test_seq)

    # Compute SHAP
    explainer = SHAPExplainer(feature_names=feature_cols, top_n=15)
    shap_vals, imp_df = explainer.explain(
        booster_model=booster,
        X_explain=X_test_flat,
        base_preds=base_test_preds_s,
        max_samples=300,
    )

    logger.info("SHAP analysis and visualization export completed successfully.")
    return shap_vals, imp_df


def main():
    parser = argparse.ArgumentParser(description="Run SHAP explainability analysis.")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    run_shap_analysis(horizon=args.horizon)


if __name__ == "__main__":
    main()
