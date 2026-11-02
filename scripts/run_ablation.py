"""
Ablation Study Experiment Script.
Quantifies marginal impact of MLP, LSTM, and Gradient Boosting Residual Correction.
Usage: python scripts/run_ablation.py --horizon 24
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.io import load_yaml
from src.data.loader import DataLoader as DL
from src.data.merger import DataMerger
from src.data.preprocessing import PreprocessingPipeline
from src.features.feature_engineering import FeatureEngineer
from src.evaluation.validation import ChronologicalSplitter, SequenceGenerator
from src.evaluation.ablation import AblationStudy
from src.models.mlp_lstm import MLPLSTMPipeline
from src.models.baselines import StandaloneLSTMPipeline
from src.models.residual_boosting import ResidualBoostingModel

logger = setup_logger("run_ablation")


class DenseMLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16], dropout: float = 0.2):
        super(DenseMLPBaseline, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_only(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, epochs: int = 40) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseMLPBaseline(input_dim=X_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_te, dtype=torch.float32).to(device)
        return model(x_t).cpu().numpy().flatten()


def run_ablation_study(horizon: int = 24, epochs: int = 40):
    config = load_yaml("config/config.yaml")
    set_seed(config.get("random_seed", 42))

    target_col = config["data"]["target_column"]

    # 1. Load Data
    loader = DL(config)
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

    seq_gen = SequenceGenerator(sequence_length=24)
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
    predictions_dict = {}

    # Variant 1: Dense MLP Only
    logger.info("Ablation 1/4: Training MLP-only...")
    mlp_preds_s = train_mlp_only(X_train_flat, y_train_seq, X_test_flat, epochs=epochs)
    predictions_dict["MLP_only"] = pipeline.inverse_transform_target(mlp_preds_s)

    # Variant 2: LSTM Only
    logger.info("Ablation 2/4: Training Standalone LSTM-only...")
    lstm_pipe = StandaloneLSTMPipeline(input_dim=len(feature_cols))
    lstm_pipe.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs)
    lstm_preds_s = lstm_pipe.predict(X_test_seq)
    predictions_dict["LSTM_only"] = pipeline.inverse_transform_target(lstm_preds_s)

    # Variant 3: Hybrid MLP-LSTM
    logger.info("Ablation 3/4: Training Hybrid MLP-LSTM...")
    mlp_lstm = MLPLSTMPipeline(input_dim=len(feature_cols), config={"epochs": epochs, "batch_size": 64})
    mlp_lstm.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs)
    base_tr_s = mlp_lstm.predict(X_train_seq)
    base_val_s = mlp_lstm.predict(X_val_seq)
    base_test_s = mlp_lstm.predict(X_test_seq)
    predictions_dict["MLP_LSTM"] = pipeline.inverse_transform_target(base_test_s)

    # Variant 4: Complete Proposed Model (MLP-LSTM + Residual Boosting)
    logger.info("Ablation 4/4: Training Residual Booster on MLP-LSTM...")
    booster = ResidualBoostingModel(n_estimators=100, max_depth=5)
    booster.fit(X_train_flat, base_tr_s, y_train_seq, X_val_flat, base_val_s, y_val_seq, feature_names=feature_cols)
    final_test_s, _ = booster.predict_final(X_test_flat, base_test_s)
    predictions_dict["MLP_LSTM_Residual_Boosting"] = pipeline.inverse_transform_target(final_test_s)

    # Evaluate
    ablation = AblationStudy()
    df_ablation, summary = ablation.evaluate_ablation(
        y_test_actual, predictions_dict, output_path="reports/metrics/ablation_results.csv"
    )

    logger.info("=== ABLATION STUDY SUMMARY ===")
    for _, r in df_ablation.iterrows():
        logger.info(f"{r['model_configuration']:<30} | RMSE: {r['RMSE']:<8.4f} | MAE: {r['MAE']:<8.4f} | R2: {r['R2']:<6.4f}")

    return df_ablation, summary


def main():
    parser = argparse.ArgumentParser(description="Run architectural ablation study.")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    run_ablation_study(horizon=args.horizon, epochs=args.epochs)


if __name__ == "__main__":
    main()
