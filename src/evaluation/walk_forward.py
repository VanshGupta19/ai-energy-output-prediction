"""
Rolling-Origin / Walk-Forward Cross-Validation for Time-Series Forecasting.
Evaluates model stability over rolling chronological origins with zero data leakage.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from src.data.preprocessing import PreprocessingPipeline
from src.features.feature_engineering import FeatureEngineer
from src.evaluation.validation import SequenceGenerator
from src.evaluation.metrics import evaluate_all_metrics
from src.models.mlp_lstm import MLPLSTMPipeline
from src.models.residual_boosting import ResidualBoostingModel
from src.utils.logger import get_logger
from src.utils.io import save_dataframe

logger = get_logger("walk_forward")


class WalkForwardValidator:
    """
    Executes expanding-window walk-forward validation:
    Fold k:
      Train: 0 -> t_k
      Validation: t_k -> t_{k+1}
    """

    def __init__(
        self,
        n_splits: int = 3,
        test_size_hours: int = 720, # 30 days
        min_train_size_hours: int = 2160, # 90 days
        sequence_length: int = 24,
        epochs: int = 15,
        horizon_hours: int = 24,
        target_column: str = "energy_output_kwh",
    ):
        self.n_splits = n_splits
        self.test_size_hours = test_size_hours
        self.min_train_size_hours = min_train_size_hours
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.horizon_hours = horizon_hours
        self.target_column = target_column

    def run_validation(
        self,
        df_engineered: pd.DataFrame,
        feature_columns: List[str],
        output_results_path: Optional[str] = "reports/metrics/walk_forward_results.csv",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute walk-forward cross-validation folds.

        Returns:
            Tuple of (fold_metrics_df, aggregate_summary_dict).
        """
        target_lead_col = f"{self.target_column}_lead_{self.horizon_hours}h"
        # Drop rows with NaN in lead target (at tail of dataset)
        valid_df = df_engineered.dropna(subset=[target_lead_col]).reset_index(drop=True)
        total_len = len(valid_df)

        required_len = self.min_train_size_hours + (self.n_splits * self.test_size_hours)
        if total_len < required_len:
            # Adjust test_size_hours or min_train_size_hours to fit data
            available_for_test = total_len - self.min_train_size_hours
            if available_for_test > 0:
                self.test_size_hours = max(240, available_for_test // self.n_splits)
            else:
                self.min_train_size_hours = int(total_len * 0.5)
                self.test_size_hours = int((total_len - self.min_train_size_hours) / self.n_splits)

        logger.info(
            f"Executing Walk-Forward Validation: {self.n_splits} folds, "
            f"min_train={self.min_train_size_hours}h, test_size={self.test_size_hours}h, epochs={self.epochs}"
        )

        fold_records = []
        seq_gen = SequenceGenerator(sequence_length=self.sequence_length)

        for fold in range(self.n_splits):
            train_end = self.min_train_size_hours + (fold * self.test_size_hours)
            val_end = train_end + self.test_size_hours

            train_df = valid_df.iloc[:train_end].copy().reset_index(drop=True)
            val_df = valid_df.iloc[train_end:val_end].copy().reset_index(drop=True)

            logger.info(f"--- FOLD {fold + 1}/{self.n_splits} [Train: 0 -> {train_end}, Val: {train_end} -> {val_end}] ---")

            # Fit Preprocessor on this fold's train data only
            pipeline = PreprocessingPipeline(target_column=self.target_column)
            train_prep = pipeline.fit_transform(train_df)
            val_prep = pipeline.transform(val_df)

            # Generate Sequences
            X_train_seq, y_train_seq, X_train_flat = seq_gen.create_sequences(
                train_prep, feature_columns, train_prep[target_lead_col]
            )
            X_val_seq, y_val_seq, X_val_flat = seq_gen.create_sequences(
                val_prep, feature_columns, val_prep[target_lead_col]
            )

            # Further split train into train/internal_val for early stopping
            val_split_idx = int(len(X_train_seq) * 0.8)
            X_tr_seq, y_tr_seq = X_train_seq[:val_split_idx], y_train_seq[:val_split_idx]
            X_tr_flat = X_train_flat[:val_split_idx]
            X_cal_seq, y_cal_seq = X_train_seq[val_split_idx:], y_train_seq[val_split_idx:]
            X_cal_flat = X_train_flat[val_split_idx:]

            # 1. Train MLP-LSTM
            mlp_lstm = MLPLSTMPipeline(
                input_dim=len(feature_columns),
                config={"epochs": self.epochs, "batch_size": 64, "early_stopping_patience": 5},
            )
            mlp_lstm.fit(X_tr_seq, y_tr_seq, X_cal_seq, y_cal_seq, epochs=self.epochs, verbose=False)

            # 2. Train Residual Booster
            base_tr_pred = mlp_lstm.predict(X_tr_seq)
            base_cal_pred = mlp_lstm.predict(X_cal_seq)

            booster = ResidualBoostingModel(
                n_estimators=60,
                max_depth=4,
                learning_rate=0.08,
                random_state=42 + fold,
            )
            booster.fit(
                X_tr_flat, base_tr_pred, y_tr_seq,
                X_cal_flat, base_cal_pred, y_cal_seq,
                feature_names=feature_columns,
            )

            # 3. Evaluate on Fold Validation set
            base_val_pred_scaled = mlp_lstm.predict(X_val_seq)
            final_val_pred_scaled, _ = booster.predict_final(X_val_flat, base_val_pred_scaled)

            # Inverse scale back to physical kWh units
            y_val_actual = pipeline.inverse_transform_target(y_val_seq)
            base_val_pred = pipeline.inverse_transform_target(base_val_pred_scaled)
            final_val_pred = pipeline.inverse_transform_target(final_val_pred_scaled)

            # Compute metrics
            m_base = evaluate_all_metrics(y_val_actual, base_val_pred)
            m_final = evaluate_all_metrics(y_val_actual, final_val_pred)

            fold_records.append({
                "fold": fold + 1,
                "train_records": len(train_df),
                "val_records": len(val_df),
                "start_val_time": str(val_df["timestamp"].min()) if "timestamp" in val_df.columns else "N/A",
                "end_val_time": str(val_df["timestamp"].max()) if "timestamp" in val_df.columns else "N/A",
                "base_MAE": m_base["MAE"],
                "base_RMSE": m_base["RMSE"],
                "base_MAPE": m_base["MAPE"],
                "base_R2": m_base["R2"],
                "final_MAE": m_final["MAE"],
                "final_RMSE": m_final["RMSE"],
                "final_MAPE": m_final["MAPE"],
                "final_R2": m_final["R2"],
            })

        results_df = pd.DataFrame(fold_records)

        # Aggregate Statistics
        summary = {
            "mean_final_MAE": round(float(results_df["final_MAE"].mean()), 4),
            "std_final_MAE": round(float(results_df["final_MAE"].std()), 4),
            "mean_final_RMSE": round(float(results_df["final_RMSE"].mean()), 4),
            "std_final_RMSE": round(float(results_df["final_RMSE"].std()), 4),
            "mean_final_MAPE": round(float(results_df["final_MAPE"].mean()), 2),
            "std_final_MAPE": round(float(results_df["final_MAPE"].std()), 2),
            "mean_final_R2": round(float(results_df["final_R2"].mean()), 4),
            "std_final_R2": round(float(results_df["final_R2"].std()), 4),
        }

        logger.info(
            f"Walk-Forward Summary ({self.n_splits} folds): "
            f"Final RMSE: {summary['mean_final_RMSE']} +/- {summary['std_final_RMSE']}, "
            f"Final MAE: {summary['mean_final_MAE']} +/- {summary['std_final_MAE']}, "
            f"Final R2: {summary['mean_final_R2']} +/- {summary['std_final_R2']}"
        )

        if output_results_path:
            save_dataframe(results_df, output_results_path)

        return results_df, summary
