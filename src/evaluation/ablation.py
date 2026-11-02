"""
Ablation Study Framework.
Quantifies the exact marginal contributions of:
1. Dense MLP feature representation
2. LSTM sequential recurrence
3. Hybrid MLP-LSTM
4. Second-stage Gradient Boosting Residual Correction
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_all_metrics
from src.utils.logger import get_logger
from src.utils.io import save_dataframe

logger = get_logger("ablation")


class AblationStudy:
    """
    Computes comparative performance across ablation configurations and calculates
    absolute and percentage performance improvements.
    """

    def __init__(self):
        self.results_ = []

    def evaluate_ablation(
        self,
        y_test_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        output_path: Optional[str] = "reports/metrics/ablation_results.csv",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate multiple model ablation variants and compute gains relative to base models.

        Args:
            y_test_true: Ground truth test energy outputs.
            predictions_dict: Dictionary mapping model name to test predictions array.
            output_path: Optional path to save CSV results.

        Returns:
            Tuple of (ablation_df, summary_dict).
        """
        logger.info("Executing comprehensive ablation analysis...")

        rows = []
        metrics_by_model = {}

        for model_name, preds in predictions_dict.items():
            m = evaluate_all_metrics(y_test_true, preds)
            metrics_by_model[model_name] = m
            rows.append({
                "model_configuration": model_name,
                "MAE": m["MAE"],
                "MSE": m["MSE"],
                "RMSE": m["RMSE"],
                "MAPE": m["MAPE"],
                "R2": m["R2"],
            })

        df_ablation = pd.DataFrame(rows)

        # Calculate marginal improvements of Proposed (MLP-LSTM + Residual Booster) over MLP-LSTM Base
        base_name = [k for k in predictions_dict.keys() if "MLP_LSTM" in k and "Residual" not in k]
        full_name = [k for k in predictions_dict.keys() if "Residual" in k]

        summary = {}
        if base_name and full_name:
            m_base = metrics_by_model[base_name[0]]
            m_full = metrics_by_model[full_name[0]]

            abs_mae_imp = m_base["MAE"] - m_full["MAE"]
            pct_mae_imp = (abs_mae_imp / m_base["MAE"]) * 100.0 if m_base["MAE"] > 0 else 0.0

            abs_rmse_imp = m_base["RMSE"] - m_full["RMSE"]
            pct_rmse_imp = (abs_rmse_imp / m_base["RMSE"]) * 100.0 if m_base["RMSE"] > 0 else 0.0

            abs_r2_imp = m_full["R2"] - m_base["R2"]

            summary = {
                "base_model": base_name[0],
                "proposed_model": full_name[0],
                "mae_absolute_improvement": round(abs_mae_imp, 4),
                "mae_percent_improvement": round(pct_mae_imp, 2),
                "rmse_absolute_improvement": round(abs_rmse_imp, 4),
                "rmse_percent_improvement": round(pct_rmse_imp, 2),
                "r2_improvement": round(abs_r2_imp, 4),
            }

            logger.info(
                f"Ablation Result: Residual Boosting improved RMSE by {abs_rmse_imp:.4f} "
                f"({pct_rmse_imp:.2f}%) and MAE by {abs_mae_imp:.4f} ({pct_mae_imp:.2f}%)"
            )

        if output_path:
            save_dataframe(df_ablation, output_path)

        return df_ablation, summary
