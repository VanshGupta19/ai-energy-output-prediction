"""
SHAP (SHapley Additive exPlanations) Explainability Module.
Explains feature attributions for the Gradient Boosting Residual Correction model
and surrogate hybrid pipeline.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from src.utils.logger import get_logger
from src.utils.io import save_dataframe

logger = get_logger("shap_analysis")


class SHAPExplainer:
    """
    Computes game-theoretic SHAP feature attributions on the Gradient Boosting
    residual model to reveal what weather, temporal, and historical lag features
    drive model corrections.
    """

    def __init__(
        self,
        feature_names: List[str],
        top_n: int = 15,
        figures_dir: str = "reports/figures/shap",
        metrics_dir: str = "reports/metrics",
    ):
        self.feature_names = feature_names
        self.top_n = top_n
        self.figures_dir = figures_dir
        self.metrics_dir = metrics_dir
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.explainer_ = None
        self.shap_values_ = None

    def explain(
        self,
        booster_model: Any,
        X_explain: np.ndarray,
        base_preds: np.ndarray,
        max_samples: int = 400,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute TreeSHAP values for residual booster.

        Args:
            booster_model: Trained XGBoost, LightGBM, or sklearn booster.
            X_explain: Tabular test features (N, F).
            base_preds: Base MLP-LSTM predictions (N,).
            max_samples: Maximum instances to compute SHAP for (efficiency).

        Returns:
            Tuple of (shap_values, feature_importance_df).
        """
        logger.info("Computing SHAP explanations for Gradient Boosting Residual Correction model...")

        # Construct input matrix matching booster feature format
        X_boost = np.hstack([X_explain, base_preds.reshape(-1, 1)])
        feature_names = list(self.feature_names)
        if len(feature_names) < X_boost.shape[1]:
            feature_names.append("base_mlp_lstm_prediction")

        # Subsample if necessary
        if len(X_boost) > max_samples:
            sub_indices = np.random.choice(len(X_boost), size=max_samples, replace=False)
            X_sample = X_boost[sub_indices]
        else:
            X_sample = X_boost

        underlying_model = booster_model.get_booster() if hasattr(booster_model, "get_booster") else booster_model

        # Compute SHAP values
        try:
            self.explainer_ = shap.TreeExplainer(underlying_model)
            shap_obj = self.explainer_(X_sample)
            if hasattr(shap_obj, "values"):
                self.shap_values_ = shap_obj.values
            else:
                self.shap_values_ = self.explainer_.shap_values(X_sample)
        except Exception as e:
            logger.warning(f"TreeExplainer encountered issue: {e}. Falling back to shap.Explainer...")
            self.explainer_ = shap.Explainer(underlying_model, X_sample[:50])
            shap_obj = self.explainer_(X_sample)
            self.shap_values_ = shap_obj.values if hasattr(shap_obj, "values") else shap_obj

        # Compute mean absolute SHAP feature importance
        mean_abs_shap = np.mean(np.abs(self.shap_values_), axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        # Save feature importance CSV
        imp_path = os.path.join(self.metrics_dir, "feature_importance.csv")
        save_dataframe(importance_df, imp_path)
        logger.info(f"Saved feature importance table to {imp_path}")

        # Generate Visualizations
        self._plot_summary(X_sample, feature_names)
        self._plot_importance_bar(importance_df)
        self._plot_local_waterfall(X_sample, feature_names, sample_idx=0)

        return self.shap_values_, importance_df

    def _plot_summary(self, X_sample: np.ndarray, feature_names: List[str]):
        """Generate beeswarm summary plot."""
        plt.figure(figsize=(10, 8), dpi=300)
        shap.summary_plot(
            self.shap_values_,
            X_sample,
            feature_names=feature_names,
            max_display=self.top_n,
            show=False,
        )
        plt.title("SHAP Global Summary (Residual Booster Corrections)", fontsize=13, pad=15)
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, "shap_summary_plot.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Saved SHAP summary plot to {save_path}")

    def _plot_importance_bar(self, importance_df: pd.DataFrame):
        """Generate top feature importance bar chart."""
        top_df = importance_df.head(self.top_n)
        plt.figure(figsize=(10, 6), dpi=300)
        plt.barh(top_df["feature"][::-1], top_df["mean_abs_shap"][::-1], color="#2b5c8f", edgecolor="black", alpha=0.85)
        plt.xlabel("Mean |SHAP Value| (Impact on Residual Magnitude)", fontsize=11)
        plt.ylabel("Features", fontsize=11)
        plt.title(f"Top {self.top_n} Features (SHAP Attribution on Residual Booster)", fontsize=13)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, "shap_feature_importance.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Saved SHAP bar plot to {save_path}")

    def _plot_local_waterfall(self, X_sample: np.ndarray, feature_names: List[str], sample_idx: int = 0):
        """Generate local feature contribution plot for a single instance."""
        plt.figure(figsize=(9, 6), dpi=300)
        sample_shap = self.shap_values_[sample_idx]
        sample_feat = X_sample[sample_idx]

        # Top features for this instance
        top_indices = np.argsort(np.abs(sample_shap))[-self.top_n:]
        y_pos = np.arange(len(top_indices))
        colors = ["#d95f02" if sample_shap[i] < 0 else "#1b9e77" for i in top_indices]

        plt.barh(y_pos, sample_shap[top_indices], color=colors, edgecolor="black", alpha=0.85)
        labels = [f"{feature_names[i]} = {sample_feat[i]:.2f}" for i in top_indices]
        plt.yticks(y_pos, labels, fontsize=9)
        plt.xlabel("SHAP Value (Contribution to Residual Error)", fontsize=11)
        plt.title(f"Local SHAP Explanation for Instance #{sample_idx}", fontsize=12)
        plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, "shap_local_explanation.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Saved SHAP local explanation to {save_path}")
