"""
Comprehensive Publication-Quality Plotting Suite for Energy Output Forecasting.
Generates all 14 mandatory research figures with clear axis labels, titles, and legends.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

logger = get_logger("visualization")

# Set standard publication styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


class Visualizer:
    """
    Automated generation of publication-ready figures for energy forecasting research.
    """

    def __init__(self, figures_dir: str = "reports/figures"):
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)

    def _save(self, fig: plt.Figure, filename: str):
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info(f"Saved figure: {path}")

    # 1. Historical Energy Output Over Time
    def plot_historical_energy(self, df: pd.DataFrame, target_col: str = "energy_output_kwh", unit: str = "kWh"):
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
        ts = pd.to_datetime(df["timestamp"])
        ax.plot(ts, df[target_col], color="#1f77b4", linewidth=1.0, alpha=0.85, label=f"Observed Energy ({unit})")
        ax.set_title("Historical Energy Output Time Series", pad=10)
        ax.set_xlabel("Date and Time")
        ax.set_ylabel(f"Energy Output [{unit}]")
        ax.legend(loc="upper right")
        self._save(fig, "01_historical_energy_output.png")

    # 2. Weather Feature Distributions
    def plot_weather_distributions(self, df: pd.DataFrame, weather_cols: List[str]):
        n_cols = len(weather_cols)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.5 * n_rows), dpi=300)
        axes = axes.flatten()

        for idx, col in enumerate(weather_cols):
            if col in df.columns:
                sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], color="#2ca02c", edgecolor="black", alpha=0.6)
                axes[idx].set_title(f"Distribution of {col}")
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel("Count")

        # Hide extra subplots
        for idx in range(len(weather_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        self._save(fig, "02_weather_feature_distributions.png")

    # 3. Missing Value Report
    def plot_missing_values(self, quality_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        top_missing = quality_df.sort_values(by="missing_pct", ascending=False)
        colors = ["#d62728" if p > 0 else "#2ca02c" for p in top_missing["missing_pct"]]
        ax.barh(top_missing["column"][::-1], top_missing["missing_pct"][::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
        ax.set_title("Data Completeness and Missing Value Audit (%)")
        ax.set_xlabel("Missing / Placeholder Percentage (%)")
        ax.set_ylabel("Dataset Columns")
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        self._save(fig, "03_missing_value_report.png")

    # 4. Correlation Heatmap
    def plot_correlation_heatmap(self, df: pd.DataFrame, feature_cols: List[str]):
        cols_to_corr = [c for c in feature_cols if c in df.columns][:14] # Top 14 features for readability
        corr = df[cols_to_corr].corr()
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        self._save(fig, "04_correlation_heatmap.png")

    # 5. Actual vs Predicted Energy Scatter Plot
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, unit: str = "kWh"):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
        ax.scatter(y_true, y_pred, alpha=0.4, color="#1f77b4", edgecolor="none", s=20, label="Predictions")
        # Ideal 1:1 line
        min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], color="#d62728", linestyle="--", linewidth=1.5, label="Perfect Agreement (y = x)")
        ax.set_title("Actual vs. Predicted Energy Output")
        ax.set_xlabel(f"Actual Energy Output [{unit}]")
        ax.set_ylabel(f"Predicted Energy Output [{unit}]")
        ax.legend(loc="upper left")
        self._save(fig, "05_actual_vs_predicted.png")

    # 6. Time-Series Forecast Comparison Overlay
    def plot_forecast_comparison(
        self,
        y_true: np.ndarray,
        base_pred: np.ndarray,
        final_pred: np.ndarray,
        timestamps: Optional[pd.Series] = None,
        n_display: int = 168, # 7 days
        unit: str = "kWh",
    ):
        fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
        n_pts = min(n_display, len(y_true))
        idx_range = slice(0, n_pts)
        if timestamps is not None and len(timestamps) >= n_pts:
            x_axis = timestamps.iloc[:n_pts]
        else:
            x_axis = np.arange(n_pts)

        ax.plot(x_axis, y_true[idx_range], label="Actual Output", color="black", linewidth=1.5)
        ax.plot(x_axis, base_pred[idx_range], label="Base (MLP-LSTM)", color="#ff7f0e", linestyle="--", linewidth=1.2)
        ax.plot(x_axis, final_pred[idx_range], label="Proposed Hybrid (MLP-LSTM + Residual Boosting)", color="#2ca02c", linewidth=1.4)

        ax.set_title(f"Time-Series Forecast Comparison (First {n_display} Hours)")
        ax.set_xlabel("Time / Timestep")
        ax.set_ylabel(f"Energy Output [{unit}]")
        ax.legend(loc="upper right")
        self._save(fig, "06_time_series_forecast_comparison.png")

    # 7. Residual Distribution Histogram
    def plot_residual_distribution(self, y_true: np.ndarray, y_pred: np.ndarray):
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        sns.histplot(residuals, kde=True, ax=ax, color="#9467bd", edgecolor="black", alpha=0.6)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.0)
        ax.set_title(f"Residual Error Distribution (Mean={residuals.mean():.3f}, Std={residuals.std():.3f})")
        ax.set_xlabel("Residual Error (Actual - Predicted)")
        ax.set_ylabel("Density / Count")
        self._save(fig, "07_residual_distribution.png")

    # 8. Residual Over Time
    def plot_residuals_over_time(self, y_true: np.ndarray, y_pred: np.ndarray, timestamps: Optional[pd.Series] = None):
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
        if timestamps is not None:
            x_axis = timestamps.iloc[:len(residuals)]
        else:
            x_axis = np.arange(len(residuals))
        ax.scatter(x_axis, residuals, alpha=0.5, color="#8c564b", s=15)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title("Residual Errors Over Time")
        ax.set_xlabel("Timestamp / Time Index")
        ax.set_ylabel("Residual Error")
        self._save(fig, "08_residual_over_time.png")

    # 9. Model Comparison Bar Chart
    def plot_model_comparison(self, comparison_df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        sns.barplot(data=comparison_df, x="model", y="RMSE", ax=axes[0], palette="Blues_d", edgecolor="black")
        axes[0].set_title("Model Comparison: RMSE (Lower is Better)")
        axes[0].tick_params(axis="x", rotation=25)

        sns.barplot(data=comparison_df, x="model", y="R2", ax=axes[1], palette="Greens_d", edgecolor="black")
        axes[1].set_title("Model Comparison: R² Score (Higher is Better)")
        axes[1].tick_params(axis="x", rotation=25)

        plt.tight_layout()
        self._save(fig, "09_model_comparison_bar_chart.png")

    # 10. Multi-Horizon Forecasting Performance Curve
    def plot_horizon_performance(self, horizon_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        for model_name, group in horizon_df.groupby("model"):
            ax.plot(group["horizon_hours"], group["RMSE"], marker="o", label=f"{model_name} (RMSE)", linewidth=1.5)

        ax.set_title("Forecast Accuracy across Horizons (1h, 3h, 6h, 12h, 24h Ahead)")
        ax.set_xlabel("Forecast Horizon [Hours]")
        ax.set_ylabel("Root Mean Squared Error (RMSE)")
        ax.set_xticks(sorted(horizon_df["horizon_hours"].unique()))
        ax.legend()
        self._save(fig, "10_forecast_horizon_performance.png")

    # 11. Walk-Forward Fold Performance
    def plot_walk_forward_performance(self, wf_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        x = np.arange(len(wf_df))
        width = 0.35
        ax.bar(x - width / 2, wf_df["base_RMSE"], width, label="Base MLP-LSTM", color="#ff7f0e", edgecolor="black", alpha=0.85)
        ax.bar(x + width / 2, wf_df["final_RMSE"], width, label="Proposed Hybrid", color="#2ca02c", edgecolor="black", alpha=0.85)
        ax.set_title("Walk-Forward Validation Accuracy Across Chronological Folds")
        ax.set_xlabel("Evaluation Fold")
        ax.set_ylabel("RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {f}" for f in wf_df["fold"]])
        ax.legend()
        self._save(fig, "11_walk_forward_performance.png")

    # 12. Prediction Intervals with Shaded Bounds
    def plot_prediction_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        timestamps: Optional[pd.Series] = None,
        n_display: int = 168,
        unit: str = "kWh",
    ):
        fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
        n_pts = min(n_display, len(y_true))
        idx_range = slice(0, n_pts)
        if timestamps is not None and len(timestamps) >= n_pts:
            x_axis = timestamps.iloc[:n_pts]
        else:
            x_axis = np.arange(n_pts)

        ax.plot(x_axis, y_true[idx_range], label="Actual Output", color="black", linewidth=1.5)
        ax.plot(x_axis, y_pred[idx_range], label="Hybrid Point Forecast", color="#1f77b4", linewidth=1.3)
        ax.fill_between(
            x_axis,
            lower_bound[idx_range],
            upper_bound[idx_range],
            color="#aec7e8",
            alpha=0.5,
            label="90% Conformal Prediction Interval",
        )

        ax.set_title(f"Uncertainty Quantification & 90% Prediction Intervals (First {n_display} Hours)")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Energy Output [{unit}]")
        ax.legend(loc="upper right")
        self._save(fig, "12_prediction_intervals.png")
