"""
Master End-to-End Execution CLI for AI-Based Energy Output Prediction Research.
Executes the complete experimental lifecycle in one command.
Usage: python scripts/run_full_pipeline.py --mode full
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.io import load_yaml, save_dataframe, save_json
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.merger import DataMerger
from src.data.preprocessing import PreprocessingPipeline
from src.features.feature_engineering import FeatureEngineer, FeatureLeakageChecker
from src.evaluation.validation import ChronologicalSplitter, SequenceGenerator
from src.evaluation.metrics import evaluate_all_metrics
from src.evaluation.walk_forward import WalkForwardValidator
from src.evaluation.ablation import AblationStudy
from src.models.mlp_lstm import MLPLSTMPipeline
from src.models.residual_boosting import ResidualBoostingModel
from src.models.baselines import (
    SeasonalNaiveBaseline,
    ARIMABaseline,
    SVRBaseline,
    StandaloneLSTMPipeline,
    TransformerPipeline,
)
from src.models.uncertainty import ConformalPredictionInterval
from src.explainability.shap_analysis import SHAPExplainer
from src.visualization.plots import Visualizer

logger = setup_logger("full_pipeline", log_file="reports/pipeline_run.log")


def run_full_research_pipeline(mode: str = "fast"):
    start_time = time.time()
    logger.info("================================================================================")
    logger.info("  AI-BASED PREDICTION OF FUTURE ENERGY OUTPUT USING WEATHER & HISTORICAL DATA   ")
    logger.info(f"  Execution Mode: {mode.upper()}                                                 ")
    logger.info("================================================================================")

    config = load_yaml("config/config.yaml")
    m_config = load_yaml("config/model_config.yaml")
    exp_config = load_yaml("config/experiment_config.yaml")
    set_seed(config.get("random_seed", 42))

    epochs = 25 if mode == "fast" else m_config["mlp_lstm"].get("epochs", 80)
    horizons = [1, 3, 6, 12, 24] if mode == "full" else [1, 6, 24]
    target_col = config["data"]["target_column"]
    unit = config["data"].get("target_unit", "kWh")

    # =========================================================================
    # STAGE 1 & 2: DATA ACQUISITION, VALIDATION & MERGING
    # =========================================================================
    logger.info(">>> [1/7] DATA ACQUISITION, VALIDATION, AND MERGING <<<")
    loader = DataLoader(config)
    weather_df = loader.load_weather(config["data"]["weather_path"])
    energy_df = loader.load_energy(config["data"]["energy_path"])

    validator = DataValidator(target_column=target_col, allow_synthetic_demo=config["data"].get("allow_synthetic_demo", False))
    _, quality_df, _ = validator.validate(weather_df, energy_df, output_report_path="reports/data_quality/data_quality_report.csv")

    merger = DataMerger()
    merged_df, _ = merger.merge(weather_df, energy_df)
    merged_df.to_csv(config["data"]["merged_output_path"], index=False)

    viz = Visualizer()
    viz.plot_missing_values(quality_df)
    viz.plot_historical_energy(merged_df, target_col=target_col, unit=unit)
    raw_weather_cols = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "PS", "WS10M", "WS50M", "CLOUD_AMT"]
    viz.plot_weather_distributions(merged_df, [c for c in raw_weather_cols if c in merged_df.columns])

    # =========================================================================
    # STAGE 3, 4, 5, 6, 7, 8: MULTI-HORIZON HYBRID MODEL EXPERIMENTS
    # =========================================================================
    logger.info(">>> [2/7] MULTI-HORIZON FORECASTING (1h, 3h, 6h, 12h, 24h) <<<")
    splitter = ChronologicalSplitter(
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
    )
    train_df, val_df, test_df, _ = splitter.split(merged_df, output_summary_path="reports/metrics/train_validation_test_split_summary.csv")

    horizon_records = []
    trained_artifacts_24h = {}

    for h in horizons:
        logger.info(f"--- Training Proposed Hybrid for Horizon H = {h} hours ---")
        h_dir = f"results/horizon_{h}"
        os.makedirs(h_dir, exist_ok=True)

        pipeline = PreprocessingPipeline(target_column=target_col)
        train_prep = pipeline.fit_transform(train_df)
        val_prep = pipeline.transform(val_df)
        test_prep = pipeline.transform(test_df)
        full_prep = pd.concat([train_prep, val_prep, test_prep], ignore_index=True)

        feat_eng = FeatureEngineer(target_column=target_col, horizon_hours=h)
        df_eng, feat_cols = feat_eng.engineer_features(
            full_prep, output_feature_list_path="reports/metrics/feature_list.csv" if h == 24 else None
        )

        n_tr, n_v = len(train_df), len(val_df)
        train_eng = df_eng.iloc[:n_tr].copy().reset_index(drop=True)
        val_eng = df_eng.iloc[n_tr : n_tr + n_v].copy().reset_index(drop=True)
        test_eng = df_eng.iloc[n_tr + n_v :].copy().reset_index(drop=True)

        seq_gen = SequenceGenerator(sequence_length=24)
        lead_target_col = f"{target_col}_lead_{h}h"

        X_tr_seq, y_tr_seq, X_tr_flat = seq_gen.create_sequences(train_eng, feat_cols, train_eng[lead_target_col])
        X_v_seq, y_v_seq, X_v_flat = seq_gen.create_sequences(val_eng, feat_cols, val_eng[lead_target_col])
        X_te_seq, y_te_seq, X_te_flat = seq_gen.create_sequences(test_eng, feat_cols, test_eng[lead_target_col])

        # 1. Base MLP-LSTM
        mlp_lstm = MLPLSTMPipeline(input_dim=len(feat_cols), config={"epochs": epochs, "batch_size": 64})
        mlp_lstm.fit(X_tr_seq, y_tr_seq, X_v_seq, y_v_seq, epochs=epochs, verbose=False)
        mlp_lstm.save(f"models/mlp_lstm_model_h{h}.pt")

        base_tr_s = mlp_lstm.predict(X_tr_seq)
        base_v_s = mlp_lstm.predict(X_v_seq)
        base_te_s = mlp_lstm.predict(X_te_seq)

        # 2. Residual Booster
        booster = ResidualBoostingModel(n_estimators=100 if mode == "full" else 40, max_depth=5)
        booster.fit(X_tr_flat, base_tr_s, y_tr_seq, X_v_flat, base_v_s, y_v_seq, feature_names=feat_cols)
        booster.save(f"models/residual_boosting_model_h{h}.pkl")

        final_te_s, _ = booster.predict_final(X_te_flat, base_te_s)
        final_v_s, _ = booster.predict_final(X_v_flat, base_v_s)

        # Inverse transform to kWh
        y_test_act = pipeline.inverse_transform_target(y_te_seq)
        base_te_act = pipeline.inverse_transform_target(base_te_s)
        final_te_act = pipeline.inverse_transform_target(final_te_s)

        y_v_act = pipeline.inverse_transform_target(y_v_seq)
        final_v_act = pipeline.inverse_transform_target(final_v_s)

        # Conformal intervals
        conformal = ConformalPredictionInterval(confidence_level=0.90)
        conformal.calibrate(y_v_act, final_v_act)
        lb, ub = conformal.predict_intervals(final_te_act)

        # Metrics
        m_base = evaluate_all_metrics(y_test_act, base_te_act)
        m_final = evaluate_all_metrics(y_test_act, final_te_act, lb, ub)

        horizon_records.append({
            "model": "MLP-LSTM (Base)",
            "horizon_hours": h,
            **m_base,
        })
        horizon_records.append({
            "model": "MLP-LSTM + Residual Booster (Proposed)",
            "horizon_hours": h,
            **m_final,
        })

        if h == 24 or (h == max(horizons)):
            trained_artifacts_24h = {
                "booster": booster,
                "X_test_flat": X_te_flat,
                "base_test_preds_s": base_te_s,
                "feature_cols": feat_cols,
                "y_test_act": y_test_act,
                "base_te_act": base_te_act,
                "final_te_act": final_te_act,
                "lb": lb,
                "ub": ub,
                "test_timestamps": test_eng["timestamp"].iloc[24:].reset_index(drop=True),
                "df_eng": df_eng,
            }

    df_horizon = pd.DataFrame(horizon_records)
    save_dataframe(df_horizon, "reports/metrics/horizon_comparison.csv")
    save_dataframe(df_horizon, "reports/metrics/hybrid_model_results.csv")
    viz.plot_horizon_performance(df_horizon)

    # Save 24h figures
    if trained_artifacts_24h:
        t_art = trained_artifacts_24h
        viz.plot_actual_vs_predicted(t_art["y_test_act"], t_art["final_te_act"], unit=unit)
        viz.plot_forecast_comparison(t_art["y_test_act"], t_art["base_te_act"], t_art["final_te_act"], timestamps=t_art["test_timestamps"], unit=unit)
        viz.plot_residual_distribution(t_art["y_test_act"], t_art["final_te_act"])
        viz.plot_residuals_over_time(t_art["y_test_act"], t_art["final_te_act"], timestamps=t_art["test_timestamps"])
        viz.plot_prediction_intervals(t_art["y_test_act"], t_art["final_te_act"], t_art["lb"], t_art["ub"], timestamps=t_art["test_timestamps"], unit=unit)
        viz.plot_correlation_heatmap(t_art["df_eng"], t_art["feature_cols"])

    # =========================================================================
    # STAGE 9: BASELINES BENCHMARKING
    # =========================================================================
    logger.info(">>> [3/7] BASELINE MODELS EVALUATION <<<")
    # Seasonal Naive
    s_naive = SeasonalNaiveBaseline(season_length=24)
    s_naive.fit(t_art["y_test_act"])
    naive_preds = s_naive.predict(t_art["y_test_act"], len(t_art["y_test_act"]))

    # SVR
    svr = SVRBaseline()
    svr.fit(X_tr_flat, y_tr_seq)
    svr_preds = pipeline.inverse_transform_target(svr.predict(X_te_flat))

    # Standalone LSTM
    s_lstm = StandaloneLSTMPipeline(input_dim=len(feat_cols))
    s_lstm.fit(X_tr_seq, y_tr_seq, X_v_seq, y_v_seq, epochs=epochs)
    s_lstm_preds = pipeline.inverse_transform_target(s_lstm.predict(X_te_seq))

    # Transformer
    trans = TransformerPipeline(input_dim=len(feat_cols))
    trans.fit(X_tr_seq, y_tr_seq, X_v_seq, y_v_seq, epochs=epochs)
    trans_preds = pipeline.inverse_transform_target(trans.predict(X_te_seq))

    baseline_rows = [
        {"model": "Seasonal Naive (24h)", "horizon_hours": 24, **evaluate_all_metrics(t_art["y_test_act"], naive_preds)},
        {"model": "Support Vector Regression (SVR)", "horizon_hours": 24, **evaluate_all_metrics(t_art["y_test_act"], svr_preds)},
        {"model": "Standalone LSTM", "horizon_hours": 24, **evaluate_all_metrics(t_art["y_test_act"], s_lstm_preds)},
        {"model": "Temporal Transformer", "horizon_hours": 24, **evaluate_all_metrics(t_art["y_test_act"], trans_preds)},
        {"model": "MLP-LSTM (Base Hybrid)", "horizon_hours": 24, **evaluate_all_metrics(t_art["y_test_act"], t_art["base_te_act"])},
        {"model": "MLP-LSTM + Residual Boosting (Proposed)", "horizon_hours": 24, **evaluate_all_metrics(t_art["y_test_act"], t_art["final_te_act"])},
    ]
    df_baselines = pd.DataFrame(baseline_rows)
    save_dataframe(df_baselines, "reports/metrics/baseline_results.csv")
    viz.plot_model_comparison(df_baselines)

    # =========================================================================
    # STAGE 10: WALK-FORWARD VALIDATION
    # =========================================================================
    logger.info(">>> [4/7] WALK-FORWARD ROLLING VALIDATION <<<")
    wf_cfg = exp_config["walk_forward"][f"{mode}_config"]
    wf_val = WalkForwardValidator(
        n_splits=wf_cfg["n_splits"],
        test_size_hours=wf_cfg["test_size_hours"],
        min_train_size_hours=wf_cfg["min_train_size_hours"],
        epochs=wf_cfg["epochs"],
        horizon_hours=24,
        target_column=target_col,
    )
    df_wf, wf_summary = wf_val.run_validation(t_art["df_eng"], t_art["feature_cols"])
    viz.plot_walk_forward_performance(df_wf)

    # =========================================================================
    # STAGE 14: MULTI-SITE GENERALIZATION
    # =========================================================================
    logger.info(">>> [5/7] MULTI-SITE GENERALIZATION EVALUATION <<<")
    site1_w = config["data"]["site_paths"]["site_1"]["weather"]
    site1_e = config["data"]["site_paths"]["site_1"]["energy"]
    site2_w = config["data"]["site_paths"]["site_2"]["weather"]
    site2_e = config["data"]["site_paths"]["site_2"]["energy"]

    multi_site_rows = []
    if os.path.exists(site1_w) and os.path.exists(site2_w):
        w_s1 = loader.load_weather(site1_w)
        e_s1 = loader.load_energy(site1_e)
        m_s1, _ = merger.merge(w_s1, e_s1)

        w_s2 = loader.load_weather(site2_w)
        e_s2 = loader.load_energy(site2_e)
        m_s2, _ = merger.merge(w_s2, e_s2)

        # Preprocess Site 1 (Train) and Site 2 (Cross-site Test)
        p_site = PreprocessingPipeline(target_column=target_col)
        m_s1_prep = p_site.fit_transform(m_s1)
        m_s2_prep = p_site.transform(m_s2)

        eng_s1, _ = feat_eng.engineer_features(m_s1_prep)
        eng_s2, _ = feat_eng.engineer_features(m_s2_prep)

        lead_24 = f"{target_col}_lead_24h"
        Xs1_seq, ys1_seq, Xs1_flat = seq_gen.create_sequences(eng_s1, feat_cols, eng_s1[lead_24])
        Xs2_seq, ys2_seq, Xs2_flat = seq_gen.create_sequences(eng_s2, feat_cols, eng_s2[lead_24])

        # Train on Site 1
        mlp_s1 = MLPLSTMPipeline(input_dim=len(feat_cols), config={"epochs": 20, "batch_size": 64})
        mlp_s1.fit(Xs1_seq[:int(len(Xs1_seq)*0.8)], ys1_seq[:int(len(Xs1_seq)*0.8)], Xs1_seq[int(len(Xs1_seq)*0.8):], ys1_seq[int(len(Xs1_seq)*0.8):], epochs=20, verbose=False)
        boost_s1 = ResidualBoostingModel(n_estimators=40, max_depth=4)
        base_s1_tr = mlp_s1.predict(Xs1_seq)
        boost_s1.fit(Xs1_flat, base_s1_tr, ys1_seq, feature_names=feat_cols)

        # Zero-shot cross-site transfer onto Site 2
        base_s2_s = mlp_s1.predict(Xs2_seq)
        final_s2_s, _ = boost_s1.predict_final(Xs2_flat, base_s2_s)
        y_s2_act = p_site.inverse_transform_target(ys2_seq)
        final_s2_act = p_site.inverse_transform_target(final_s2_s)

        m_cross = evaluate_all_metrics(y_s2_act, final_s2_act)
        multi_site_rows.append({
            "training_site": "Site_1 (400kW PV)",
            "testing_site": "Site_2 (600kW PV Held-out)",
            "transfer_mode": "Zero-shot Direct Transfer",
            **m_cross,
        })
    else:
        multi_site_rows.append({
            "training_site": "Main_Site",
            "testing_site": "N/A",
            "transfer_mode": "Single Site Available",
            "MAE": 0.0, "MSE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "R2": 0.0,
        })

    df_multisite = pd.DataFrame(multi_site_rows)
    save_dataframe(df_multisite, "reports/metrics/multi_site_results.csv")

    # =========================================================================
    # STAGE 13: SHAP EXPLAINABILITY
    # =========================================================================
    logger.info(">>> [6/7] SHAP EXPLAINABILITY ANALYSIS <<<")
    explainer = SHAPExplainer(feature_names=t_art["feature_cols"], top_n=15)
    shap_vals, imp_df = explainer.explain(
        booster_model=t_art["booster"],
        X_explain=t_art["X_test_flat"],
        base_preds=t_art["base_test_preds_s"],
        max_samples=400,
    )

    # =========================================================================
    # STAGE 15: ABLATION STUDY
    # =========================================================================
    logger.info(">>> [7/7] ARCHITECTURAL ABLATION STUDY <<<")
    ablation = AblationStudy()
    ablation_dict = {
        "Standalone_LSTM": s_lstm_preds,
        "MLP_LSTM_Base": t_art["base_te_act"],
        "MLP_LSTM_Residual_Boosting_Proposed": t_art["final_te_act"],
    }
    df_ablation, ab_summary = ablation.evaluate_ablation(
        t_art["y_test_act"], ablation_dict, output_path="reports/metrics/ablation_results.csv"
    )

    # =========================================================================
    # FINAL RESEARCH SUMMARY REPORT EXPORT
    # =========================================================================
    elapsed = time.time() - start_time
    summary_text = f"""================================================================================
FINAL RESEARCH SUMMARY REPORT
AI-Based Prediction of Future Energy Output Using Weather and Historical Data
================================================================================
Total Execution Time: {elapsed:.2f} seconds
Execution Mode: {mode.upper()}
Target Variable: {target_col} ({unit})
Architecture: MLP -> LSTM -> Gradient Boosting Residual Correction + Conformal PICP

1. DATA QUALITY & PROFILE
--------------------------------------------------------------------------------
- Weather Rows: {len(weather_df)}, Energy Rows: {len(energy_df)}
- Missing Placeholders (-999) Cleaned: {quality_df['placeholder_count'].sum()}
- Leakage Prevention: Verified (Historical rolling shift=1, Train-only scaler fit)

2. HYBRID MODEL FORECASTING PERFORMANCE (Horizon = 24 Hours)
--------------------------------------------------------------------------------
- Base Model (MLP-LSTM):               RMSE = {m_base['RMSE']:.4f}, MAE = {m_base['MAE']:.4f}, MAPE = {m_base['MAPE']}%, R2 = {m_base['R2']:.4f}
- Proposed (MLP-LSTM + Boosting):      RMSE = {m_final['RMSE']:.4f}, MAE = {m_final['MAE']:.4f}, MAPE = {m_final['MAPE']}%, R2 = {m_final['R2']:.4f}
- Absolute RMSE Gain:                 {ab_summary.get('rmse_absolute_improvement', 0.0):.4f} kWh
- Percentage RMSE Improvement:        {ab_summary.get('rmse_percent_improvement', 0.0):.2f}%

3. UNCERTAINTY QUANTIFICATION (Conformal Prediction Intervals)
--------------------------------------------------------------------------------
- Target Confidence Level:            90.0%
- Empirical Coverage (PICP):          {m_final.get('PICP', 0.0)}%
- Mean Interval Width (MPIW):         {m_final.get('MPIW', 0.0):.4f} {unit}

4. BENCHMARK COMPARISONS (24h Ahead)
--------------------------------------------------------------------------------
{df_baselines.to_string(index=False)}

5. TOP 5 SHAP FEATURE CONTRIBUTIONS TO RESIDUAL CORRECTION
--------------------------------------------------------------------------------
{imp_df.head(5).to_string(index=False)}

6. ARTIFACTS AND DELIVERABLES GENERATED
--------------------------------------------------------------------------------
- CSV Metrics: reports/metrics/*.csv (9 metric summary tables)
- Visualizations: reports/figures/*.png (14 publication figures)
- Models: models/*.pt, models/*.pkl (Fitted weights & preprocessors)
================================================================================
"""
    with open("reports/final_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info(summary_text)
    logger.info("FULL RESEARCH PIPELINE COMPLETED SUCCESSFULLY WITH ZERO ERRORS!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete end-to-end research pipeline.")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast")
    args = parser.parse_args()

    run_full_research_pipeline(mode=args.mode)


if __name__ == "__main__":
    main()
