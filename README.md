# AI-Based Prediction of Future Energy Output Using Weather and Historical Data

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/pytest-20%20passed-success.svg)](https://pytest.org/)

---

## 1. Project Overview & Research Problem

Renewable energy generation, particularly solar photovoltaic (PV) and wind turbine power, exhibits strong non-linear and time-varying dependencies on meteorological variables:
* **Solar Variables**: Surface downwelling solar irradiance (`ALLSKY_SFC_SW_DWN`), air temperature at 2m (`T2M`), cloud amount fraction (`CLOUD_AMT`), relative humidity (`RH2M`), surface pressure (`PS`).
* **Wind Variables**: Wind speed at 10m & 50m (`WS10M`, `WS50M`), wind direction (`WD10M`, `WD50M`), air density (`RHOA`), specific humidity (`QV10M`).

Because meteorological dynamics fluctuate continuously, raw renewable generation is intermittent and volatile. This project implements a **clean, leak-free, reproducible, end-to-end research platform** for short-term renewable energy forecasting ($H \in \{1, 3, 6, 12, 24\}$ hours ahead) with **conformal uncertainty quantification (PICP)** and **game-theoretic SHAP explainability**.

### Scientific & Methodological Integrity Rules
* **NO CNN**: Because data is tabular and sequential time-series (not image/spatial grids), Convolutional Neural Networks are strictly avoided.
* **Strict Distinction Between Irradiance and Energy**: Solar irradiance ($kW/m^2$) is never falsely labeled as generated power ($kWh$). Actual energy generation datasets are required and validated.
* **Zero Data Leakage**: Scalers are fitted **strictly on training partitions**; rolling statistics only look backwards ($\text{shift}(1)$); chronological splitting is strictly preserved (no random shuffling).

---

## 2. Core Hybrid Architecture

$$\text{Final Forecast } \hat{y}(t + H) = \hat{y}_{\text{MLP-LSTM}}(t + H) + \hat{\epsilon}_{\text{Residual}}(t + H)$$

```
                               ┌────────────────────────────────────────────────────────┐
                               │                 Input Data at time t                   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                      ┌────────────────────────────────────┴───────────────────────────────────┐
                      │                                                                        │
                      ▼                                                                        ▼
         ┌─────────────────────────┐                                              ┌─────────────────────────┐
         │ Sliding Lookback Tensor │                                              │ Flat Tabular State at t │
         │   (Batch, Window, F)    │                                              │      (Batch, F)         │
         └────────────┬────────────┘                                              └────────────┬────────────┘
                      │                                                                        │
                      ▼                                                                        │
         ┌─────────────────────────┐                                                           │
         │ TimeDistributed MLP     │                                                           │
         │ Feature Encoder (Dense) │                                                           │
         └────────────┬────────────┘                                                           │
                      │                                                                        │
                      ▼                                                                        │
         ┌─────────────────────────┐                                                           │
         │ Sequential LSTM Network │                                                           │
         │  (Temporal Recurrence)  │                                                           │
         └────────────┬────────────┘                                                           │
                      │                                                                        │
                      ▼                                                                        │
         ┌─────────────────────────┐                                                           │
         │ Dense Regression Head   │                                                           │
         └────────────┬────────────┘                                                           │
                      │                                                                        │
                      ▼                                                                        │
         ┌─────────────────────────┐                                                           │
         │ Base Forecast           │ ────────────────────────────────────────────────────────┐ │
         │   y_hat_MLP-LSTM        │                                                         │ │
         └────────────┬────────────┘                                                         │ │
                      │                                                                      ▼ ▼
                      │                                                        ┌─────────────────────────────┐
                      │                                                        │ Gradient Boosting Residual  │
                      │                                                        │  Corrector (XGBoost/LGBM)   │
                      │                                                        └─────────────┬───────────────┘
                      │                                                                      │
                      │                                                                      ▼
                      │                                                        ┌─────────────────────────────┐
                      │                                                        │ Predicted Residual Error    │
                      │                                                        │       epsilon_hat           │
                      │                                                        └─────────────┬───────────────┘
                      │                                                                      │
                      └──────────────────────────────────┬───────────────────────────────────┘
                                                         │
                                                         ▼
                                       ┌───────────────────────────────────┐
                                       │ Final Hybrid Forecast             │
                                       │ y_hat = y_base + epsilon_hat      │
                                       └─────────────────┬─────────────────┘
                                                         │
                                 ┌───────────────────────┴───────────────────────┐
                                 │                                               │
                                 ▼                                               ▼
               ┌───────────────────────────────────┐           ┌───────────────────────────────────┐
               │ Conformal Prediction Intervals    │           │ SHAP Attribution & Explainability │
               │   [Lower_90, Upper_90] (PICP)     │           │   (Global Summary & Feature Imp)  │
               └───────────────────────────────────┘           └───────────────────────────────────┘
```

1. **MLP Feature Encoder**: Encodes non-linear simultaneous interactions among simultaneous weather variables and physical indexes.
2. **LSTM Recurrent Sequence Layer**: Models multi-step sequential lookback dependencies ($t-23 \dots t$).
3. **Gradient Boosting Residual Corrector (XGBoost / LightGBM fallback)**: Learns the systematic residual error $\text{residual} = y - \hat{y}_{\text{MLP-LSTM}}$ using tabular feature representations and base prediction inputs available at forecast time.
4. **Conformal Uncertainty Quantification**: Computes finite-sample calibrated 90% prediction intervals $[L, U]$ and evaluates Prediction Interval Coverage Probability (PICP).
5. **TreeSHAP Explainability**: Computes game-theoretic Shapley values on the residual correction booster to identify primary drivers of model adjustments.

---

## 3. Project Directory Structure

```
energy_forecasting_project/
├── config/
│   ├── config.yaml                     # Primary pipeline & data config
│   ├── model_config.yaml               # MLP, LSTM, Boosting, Baseline hyperparameters
│   └── experiment_config.yaml          # Horizons, walk-forward folds, ablation settings
├── data/
│   ├── raw/
│   │   ├── weather/                    # NASA POWER weather hourly CSVs
│   │   ├── energy/                     # Solar/wind actual generation CSVs
│   │   ├── site_1/                     # Multi-site: Site 1 data
│   │   └── site_2/                     # Multi-site: Site 2 data
│   ├── interim/                        # Merged weather-energy time series
│   └── processed/                      # Preprocessed & scaled datasets
├── src/
│   ├── data/                           # Loader, validator, merger, preprocessor
│   ├── features/                       # Lag generator, temporal encodings, leakage checker
│   ├── models/                         # MLP-LSTM, Residual Booster, Baselines, Uncertainty
│   ├── evaluation/                     # Metrics, Sequence generator, Walk-forward, Ablation
│   ├── explainability/                 # TreeSHAP explainer & visualization
│   ├── visualization/                  # 14 publication-grade figures
│   └── utils/                          # Logging, seed control, I/O
├── scripts/
│   ├── run_data_validation.py          # Data profiling & validation script
│   ├── run_training.py                 # Hybrid model training & evaluation script
│   ├── run_baselines.py                # Comparative baseline experiments
│   ├── run_walk_forward.py             # Walk-forward cross-validation script
│   ├── run_ablation.py                 # Ablation study execution
│   ├── run_shap.py                     # Explainability figure generation
│   ├── run_smoke_test.py               # Fast lightweight smoke test (<30s)
│   └── run_full_pipeline.py            # Master CLI orchestrating all stages end-to-end
├── tests/                              # Pytest test suite (20 unit & integration tests)
├── notebooks/                          # 3 Jupyter research walkthrough notebooks
├── models/                             # Saved model checkpoints (*.pt, *.pkl)
├── reports/
│   ├── figures/                        # 14 publication figures + SHAP plots
│   ├── metrics/                        # 9 CSV experimental summary tables
│   └── data_quality/                   # Data validation report
├── results/                            # Multi-horizon forecast result folders
├── README.md                           # Documentation & user manual
├── REQUIREMENTS_CHECKLIST.md           # Stage-by-stage requirement verification matrix
├── pyproject.toml
└── requirements.txt
```

---

## 4. Installation & Environment Setup

### 4.1 Prerequisites
* Python 3.9, 3.10, or 3.11
* CUDA-compatible GPU (optional, automatically used if available)

### 4.2 Installation
```bash
# Clone or navigate to the workspace directory
cd pbl_project_sem7

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Dataset Configuration & Preparation

### 5.1 Weather Dataset Format (NASA POWER Standard)
Place weather hourly CSVs in `data/raw/weather/nasa_power_weather.csv` containing:
* Date/Time identifiers: `YEAR`, `MO`, `DY`, `HR` (or unified `timestamp`)
* Weather variables: `ALLSKY_SFC_SW_DWN`, `T2M`, `RH2M`, `PS`, `WS10M`, `WD10M`, `WS50M`, `WD50M`, `RHOA`, `QV10M`, `CLOUD_AMT`.

### 5.2 Energy Generation Dataset Format
Place energy generation CSVs in `data/raw/energy/solar_energy_generation.csv` containing:
* `timestamp`: Formatted datetime (e.g. `YYYY-MM-DD HH:MM:SS`)
* `energy_output_kwh`: Actual measured generation in kWh, kW, or MW.
* `unit`: Energy unit indicator (`kWh`, `kW`, or `MW`).

> **Target Validation Rule**: If no energy generation target is provided, the data validator explicitly halts execution. Synthetic demo mode (`allow_synthetic_demo: true`) is only permitted for testing software pipelines in isolation.

---

## 6. How to Run the Project

### 6.1 Run Automated Tests
```bash
# Run complete pytest test suite (20 unit and integration tests)
python -m pytest tests/ -v
```

### 6.2 Run Fast Smoke Test
```bash
python scripts/run_smoke_test.py
```

### 6.3 Run Data Validation & Audit
```bash
python scripts/run_data_validation.py
```

### 6.4 Train Proposed Hybrid Model
```bash
# Train hybrid MLP-LSTM + Residual Booster for Horizon = 24 hours
python scripts/run_training.py --horizon 24
```

### 6.5 Run Comparative Baseline Benchmarks
```bash
# Evaluates Seasonal Naive, SVR, Standalone LSTM, Transformer, and Proposed Hybrid
python scripts/run_baselines.py --horizon 24
```

### 6.6 Run Rolling-Origin Walk-Forward Validation
```bash
# Fast mode (3 folds) or Full mode (5 folds)
python scripts/run_walk_forward.py --mode full --horizon 24
```

### 6.7 Run Architectural Ablation Study
```bash
python scripts/run_ablation.py --horizon 24
```

### 6.8 Generate SHAP Explainability Plots
```bash
python scripts/run_shap.py --horizon 24
```

### 6.9 Run Master Full Pipeline End-to-End
```bash
# Orchestrates all stages, models, horizons, walk-forward folds, and figures in one command
python scripts/run_full_pipeline.py --mode full
```

---

## 7. Experimental Results & Findings

### 7.1 Benchmark Comparison (24-Hour Ahead Forecast)
| Model | Horizon | MAE ($kWh$) | RMSE ($kWh$) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Seasonal Naive (24h) | 24h | 117.40 | 136.95 | 100.00% | -1.3287 |
| Support Vector Regression (SVR) | 24h | 21.06 | 28.99 | 86.96% | 0.8956 |
| Temporal Transformer | 24h | 10.01 | 15.93 | 18.48% | 0.9685 |
| Standalone LSTM | 24h | 8.85 | 15.76 | 16.97% | 0.9692 |
| MLP-LSTM (Base Hybrid) | 24h | 11.28 | 16.78 | 21.98% | 0.9650 |
| **MLP-LSTM + Residual Boosting (Proposed)** | **24h** | **8.32** | **15.65** | **17.15%** | **0.9696** |

### 7.2 Multi-Horizon Forecasting Accuracy
| Horizon | Base MLP-LSTM RMSE ($kWh$) | Proposed Hybrid RMSE ($kWh$) | Empirical PICP (90% Interval) | Mean Interval Width ($kWh$) |
| :---: | :---: | :---: | :---: | :---: |
| **1 Hour Ahead** | 16.95 | **15.82** | 96.94% | 63.81 |
| **3 Hours Ahead** | 16.89 | **15.79** | 96.94% | 63.81 |
| **6 Hours Ahead** | 16.82 | **15.74** | 96.94% | 63.81 |
| **12 Hours Ahead** | 16.80 | **15.70** | 96.94% | 63.81 |
| **24 Hours Ahead** | 16.78 | **15.65** | 97.09% | 63.61 |

### 7.3 Ablation Study Insights
* Adding Gradient Boosting Residual Correction to MLP-LSTM yielded a **$1.1324\text{ kWh}$ absolute reduction in RMSE** (**$6.75\%$ relative improvement**) and a **$2.9527\text{ kWh}$ reduction in MAE** (**$26.17\%$ relative improvement**).
* Conformal prediction intervals achieved **$97.09\%$ empirical coverage** against a $90\%$ nominal target.

---

## 8. Publication-Quality Figures Generated

All figures are automatically generated at 300 DPI in `reports/figures/`:
1. `01_historical_energy_output.png` — Historical time-series trace
2. `02_weather_feature_distributions.png` — Histograms and KDE of all meteorological variables
3. `03_missing_value_report.png` — Missingness & placeholder audit
4. `04_correlation_heatmap.png` — Pearson correlation matrix across top predictors
5. `05_actual_vs_predicted.png` — Scatter agreement plot with $y=x$ ideal line
6. `06_time_series_forecast_comparison.png` — Time-series forecast overlay
7. `07_residual_distribution.png` — Residual error distribution
8. `08_residual_over_time.png` — Residual errors across chronological time
9. `09_model_comparison_bar_chart.png` — RMSE and $R^2$ comparative benchmark bar chart
10. `10_forecast_horizon_performance.png` — Multi-horizon error progression curve
11. `11_walk_forward_performance.png` — Cross-validation performance across rolling folds
12. `12_prediction_intervals.png` — 90% Conformal uncertainty interval shading
13. `shap/shap_feature_importance.png` — Mean absolute SHAP feature importance bar plot
14. `shap/shap_summary_plot.png` — Global SHAP beeswarm summary plot
15. `shap/shap_local_explanation.png` — Instance-level local feature contribution waterfall

---

## 9. Limitations & Research Notes

1. **Physical Power Bounds**: Generation is physically non-negative ($P \ge 0$). Predictions and lower bounds are automatically truncated at zero.
2. **Nighttime Zero Generation**: Solar PV generation is zero during night hours ($G = 0$). Standard MAPE causes division by zero; this codebase utilizes a robust, zero-safe, masked MAPE implementation ($\epsilon$-stabilized).
3. **Multi-Site Transfer**: Zero-shot cross-site transfer between sites with differing capacity ratings ($400\text{ kW}$ vs $600\text{ kW}$) requires capacity normalization or fine-tuning for optimal transfer accuracy.
