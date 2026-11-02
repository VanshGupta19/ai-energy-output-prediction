"""
Rolling-Origin Walk-Forward Validation Script.
Usage: python scripts/run_walk_forward.py --mode fast
"""

import argparse
import os
import sys

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.io import load_yaml
from src.data.loader import DataLoader
from src.data.merger import DataMerger
from src.features.feature_engineering import FeatureEngineer
from src.evaluation.walk_forward import WalkForwardValidator
from src.visualization.plots import Visualizer

logger = setup_logger("run_walk_forward")


def main():
    parser = argparse.ArgumentParser(description="Run rolling-origin walk-forward validation.")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    config = load_yaml("config/config.yaml")
    exp_config = load_yaml("config/experiment_config.yaml")
    set_seed(config.get("random_seed", 42))

    mode_cfg = exp_config["walk_forward"][f"{args.mode}_config"]
    target_col = config["data"]["target_column"]

    loader = DataLoader(config)
    weather_df = loader.load_weather(config["data"]["weather_path"])
    energy_df = loader.load_energy(config["data"]["energy_path"])
    merger = DataMerger()
    merged_df, _ = merger.merge(weather_df, energy_df)

    feat_eng = FeatureEngineer(target_column=target_col, horizon_hours=args.horizon)
    df_eng, feature_cols = feat_eng.engineer_features(merged_df)

    validator = WalkForwardValidator(
        n_splits=mode_cfg["n_splits"],
        test_size_hours=mode_cfg["test_size_hours"],
        min_train_size_hours=mode_cfg["min_train_size_hours"],
        epochs=mode_cfg["epochs"],
        horizon_hours=args.horizon,
        target_column=target_col,
    )
    results_df, summary = validator.run_validation(df_eng, feature_cols)

    viz = Visualizer()
    viz.plot_walk_forward_performance(results_df)

    logger.info(f"Walk-forward validation completed in {args.mode.upper()} mode.")


if __name__ == "__main__":
    main()
