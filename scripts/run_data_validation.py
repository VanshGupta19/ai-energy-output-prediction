"""
Data Validation and Quality Audit Script.
Usage: python scripts/run_data_validation.py --weather data/raw/weather/nasa_power_weather.csv --energy data/raw/energy/solar_energy_generation.csv
"""

import argparse
import os
import sys

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.io import load_yaml
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.merger import DataMerger
from src.visualization.plots import Visualizer

logger = setup_logger("run_data_validation")


def main():
    parser = argparse.ArgumentParser(description="Audit and validate meteorological and energy datasets.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--weather", type=str, default=None, help="Path to raw weather CSV")
    parser.add_argument("--energy", type=str, default=None, help="Path to raw energy CSV")
    args = parser.parse_args()

    config = load_yaml(args.config)
    weather_path = args.weather or config["data"]["weather_path"]
    energy_path = args.energy or config["data"]["energy_path"]
    target_col = config["data"]["target_column"]
    allow_synthetic = config["data"].get("allow_synthetic_demo", False)

    logger.info("=== STEP 1: LOADING DATASETS ===")
    loader = DataLoader(config)
    weather_df = loader.load_weather(weather_path)
    energy_df = loader.load_energy(energy_path)

    logger.info("=== STEP 2: DATA VALIDATION & QUALITY AUDIT ===")
    validator = DataValidator(target_column=target_col, allow_synthetic_demo=allow_synthetic)
    is_valid, quality_df, summary = validator.validate(
        weather_df, energy_df, output_report_path="reports/data_quality/data_quality_report.csv"
    )

    logger.info("=== STEP 3: TEMPORAL MERGING ===")
    merger = DataMerger()
    merged_df, merge_summary = merger.merge(weather_df, energy_df)

    # Save merged data
    merged_out = config["data"]["merged_output_path"]
    os.makedirs(os.path.dirname(merged_out), exist_ok=True)
    merged_df.to_csv(merged_out, index=False)
    logger.info(f"Saved merged dataset to: {merged_out}")

    logger.info("=== STEP 4: GENERATING DATA QUALITY PLOTS ===")
    viz = Visualizer()
    viz.plot_missing_values(quality_df)
    viz.plot_historical_energy(merged_df, target_col=target_col, unit=config["data"].get("target_unit", "kWh"))
    raw_weather_cols = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "PS", "WS10M", "WS50M", "CLOUD_AMT"]
    viz.plot_weather_distributions(merged_df, [c for c in raw_weather_cols if c in merged_df.columns])

    logger.info("Data validation and preparation completed successfully!")


if __name__ == "__main__":
    main()
