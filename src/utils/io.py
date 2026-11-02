"""
File and Artifact I/O utilities.
"""

import os
import json
import pickle
from typing import Any, Dict, Optional
import pandas as pd
import yaml
import torch
from src.utils.logger import get_logger

logger = get_logger("io_utils")


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        file_path: Path to YAML file.

    Returns:
        Dictionary with parsed configuration.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary to YAML file.

    Args:
        data: Data to save.
        file_path: Destination path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved YAML to {file_path}")


def save_json(data: Any, file_path: str, indent: int = 4) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to serialize.
        file_path: Destination path.
        indent: JSON indentation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info(f"Saved JSON to {file_path}")


def load_json(file_path: str) -> Any:
    """
    Load data from JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Loaded object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        df: Pandas DataFrame.
        file_path: Destination path.
        index: Whether to write index.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    df.to_csv(file_path, index=index)
    logger.info(f"Saved DataFrame ({df.shape[0]} rows, {df.shape[1]} cols) to {file_path}")


def load_dataframe(file_path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.

    Args:
        file_path: Path to CSV file.
        parse_dates: List of date columns to parse.

    Returns:
        Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path, parse_dates=parse_dates)


def save_model(model: Any, file_path: str) -> None:
    """
    Save PyTorch or Scikit-learn/XGBoost/LightGBM model.

    Args:
        model: Trained model object.
        file_path: Destination path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), file_path)
    else:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
    logger.info(f"Saved model to {file_path}")


def load_model(file_path: str, model_class: Optional[Any] = None, *args, **kwargs) -> Any:
    """
    Load PyTorch or Scikit-learn/XGBoost/LightGBM model.

    Args:
        file_path: Path to saved model file.
        model_class: PyTorch model class if loading PyTorch weights.

    Returns:
        Loaded model object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    if model_class is not None:
        model = model_class(*args, **kwargs)
        state_dict = torch.load(file_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        return model
    else:
        with open(file_path, "rb") as f:
            return pickle.load(f)
