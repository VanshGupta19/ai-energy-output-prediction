"""
Seed control for strict experimental reproducibility.
"""

import os
import random
import numpy as np
import torch
from src.utils.logger import get_logger

logger = get_logger("seed_control")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, PyTorch (CPU and CUDA), and environment.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set deterministically to {seed}")
