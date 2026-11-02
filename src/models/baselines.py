"""
Comparative Baseline Forecasting Models.
Includes:
1. Seasonal Naive / ARIMA baseline
2. Support Vector Regression (SVR)
3. Standalone LSTM (without MLP encoder)
4. Transformer / Temporal Attention baseline
"""

import math
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVR
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("baselines")


# ==========================================
# 1. Seasonal Naive / ARIMA Baselines
# ==========================================

class SeasonalNaiveBaseline:
    """
    Seasonal persistence benchmark: predicts value from the same hour in previous cycle (e.g. 24h prior).
    """

    def __init__(self, season_length: int = 24):
        self.season_length = season_length

    def fit(self, y_train: np.ndarray):
        """Fit seasonal baseline (stores last observed seasonal profile)."""
        self.last_cycle_ = y_train[-self.season_length:]
        return self

    def predict(self, y_history: np.ndarray, n_steps: int) -> np.ndarray:
        """Predict using seasonal repeats."""
        repeats = math.ceil(n_steps / self.season_length)
        preds = np.tile(self.last_cycle_, repeats)[:n_steps]
        return preds


class ARIMABaseline:
    """
    Statistical Autoregressive Integrated Moving Average with seasonality.
    """

    def __init__(self, order: Tuple[int, int, int] = (2, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 0, 0, 24)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model_ = None

    def fit(self, y_train: np.ndarray, max_samples: int = 1500):
        """Fit ARIMA model on recent training series slice."""
        from statsmodels.tsa.arima.model import ARIMA
        y_slice = y_train[-max_samples:] if len(y_train) > max_samples else y_train
        try:
            model = ARIMA(y_slice, order=self.order)
            self.fitted_model_ = model.fit()
            logger.info("Fitted ARIMA baseline successfully.")
        except Exception as e:
            logger.warning(f"ARIMA fit failed with error: {e}. Using mean fallback.")
            self.fitted_model_ = None
            self.fallback_mean_ = np.mean(y_slice)
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        """Forecast n steps ahead."""
        if self.fitted_model_ is not None:
            try:
                return self.fitted_model_.forecast(steps=n_steps)
            except Exception:
                pass
        return np.full(n_steps, getattr(self, "fallback_mean_", 0.0))


# ==========================================
# 2. Support Vector Regression (SVR)
# ==========================================

class SVRBaseline:
    """
    Support Vector Regression baseline with RBF kernel on tabular features.
    """

    def __init__(self, C: float = 10.0, epsilon: float = 0.1, gamma: str = "scale"):
        self.model = SVR(C=C, epsilon=epsilon, gamma=gamma)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, max_samples: int = 5000):
        """Fit SVR model."""
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), size=max_samples, replace=False)
            indices.sort()
            X_fit, y_fit = X_train[indices], y_train[indices]
        else:
            X_fit, y_fit = X_train, y_train

        logger.info(f"Fitting SVR baseline on {len(X_fit)} samples...")
        self.model.fit(X_fit, y_fit)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict SVR."""
        return np.maximum(0.0, self.model.predict(X))


# ==========================================
# 3. Standalone LSTM (No MLP Encoder)
# ==========================================

class StandaloneLSTMNet(nn.Module):
    """
    Standard LSTM network directly ingesting sequential tabular input without MLP encoder.
    """

    def __init__(self, input_dim: int, lstm_units: int = 64, lstm_layers: int = 2, dropout: float = 0.2):
        super(StandaloneLSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        last_h = h_n[-1]
        return self.fc(last_h)


class StandaloneLSTMPipeline:
    """Trainer for standalone LSTM baseline."""

    def __init__(self, input_dim: int, lstm_units: int = 64, lstm_layers: int = 2, dropout: float = 0.2, lr: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StandaloneLSTMNet(input_dim, lstm_units, lstm_layers, dropout).to(self.device)
        self.lr = lr
        self.criterion = nn.MSELoss()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50, batch_size: int = 64):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_x = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        best_val = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = self.criterion(pred, by)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.model(val_x), val_y).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(xt).cpu().numpy().flatten()


# ==========================================
# 4. Transformer / Self-Attention Baseline
# ==========================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequential transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerForecaster(nn.Module):
    """
    Lightweight Temporal Transformer baseline.
    Linear projection -> Positional Encoding -> Multi-Head Self-Attention Transformer Encoder -> Linear Head.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super(TransformerForecaster, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer_encoder(x_pos)
        # Pool across sequence or use last timestep
        last_step = encoded[:, -1, :]
        return self.decoder(last_step)


class TransformerPipeline:
    """Trainer for Transformer baseline."""

    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2, lr: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerForecaster(input_dim, d_model, nhead, num_layers).to(self.device)
        self.lr = lr
        self.criterion = nn.MSELoss()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50, batch_size: int = 64):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_x = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        best_val = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = self.criterion(pred, by)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.model(val_x), val_y).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(xt).cpu().numpy().flatten()
