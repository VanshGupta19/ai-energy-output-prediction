"""
Hybrid MLP-LSTM Neural Network Architecture.
Combines Dense MLP tabular representation encoding with LSTM temporal sequence modeling.
"""

import os
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("mlp_lstm")


class MLPLSTMNet(nn.Module):
    """
    PyTorch Hybrid Architecture:
    Input (Batch, Lookback, Features)
    ↓
    TimeDistributed MLP Feature Encoder (Dense -> ReLU -> Dropout)
    ↓
    LSTM Sequential Dynamics (LSTM Layer(s))
    ↓
    Fully-Connected Regression Head (Dense -> ReLU -> Dropout -> Linear Output)
    ↓
    Base Energy Prediction
    """

    def __init__(
        self,
        input_dim: int,
        mlp_hidden_dims: List[int] = [64, 32],
        lstm_units: int = 64,
        lstm_layers: int = 1,
        dense_output_dims: List[int] = [32, 16],
        dropout: float = 0.2,
    ):
        super(MLPLSTMNet, self).__init__()
        self.input_dim = input_dim

        # 1. MLP Feature Encoder (processes features at each timestep)
        mlp_layers = []
        in_dim = input_dim
        for h_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, h_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.mlp_encoder = nn.Sequential(*mlp_layers)
        lstm_input_dim = in_dim

        # 2. LSTM Sequence Layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 3. Dense Regression Head
        head_layers = []
        curr_dim = lstm_units
        for d_dim in dense_output_dims:
            head_layers.append(nn.Linear(curr_dim, d_dim))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.Dropout(dropout))
            curr_dim = d_dim
        head_layers.append(nn.Linear(curr_dim, 1)) # Scalar prediction output
        self.regression_head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            Tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape

        # Pass through TimeDistributed MLP encoder: flatten time, encode, unflatten
        x_reshaped = x.view(-1, self.input_dim)
        mlp_out = self.mlp_encoder(x_reshaped)
        mlp_out = mlp_out.view(batch_size, seq_len, -1)

        # Pass through LSTM
        lstm_out, (h_n, _) = self.lstm(mlp_out)
        # Use final hidden state from top LSTM layer
        last_hidden = h_n[-1]

        # Pass through regression head
        prediction = self.regression_head(last_hidden)
        return prediction


class MLPLSTMPipeline:
    """
    Orchestrates training, validation, early stopping, and inference for the MLP-LSTM hybrid model.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
    ):
        self.input_dim = input_dim
        self.config = config or {}
        self.seed = random_seed
        set_seed(self.seed)

        # Hyperparameters
        self.mlp_hidden_dims = self.config.get("mlp_hidden_dims", [64, 32])
        self.lstm_units = self.config.get("lstm_units", 64)
        self.lstm_layers = self.config.get("lstm_layers", 1)
        self.dense_output_dims = self.config.get("dense_output_dims", [32, 16])
        self.dropout = self.config.get("dropout", 0.2)
        self.lr = self.config.get("learning_rate", 0.001)
        self.weight_decay = self.config.get("weight_decay", 1e-5)
        self.batch_size = self.config.get("batch_size", 64)
        self.epochs = self.config.get("epochs", 100)
        self.early_stopping_patience = self.config.get("early_stopping_patience", 12)
        self.lr_reduce_patience = self.config.get("lr_reduce_patience", 5)
        self.lr_reduce_factor = self.config.get("lr_reduce_factor", 0.5)
        self.min_lr = self.config.get("min_lr", 1e-5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPLSTMNet(
            input_dim=self.input_dim,
            mlp_hidden_dims=self.mlp_hidden_dims,
            lstm_units=self.lstm_units,
            lstm_layers=self.lstm_layers,
            dense_output_dims=self.dense_output_dims,
            dropout=self.dropout,
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.best_model_state_: Optional[Dict[str, Any]] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train hybrid model with early stopping and learning rate scheduler.
        """
        max_epochs = epochs or self.epochs
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.lr_reduce_factor, patience=self.lr_reduce_patience, min_lr=self.min_lr
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_x = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting MLP-LSTM training on {self.device} for up to {max_epochs} epochs...")

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            train_loss_accum = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item() * len(batch_x)

            epoch_train_loss = train_loss_accum / len(X_train)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_x)
                epoch_val_loss = self.criterion(val_pred, val_y).item()

            scheduler.step(epoch_val_loss)
            self.history_["train_loss"].append(epoch_train_loss)
            self.history_["val_loss"].append(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                self.best_model_state_ = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == max_epochs):
                logger.info(
                    f"Epoch [{epoch:03d}/{max_epochs:03d}] - Train Loss: {epoch_train_loss:.6f}, "
                    f"Val Loss: {epoch_val_loss:.6f} {'(Best)' if patience_counter == 0 else ''}"
                )

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best checkpoint.")
                break

        # Load best weights
        if self.best_model_state_ is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state_.items()})

        return self.history_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for sequential inputs.
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(x_tensor).cpu().numpy().flatten()
        return preds

    def save(self, file_path: str) -> None:
        """Save model checkpoint to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "config": self.config,
        }, file_path)
        logger.info(f"Saved MLP-LSTM model checkpoint to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "MLPLSTMPipeline":
        """Load model checkpoint from disk."""
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        pipeline = cls(
            input_dim=checkpoint["input_dim"],
            config=checkpoint["config"],
        )
        pipeline.model.load_state_dict(checkpoint["model_state_dict"])
        return pipeline
