from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    def __init__(
        self, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_data: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x_data)
        return self.head(output[:, -1, :])


@dataclass(frozen=True)
class LSTMArtifacts:
    model: LSTMForecaster
    scaler: StandardScaler
    window_size: int
    metadata: dict[str, float | int]


def _build_sequences(
    values: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    x_data, y_data = [], []
    for idx in range(window_size, len(values)):
        x_data.append(values[idx - window_size : idx])
        y_data.append(values[idx])
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)


def train_lstm(
    train_series: pd.Series,
    window_size: int = 12,
    max_epochs: int = 250,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
    patience: int = 25,
) -> LSTMArtifacts:
    values = train_series.to_numpy(dtype=np.float32).reshape(-1, 1)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values).flatten()

    x_data, y_data = _build_sequences(scaled_values, window_size=window_size)
    if len(x_data) < 24:
        raise ValueError("Not enough samples for LSTM sequence training.")

    split_idx = int(0.8 * len(x_data))
    x_train, y_train = x_data[:split_idx], y_data[:split_idx]
    x_valid, y_valid = x_data[split_idx:], y_data[split_idx:]

    train_ds = TensorDataset(
        torch.tensor(x_train).unsqueeze(-1), torch.tensor(y_train).unsqueeze(-1)
    )
    valid_ds = TensorDataset(
        torch.tensor(x_valid).unsqueeze(-1), torch.tensor(y_valid).unsqueeze(-1)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_state = None
    best_valid_loss = float("inf")
    stale_epochs = 0
    epochs_trained = 0

    for epoch in range(max_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                valid_losses.append(float(criterion(pred, y_batch).item()))

        avg_valid_loss = float(np.mean(valid_losses))
        epochs_trained = epoch + 1

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("LSTM training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    return LSTMArtifacts(
        model=model,
        scaler=scaler,
        window_size=window_size,
        metadata={
            "epochs_trained": epochs_trained,
            "best_valid_mse": round(best_valid_loss, 5),
        },
    )


def recursive_lstm_forecast(
    artifacts: LSTMArtifacts,
    train_series: pd.Series,
    forecast_horizon: int,
) -> np.ndarray:
    model = artifacts.model
    scaler = artifacts.scaler
    window_size = artifacts.window_size
    device = next(model.parameters()).device

    history_scaled = (
        scaler.transform(train_series.to_numpy(dtype=np.float32).reshape(-1, 1))
        .flatten()
        .tolist()
    )
    preds_scaled: list[float] = []

    with torch.no_grad():
        for _ in range(forecast_horizon):
            x_next = np.array(history_scaled[-window_size:], dtype=np.float32)
            x_tensor = torch.tensor(x_next).unsqueeze(0).unsqueeze(-1).to(device)
            y_next = float(model(x_tensor).cpu().numpy().ravel()[0])
            preds_scaled.append(y_next)
            history_scaled.append(y_next)

    preds = scaler.inverse_transform(
        np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    ).flatten()
    return preds
