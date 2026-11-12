"""A leakage-safe direct multi-horizon demand-forecasting example."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def make_demand_series(days: int = 420) -> tuple[torch.Tensor, torch.Tensor]:
    """Create demand and known-ahead calendar features for a small demo."""
    day = torch.arange(days, dtype=torch.float32)
    weekly_phase = 2 * torch.pi * day / 7
    promotions = ((day.remainder(35) >= 28) & (day.remainder(35) <= 30)).float()
    demand = 80 + 0.08 * day + 14 * torch.sin(weekly_phase) + 18 * promotions
    demand = demand + 2.0 * torch.randn_like(demand)
    known_future = torch.stack(
        (torch.sin(weekly_phase), torch.cos(weekly_phase), promotions), dim=-1
    )
    return demand, known_future


def build_windows(
    values: torch.Tensor,
    known_future: torch.Tensor,
    history: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build windows and return the first target index for each example."""
    if values.ndim != 1 or known_future.shape[0] != values.numel():
        raise ValueError("values and known_future must share one time dimension")
    if values.numel() < history + horizon:
        raise ValueError("series is shorter than the requested history plus horizon")

    starts = torch.arange(values.numel() - history - horizon + 1)
    past = torch.stack([values[start : start + history] for start in starts])
    future = torch.stack(
        [known_future[start + history : start + history + horizon] for start in starts]
    )
    targets = torch.stack(
        [values[start + history : start + history + horizon] for start in starts]
    )
    return past.unsqueeze(-1), future, targets, starts + history


class DirectDemandForecaster(nn.Module):
    """Forecast a fixed horizon from history and known-ahead features."""

    def __init__(self, hidden_size: int, horizon: int, future_features: int) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + horizon * future_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, past: torch.Tensor, known_future: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(past)
        features = torch.cat((hidden[-1], known_future.flatten(start_dim=1)), dim=-1)
        return self.head(features)


def main() -> None:
    torch.manual_seed(0)
    history, horizon, train_end = 28, 7, 300
    demand, known_future = make_demand_series()

    train_mean = demand[:train_end].mean()
    train_std = demand[:train_end].std().clamp_min(1e-6)
    scaled = (demand - train_mean) / train_std
    past, future, targets, target_starts = build_windows(scaled, known_future, history, horizon)

    train_rows = target_starts + horizon <= train_end
    validation_rows = target_starts >= train_end
    model = DirectDemandForecaster(hidden_size=32, horizon=horizon, future_features=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for _ in range(300):
        optimizer.zero_grad()
        loss = F.mse_loss(model(past[train_rows], future[train_rows]), targets[train_rows])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        forecast = model(past[validation_rows], future[validation_rows])
        forecast = forecast * train_std + train_mean
        actual = targets[validation_rows] * train_std + train_mean
        mae = (forecast - actual).abs().mean()
    print(f"Validation MAE over a {horizon}-day horizon: {mae:.2f} demand units")


if __name__ == "__main__":
    main()
