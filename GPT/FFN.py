from __future__ import annotations

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
