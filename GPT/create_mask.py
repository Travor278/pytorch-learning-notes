from __future__ import annotations

import torch


def make_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )

