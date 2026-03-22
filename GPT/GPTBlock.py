from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    from FFN import PositionwiseFeedForward
    from MHA import MultiHeadAttention
else:
    from .FFN import PositionwiseFeedForward
    from .MHA import MultiHeadAttention


class GPTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_len=max_len,
        )
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        normed_x = self.norm1(x)
        attn_out, attn_weights, present_kv = self.self_attn(
            normed_x,
            normed_x,
            normed_x,
            mask=causal_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + self.dropout1(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x, attn_weights, present_kv
