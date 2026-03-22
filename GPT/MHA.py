from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    from RoPE import RotaryEmbedding, apply_rotary_pos_emb
else:
    from .RoPE import RotaryEmbedding, apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        use_rope: bool = True,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(dim=self.d_k, max_len=max_len) if use_rope else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        Q = self._split_heads(self.W_q(query))
        K_current = self._split_heads(self.W_k(key))
        V_current = self._split_heads(self.W_v(value))

        past_len = 0
        if past_kv is not None:
            past_len = past_kv[0].size(-2)

        if self.rope is not None:
            cos, sin = self.rope(Q.size(-2), Q.device, offset=past_len)
            Q, K_current = apply_rotary_pos_emb(Q, K_current, cos, sin)

        if past_kv is not None:
            K = torch.cat((past_kv[0], K_current), dim=-2)
            V = torch.cat((past_kv[1], V_current), dim=-2)
        else:
            K = K_current
            V = V_current

        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = self._merge_heads(attn_output)
        output = self.W_o(attn_output)
        present_kv = (K, V) if use_cache else None
        return output, attn_weights, present_kv
