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
    from GPTBlock import GPTBlock
    from create_mask import make_causal_mask
else:
    from .GPTBlock import GPTBlock
    from .create_mask import make_causal_mask


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.use_rope = use_rope

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 原来：use_rope=False 时 pos_encoding=None，等于没有任何位置信息
        # self.pos_encoding = None
        # 修改：use_rope=False 时用 Learned Positional Embedding 作为 fallback
        # 每个位置学一个独立的 d_model 维向量，加到 token embedding 上
        if not use_rope:
            self.pos_encoding: nn.Embedding | None = nn.Embedding(max_len, d_model)
        else:
            self.pos_encoding = None
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                GPTBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_rope=use_rope,
                    max_len=max_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        past_len: int = 0,
    ) -> torch.Tensor:
        if past_len == 0:
            return make_causal_mask(seq_len, device=device)
        current_mask = make_causal_mask(seq_len, device=device)
        past_mask = torch.zeros(seq_len, past_len, dtype=torch.bool, device=device)
        return torch.cat((past_mask, current_mask), dim=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        x = self.token_embedding(token_ids) * math.sqrt(self.d_model)

        # 先算 past_len，use_rope=False 时需要用它来确定当前 token 的绝对位置
        past_len = 0
        if past_kvs is not None and len(past_kvs) > 0:
            past_len = past_kvs[0][0].size(-2)

        # 原来：pos_encoding 始终为 None，use_rope=False 时没有任何位置信息
        # 修改：use_rope=False 时，把 Learned PE 加到 token embedding 上
        if self.pos_encoding is not None:
            seq_len = token_ids.size(1)
            positions = torch.arange(past_len, past_len + seq_len, device=token_ids.device)
            x = x + self.pos_encoding(positions)

        x = self.embed_dropout(x)

        if causal_mask is None:
            causal_mask = self._build_causal_mask(token_ids.size(1), token_ids.device, past_len=past_len)

        all_attn_weights: list[torch.Tensor] = []
        present_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            layer_past = None if past_kvs is None else past_kvs[idx]
            x, attn_weights, layer_present = layer(
                x,
                causal_mask=causal_mask,
                past_kv=layer_past,
                use_cache=use_cache,
            )
            all_attn_weights.append(attn_weights)
            if use_cache and layer_present is not None:
                present_kvs.append(layer_present)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, all_attn_weights, (present_kvs if use_cache else None)


def demo_gpt_shapes() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 16
    vocab_size = 100

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = GPTModel(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=256,
        dropout=0.1,
        max_len=64,
    )

    logits, all_attn_weights, _ = model(token_ids)

    print("token_ids shape        :", token_ids.shape)
    print("logits shape           :", logits.shape)
    print("num layers             :", len(all_attn_weights))
    print("layer1 self-attn shape :", all_attn_weights[0].shape)


if __name__ == "__main__":
    demo_gpt_shapes()
