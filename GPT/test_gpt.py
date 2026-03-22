from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    from GPT import GPTModel
    from GPTBlock import GPTBlock
    from RoPE import rotate_half
else:
    from .GPT import GPTModel
    from .GPTBlock import GPTBlock
    from .RoPE import rotate_half


def test_gelu_and_weight_tying() -> None:
    model = GPTModel(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=2,
        d_ff=32,
        dropout=0.0,
        max_len=16,
    )
    assert isinstance(model.layers[0].ffn.activation, nn.GELU)
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_pre_ln_block_behavior() -> None:
    torch.manual_seed(0)
    block = GPTBlock(
        d_model=8,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
    )

    def fake_self_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None, past_kv=None, use_cache=False):
        heads = torch.zeros(q.size(0), block.self_attn.num_heads, q.size(1), q.size(1))
        return q, heads, None

    block.self_attn.forward = fake_self_attn  # type: ignore[method-assign]
    block.ffn.forward = lambda x: x  # type: ignore[method-assign]

    x = torch.randn(2, 4, 8)
    y, _, _ = block(x)

    expected = x + block.norm1(x)
    expected = expected + block.norm2(expected)

    assert torch.allclose(y, expected, atol=1e-6)


def test_forward_shapes() -> None:
    torch.manual_seed(0)
    model = GPTModel(
        vocab_size=40,
        d_model=16,
        num_heads=4,
        num_layers=2,
        d_ff=32,
        dropout=0.0,
        max_len=16,
    )
    token_ids = torch.randint(0, 40, (2, 10))
    logits, attn, _ = model(token_ids)
    assert logits.shape == (2, 10, 40)
    assert len(attn) == 2
    assert attn[0].shape == (2, 4, 10, 10)


def test_rope_replaces_absolute_position_encoding() -> None:
    model = GPTModel(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=2,
        d_ff=32,
        dropout=0.0,
        max_len=16,
    )
    assert model.use_rope is True
    assert hasattr(model.layers[0].self_attn, "rope")
    assert model.pos_encoding is None


def test_rotate_half_uses_llama_pairing() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = rotate_half(x)
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    assert torch.equal(y, expected)


def test_kv_cache_matches_full_forward() -> None:
    torch.manual_seed(0)
    model = GPTModel(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=2,
        d_ff=32,
        dropout=0.0,
        max_len=16,
    )
    model.eval()

    token_ids = torch.randint(0, 32, (1, 6))
    full_logits, _, _ = model(token_ids, use_cache=False)

    cached_logits = []
    past_kvs = None
    for t in range(token_ids.size(1)):
        step_logits, _, past_kvs = model(
            token_ids[:, t : t + 1],
            past_kvs=past_kvs,
            use_cache=True,
        )
        cached_logits.append(step_logits)

    stacked_logits = torch.cat(cached_logits, dim=1)
    assert torch.allclose(full_logits, stacked_logits, atol=1e-5)
    assert past_kvs is not None
    assert len(past_kvs) == 2
    assert past_kvs[0][0].shape == (1, 4, 6, 4)
    assert past_kvs[0][1].shape == (1, 4, 6, 4)


if __name__ == "__main__":
    test_gelu_and_weight_tying()
    test_pre_ln_block_behavior()
    test_forward_shapes()
    test_rope_replaces_absolute_position_encoding()
    test_rotate_half_uses_llama_pairing()
    test_kv_cache_matches_full_forward()
    print("All GPT checks passed.")
