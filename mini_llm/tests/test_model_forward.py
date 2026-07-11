from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_llm.model import MiniMindConfig, MiniMindForCausalLM, build_config_from_preset
from mini_llm.trainer.trainer_utils import count_parameters


def test_default_parameter_count() -> None:
    model = MiniMindForCausalLM(MiniMindConfig())
    params = count_parameters(model)
    assert 63_000_000 <= params <= 65_000_000, params


def test_mini310m_parameter_count() -> None:
    config = build_config_from_preset(
        "mini310m",
        tokenizer_vocab_size=260,
        block_size=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = MiniMindForCausalLM(config)
    params = count_parameters(model)
    assert 309_000_000 <= params <= 312_000_000, params


def test_forward_loss_backward() -> None:
    config = MiniMindConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = MiniMindForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels)
    assert output.logits.shape == (2, 16, config.vocab_size)
    assert output.loss is not None

    output.loss.backward()
    assert model.model.embed_tokens.weight.grad is not None


def test_generate_greedy_shape() -> None:
    config = MiniMindConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = MiniMindForCausalLM(config)
    input_ids = torch.randint(4, config.vocab_size, (2, 8))

    generated = model.generate(
        input_ids,
        max_new_tokens=4,
        temperature=0.0,
        vocab_size_limit=128,
        suppress_token_ids=[config.pad_token_id],
    )
    assert generated.shape == (2, 12)
    assert generated.max().item() < config.vocab_size


def test_moe_forward_has_aux_loss() -> None:
    config = MiniMindConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = MiniMindForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    output = model(input_ids=input_ids, labels=input_ids.clone())

    assert output.loss is not None
    assert output.aux_loss is not None
    output.loss.backward()
    assert model.model.layers[0].mlp.gate.weight.grad is not None


if __name__ == "__main__":
    test_default_parameter_count()
    test_mini310m_parameter_count()
    test_forward_loss_backward()
    test_generate_greedy_shape()
    test_moe_forward_has_aux_loss()
    print("mini_llm M1 smoke tests passed.")
