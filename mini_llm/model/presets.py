from __future__ import annotations

from dataclasses import asdict

from .model_minimind import MiniMindConfig


MODEL_PRESETS: dict[str, dict] = {
    "tiny": {
        "vocab_size": None,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    },
    "mini64m": {
        "vocab_size": 6400,
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
    },
    "mini123m": {
        "vocab_size": 6400,
        "hidden_size": 768,
        "num_hidden_layers": 16,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
    },
    "mini209m": {
        "vocab_size": 6400,
        "hidden_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
    },
    "mini310m": {
        "vocab_size": 6400,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
    },
    "mini310m_8k": {
        "vocab_size": 8192,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
    },
    "mini320m_16k": {
        "vocab_size": 16000,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
    },
    "tiny_moe": {
        "vocab_size": None,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "use_moe": True,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    },
}

MODEL_PRESET_ALIASES = {
    "minimind3": "mini64m",
}


def list_model_presets() -> list[str]:
    return [*MODEL_PRESETS.keys(), *MODEL_PRESET_ALIASES.keys()]


def resolve_model_preset(name: str) -> str:
    return MODEL_PRESET_ALIASES.get(name, name)


def build_config_from_preset(
    name: str,
    *,
    tokenizer_vocab_size: int,
    block_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> MiniMindConfig:
    resolved = resolve_model_preset(name)
    if resolved not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {name}")

    values = dict(MODEL_PRESETS[resolved])
    vocab_size = values.pop("vocab_size")
    if vocab_size is None:
        vocab_size = tokenizer_vocab_size
    if tokenizer_vocab_size > vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer_vocab_size} exceeds model vocab_size={vocab_size}."
        )

    return MiniMindConfig(
        **values,
        vocab_size=vocab_size,
        max_position_embeddings=max(block_size, 32768 if resolved != "tiny" else block_size),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def preset_config_dict(name: str) -> dict:
    resolved = resolve_model_preset(name)
    values = dict(MODEL_PRESETS[resolved])
    vocab_size = values.pop("vocab_size")
    config = MiniMindConfig(**values, vocab_size=vocab_size if vocab_size is not None else 260)
    payload = asdict(config)
    payload["preset_name"] = name
    return payload
