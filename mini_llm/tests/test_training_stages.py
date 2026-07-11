from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_llm.model import MiniMindConfig, MiniMindForCausalLM, apply_lora, lora_parameter_count
from mini_llm.trainer.preference_data import PreferenceCollator, PreferenceJsonlDataset
from mini_llm.trainer.sft_data import SFTCollator, SFTJsonlDataset, StreamingSFTJsonlDataset
from mini_llm.trainer.train_dpo import main as train_dpo_main
from mini_llm.trainer.train_full_sft import main as train_sft_main
from mini_llm.trainer.train_lora import main as train_lora_main
from mini_llm.trainer.tokenizer import BPETokenizer, ByteTokenizer


def write_sft_jsonl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"conversations":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]}',
                '{"instruction":"2+2?","output":"4"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_dpo_jsonl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"prompt":"2+2?","chosen":"4","rejected":"5"}',
                '{"prompt":"Say hello","chosen":"hello","rejected":"goodbye"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_sft_dataset_masks_user_tokens() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sft.jsonl"
        write_sft_jsonl(path)
        tokenizer = ByteTokenizer()
        dataset = SFTJsonlDataset(path, tokenizer, block_size=64)
        batch = SFTCollator(tokenizer.pad_token_id)([dataset[0], dataset[1]])

        assert batch["input_ids"].shape == batch["labels"].shape
        assert batch["labels"].ne(-100).any()
        assert batch["labels"].eq(-100).any()


def test_streaming_sft_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sft.jsonl"
        write_sft_jsonl(path)
        tokenizer = ByteTokenizer()
        dataset = StreamingSFTJsonlDataset(path, tokenizer, block_size=64)
        first = next(iter(dataset))

        assert first["input_ids"].numel() == first["labels"].numel()
        assert first["labels"].ne(-100).any()


def test_lora_injection_freezes_base() -> None:
    model = MiniMindForCausalLM(
        MiniMindConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    apply_lora(model, rank=4, alpha=8, target_modules=("q_proj", "v_proj"))

    assert lora_parameter_count(model) > 0
    assert all(("lora_" in name) == param.requires_grad for name, param in model.named_parameters())


def test_preference_dataset_collator() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dpo.jsonl"
        write_dpo_jsonl(path)
        tokenizer = ByteTokenizer()
        dataset = PreferenceJsonlDataset(path, tokenizer, block_size=64)
        batch = PreferenceCollator(tokenizer.pad_token_id)([dataset[0], dataset[1]])

        assert batch["chosen_input_ids"].ndim == 2
        assert batch["rejected_input_ids"].ndim == 2
        assert batch["chosen_labels"].ne(-100).any()


def test_bpe_tokenizer_roundtrip() -> None:
    tokenizer = BPETokenizer.train(
        "你好世界\n你好模型\nhello model\nhello world",
        vocab_size=64,
        min_freq=1,
    )
    text = "你好世界"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size <= 64


def test_sft_lora_dpo_train_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sft_path = tmp_path / "sft.jsonl"
        dpo_path = tmp_path / "dpo.jsonl"
        tokenizer_path = tmp_path / "tokenizer.json"
        out_dir = tmp_path / "checkpoints"
        write_sft_jsonl(sft_path)
        write_dpo_jsonl(dpo_path)

        common = [
            "--tokenizer-path",
            str(tokenizer_path),
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--model-size",
            "tiny",
            "--block-size",
            "64",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--log-every",
            "1",
        ]
        train_sft_main(["--data-path", str(sft_path), *common, "--in-memory"])
        assert (out_dir / "sft_last.pt").exists()

        train_lora_main(["--data-path", str(sft_path), *common, "--init-from", str(out_dir / "sft_last.pt")])
        assert (out_dir / "lora_merged.pt").exists()

        train_dpo_main(["--data-path", str(dpo_path), *common, "--init-from", str(out_dir / "sft_last.pt")])
        assert (out_dir / "dpo_last.pt").exists()

        checkpoint = torch.load(out_dir / "dpo_last.pt", map_location="cpu")
        assert checkpoint["step"] == 1


if __name__ == "__main__":
    test_sft_dataset_masks_user_tokens()
    test_streaming_sft_dataset()
    test_lora_injection_freezes_base()
    test_preference_dataset_collator()
    test_bpe_tokenizer_roundtrip()
    test_sft_lora_dpo_train_smoke()
    print("mini_llm M4-M7 training stage tests passed.")
