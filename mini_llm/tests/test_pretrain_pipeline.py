from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_llm.trainer.pretrain_data import CausalLMCollator, PretrainJsonlDataset
from mini_llm.trainer.pretrain_data import StreamingPretrainJsonlDataset
from mini_llm.trainer.tokenizer import ByteTokenizer
from mini_llm.trainer.train_pretrain import main as train_pretrain_main


def write_demo_jsonl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"text": "hello mini llm"}',
                '{"text": "春眠不觉晓，处处闻啼鸟。"}',
                '{"text": "causal language modeling predicts the next token."}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_pretrain_dataset_and_collator() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "pretrain.jsonl"
        write_demo_jsonl(data_path)
        tokenizer = ByteTokenizer()
        dataset = PretrainJsonlDataset(data_path, tokenizer, block_size=16)
        batch = CausalLMCollator(tokenizer.pad_token_id)([dataset[0], dataset[1]])

        assert batch["input_ids"].ndim == 2
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_streaming_pretrain_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "pretrain.jsonl"
        write_demo_jsonl(data_path)
        tokenizer = ByteTokenizer()
        dataset = StreamingPretrainJsonlDataset(data_path, tokenizer, block_size=16)
        first = next(iter(dataset))

        assert first["input_ids"].numel() == 16
        assert first["labels"].shape == first["input_ids"].shape


def test_pretrain_train_loop_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "pretrain.jsonl"
        tokenizer_path = tmp_path / "tokenizer.json"
        out_dir = tmp_path / "checkpoints"
        write_demo_jsonl(data_path)

        train_pretrain_main(
            [
                "--data-path",
                str(data_path),
                "--tokenizer-path",
                str(tokenizer_path),
                "--out-dir",
                str(out_dir),
                "--device",
                "cpu",
                "--model-size",
                "tiny",
                "--block-size",
                "16",
                "--batch-size",
                "2",
                "--gradient-accumulation-steps",
                "2",
                "--max-steps",
                "2",
                "--log-every",
                "1",
                "--precision",
                "fp32",
                "--in-memory",
            ]
        )

        checkpoint_path = out_dir / "pretrain_last.pt"
        assert checkpoint_path.exists()
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        assert ckpt["step"] == 2
        assert ckpt["micro_step"] == 4
        assert ckpt["training_args"]["gradient_accumulation_steps"] == 2
        assert "model_state_dict" in ckpt


def test_pretrain_resume_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "pretrain.jsonl"
        tokenizer_path = tmp_path / "tokenizer.json"
        out_dir = tmp_path / "checkpoints"
        write_demo_jsonl(data_path)

        base_args = [
            "--data-path",
            str(data_path),
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
            "16",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--save-every",
            "1",
            "--in-memory",
        ]
        train_pretrain_main(base_args)

        first_checkpoint = out_dir / "pretrain_last.pt"
        assert first_checkpoint.exists()
        first = torch.load(first_checkpoint, map_location="cpu")
        assert first["step"] == 1

        train_pretrain_main(
            [
                *base_args,
                "--max-steps",
                "2",
                "--resume-from",
                str(first_checkpoint),
            ]
        )
        resumed = torch.load(first_checkpoint, map_location="cpu")
        assert resumed["step"] == 2


if __name__ == "__main__":
    test_pretrain_dataset_and_collator()
    test_streaming_pretrain_dataset()
    test_pretrain_train_loop_smoke()
    test_pretrain_resume_smoke()
    print("mini_llm M2 pretrain pipeline tests passed.")
