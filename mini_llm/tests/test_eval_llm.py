from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_llm.eval_llm import main as eval_main
from mini_llm.trainer.train_pretrain import main as train_pretrain_main


def write_demo_jsonl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"text": "hello mini llm"}',
                '{"text": "checkpoint loading should generate text"}',
                '{"text": "春眠不觉晓，处处闻啼鸟。"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_eval_loads_checkpoint_and_generates() -> None:
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
            ]
        )

        checkpoint_path = out_dir / "pretrain_last.pt"
        assert checkpoint_path.exists()
        text = eval_main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--tokenizer-path",
                str(tokenizer_path),
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "4",
                "--temperature",
                "0",
            ]
        )
        assert isinstance(text, str)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["step"] == 1


if __name__ == "__main__":
    test_eval_loads_checkpoint_and_generates()
    print("mini_llm M3 eval/generate tests passed.")
