from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .tokenizer import BPETokenizer, ByteTokenizer
    from .trainer_utils import DATASET_DIR, PROJECT_ROOT, iter_jsonl
except ImportError:
    from tokenizer import BPETokenizer, ByteTokenizer
    from trainer_utils import DATASET_DIR, PROJECT_ROOT, iter_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a byte or BPE tokenizer for mini_llm.")
    parser.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte")
    parser.add_argument("--data-path", type=Path, default=DATASET_DIR / "pretrain.jsonl")
    parser.add_argument("--tokenizer-path", type=Path, default=PROJECT_ROOT / "tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=5_000_000)
    return parser


def _iter_strings(value: Any):
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def collect_text(path: Path, max_chars: int) -> str:
    pieces: list[str] = []
    total = 0
    for record in iter_jsonl(path):
        for text in _iter_strings(record):
            if total >= max_chars:
                break
            remaining = max_chars - total
            piece = text[:remaining]
            pieces.append(piece)
            total += len(piece)
        if total >= max_chars:
            break
    if not pieces:
        raise ValueError(f"No usable text found in {path}")
    return "\n".join(pieces)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.tokenizer == "byte":
        tokenizer = ByteTokenizer()
    else:
        if not args.data_path.exists():
            raise FileNotFoundError(f"Missing tokenizer training data: {args.data_path}")
        text = collect_text(args.data_path, args.max_chars)
        tokenizer = BPETokenizer.train(
            text,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq,
        )
    tokenizer.save(args.tokenizer_path)
    print(f"saved tokenizer: {args.tokenizer_path}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"vocab_size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
