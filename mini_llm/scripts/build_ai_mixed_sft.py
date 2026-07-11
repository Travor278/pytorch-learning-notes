from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "mini_llm" / "dataset"


def parse_json_line(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def turn_role(turn: dict[str, Any]) -> str:
    return str(turn.get("role", turn.get("from", ""))).strip().lower()


def has_conversation(record: dict[str, Any]) -> bool:
    conversations = record.get("conversations") or record.get("messages")
    if not isinstance(conversations, list):
        return False
    has_user = any(isinstance(turn, dict) and turn_role(turn) in {"user", "human"} for turn in conversations)
    has_assistant = any(
        isinstance(turn, dict) and turn_role(turn) in {"assistant", "gpt"} for turn in conversations
    )
    return has_user and has_assistant


def normalize_conversations(record: dict[str, Any]) -> dict[str, Any] | None:
    conversations = record.get("conversations") or record.get("messages")
    if not isinstance(conversations, list):
        return None

    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
    }
    normalized: list[dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = role_map.get(turn_role(turn))
        content = turn.get("content", turn.get("value", ""))
        if role is None or not isinstance(content, str) or not content.strip():
            continue
        normalized.append({"role": role, "content": content.strip()})

    candidate = dict(record)
    candidate["conversations"] = normalized
    candidate.pop("messages", None)
    return candidate if has_conversation(candidate) else None


def load_jsonl(path: Path, default_source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = parse_json_line(line)
            if record is None:
                continue
            record = normalize_conversations(record)
            if record is None:
                continue
            record.setdefault("source", default_source)
            records.append(record)
    if not records:
        raise ValueError(f"No usable conversation records found in {path}")
    return records


def reservoir_sample_jsonl(path: Path, sample_size: int, rng: random.Random) -> tuple[list[dict[str, Any]], int]:
    sample: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = parse_json_line(line)
            if record is None:
                continue
            record = normalize_conversations(record)
            if record is None:
                continue
            record.setdefault("source", "minimind_sft_t2t_mini")
            seen += 1
            if len(sample) < sample_size:
                sample.append(record)
                continue
            index = rng.randrange(seen)
            if index < sample_size:
                sample[index] = record
    return sample, seen


def sample_records(records: list[dict[str, Any]], sample_size: int, rng: random.Random) -> list[dict[str, Any]]:
    if sample_size >= len(records):
        return [dict(record) for record in records]
    return [dict(record) for record in rng.sample(records, sample_size)]


def repeated_records(records: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for _ in range(repeat):
        output.extend(dict(record) for record in records)
    return output


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MiniMind + strict AI HF + curated-anchor SFT mix.")
    parser.add_argument("--base-path", type=Path, default=DATASET_DIR / "sft_t2t_mini.jsonl")
    parser.add_argument("--hf-ai-path", type=Path, default=DATASET_DIR / "ai_hf_sft_strict.jsonl")
    parser.add_argument("--curated-path", type=Path, default=DATASET_DIR / "ai_knowledge_sft.jsonl")
    parser.add_argument("--out-path", type=Path, default=DATASET_DIR / "sft_ai_hf_mix.jsonl")
    parser.add_argument("--base-samples", type=int, default=50000)
    parser.add_argument("--hf-ai-samples", type=int, default=5000)
    parser.add_argument("--curated-repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ["base_samples", "curated_repeat"]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.hf_ai_samples < 0:
        raise ValueError("--hf-ai-samples must be non-negative.")
    required_paths = [args.base_path, args.curated_path]
    if args.hf_ai_samples > 0:
        required_paths.append(args.hf_ai_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

    rng = random.Random(args.seed)
    base_records, seen_base = reservoir_sample_jsonl(args.base_path, args.base_samples, rng)
    hf_records = load_jsonl(args.hf_ai_path, "ai_hf_sft_strict") if args.hf_ai_samples > 0 else []
    curated_records = load_jsonl(args.curated_path, "curated_ai_knowledge_v1")

    hf_sample = sample_records(hf_records, args.hf_ai_samples, rng) if args.hf_ai_samples > 0 else []
    curated = repeated_records(curated_records, args.curated_repeat)
    mixed = [*base_records, *hf_sample, *curated]
    rng.shuffle(mixed)
    write_jsonl(args.out_path, mixed)

    source_counts = Counter(str(record.get("source", "unknown")) for record in mixed)
    category_counts = Counter(str(record.get("category", record.get("topic", "base"))) for record in mixed)
    total = max(len(mixed), 1)
    print(f"base usable records seen: {seen_base:,}")
    print(f"base sampled records: {len(base_records):,} ({len(base_records) / total:.2%})")
    print(f"HF AI pool records: {len(hf_records):,}")
    print(f"HF AI sampled records: {len(hf_sample):,} ({len(hf_sample) / total:.2%})")
    print(f"curated records: {len(curated_records):,} x {args.curated_repeat} = {len(curated):,} ({len(curated) / total:.2%})")
    print(f"mixed records: {len(mixed):,}")
    print(f"wrote: {args.out_path}")
    print("top sources: " + ", ".join(f"{key}={value}" for key, value in source_counts.most_common(12)))
    print("top categories: " + ", ".join(f"{key}={value}" for key, value in category_counts.most_common(12)))


if __name__ == "__main__":
    main()
