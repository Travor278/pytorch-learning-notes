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


def has_conversation(record: dict[str, Any]) -> bool:
    conversations = record.get("conversations") or record.get("messages")
    if isinstance(conversations, list):
        has_user = any(isinstance(turn, dict) and turn.get("role") == "user" for turn in conversations)
        has_assistant = any(
            isinstance(turn, dict) and turn.get("role") == "assistant" for turn in conversations
        )
        return has_user and has_assistant
    return False


def reservoir_sample_jsonl(path: Path, sample_size: int, rng: random.Random) -> tuple[list[dict[str, Any]], int]:
    sample: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = parse_json_line(line)
            if record is None or not has_conversation(record):
                continue
            record = dict(record)
            record.setdefault("source", "minimind_sft_t2t_mini")
            seen += 1
            if len(sample) < sample_size:
                sample.append(record)
                continue
            index = rng.randrange(seen)
            if index < sample_size:
                sample[index] = record
    return sample, seen


def load_domain_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = parse_json_line(line)
            if record is None or not has_conversation(record):
                continue
            record = dict(record)
            record.setdefault("source", path.stem)
            records.append(record)
    if not records:
        raise ValueError(f"No usable domain SFT records found in {path}")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a shuffled MiniMind + AI knowledge SFT mix.")
    parser.add_argument("--base-path", type=Path, default=DATASET_DIR / "sft_t2t_mini.jsonl")
    parser.add_argument("--domain-path", type=Path, default=DATASET_DIR / "ai_knowledge_sft.jsonl")
    parser.add_argument("--out-path", type=Path, default=DATASET_DIR / "sft_ai_mix.jsonl")
    parser.add_argument("--base-samples", type=int, default=20000)
    parser.add_argument("--domain-repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.base_samples <= 0:
        raise ValueError("--base-samples must be positive.")
    if args.domain_repeat <= 0:
        raise ValueError("--domain-repeat must be positive.")
    if not args.base_path.exists():
        raise FileNotFoundError(f"Missing base SFT file: {args.base_path}")
    if not args.domain_path.exists():
        raise FileNotFoundError(f"Missing domain SFT file: {args.domain_path}")

    rng = random.Random(args.seed)
    base_records, seen_base = reservoir_sample_jsonl(args.base_path, args.base_samples, rng)
    if len(base_records) < args.base_samples:
        print(f"warning: requested {args.base_samples} base samples, found {len(base_records)} usable records")
    domain_records = load_domain_records(args.domain_path)

    mixed: list[dict[str, Any]] = []
    mixed.extend(base_records)
    for _ in range(args.domain_repeat):
        for record in domain_records:
            mixed.append(dict(record))
    rng.shuffle(mixed)
    write_jsonl(args.out_path, mixed)

    source_counts = Counter(str(record.get("source", "unknown")) for record in mixed)
    category_counts = Counter(str(record.get("category", "base")) for record in mixed)
    domain_count = len(domain_records) * args.domain_repeat
    print(f"base usable records seen: {seen_base:,}")
    print(f"base sampled records: {len(base_records):,}")
    print(f"domain records: {len(domain_records):,}")
    print(f"domain repeat: {args.domain_repeat}")
    print(f"mixed records: {len(mixed):,}")
    print(f"domain ratio: {domain_count / max(len(mixed), 1):.2%}")
    print(f"wrote: {args.out_path}")
    print("sources: " + ", ".join(f"{key}={value}" for key, value in source_counts.most_common()))
    print("top categories: " + ", ".join(f"{key}={value}" for key, value in category_counts.most_common(10)))


if __name__ == "__main__":
    main()
