from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    from mini_llm.eval_llm import (
        autocast_context,
        load_model_from_checkpoint,
        resolve_checkpoint_path,
        resolve_device,
        resolve_precision,
        tokenizer_path_from_args,
    )
    from mini_llm.trainer.tokenizer import load_or_create_tokenizer
    from mini_llm.trainer.trainer_utils import PROJECT_ROOT
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mini_llm.eval_llm import (
        autocast_context,
        load_model_from_checkpoint,
        resolve_checkpoint_path,
        resolve_device,
        resolve_precision,
        tokenizer_path_from_args,
    )
    from mini_llm.trainer.tokenizer import load_or_create_tokenizer
    from mini_llm.trainer.trainer_utils import PROJECT_ROOT


DEFAULT_EVAL_SET = PROJECT_ROOT / "evals" / "ai_expert_eval_v2.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_index}: expected JSON object")
            records.append(value)
    return records


def chat_prompt(user_prompt: str) -> str:
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = normalize_text(text)
    hits = []
    for keyword in keywords:
        needle = normalize_text(str(keyword))
        if not needle:
            continue
        if len(needle) == 1 and needle.isascii() and needle.isalnum():
            if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", lowered):
                hits.append(str(keyword))
        elif needle in lowered:
            hits.append(str(keyword))
    return hits


def forbidden_keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = normalize_text(text)
    hits = []
    negation_markers = ["不是", "并非", "不应", "不要", "不能", "不等于", "不是一种", "not ", "not a"]
    for keyword in keywords:
        needle = normalize_text(str(keyword))
        if not needle:
            continue
        start = 0
        while True:
            if len(needle) == 1 and needle.isascii() and needle.isalnum():
                match = re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", lowered[start:])
                if match is None:
                    break
                index = start + match.start()
            else:
                index = lowered.find(needle, start)
            if index < 0:
                break
            left = lowered[max(0, index - 12) : index]
            if not any(marker in left for marker in negation_markers):
                hits.append(str(keyword))
                break
            start = index + len(needle)
    return hits


def has_repetition(text: str) -> bool:
    compact = normalize_text(text)
    if len(compact) < 30:
        return False
    pieces = re.split(r"[。！？.!?\n]", compact)
    pieces = [piece.strip() for piece in pieces if len(piece.strip()) >= 8]
    seen: set[str] = set()
    for piece in pieces:
        if piece in seen:
            return True
        seen.add(piece)
    for n in range(8, 18):
        chunks = [compact[i : i + n] for i in range(0, max(len(compact) - n + 1, 0), n)]
        if chunks and max(chunks.count(chunk) for chunk in set(chunks)) >= 4:
            return True
    return False


def score_completion(record: dict[str, Any], completion: str) -> dict[str, Any]:
    required = [str(item) for item in record.get("required_keywords", [])]
    forbidden = [str(item) for item in record.get("forbidden_keywords", [])]
    required_hits = keyword_hits(completion, required)
    forbidden_hits = forbidden_keyword_hits(completion, forbidden)
    recall = len(required_hits) / max(len(required), 1)
    min_recall = float(record.get("min_required_keyword_recall", 0.55))
    length = len(completion.strip())
    repetition = has_repetition(completion)
    passed = recall >= min_recall and not forbidden_hits and 20 <= length <= 900 and not repetition
    return {
        "required_hits": required_hits,
        "forbidden_hits": forbidden_hits,
        "keyword_recall": round(recall, 4),
        "min_required_keyword_recall": min_recall,
        "length": length,
        "repetition": repetition,
        "passed": passed,
    }


def generate_one(
    *,
    model,
    tokenizer,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    prompt_ids = tokenizer.encode(chat_prompt(prompt), add_bos=True, add_eos=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with autocast_context(device, autocast_dtype):
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            vocab_size_limit=tokenizer.vocab_size,
            suppress_token_ids=[tokenizer.pad_token_id],
        )
    generated = generated_ids[0].detach().cpu().tolist()
    return tokenizer.decode(generated[len(prompt_ids) :]).strip()


def default_out_path(checkpoint_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = checkpoint_path.parent.name + "_" + checkpoint_path.stem
    return PROJECT_ROOT / "evals" / f"{safe_name}_ai_expert_v2_{timestamp}.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed AI expert eval set with keyword scoring.")
    parser.add_argument("--checkpoint", "--load-from", dest="checkpoint", default=None)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Only validate and summarize the eval set.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cases = read_jsonl(args.eval_set)
    topics = sorted({str(case.get("topic", "unknown")) for case in cases})
    print(f"eval_set: {args.eval_set}")
    print(f"cases: {len(cases)}")
    print("topics: " + ", ".join(topics))
    if args.dry_run:
        return
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --dry-run is set.")

    device = resolve_device(args.device)
    precision_name, autocast_dtype = resolve_precision(device, args.precision)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    tokenizer = load_or_create_tokenizer(tokenizer_path_from_args(args, checkpoint))
    out_path = args.out_path or default_out_path(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "type": "metadata",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step", 0),
        "eval_set": str(args.eval_set),
        "device": str(device),
        "precision": precision_name,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "num_cases": len(cases),
    }

    passed_count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n")
        for case in cases:
            completion = generate_one(
                model=model,
                tokenizer=tokenizer,
                device=device,
                autocast_dtype=autocast_dtype,
                prompt=str(case["prompt"]),
                max_new_tokens=int(case.get("max_new_tokens", args.max_new_tokens)),
                temperature=args.temperature,
                top_k=args.top_k,
            )
            score = score_completion(case, completion)
            passed_count += int(bool(score["passed"]))
            output = {
                "type": "case",
                **case,
                "completion": completion,
                "score": score,
            }
            f.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            marker = "PASS" if score["passed"] else "FAIL"
            print(
                f"[{marker}] {case['id']} {case['topic']}/{case['subtopic']} "
                f"recall={score['keyword_recall']:.2f} forbidden={len(score['forbidden_hits'])}"
            )

    pass_rate = passed_count / max(len(cases), 1)
    print(f"passed: {passed_count}/{len(cases)} ({pass_rate:.2%})")
    print(f"out: {out_path}")


if __name__ == "__main__":
    main()
