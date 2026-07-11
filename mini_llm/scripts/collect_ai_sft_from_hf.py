from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "mini_llm" / "dataset"
HF_CACHE_DIR = PROJECT_ROOT / "mini_llm" / ".hf_cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


KEYWORDS = [
    "agent",
    "artificial intelligence",
    "attention",
    "backpropagation",
    "bert",
    "bm25",
    "clip",
    "computer vision",
    "deep learning",
    "diffusion model",
    "embedding",
    "fine-tuning",
    "finetuning",
    "function calling",
    "gpt",
    "gradient",
    "inference",
    "language model",
    "llama",
    "llava",
    "llm",
    "lora",
    "machine learning",
    "moe",
    "multimodal",
    "neural network",
    "nlp",
    "pytorch",
    "rag",
    "reinforcement learning",
    "retrieval",
    "robot",
    "robot learning",
    "robotics",
    "tokenizer",
    "tool calling",
    "tool use",
    "transformer",
    "vla",
    "vision-language",
    "vlm",
    "人工智能",
    "机器学习",
    "深度学习",
    "神经网络",
    "大模型",
    "语言模型",
    "多模态",
    "视觉语言",
    "机器人",
    "智能体",
    "代理",
    "工具调用",
    "函数调用",
    "检索",
    "向量",
    "嵌入",
    "注意力",
    "微调",
    "预训练",
    "强化学习",
    "生成模型",
    "扩散模型",
    "计算机视觉",
    "自然语言处理",
]

NOISE_PATTERNS = [
    "As an AI language model",
    "作为一个AI语言模型",
    "I cannot browse",
    "I don't have personal experiences",
]


@dataclass(frozen=True)
class Recipe:
    name: str
    repo: str
    config: str | None
    split: str
    adapter: str
    target: int
    max_scan: int
    min_chars: int = 80
    max_chars: int = 5000


RECIPES = [
    Recipe(
        name="finetome_ai_code",
        repo="mlabonne/FineTome-100k",
        config=None,
        split="train",
        adapter="sharegpt",
        target=2500,
        max_scan=100000,
    ),
    Recipe(
        name="smol_smoltalk_ai",
        repo="HuggingFaceTB/smol-smoltalk",
        config=None,
        split="train",
        adapter="messages",
        target=1200,
        max_scan=180000,
    ),
    Recipe(
        name="openorca_ai",
        repo="Open-Orca/OpenOrca",
        config=None,
        split="train",
        adapter="orca",
        target=1200,
        max_scan=250000,
    ),
    Recipe(
        name="coig_segmentfault_tech",
        repo="m-a-p/COIG-CQIA",
        config="segmentfault",
        split="train",
        adapter="alpaca",
        target=1200,
        max_scan=80000,
        min_chars=60,
    ),
    Recipe(
        name="coig_zhihu_tech",
        repo="m-a-p/COIG-CQIA",
        config="zhihu",
        split="train",
        adapter="alpaca",
        target=800,
        max_scan=80000,
        min_chars=60,
    ),
    Recipe(
        name="tiger_sft_zh_ai",
        repo="TigerResearch/sft_zh",
        config=None,
        split="train",
        adapter="alpaca",
        target=1000,
        max_scan=200000,
        min_chars=60,
    ),
    Recipe(
        name="alpaca_gpt4_chinese_ai",
        repo="FreedomIntelligence/alpaca-gpt4-chinese",
        config=None,
        split="train",
        adapter="sharegpt",
        target=600,
        max_scan=60000,
        min_chars=60,
    ),
    Recipe(
        name="glaive_tool_calling",
        repo="hiyouga/glaive-function-calling-v2-sharegpt",
        config=None,
        split="train",
        adapter="tool_sharegpt",
        target=700,
        max_scan=120000,
        min_chars=80,
    ),
]


def compact_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_score(text: str) -> int:
    lowered = text.lower()
    score = 0
    for keyword in KEYWORDS:
        needle = keyword.lower()
        if needle.isascii() and re.fullmatch(r"[a-z0-9_+-]+", needle):
            if re.search(rf"(?<![a-z0-9_+-]){re.escape(needle)}(?![a-z0-9_+-])", lowered):
                score += 1
        elif needle in lowered:
            score += 1
    return score


def noisy(text: str) -> bool:
    return any(pattern.lower() in text.lower() for pattern in NOISE_PATTERNS)


def normalize_turns(turns: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
        "function_call": "assistant",
        "observation": "tool",
        "tool": "tool",
    }
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        raw_role = str(turn.get("role", turn.get("from", ""))).strip().lower()
        content = turn.get("content", turn.get("value", ""))
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        role = role_map.get(raw_role)
        if role is None:
            continue
        content = compact_text(content)
        if not content:
            continue
        if raw_role == "function_call":
            content = "<tool_call>" + content + "</tool_call>"
        normalized.append({"role": role, "content": content})
    return normalized


def adapt_sharegpt(row: dict[str, Any]) -> list[dict[str, str]]:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        return normalize_turns(conversations)
    return []


def adapt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        return normalize_turns(messages)
    return []


def adapt_orca(row: dict[str, Any]) -> list[dict[str, str]]:
    turns = []
    system_prompt = compact_text(str(row.get("system_prompt") or ""))
    question = compact_text(str(row.get("question") or ""))
    response = compact_text(str(row.get("response") or ""))
    if system_prompt:
        turns.append({"role": "system", "content": system_prompt})
    if question and response:
        turns.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        )
    return turns


def adapt_alpaca(row: dict[str, Any]) -> list[dict[str, str]]:
    instruction = compact_text(str(row.get("instruction") or row.get("prompt") or row.get("question") or ""))
    input_text = compact_text(str(row.get("input") or ""))
    output = compact_text(str(row.get("output") or row.get("answer") or row.get("response") or ""))
    if input_text:
        instruction = instruction + "\n\n" + input_text
    if instruction and output:
        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
    return []


def adapt_tool_sharegpt(row: dict[str, Any]) -> list[dict[str, str]]:
    conversations = normalize_turns(row.get("conversations") or [])
    tools = compact_text(str(row.get("tools") or ""))
    if tools and conversations:
        conversations.insert(0, {"role": "system", "content": "可用工具定义：" + tools})
    return conversations


ADAPTERS: dict[str, Callable[[dict[str, Any]], list[dict[str, str]]]] = {
    "sharegpt": adapt_sharegpt,
    "messages": adapt_messages,
    "orca": adapt_orca,
    "alpaca": adapt_alpaca,
    "tool_sharegpt": adapt_tool_sharegpt,
}


def conversation_text(conversations: list[dict[str, str]]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversations)


def first_user_and_assistant(conversations: list[dict[str, str]]) -> tuple[str, str] | None:
    user = next((turn["content"] for turn in conversations if turn["role"] == "user"), "")
    assistant = next((turn["content"] for turn in conversations if turn["role"] == "assistant"), "")
    if not user or not assistant:
        return None
    return user, assistant


def valid_conversation(conversations: list[dict[str, str]], recipe: Recipe) -> bool:
    pair = first_user_and_assistant(conversations)
    if pair is None:
        return False
    text = conversation_text(conversations)
    if len(text) < recipe.min_chars or len(text) > recipe.max_chars:
        return False
    if noisy(text):
        return False
    if recipe.name == "glaive_tool_calling":
        return True
    return text_score(text) > 0


def load_local_curated(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            conversations = record.get("conversations")
            if isinstance(conversations, list):
                records.append(
                    {
                        "source": record.get("source", "curated_ai_knowledge_v1"),
                        "category": record.get("category", "curated"),
                        "conversations": normalize_turns(conversations),
                    }
                )
    return records


def collect_recipe(recipe: Recipe, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": recipe.split, "streaming": True}
    if recipe.config is not None:
        dataset = load_dataset(recipe.repo, recipe.config, **kwargs)
    else:
        dataset = load_dataset(recipe.repo, **kwargs)
    adapter = ADAPTERS[recipe.adapter]
    records: list[dict[str, Any]] = []
    scanned = 0
    accepted = 0
    rng = random.Random(seed)
    for row in dataset:
        scanned += 1
        conversations = adapter(row)
        if valid_conversation(conversations, recipe):
            category = infer_category(conversations, recipe.name)
            if recipe.name != "glaive_tool_calling" and category == recipe.name:
                continue
            accepted += 1
            records.append(
                {
                    "source": recipe.name,
                    "hf_dataset": recipe.repo,
                    "hf_config": recipe.config,
                    "category": category,
                    "conversations": conversations,
                }
            )
            if len(records) >= recipe.target:
                break
        if scanned >= recipe.max_scan:
            break
    rng.shuffle(records)
    return records, {"scanned": scanned, "accepted": accepted, "kept": len(records)}


def infer_category(conversations: list[dict[str, str]], fallback: str) -> str:
    text = conversation_text(conversations).lower()
    checks = [
        ("agent", ["agent", "tool calling", "tool use", "function calling", "工具调用", "函数调用", "<tool_call>"]),
        ("vlm", ["vision-language", "multimodal", "clip", "llava", "视觉语言", "多模态"]),
        ("vla", ["vla", "robot learning", "robotics", "vision-language-action", "机器人"]),
        ("rag", ["retrieval", "embedding", "bm25", "rag", "检索", "向量"]),
        ("peft", ["lora", "qlora", "fine-tuning", "微调"]),
        ("moe", ["moe", "expert", "routing", "专家", "路由"]),
        ("llm_architecture", ["transformer", "attention", "tokenizer", "language model", "注意力", "大模型"]),
        ("ml", ["machine learning", "deep learning", "neural network", "机器学习", "深度学习", "神经网络"]),
    ]
    for category, needles in checks:
        if any(needle in text for needle in needles):
            return category
    return fallback


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        conversations = record.get("conversations")
        if not isinstance(conversations, list):
            continue
        pair = first_user_and_assistant(conversations)
        if pair is None:
            continue
        key = re.sub(r"\s+", " ", (pair[0] + "\n" + pair[1]).lower())[:1000]
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect AI-related SFT records from public HF datasets.")
    parser.add_argument("--out-path", type=Path, default=DATASET_DIR / "ai_hf_sft.jsonl")
    parser.add_argument("--curated-path", type=Path, default=DATASET_DIR / "ai_knowledge_sft.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sources", type=int, default=0, help="Debug: only run the first N recipes.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    recipes = RECIPES[: args.max_sources] if args.max_sources > 0 else RECIPES
    all_records = load_local_curated(args.curated_path)
    print(f"local curated: {len(all_records)}")
    for index, recipe in enumerate(recipes):
        print(f"\ncollecting {recipe.name} from {recipe.repo}...")
        records, stats = collect_recipe(recipe, seed=args.seed + index)
        all_records.extend(records)
        print(
            f"{recipe.name}: scanned={stats['scanned']:,} "
            f"accepted={stats['accepted']:,} kept={stats['kept']:,}"
        )
    random.Random(args.seed).shuffle(all_records)
    all_records = dedupe(all_records)
    if not args.dry_run:
        write_jsonl(args.out_path, all_records)
    print(f"\ntotal after dedupe: {len(all_records):,}")
    print(f"out: {args.out_path}")


if __name__ == "__main__":
    main()
