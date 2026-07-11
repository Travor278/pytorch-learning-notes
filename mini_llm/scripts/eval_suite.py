from __future__ import annotations

import argparse
import json
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


DEFAULT_CASES: list[dict[str, str]] = [
    {
        "category": "identity",
        "name": "model_identity",
        "prompt": "你是谁？你由谁开发？",
    },
    {
        "category": "chat",
        "name": "movie_recommendation",
        "prompt": "你最近有没有看过什么好电影？推荐三部并简单说明。",
    },
    {
        "category": "writing",
        "name": "ai_short_essay",
        "prompt": "写一篇关于人工智能对未来影响的短文，控制在150字以内。",
    },
    {
        "category": "summary",
        "name": "ai_governance_summary",
        "prompt": "请摘要这段文字：人工智能正在改变医疗、教育和制造业。它可以提高效率，也会带来就业结构变化和隐私风险，因此需要技术发展与治理同步推进。",
    },
    {
        "category": "code",
        "name": "word_count_function",
        "prompt": "请写一个 Python 函数，将一段文本转换为包含单词及其出现次数的字典。",
    },
    {
        "category": "math",
        "name": "simple_arithmetic_en",
        "prompt": "What is the result of 25 multiplied by 3 plus 10 divided by 2?",
    },
    {
        "category": "classification",
        "name": "junk_food_examples",
        "prompt": "生成垃圾食品的类型和常见例子。",
    },
    {
        "category": "llm",
        "name": "lora_core",
        "prompt": "请简单解释一下 LoRA 的核心思想。",
    },
    {
        "category": "llm",
        "name": "attention_mqa_gqa",
        "prompt": "MHA、MQA 和 GQA 有什么区别？",
    },
    {
        "category": "llm",
        "name": "moe_routing",
        "prompt": "MoE 中 routing 是怎么工作的？",
    },
    {
        "category": "rag",
        "name": "bm25",
        "prompt": "BM25 的核心思想是什么？",
    },
    {
        "category": "vlm",
        "name": "llava_structure",
        "prompt": "LLaVA 如何把视觉特征接入大语言模型？",
    },
    {
        "category": "vla",
        "name": "vla_definition",
        "prompt": "VLA 模型是什么？",
    },
    {
        "category": "agent",
        "name": "agent_components",
        "prompt": "AI Agent 通常由哪些组件组成？",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fixed mini_llm generation eval suite.")
    parser.add_argument("--checkpoint", "--load-from", dest="checkpoint", required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--show-prompts", action="store_true")
    return parser


def default_out_path(checkpoint_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = checkpoint_path.parent.name + "_" + checkpoint_path.stem
    return PROJECT_ROOT / "evals" / f"{safe_name}_{timestamp}.jsonl"


def chat_prompt(user_prompt: str) -> str:
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n"


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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
        "device": str(device),
        "precision": precision_name,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "num_cases": len(DEFAULT_CASES),
    }

    print(f"checkpoint: {checkpoint_path}")
    print(f"step: {metadata['checkpoint_step']}")
    print(f"device: {device}")
    print(f"precision: {precision_name}")
    print(f"out: {out_path}")

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n")
        for case in DEFAULT_CASES:
            completion = generate_one(
                model=model,
                tokenizer=tokenizer,
                device=device,
                autocast_dtype=autocast_dtype,
                prompt=case["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            record = {
                "type": "case",
                **case,
                "completion": completion,
            }
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(f"\n[{case['category']}/{case['name']}]")
            if args.show_prompts:
                print(f"Q: {case['prompt']}")
            print(completion)


if __name__ == "__main__":
    main()
