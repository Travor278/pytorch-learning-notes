from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

try:
    from mini_llm.model import MiniMindConfig, MiniMindForCausalLM
    from mini_llm.trainer.tokenizer import load_or_create_tokenizer
    from mini_llm.trainer.trainer_utils import CHECKPOINT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mini_llm.model import MiniMindConfig, MiniMindForCausalLM
    from mini_llm.trainer.tokenizer import load_or_create_tokenizer
    from mini_llm.trainer.trainer_utils import CHECKPOINT_DIR, PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a mini_llm checkpoint and generate text.")
    parser.add_argument(
        "--checkpoint",
        "--load-from",
        dest="checkpoint",
        default=CHECKPOINT_DIR / "pretrain_last.pt",
        help="Checkpoint path, or 'latest' for mini_llm/checkpoints/pretrain_last.pt.",
    )
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--prompt", default="你好", help="Prompt for smoke testing.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Inference precision. 'auto' uses bf16 on supported CUDA devices.",
    )
    parser.add_argument(
        "--show-full-text",
        action="store_true",
        help="Print prompt plus completion instead of only the generated completion.",
    )
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_precision(device: torch.device, precision_arg: str) -> tuple[str, torch.dtype | None]:
    if precision_arg == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return "bf16", torch.bfloat16
            return "fp16", torch.float16
        return "fp32", None
    if precision_arg == "fp32":
        return "fp32", None
    if precision_arg == "bf16":
        return "bf16", torch.bfloat16
    if precision_arg == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 inference is only enabled for CUDA.")
        return "fp16", torch.float16
    raise ValueError(f"Unsupported precision: {precision_arg}")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def resolve_checkpoint_path(value: str | Path) -> Path:
    if str(value) == "latest":
        return CHECKPOINT_DIR / "pretrain_last.pt"
    return Path(value)


def config_from_checkpoint(payload: dict[str, Any]) -> MiniMindConfig:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Checkpoint does not contain a config dictionary.")

    valid_keys = {field.name for field in fields(MiniMindConfig)}
    clean_payload = {key: value for key, value in config_payload.items() if key in valid_keys}
    return MiniMindConfig(**clean_payload)


def tokenizer_path_from_args(args: argparse.Namespace, checkpoint: dict[str, Any]) -> Path:
    if args.tokenizer_path is not None:
        return args.tokenizer_path

    training_args = checkpoint.get("training_args", {})
    if isinstance(training_args, dict) and training_args.get("tokenizer_path"):
        return Path(training_args["tokenizer_path"])
    return PROJECT_ROOT / "tokenizer.json"


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MiniMindForCausalLM, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = config_from_checkpoint(checkpoint)
    model = MiniMindForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main(argv: list[str] | None = None) -> str:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    precision_name, autocast_dtype = resolve_precision(device, args.precision)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    tokenizer = load_or_create_tokenizer(tokenizer_path_from_args(args, checkpoint))

    prompt_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with autocast_context(device, autocast_dtype):
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            vocab_size_limit=tokenizer.vocab_size,
            suppress_token_ids=[tokenizer.pad_token_id],
        )

    generated = generated_ids[0].detach().cpu().tolist()
    completion = tokenizer.decode(generated[len(prompt_ids) :])
    text = tokenizer.decode(generated) if args.show_full_text else completion

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print(f"precision: {precision_name}")
    print(f"step: {checkpoint.get('step', 0)}")
    print(text)
    return text


if __name__ == "__main__":
    main()
