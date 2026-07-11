from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from mini_llm.model import (
        MiniMindConfig,
        MiniMindForCausalLM,
        build_config_from_preset,
        list_model_presets,
    )
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mini_llm.model import (
        MiniMindConfig,
        MiniMindForCausalLM,
        build_config_from_preset,
        list_model_presets,
    )

try:
    from .pretrain_data import CausalLMCollator, PretrainJsonlDataset, StreamingPretrainJsonlDataset
    from .tokenizer import load_or_create_tokenizer
    from .trainer_utils import CHECKPOINT_DIR, DATASET_DIR, count_parameters, set_seed
except ImportError:
    from pretrain_data import CausalLMCollator, PretrainJsonlDataset, StreamingPretrainJsonlDataset
    from tokenizer import load_or_create_tokenizer
    from trainer_utils import CHECKPOINT_DIR, DATASET_DIR, count_parameters, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-card pretraining for mini_llm.")
    parser.add_argument("--data-path", type=Path, default=DATASET_DIR / "pretrain.jsonl")
    parser.add_argument("--tokenizer-path", type=Path, default=DATASET_DIR.parent / "tokenizer.json")
    parser.add_argument("--out-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--model-size", choices=list_model_presets(), default="tiny")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Load all pretrain tokens into memory. Streaming is the default for large JSONL files.",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Training precision. 'auto' uses bf16 on supported CUDA devices.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Checkpoint path to resume from, or 'latest' to use out-dir/pretrain_last.pt.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build data/model and print stats without training.")
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
        if device.type not in {"cuda", "cpu"}:
            raise ValueError("bf16 autocast is only supported here on CUDA or CPU.")
        return "bf16", torch.bfloat16
    if precision_arg == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 training is only enabled for CUDA.")
        return "fp16", torch.float16

    raise ValueError(f"Unsupported precision: {precision_arg}")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def build_config(model_size: str, vocab_size: int, block_size: int, tokenizer) -> MiniMindConfig:
    return build_config_from_preset(
        model_size,
        tokenizer_vocab_size=vocab_size,
        block_size=block_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def save_checkpoint(
    path: Path,
    model: MiniMindForCausalLM,
    optimizer: torch.optim.Optimizer,
    config: MiniMindConfig,
    step: int,
    loss: float,
    micro_step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "micro_step": micro_step,
            "loss": loss,
            "training_args": serialize_training_args(args),
        },
        path,
    )


def serialize_training_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def resolve_resume_path(resume_from: str | None, out_dir: Path) -> Path | None:
    if resume_from is None:
        return None
    if resume_from == "latest":
        path = out_dir / "pretrain_last.pt"
    else:
        path = Path(resume_from)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for resume: {path}")
    return path


def load_checkpoint(
    path: Path,
    model: MiniMindForCausalLM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Missing data file: {args.data_path}. "
            "Create JSONL lines like {\"text\": \"...\"} or pass --data-path."
        )

    tokenizer = load_or_create_tokenizer(args.tokenizer_path)
    dataset = (
        PretrainJsonlDataset(
            path=args.data_path,
            tokenizer=tokenizer,
            block_size=args.block_size,
        )
        if args.in_memory
        else StreamingPretrainJsonlDataset(
            path=args.data_path,
            tokenizer=tokenizer,
            block_size=args.block_size,
        )
    )
    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.in_memory,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    device = resolve_device(args.device)
    precision_name, autocast_dtype = resolve_precision(device, args.precision)
    config = build_config(
        model_size=args.model_size,
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        tokenizer=tokenizer,
    )
    model = MiniMindForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    resume_path = resolve_resume_path(args.resume_from, args.out_dir)
    start_step = 0
    micro_step = 0
    last_loss = 0.0
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, optimizer, device)
        start_step = int(checkpoint.get("step", 0))
        micro_step = int(checkpoint.get("micro_step", 0))
        last_loss = float(checkpoint.get("loss", 0.0))

    print(f"device: {device}")
    print(f"precision: {precision_name}")
    print(f"data mode: {'in_memory' if args.in_memory else 'streaming'}")
    if args.in_memory:
        print(f"data examples: {len(dataset)}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"model params: {count_parameters(model):,}")
    print(f"model_size: {args.model_size}")
    print(f"micro_batch_size: {args.batch_size}")
    print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"effective_batch_size: {args.batch_size * args.gradient_accumulation_steps}")
    if resume_path is not None:
        print(f"resumed from: {resume_path} at step {start_step}")

    if args.dry_run:
        print("dry_run: built tokenizer, dataset, and model; skipped training.")
        return

    model.train()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=precision_name == "fp16" and device.type == "cuda",
    )
    step = start_step
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    accumulated_batches = 0
    while step < args.max_steps:
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, autocast_dtype):
                output = model(**batch)
            if output.loss is None:
                raise RuntimeError("Model did not return loss.")

            raw_loss = output.loss
            loss = raw_loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1
            accumulated_batches += 1
            accumulated_loss += float(raw_loss.detach().cpu())
            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            if args.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            last_loss = accumulated_loss / max(accumulated_batches, 1)
            accumulated_loss = 0.0
            accumulated_batches = 0
            if step % args.log_every == 0 or step == 1:
                print(f"step {step:05d} | loss {last_loss:.4f}")

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    args.out_dir / f"pretrain_step_{step}.pt",
                    model,
                    optimizer,
                    config,
                    step,
                    last_loss,
                    micro_step,
                    args,
                )

            if step >= args.max_steps:
                break

    save_checkpoint(
        args.out_dir / "pretrain_last.pt",
        model,
        optimizer,
        config,
        step,
        last_loss,
        micro_step,
        args,
    )
    print(f"saved checkpoint: {args.out_dir / 'pretrain_last.pt'}")


if __name__ == "__main__":
    main()
