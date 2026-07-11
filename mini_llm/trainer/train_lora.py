from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from mini_llm.model import (
        apply_lora,
        build_config_from_preset,
        list_model_presets,
        lora_parameter_count,
        merge_lora_weights,
    )
    from mini_llm.trainer.train_full_sft import _config_from_checkpoint
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mini_llm.model import (
        apply_lora,
        build_config_from_preset,
        list_model_presets,
        lora_parameter_count,
        merge_lora_weights,
    )
    from mini_llm.trainer.train_full_sft import _config_from_checkpoint

try:
    from .sft_data import SFTCollator, SFTJsonlDataset
    from .tokenizer import load_or_create_tokenizer
    from .train_pretrain import autocast_context, resolve_device, resolve_precision, save_checkpoint
    from .trainer_utils import CHECKPOINT_DIR, DATASET_DIR, count_parameters, set_seed
    from mini_llm.model import MiniMindForCausalLM
except ImportError:
    from sft_data import SFTCollator, SFTJsonlDataset
    from tokenizer import load_or_create_tokenizer
    from train_pretrain import autocast_context, resolve_device, resolve_precision, save_checkpoint
    from trainer_utils import CHECKPOINT_DIR, DATASET_DIR, count_parameters, set_seed
    from mini_llm.model import MiniMindForCausalLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native LoRA SFT for mini_llm.")
    parser.add_argument("--data-path", type=Path, default=DATASET_DIR / "sft.jsonl")
    parser.add_argument("--tokenizer-path", type=Path, default=DATASET_DIR.parent / "tokenizer.json")
    parser.add_argument("--out-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--model-size", choices=list_model_presets(), default="tiny")
    parser.add_argument("--init-from", default=None, help="Base checkpoint path, or 'latest' for pretrain_last.pt.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_init_path(value: str | None, out_dir: Path) -> Path | None:
    if value is None:
        return None
    return out_dir / "pretrain_last.pt" if value == "latest" else Path(value)


def build_base_model(args: argparse.Namespace, tokenizer, device: torch.device) -> MiniMindForCausalLM:
    init_path = _resolve_init_path(args.init_from, args.out_dir)
    if init_path is not None:
        checkpoint = torch.load(init_path, map_location=device)
        model = MiniMindForCausalLM(_config_from_checkpoint(checkpoint)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    config = build_config_from_preset(
        args.model_size,
        tokenizer_vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return MiniMindForCausalLM(config).to(device)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Missing SFT data file: {args.data_path}")

    tokenizer = load_or_create_tokenizer(args.tokenizer_path)
    dataset = SFTJsonlDataset(args.data_path, tokenizer, args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=SFTCollator(tokenizer.pad_token_id),
    )
    device = resolve_device(args.device)
    precision_name, autocast_dtype = resolve_precision(device, args.precision)
    model = build_base_model(args, tokenizer, device)
    target_modules = tuple(item.strip() for item in args.target_modules.split(",") if item.strip())
    apply_lora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"device: {device}")
    print(f"precision: {precision_name}")
    print(f"sft examples: {len(dataset)}")
    print(f"model params: {count_parameters(model):,}")
    print(f"lora trainable params: {lora_parameter_count(model):,}")
    print(f"target_modules: {','.join(target_modules)}")
    if args.dry_run:
        print("dry_run: built LoRA model and dataset; skipped training.")
        return

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=precision_name == "fp16" and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    last_loss = 0.0
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
                print(f"step {step:05d} | lora_sft_loss {last_loss:.4f}")
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(args.out_dir / f"lora_step_{step}.pt", model, optimizer, model.config, step, last_loss, micro_step, args)
            if step >= args.max_steps:
                break

    save_checkpoint(args.out_dir / "lora_last.pt", model, optimizer, model.config, step, last_loss, micro_step, args)
    merge_lora_weights(model)
    merged_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    save_checkpoint(args.out_dir / "lora_merged.pt", model, merged_optimizer, model.config, step, last_loss, micro_step, args)
    print(f"saved checkpoint: {args.out_dir / 'lora_last.pt'}")
    print(f"saved merged checkpoint: {args.out_dir / 'lora_merged.pt'}")


if __name__ == "__main__":
    main()
