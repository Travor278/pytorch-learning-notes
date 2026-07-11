from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from mini_llm.model import MiniMindForCausalLM, build_config_from_preset, list_model_presets
    from mini_llm.trainer.train_full_sft import _config_from_checkpoint
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mini_llm.model import MiniMindForCausalLM, build_config_from_preset, list_model_presets
    from mini_llm.trainer.train_full_sft import _config_from_checkpoint

try:
    from .preference_data import PreferenceCollator, PreferenceJsonlDataset
    from .tokenizer import load_or_create_tokenizer
    from .train_pretrain import autocast_context, resolve_device, resolve_precision, save_checkpoint
    from .trainer_utils import CHECKPOINT_DIR, DATASET_DIR, count_parameters, set_seed
except ImportError:
    from preference_data import PreferenceCollator, PreferenceJsonlDataset
    from tokenizer import load_or_create_tokenizer
    from train_pretrain import autocast_context, resolve_device, resolve_precision, save_checkpoint
    from trainer_utils import CHECKPOINT_DIR, DATASET_DIR, count_parameters, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO preference training for mini_llm.")
    parser.add_argument("--data-path", type=Path, default=DATASET_DIR / "dpo.jsonl")
    parser.add_argument("--tokenizer-path", type=Path, default=DATASET_DIR.parent / "tokenizer.json")
    parser.add_argument("--out-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--model-size", choices=list_model_presets(), default="tiny")
    parser.add_argument("--init-from", default=None, help="Policy checkpoint, or 'latest' for sft_last.pt.")
    parser.add_argument("--reference-from", default=None, help="Reference checkpoint. Defaults to init-from.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_checkpoint(value: str | None, out_dir: Path, default_name: str) -> Path | None:
    if value is None:
        return None
    return out_dir / default_name if value == "latest" else Path(value)


def build_model(
    checkpoint_path: Path | None,
    args: argparse.Namespace,
    tokenizer,
    device: torch.device,
) -> MiniMindForCausalLM:
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
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


def sequence_log_probs(
    model: MiniMindForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = output.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Missing DPO data file: {args.data_path}")

    tokenizer = load_or_create_tokenizer(args.tokenizer_path)
    dataset = PreferenceJsonlDataset(args.data_path, tokenizer, args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=PreferenceCollator(tokenizer.pad_token_id),
    )
    device = resolve_device(args.device)
    precision_name, autocast_dtype = resolve_precision(device, args.precision)
    policy_path = _resolve_checkpoint(args.init_from, args.out_dir, "sft_last.pt")
    reference_path = _resolve_checkpoint(args.reference_from, args.out_dir, "sft_last.pt") or policy_path
    policy = build_model(policy_path, args, tokenizer, device)
    reference = build_model(reference_path, args, tokenizer, device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"device: {device}")
    print(f"precision: {precision_name}")
    print(f"dpo examples: {len(dataset)}")
    print(f"policy params: {count_parameters(policy):,}")
    print("note: full DPO keeps policy and reference in memory; use tiny/LoRA on 8GB for real runs.")
    if args.dry_run:
        print("dry_run: built policy/reference models and dataset; skipped DPO.")
        return

    policy.train()
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
                policy_chosen = sequence_log_probs(
                    policy,
                    batch["chosen_input_ids"],
                    batch["chosen_labels"],
                    batch["chosen_attention_mask"],
                )
                policy_rejected = sequence_log_probs(
                    policy,
                    batch["rejected_input_ids"],
                    batch["rejected_labels"],
                    batch["rejected_attention_mask"],
                )
                with torch.no_grad():
                    ref_chosen = sequence_log_probs(
                        reference,
                        batch["chosen_input_ids"],
                        batch["chosen_labels"],
                        batch["chosen_attention_mask"],
                    )
                    ref_rejected = sequence_log_probs(
                        reference,
                        batch["rejected_input_ids"],
                        batch["rejected_labels"],
                        batch["rejected_attention_mask"],
                    )
                raw_loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta)

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
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
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
                print(f"step {step:05d} | dpo_loss {last_loss:.4f}")
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(args.out_dir / f"dpo_step_{step}.pt", policy, optimizer, policy.config, step, last_loss, micro_step, args)
            if step >= args.max_steps:
                break

    save_checkpoint(args.out_dir / "dpo_last.pt", policy, optimizer, policy.config, step, last_loss, micro_step, args)
    print(f"saved checkpoint: {args.out_dir / 'dpo_last.pt'}")


if __name__ == "__main__":
    main()
