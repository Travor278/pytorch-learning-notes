from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn

if __package__ in (None, ""):
    import sys

    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    from GPT import GPTModel
    from tokenizer import CharTokenizer
    from bpe_tokenizer import BPETokenizer
else:
    from .GPT import GPTModel
    from .tokenizer import CharTokenizer
    from .bpe_tokenizer import BPETokenizer


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine decay with linear warmup.

    - step < warmup_steps : LR 从 0 线性爬升到 max_lr
    - warmup_steps <= step < max_steps : 按余弦曲线从 max_lr 衰减到 min_lr
    - step >= max_steps : 保持 min_lr
    """
    if step < warmup_steps:
        # 线性 warmup：让模型在训练初期用小 LR "热身"，避免初始梯度过大导致震荡
        return max_lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    # cosine 衰减：progress 从 0 → 1，cos 从 1 → -1，LR 从 max_lr → min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_data(data: torch.Tensor, val_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    split_idx = int(len(data) * (1.0 - val_ratio))
    split_idx = max(2, min(split_idx, len(data) - 2))
    return data[:split_idx], data[split_idx:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError("data length must be larger than block_size")

    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPTModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_steps: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()

    def _avg_loss(data: torch.Tensor) -> float:
        losses: list[float] = []
        for _ in range(eval_steps):
            xb, yb = get_batch(data, batch_size, block_size, device)
            logits, _, _ = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            losses.append(loss.item())
        return sum(losses) / len(losses)

    train_loss = _avg_loss(train_data)
    val_loss = _avg_loss(val_data)
    model.train()
    return train_loss, val_loss


def save_checkpoint(
    path: Path,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer | BPETokenizer,
    config: dict,
    step: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )
    tokenizer.save(path.with_suffix(".tokenizer.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal char-level GPT.")
    parser.add_argument("--text-path", type=Path, required=True, help="Path to a plain text file.")
    parser.add_argument("--out-dir", type=Path, default=Path("GPT/checkpoints"), help="Checkpoint output directory.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (max_lr for cosine scheduler).")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum LR at end of cosine decay (default: lr/10).")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Linear warmup steps before cosine decay.")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a checkpoint (.pt file).")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    # tokenizer 选择：char（字符级）或 bpe（Byte Pair Encoding）
    parser.add_argument(
        "--tokenizer", type=str, default="char", choices=["char", "bpe"],
        help="tokenizer 类型：char（默认）或 bpe",
    )
    parser.add_argument(
        "--bpe-vocab-size", type=int, default=None,
        help="BPE 目标词表大小（仅 --tokenizer bpe 时有效；默认 = 初始字符数 + 2000）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.max_steps = min(args.max_steps, 80)
        args.eval_interval = min(args.eval_interval, 20)
        args.eval_steps = min(args.eval_steps, 8)
        args.batch_size = min(args.batch_size, 8)
        args.block_size = min(args.block_size, 48)
        args.d_model = min(args.d_model, 96)
        args.num_layers = min(args.num_layers, 3)
        args.d_ff = min(args.d_ff, 256)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    text = read_text(args.text_path)

    # ── 构建 tokenizer ─────────────────────────────────────────────────────
    if args.tokenizer == "bpe":
        # BPE：先统计初始字符数，再决定目标词表大小
        init_vocab = len(set(text)) + len(("  <pad>", "<bos>", "<eos>", "<unk>"))
        bpe_vocab_size = args.bpe_vocab_size or (init_vocab + 2000)
        print(f"[BPE] 训练 tokenizer  初始词表 {init_vocab} → 目标 {bpe_vocab_size} ...")
        tokenizer: CharTokenizer | BPETokenizer = BPETokenizer.train(
            text, vocab_size=bpe_vocab_size, min_freq=args.min_freq, verbose=False
        )
        print(f"[BPE] 完成，词表大小 : {tokenizer.vocab_size}")
    else:
        tokenizer = CharTokenizer.from_text(text, min_freq=args.min_freq)

    # 编码整段语料；开头加 <bos> 与推理对齐
    # 原来：data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    data = torch.tensor(tokenizer.encode(text, add_bos=True), dtype=torch.long)

    if len(data) <= args.block_size + 1:
        raise ValueError("text is too short for the chosen block_size")

    train_data, val_data = split_data(data, args.val_ratio)
    if len(val_data) <= args.block_size + 1:
        val_data = train_data[-(args.block_size + 2) :].clone()

    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.block_size + 1,
    ).to(device)

    # 原来：固定 LR，不随训练步数变化
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 修改：AdamW 初始 LR 设为 0，后续在训练循环里用 get_lr 手动更新（cosine warmup）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,  # 由 get_lr 控制，初始为 0 对应 warmup 起点
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "max_len": args.block_size + 1,
        "tokenizer_type": args.tokenizer,   # sample.py 加载时按此选择 tokenizer 类
    }

    best_val_loss = float("inf")
    start_step = 0
    ckpt_path = args.out_dir / "gpt_char_best.pt"

    # 从 checkpoint 恢复训练
    if args.resume is not None:
        print(f"resuming from    : {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_step = resume_ckpt["step"]
        best_val_loss = resume_ckpt["val_loss"]
        print(f"resumed at step  : {start_step}  (best val_loss={best_val_loss:.4f})")

    print(f"text_path        : {args.text_path}")
    print(f"text length      : {len(text)} chars")
    print(f"vocab size       : {tokenizer.vocab_size}")
    print(f"train tokens     : {len(train_data)}")
    print(f"val tokens       : {len(val_data)}")
    print(f"device           : {device}")
    print(f"warmup_steps     : {args.warmup_steps}")
    print(f"lr (max → min)   : {args.lr} → {args.min_lr}")

    start_time = time.time()
    model.train()

    pbar = tqdm(
        range(start_step + 1, args.max_steps + 1),
        initial=start_step,
        total=args.max_steps,
        desc="training",
        unit="step",
        dynamic_ncols=True,   # 自动适应终端宽度
    )

    for step in pbar:
        # ── cosine warmup ────────────────────────────────────────────────────
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
        logits, _, _ = model(xb)
        loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # 进度条右侧实时显示 train_loss 和 lr
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        if step % args.eval_interval == 0 or step == 1 or step == args.max_steps:
            train_loss, val_loss = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                eval_steps=args.eval_steps,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
                criterion=criterion,
            )
            elapsed = time.time() - start_time
            # tqdm.write 不会打乱进度条，相当于在进度条上方插一行
            saved = " ★" if val_loss < best_val_loss else ""
            tqdm.write(
                f"step {step:4d} | train={train_loss:.4f} | "
                f"val={val_loss:.4f} | {elapsed:.0f}s{saved}"
            )
            # 同步更新进度条 postfix，加入 val_loss
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", val=f"{val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    config=config,
                    step=step,
                    val_loss=val_loss,
                )

    pbar.close()
    print(f"best val loss    : {best_val_loss:.4f}")
    print(f"checkpoint saved : {ckpt_path}")
    print(f"tokenizer saved  : {ckpt_path.with_suffix('.tokenizer.json')}")


if __name__ == "__main__":
    main()
