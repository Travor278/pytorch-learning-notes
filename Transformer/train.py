# %% [markdown]
# # Train Transformer On Small Real Text
# 这个脚本把前面实现的 Transformer 主干真正接到一个小型文本训练流程上。
#
# 设计目标：
# - 用真实短文本而不是手写 tensor 训练 Transformer
# - 让 CPU 上也能稳定跑通完整训练链路
# - 保持代码可读，重点展示 text -> id -> batch -> mask -> model -> loss 的全过程
#
# 数据与词表：
# - 数据来自 `Transformer/data/parallel_toy.tsv`
# - 使用空格分词，分别构建 source / target vocabulary
# - 特殊 token：`<pad> <bos> <eos> <unk>`
#
# 训练链路是：
# - 读取 `parallel_toy.tsv`
# - 构建 source / target vocabulary
# - 文本转 token id
# - padding 成 batch
# - teacher forcing
# - mask 构造
# - 前向、loss、反向传播、参数更新
# - greedy decode 观察效果
#
# 训练输出：
# - train / val loss 日志
# - 若干样本的 greedy decode 预测
# - `Transformer/checkpoints/best.pt` 最优检查点
#
# 第一版非目标：
# - 不接外部 tokenizer
# - 不做 beam search / BLEU
# - 不做 resume / 混合精度 / 分布式训练

from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

try:
    from Transformer import Transformer
    from create_mask import (
        create_memory_mask,
        create_src_padding_mask,
        create_tgt_mask,
    )
except ImportError:
    from Transformer.Transformer import Transformer
    from Transformer.create_mask import (
        create_memory_mask,
        create_src_padding_mask,
        create_tgt_mask,
    )


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
SOURCE_SUBJECTS = (
    "the teacher",
    "the child",
    "our coach",
    "my friend",
    "they",
    "she",
    "you",
    "we",
    "he",
    "i",
)
TARGET_SUBJECTS = (
    "我们的 教练",
    "我的 朋友",
    "老师",
    "孩子",
    "他们",
    "我们",
    "她",
    "你",
    "他",
    "我",
)
TIME_PREFIXES = (
    (("today",), ("今天",)),
    (("now",), ("现在",)),
    (("in", "the", "morning"), ("在", "早上")),
    (("at", "night"), ("在", "夜里")),
    (("after", "class"), ("在", "下课", "后")),
)
PLACE_SUFFIXES = (
    (("at", "home"), ("在", "家里")),
    (("at", "school"), ("在", "学校")),
    (("in", "the", "library"), ("在", "图书馆")),
    (("in", "the", "park"), ("在", "公园", "里")),
    (("in", "the", "classroom"), ("在", "教室", "里")),
)
PARTNER_PAIRS = (
    (("the", "teacher"), ("老师",)),
    (("the", "child"), ("孩子",)),
    (("my", "friend"), ("我的", "朋友")),
    (("our", "coach"), ("我们的", "教练")),
)
VERB_TO_BASE = {
    "likes": "like",
    "reads": "read",
    "watches": "watch",
    "drinks": "drink",
    "eats": "eat",
    "learns": "learn",
    "writes": "write",
    "opens": "open",
    "closes": "close",
    "carries": "carry",
    "draws": "draw",
    "cleans": "clean",
    "visits": "visit",
    "buys": "buy",
}


def tokenize(text: str) -> list[str]:
    return text.strip().split()


def split_prefix(tokens: list[str], candidates: tuple[str, ...]) -> tuple[list[str], list[str]] | None:
    for candidate in sorted(candidates, key=lambda item: len(item.split()), reverse=True):
        candidate_tokens = candidate.split()
        if tokens[: len(candidate_tokens)] == candidate_tokens:
            return candidate_tokens, tokens[len(candidate_tokens) :]
    return None


def to_base_predicate(predicate_tokens: list[str]) -> list[str]:
    if not predicate_tokens:
        return predicate_tokens

    first = predicate_tokens[0]
    return [VERB_TO_BASE.get(first, first), *predicate_tokens[1:]]


def augment_parallel_pairs(
    pairs: list[tuple[str, str]],
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    augmented: list[tuple[str, str]] = []

    for src_text, tgt_text in pairs:
        augmented.append((src_text, tgt_text))

        src_tokens = tokenize(src_text)
        tgt_tokens = tokenize(tgt_text)
        src_split = split_prefix(src_tokens, SOURCE_SUBJECTS)
        tgt_split = split_prefix(tgt_tokens, TARGET_SUBJECTS)

        if src_split is None or tgt_split is None:
            continue

        src_subject, src_predicate = src_split
        tgt_subject, tgt_predicate = tgt_split
        if not src_predicate or not tgt_predicate:
            continue

        base_predicate = to_base_predicate(src_predicate)

        for time_src, time_tgt in rng.sample(TIME_PREFIXES, k=min(3, len(TIME_PREFIXES))):
            augmented.append(
                (
                    " ".join([*time_src, *src_subject, *src_predicate]),
                    " ".join([*time_tgt, *tgt_subject, *tgt_predicate]),
                )
            )

        for place_src, place_tgt in rng.sample(PLACE_SUFFIXES, k=min(3, len(PLACE_SUFFIXES))):
            augmented.append(
                (
                    " ".join([*src_subject, *src_predicate, *place_src]),
                    " ".join([*tgt_subject, *place_tgt, *tgt_predicate]),
                )
            )

        augmented.append(
            (
                " ".join([*src_subject, "cannot", *base_predicate]),
                " ".join([*tgt_subject, "不能", *tgt_predicate]),
            )
        )
        augmented.append(
            (
                " ".join(["can", *src_subject, *base_predicate]),
                " ".join([*tgt_subject, "能", *tgt_predicate, "吗"]),
            )
        )

        partner_src, partner_tgt = rng.choice(PARTNER_PAIRS)
        if partner_src != tuple(src_subject):
            augmented.append(
                (
                    " ".join([*src_subject, *src_predicate, "with", *partner_src]),
                    " ".join([*tgt_subject, "和", *partner_tgt, "一起", *tgt_predicate]),
                )
            )

    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for pair in augmented:
        if pair not in seen:
            seen.add(pair)
            deduped.append(pair)

    return deduped


def load_parallel_tsv(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue

            src_text, tgt_text = parts[0].strip(), parts[1].strip()
            if not src_text or not tgt_text:
                continue

            pairs.append((src_text, tgt_text))

    return pairs


def build_vocab(texts: list[str], special_tokens: tuple[str, ...] = SPECIAL_TOKENS) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab: dict[str, int] = {token: idx for idx, token in enumerate(special_tokens)}
    sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    for token, _ in sorted_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {idx: token for token, idx in vocab.items()}


def encode_source(text: str, vocab: dict[str, int]) -> list[int]:
    unk_idx = vocab["<unk>"]
    return [vocab.get(token, unk_idx) for token in tokenize(text)]


def encode_target(text: str, vocab: dict[str, int]) -> list[int]:
    unk_idx = vocab["<unk>"]
    bos_idx = vocab["<bos>"]
    eos_idx = vocab["<eos>"]

    token_ids = [vocab.get(token, unk_idx) for token in tokenize(text)]
    return [bos_idx, *token_ids, eos_idx]


def build_parallel_dataset(
    pairs: list[tuple[str, str]],
    src_vocab: dict[str, int],
    tgt_vocab: dict[str, int],
) -> list[tuple[list[int], list[int]]]:
    dataset: list[tuple[list[int], list[int]]] = []
    for src_text, tgt_text in pairs:
        dataset.append((encode_source(src_text, src_vocab), encode_target(tgt_text, tgt_vocab)))
    return dataset


def pad_sequences(seqs: list[list[int]], pad_idx: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in seqs]
    return torch.tensor(padded, dtype=torch.long)


def collate_batch(
    examples: list[tuple[list[int], list[int]]],
    pad_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_batch = [example[0] for example in examples]
    tgt_batch = [example[1] for example in examples]
    return pad_sequences(src_batch, pad_idx), pad_sequences(tgt_batch, pad_idx)


def split_teacher_forcing_targets(tgt_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return tgt_batch[:, :-1], tgt_batch[:, 1:]


def make_batches(
    dataset: list[tuple[list[int], list[int]]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[list[tuple[list[int], list[int]]]]:
    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    batches: list[list[tuple[list[int], list[int]]]] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batches.append([dataset[idx] for idx in batch_indices])

    return batches


def decode_ids(ids: list[int], inv_vocab: dict[int, str]) -> str:
    tokens: list[str] = []
    for idx in ids:
        token = inv_vocab.get(idx, "<unk>")
        if token in {"<pad>", "<bos>", "<eos>"}:
            continue
        tokens.append(token)
    return " ".join(tokens)


def greedy_decode(
    model: Transformer,
    src_tensor: torch.Tensor,
    src_pad_idx: int,
    tgt_vocab: dict[str, int],
    max_decode_len: int,
) -> list[int]:
    model.eval()

    bos_idx = tgt_vocab["<bos>"]
    eos_idx = tgt_vocab["<eos>"]

    src_mask = create_src_padding_mask(src_tensor, src_pad_idx)
    generated = torch.tensor([[bos_idx]], dtype=torch.long)
    memory, _ = model.encode(src_tensor, src_mask=src_mask)

    with torch.no_grad():
        for _ in range(max_decode_len):
            tgt_mask = create_tgt_mask(generated, tgt_vocab["<pad>"])
            memory_mask = create_memory_mask(generated, src_tensor, src_pad_idx)
            decoder_out, _, _ = model.decode(
                generated,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
            logits = model.generator(decoder_out)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if int(next_token.item()) == eos_idx:
                break

    return generated[0].tolist()


def train_one_epoch(
    model: Transformer,
    dataset: list[tuple[list[int], list[int]]],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pad_idx: int,
    epoch_seed: int,
) -> float:
    model.train()

    total_loss = 0.0
    total_batches = 0
    batches = make_batches(dataset, batch_size=batch_size, shuffle=True, seed=epoch_seed)

    for batch in batches:
        src_batch, tgt_batch = collate_batch(batch, pad_idx=pad_idx)
        tgt_input, tgt_output = split_teacher_forcing_targets(tgt_batch)

        src_mask = create_src_padding_mask(src_batch, pad_idx)
        tgt_mask = create_tgt_mask(tgt_input, pad_idx)
        memory_mask = create_memory_mask(tgt_input, src_batch, pad_idx)

        optimizer.zero_grad()
        logits, _, _, _ = model(
            src_batch,
            tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate(
    model: Transformer,
    dataset: list[tuple[list[int], list[int]]],
    batch_size: int,
    criterion: nn.Module,
    pad_idx: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        batches = make_batches(dataset, batch_size=batch_size, shuffle=False, seed=0)
        for batch in batches:
            src_batch, tgt_batch = collate_batch(batch, pad_idx=pad_idx)
            tgt_input, tgt_output = split_teacher_forcing_targets(tgt_batch)

            src_mask = create_src_padding_mask(src_batch, pad_idx)
            tgt_mask = create_tgt_mask(tgt_input, pad_idx)
            memory_mask = create_memory_mask(tgt_input, src_batch, pad_idx)

            logits, _, _, _ = model(
                src_batch,
                tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


def preview_predictions(
    model: Transformer,
    examples: list[tuple[str, str]],
    src_vocab: dict[str, int],
    tgt_vocab: dict[str, int],
    inv_tgt_vocab: dict[int, str],
    max_decode_len: int,
    count: int,
) -> None:
    print("  sample predictions:")

    for src_text, tgt_text in examples[:count]:
        src_ids = encode_source(src_text, src_vocab)
        src_tensor = torch.tensor([src_ids], dtype=torch.long)
        pred_ids = greedy_decode(
            model,
            src_tensor,
            src_pad_idx=src_vocab["<pad>"],
            tgt_vocab=tgt_vocab,
            max_decode_len=max_decode_len,
        )
        pred_text = decode_ids(pred_ids, inv_tgt_vocab)

        print(f"    src : {src_text}")
        print(f"    tgt : {tgt_text}")
        print(f"    pred: {pred_text}")


def save_checkpoint(
    path: Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    src_vocab: dict[str, int],
    tgt_vocab: dict[str, int],
    config: argparse.Namespace,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "config": vars(config),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Train the teaching Transformer on a tiny parallel text dataset.")
    parser.add_argument("--data-path", type=Path, default=root / "data" / "parallel_toy.tsv")
    parser.add_argument("--checkpoint-path", type=Path, default=root / "checkpoints" / "best.pt")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--preview-every", type=int, default=24)
    parser.add_argument("--max-decode-len", type=int, default=12)
    parser.add_argument("--quick", action="store_true", help="Run a shorter smoke-training version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.quick:
        args.epochs = min(args.epochs, 12)
        args.batch_size = min(args.batch_size, 8)
        args.d_model = min(args.d_model, 96)
        args.d_ff = min(args.d_ff, 256)
        args.preview_every = 4
        args.patience = min(args.patience, 8)

    print("Starting Transformer training...")
    if not args.quick:
        print("  profile      : default profile aims for a longer CPU training run; use --quick for a short smoke test")
    print(f"  data path    : {args.data_path}")
    print(f"  epochs       : {args.epochs}")
    print(f"  batch size   : {args.batch_size}")
    print(f"  d_model      : {args.d_model}")
    print(f"  enc/dec      : {args.num_encoder_layers}/{args.num_decoder_layers}")

    raw_pairs = load_parallel_tsv(args.data_path)
    if not raw_pairs:
        raise ValueError(f"No valid parallel pairs found in {args.data_path}")

    rng = random.Random(args.seed)
    shuffled_pairs = raw_pairs[:]
    rng.shuffle(shuffled_pairs)

    split_idx = max(1, int(len(shuffled_pairs) * (1.0 - args.val_ratio)))
    train_pairs_raw = shuffled_pairs[:split_idx]
    val_pairs = shuffled_pairs[split_idx:]
    if not val_pairs:
        val_pairs = train_pairs_raw[: max(1, min(8, len(train_pairs_raw)))]

    train_pairs = augment_parallel_pairs(train_pairs_raw, seed=args.seed + 99)

    src_vocab = build_vocab([src for src, _ in train_pairs], SPECIAL_TOKENS)
    tgt_vocab = build_vocab([tgt for _, tgt in train_pairs], SPECIAL_TOKENS)
    inv_tgt_vocab = invert_vocab(tgt_vocab)

    train_dataset = build_parallel_dataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = build_parallel_dataset(val_pairs, src_vocab, tgt_vocab)

    max_src_len = max(len(src_ids) for src_ids, _ in train_dataset + val_dataset)
    max_tgt_len = max(len(tgt_ids) for _, tgt_ids in train_dataset + val_dataset)
    max_len = max(max_src_len, max_tgt_len) + 4

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=max_len,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab["<pad>"],
        label_smoothing=args.label_smoothing,
    )

    best_val_loss = math.inf
    epochs_without_improve = 0
    start_time = time.perf_counter()

    print(f"  raw train    : {len(train_pairs_raw)}")
    print(f"  aug train    : {len(train_pairs)}")
    print(f"  val pairs    : {len(val_pairs)}")
    print(f"  src vocab    : {len(src_vocab)}")
    print(f"  tgt vocab    : {len(tgt_vocab)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataset=train_dataset,
            batch_size=args.batch_size,
            optimizer=optimizer,
            criterion=criterion,
            pad_idx=tgt_vocab["<pad>"],
            epoch_seed=args.seed + epoch,
        )
        val_loss = evaluate(
            model=model,
            dataset=val_dataset,
            batch_size=args.batch_size,
            criterion=criterion,
            pad_idx=tgt_vocab["<pad>"],
        )

        elapsed = time.perf_counter() - start_time
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"elapsed={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            save_checkpoint(
                args.checkpoint_path,
                model,
                optimizer,
                src_vocab,
                tgt_vocab,
                args,
                best_val_loss,
            )
            print(f"  saved new best checkpoint -> {args.checkpoint_path}")
        else:
            epochs_without_improve += 1

        if epoch == 1 or epoch % args.preview_every == 0 or epoch == args.epochs:
            preview_predictions(
                model=model,
                examples=val_pairs,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                inv_tgt_vocab=inv_tgt_vocab,
                max_decode_len=args.max_decode_len,
                count=args.preview_count,
            )

        if epochs_without_improve >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs without validation improvement.")
            break

    total_elapsed = time.perf_counter() - start_time
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {args.checkpoint_path}")
    print(f"Total elapsed time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
