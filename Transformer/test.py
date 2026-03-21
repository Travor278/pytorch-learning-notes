import torch
import torch.nn as nn
from pathlib import Path

try:
    from Transformer import Transformer
    from create_mask import (
        create_memory_mask,
        create_src_padding_mask,
        create_tgt_causal_mask,
        create_tgt_mask,
        create_tgt_padding_mask,
    )
except ImportError:
    from Transformer.Transformer import Transformer
    from Transformer.create_mask import (
        create_memory_mask,
        create_src_padding_mask,
        create_tgt_causal_mask,
        create_tgt_mask,
        create_tgt_padding_mask,
    )


def assert_tensor_finite(x: torch.Tensor, name: str) -> None:
    assert torch.isfinite(x).all(), f"{name} contains NaN or Inf"


def assert_shape(actual: torch.Size, expected: tuple[int, ...], name: str) -> None:
    assert actual == expected, f"{name} shape mismatch: got {actual}, expected {expected}"


def run_text_data_pipeline_checks() -> None:
    print("[text pipeline] train helpers")

    try:
        from train import (
            SPECIAL_TOKENS,
            build_parallel_dataset,
            build_vocab,
            collate_batch,
            encode_source,
            encode_target,
            invert_vocab,
            load_parallel_tsv,
            split_teacher_forcing_targets,
        )
    except ImportError:
        from Transformer.train import (
            SPECIAL_TOKENS,
            build_parallel_dataset,
            build_vocab,
            collate_batch,
            encode_source,
            encode_target,
            invert_vocab,
            load_parallel_tsv,
            split_teacher_forcing_targets,
        )

    data_path = Path(__file__).resolve().parent / "_mini_parallel_test.tsv"
    data_path.write_text(
        "i like apples\t我 喜欢 苹果\n"
        "he likes tea\t他 喜欢 茶\n"
        "we read books\t我们 读 书\n",
        encoding="utf-8",
    )

    try:
        pairs = load_parallel_tsv(data_path)
        assert len(pairs) == 3, f"expected 3 text pairs, got {len(pairs)}"
        assert pairs[0] == ("i like apples", "我 喜欢 苹果")

        src_vocab = build_vocab([src for src, _ in pairs], SPECIAL_TOKENS)
        tgt_vocab = build_vocab([tgt for _, tgt in pairs], SPECIAL_TOKENS)
        inv_tgt_vocab = invert_vocab(tgt_vocab)

        for token in SPECIAL_TOKENS:
            assert token in src_vocab, f"missing special token in source vocab: {token}"
            assert token in tgt_vocab, f"missing special token in target vocab: {token}"

        encoded_src = encode_source("i like apples", src_vocab)
        encoded_tgt = encode_target("我 喜欢 苹果", tgt_vocab)

        assert len(encoded_src) == 3, "source encoding should not add BOS/EOS"
        assert encoded_tgt[0] == tgt_vocab["<bos>"], "target encoding should start with BOS"
        assert encoded_tgt[-1] == tgt_vocab["<eos>"], "target encoding should end with EOS"

        dataset = build_parallel_dataset(pairs, src_vocab, tgt_vocab)
        src_batch, tgt_batch = collate_batch(dataset[:2], pad_idx=tgt_vocab["<pad>"])
        tgt_input, tgt_output = split_teacher_forcing_targets(tgt_batch)

        assert src_batch.dim() == 2, "source batch should be rank-2"
        assert tgt_batch.dim() == 2, "target batch should be rank-2"
        assert_shape(tgt_input.shape, (2, tgt_batch.size(1) - 1), "tgt_input")
        assert_shape(tgt_output.shape, (2, tgt_batch.size(1) - 1), "tgt_output")

        recovered_first_target = [
            inv_tgt_vocab[idx]
            for idx in tgt_batch[0].tolist()
            if idx != tgt_vocab["<pad>"]
        ]
        assert recovered_first_target[0] == "<bos>", "decoded target should begin with BOS"
        assert recovered_first_target[-1] == "<eos>", "decoded target should end with EOS"
    finally:
        if data_path.exists():
            data_path.unlink()


def run_mask_checks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int) -> None:
    src_mask = create_src_padding_mask(src, pad_idx)
    tgt_pad_mask = create_tgt_padding_mask(tgt, pad_idx)
    tgt_causal_mask = create_tgt_causal_mask(tgt.size(1), device=tgt.device)
    tgt_mask = create_tgt_mask(tgt, pad_idx)
    memory_mask = create_memory_mask(tgt, src, pad_idx)

    batch_size, src_len = src.shape
    tgt_len = tgt.size(1)

    assert_shape(src_mask.shape, (batch_size, 1, 1, src_len), "src_mask")
    assert_shape(tgt_pad_mask.shape, (batch_size, 1, 1, tgt_len), "tgt_pad_mask")
    assert_shape(tgt_causal_mask.shape, (tgt_len, tgt_len), "tgt_causal_mask")
    assert_shape(tgt_mask.shape, (batch_size, 1, tgt_len, tgt_len), "tgt_mask")
    assert_shape(memory_mask.shape, (batch_size, 1, tgt_len, src_len), "memory_mask")

    expected_causal = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    )
    assert torch.equal(tgt_causal_mask, expected_causal), "causal mask is incorrect"

    src_pad_positions = src == pad_idx
    for b in range(batch_size):
        for s in range(src_len):
            if src_pad_positions[b, s]:
                assert memory_mask[b, 0, :, s].all(), "memory mask missed a PAD source position"

    tgt_pad_positions = tgt == pad_idx
    for b in range(batch_size):
        for t in range(tgt_len):
            if tgt_pad_positions[b, t]:
                assert tgt_mask[b, 0, :, t].all(), "target PAD position should be masked for all queries"


def build_toy_batch(case: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    pad_idx = 0

    if case == "small":
        src = torch.tensor(
            [
                [11, 12, 13, 14, 15, 0, 0, 0],
                [21, 22, 23, 24, 25, 26, 27, 0],
            ],
            dtype=torch.long,
        )
        tgt = torch.tensor(
            [
                [1, 31, 32, 33, 34, 0, 0],
                [1, 41, 42, 43, 44, 45, 0],
            ],
            dtype=torch.long,
        )
    elif case == "medium":
        src = torch.tensor(
            [
                [5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0],
                [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 0],
                [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 0],
            ],
            dtype=torch.long,
        )
        tgt = torch.tensor(
            [
                [1, 40, 41, 42, 43, 44, 0, 0, 0],
                [1, 50, 51, 52, 53, 54, 55, 0, 0],
                [1, 56, 57, 58, 59, 60, 61, 62, 0],
            ],
            dtype=torch.long,
        )
    else:
        raise ValueError(f"Unknown case: {case}")

    return src, tgt, pad_idx


def run_forward_case(
    *,
    case_name: str,
    src: torch.Tensor,
    tgt: torch.Tensor,
    pad_idx: int,
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int,
    num_heads: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    d_ff: int,
) -> None:
    print(f"[forward case] {case_name}")

    run_mask_checks(src, tgt, pad_idx)

    batch_size, src_len = src.shape
    tgt_len = tgt.size(1)

    src_mask = create_src_padding_mask(src, pad_idx)
    tgt_mask = create_tgt_mask(tgt, pad_idx)
    memory_mask = create_memory_mask(tgt, src, pad_idx)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=0.0,
        max_len=max(src_len, tgt_len) + 8,
    )

    logits, enc_attn_list, dec_self_attn_list, dec_cross_attn_list = model(
        src,
        tgt,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
    )

    assert_shape(logits.shape, (batch_size, tgt_len, tgt_vocab_size), "logits")
    assert len(enc_attn_list) == num_encoder_layers
    assert len(dec_self_attn_list) == num_decoder_layers
    assert len(dec_cross_attn_list) == num_decoder_layers

    assert_shape(
        enc_attn_list[0].shape,
        (batch_size, num_heads, src_len, src_len),
        "encoder attention",
    )
    assert_shape(
        dec_self_attn_list[0].shape,
        (batch_size, num_heads, tgt_len, tgt_len),
        "decoder self attention",
    )
    assert_shape(
        dec_cross_attn_list[0].shape,
        (batch_size, num_heads, tgt_len, src_len),
        "decoder cross attention",
    )

    assert_tensor_finite(logits, "logits")
    assert_tensor_finite(enc_attn_list[0], "encoder attention")
    assert_tensor_finite(dec_self_attn_list[0], "decoder self attention")
    assert_tensor_finite(dec_cross_attn_list[0], "decoder cross attention")

    print(
        f"  logits={tuple(logits.shape)}, "
        f"enc_layers={len(enc_attn_list)}, "
        f"dec_layers={len(dec_self_attn_list)}"
    )


def run_train_loop_check() -> None:
    print("[train loop] mini integration")
    torch.manual_seed(11)

    src = torch.tensor(
        [
            [11, 12, 13, 14, 15, 16, 0, 0, 0, 0],
            [21, 22, 23, 24, 25, 26, 27, 28, 0, 0],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 0],
        ],
        dtype=torch.long,
    )
    tgt = torch.tensor(
        [
            [1, 41, 42, 43, 44, 45, 2, 0],
            [1, 51, 52, 53, 54, 55, 56, 2],
            [1, 57, 58, 59, 60, 61, 2, 0],
        ],
        dtype=torch.long,
    )

    pad_idx = 0
    src_vocab_size = 80
    tgt_vocab_size = 90

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    src_mask = create_src_padding_mask(src, pad_idx)
    tgt_mask = create_tgt_mask(tgt_input, pad_idx)
    memory_mask = create_memory_mask(tgt_input, src, pad_idx)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=64,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=256,
        dropout=0.0,
        max_len=32,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    losses = []
    for step in range(100):
        optimizer.zero_grad()
        logits, _, _, _ = model(
            src,
            tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
        assert torch.isfinite(loss), f"loss at step {step} is NaN or Inf"
        loss.backward()

        grad_count = 0
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                assert_tensor_finite(param.grad, "parameter gradient")
                grad_count += 1
                total_grad_norm += float(param.grad.norm().item())

        assert grad_count > 0, "no gradients were produced"
        optimizer.step()
        losses.append(float(loss.item()))
        print(
            f"  step {step + 1}: loss={loss.item():.4f}, "
            f"grad_norm_sum={total_grad_norm:.4f}"
        )

    assert all(torch.isfinite(torch.tensor(losses))), "non-finite loss history"


def main() -> None:
    torch.manual_seed(7)
    print("Running stronger Transformer integration checks...")
    run_text_data_pipeline_checks()

    src_small, tgt_small, pad_idx_small = build_toy_batch("small")
    run_forward_case(
        case_name="small",
        src=src_small,
        tgt=tgt_small,
        pad_idx=pad_idx_small,
        src_vocab_size=50,
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
    )

    src_medium, tgt_medium, pad_idx_medium = build_toy_batch("medium")
    run_forward_case(
        case_name="medium",
        src=src_medium,
        tgt=tgt_medium,
        pad_idx=pad_idx_medium,
        src_vocab_size=80,
        tgt_vocab_size=90,
        d_model=64,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=256,
    )

    run_train_loop_check()
    print("All stronger integration checks passed.")


if __name__ == "__main__":
    main()
