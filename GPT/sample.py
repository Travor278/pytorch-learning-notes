from __future__ import annotations

import argparse
from pathlib import Path

import torch

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


@torch.no_grad()
def generate(
    model: GPTModel,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    model.eval()
    past_kvs = None

    for _ in range(max_new_tokens):
        if past_kvs is None or past_kvs[0][0].size(-2) >= model.max_len:
            idx_cond = token_ids[:, -model.max_len :]
            logits, _, past_kvs = model(idx_cond, use_cache=True)
        else:
            logits, _, past_kvs = model(token_ids[:, -1:], past_kvs=past_kvs, use_cache=True)
        next_token_logits = logits[:, -1, :]

        if temperature <= 0:
            raise ValueError("temperature must be positive")
        next_token_logits = next_token_logits / temperature

        if top_k is not None and top_k > 0:
            values, _ = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
            threshold = values[:, [-1]]
            next_token_logits = next_token_logits.masked_fill(next_token_logits < threshold, float("-inf"))

        # top-p (nucleus) 采样：
        # 把所有 token 按概率从大到小排序，只保留累积概率刚好超过 p 的最小 token 集合，
        # 其余 token 的 logit 设为 -inf。
        # 好处：top-k 固定保留 k 个候选，top-p 则根据概率分布的"尖锐程度"自动调整候选数量：
        #   - 分布尖锐时（某个词概率很高）→ 只保留少数几个词，生成更确定
        #   - 分布平坦时（很多词概率差不多）→ 保留更多词，生成更多样
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # 把累积概率已经超过 p 的位置标记为 True，再右移一位，
            # 保证累积概率刚超过 p 的那个 token 本身还被保留
            sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            # 把排序后的 logits 还原回原始 token 顺序
            next_token_logits = next_token_logits.scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_ids = torch.cat([token_ids, next_token], dim=1)

    return token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained mini GPT checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling probability threshold (e.g. 0.9).")
    # 原来：add_bos 硬编码为 True，但训练时序列不以 <bos> 开头，存在 train/inference 不一致
    # --add-bos 现在默认 False，与训练保持一致；如果训练时也加了 <bos>（train_gpt.py 里已修改），则传 --add-bos
    parser.add_argument("--add-bos", action="store_true", default=True,
                        help="Prepend <bos> token before prompt (default True, matches train_gpt.py fix).")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    tokenizer_path = args.ckpt.with_suffix(".tokenizer.json")

    # 按训练时记录的类型加载对应 tokenizer
    tokenizer_type = config.get("tokenizer_type", "char")
    if tokenizer_type == "bpe":
        tokenizer: CharTokenizer | BPETokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = CharTokenizer.load(tokenizer_path)

    # GPTModel 的构造参数不包含 tokenizer_type，传入前过滤掉
    model_config = {k: v for k, v in config.items() if k != "tokenizer_type"}
    model = GPTModel(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 原来：add_bos=True 硬编码，但训练数据没有 <bos>，推理和训练分布不一致
    # input_ids = tokenizer.encode(args.prompt, add_bos=True)
    # 修改：由 --add-bos 参数控制，默认 True 与 train_gpt.py 里 add_bos=True 对齐
    input_ids = tokenizer.encode(args.prompt, add_bos=args.add_bos)
    token_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    output_ids = generate(
        model=model,
        token_ids=token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )[0].tolist()

    print("prompt   :", args.prompt)
    print("generated:")
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
