# %% [markdown]
# # Transformer Mask Utilities
# Transformer 里常见的 mask 可以粗略分成三类：
#
# 1. Source Padding Mask
# 2. Target Causal Mask
# 3. Memory Mask
#
# 它们分别回答：
# - Encoder 里哪些源位置无效（通常是 PAD）
# - Decoder 里哪些未来目标位置不能看
# - Decoder 在 cross-attention 时，哪些 Encoder 位置不能看

# %%
import torch

try:
    from MHA import make_causal_mask
except ImportError:
    from Transformer.MHA import make_causal_mask


# %% [markdown]
# ## Source Padding Mask
# 如果源序列是：
#
# $$
# src \in \mathbb{R}^{B \times S}
# $$
#
# 那么一个常见的 padding mask 形状是：
#
# $$
# (B, 1, 1, S)
# $$
#
# 它可以广播到 Encoder self-attention 的分数矩阵：
#
# $$
# (B, h, S, S)
# $$
#
# 其中最后一个维度表示 key 位置，因此 PAD 位置会被统一屏蔽。

# %%
def create_src_padding_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    src shape: (batch_size, src_len)
    return  : (batch_size, 1, 1, src_len)
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


# %% [markdown]
# ## Target Padding Mask
# target padding mask 的直觉与 source padding mask 类似：
# - 用来标记目标序列里哪些位置是 PAD
# - 常与 causal mask 组合，得到 Decoder self-attention 的最终 mask

# %%
def create_tgt_padding_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    tgt shape: (batch_size, tgt_len)
    return  : (batch_size, 1, 1, tgt_len)
    """
    return (tgt == pad_idx).unsqueeze(1).unsqueeze(2)


# %% [markdown]
# ## Target Causal Mask
# causal mask 只依赖目标长度：
#
# $$
# (T, T)
# $$
#
# 其中：
# - 对角线以上为 `True`
# - 表示未来位置要被屏蔽

# %%
def create_tgt_causal_mask(tgt_len: int, device: torch.device | str) -> torch.Tensor:
    """
    return shape: (tgt_len, tgt_len)
    """
    return make_causal_mask(tgt_len, device=device)


# %% [markdown]
# ## Target Mask
# Decoder self-attention 通常会把：
# - causal mask
# - target padding mask
#
# 合并起来，得到：
#
# $$
# (B, 1, T, T)
# $$
#
# 其中：
# - 最后一个维度表示 key 位置
# - `True` 表示对应位置不能看

# %%
def create_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    tgt shape: (batch_size, tgt_len)
    return  : (batch_size, 1, tgt_len, tgt_len)
    """
    batch_size, tgt_len = tgt.shape
    causal_mask = create_tgt_causal_mask(tgt_len, device=tgt.device).unsqueeze(0).unsqueeze(0)
    padding_mask = create_tgt_padding_mask(tgt, pad_idx)
    return causal_mask | padding_mask.expand(batch_size, 1, tgt_len, tgt_len)


# %% [markdown]
# ## Memory Mask
# cross-attention 的分数矩阵形状是：
#
# $$
# (B, h, T, S)
# $$
#
# 所以 memory mask 常见作用是：
# - 屏蔽 Encoder memory 中对应 PAD 的源位置
# - 让所有目标 query 都不要去看这些无效 source 位置

# %%
def create_memory_mask(
    tgt: torch.Tensor,
    src: torch.Tensor,
    src_pad_idx: int,
) -> torch.Tensor:
    """
    tgt shape: (batch_size, tgt_len)
    src shape: (batch_size, src_len)
    return  : (batch_size, 1, tgt_len, src_len)
    """
    batch_size, tgt_len = tgt.shape
    src_len = src.size(1)
    src_padding_mask = (src == src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_padding_mask.expand(batch_size, 1, tgt_len, src_len)


# %% [markdown]
# ## 一个小例子
# 如果：
#
# - `src = [[5, 6, 0, 0]]`
# - `tgt = [[1, 2, 3, 0]]`
# - `pad_idx = 0`
#
# 那么：
# - source padding mask 会遮掉源位置 2、3
# - target mask 会同时遮掉未来位置和目标 PAD
# - memory mask 会让所有目标位置都不去看源序列中的 PAD

# %%
def demo_mask_shapes():
    src = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]])
    tgt = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    pad_idx = 0

    src_mask = create_src_padding_mask(src, pad_idx)
    tgt_pad_mask = create_tgt_padding_mask(tgt, pad_idx)
    tgt_causal_mask = create_tgt_causal_mask(tgt.size(1), device=tgt.device)
    tgt_mask = create_tgt_mask(tgt, pad_idx)
    memory_mask = create_memory_mask(tgt, src, pad_idx)

    print("src shape       :", src.shape)
    print("tgt shape       :", tgt.shape)
    print("src_mask shape  :", src_mask.shape)
    print("tgt_pad shape   :", tgt_pad_mask.shape)
    print("tgt_causal shape:", tgt_causal_mask.shape)
    print("tgt_mask shape  :", tgt_mask.shape)
    print("memory_mask shape:", memory_mask.shape)
    print()
    print("tgt_causal_mask:")
    print(tgt_causal_mask)


# %%
if __name__ == "__main__":
    demo_mask_shapes()
