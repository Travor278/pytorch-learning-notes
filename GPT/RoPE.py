# %% [markdown]
# # Rotary Position Embedding (RoPE)
# RoPE 和固定正弦位置编码的共同点是：都给不同维度分配不同频率。
# 但它不是把位置向量直接加到输入上，而是把位置以“旋转”的形式注入到 attention 的 $Q/K$ 里。
# 所以它不再是：
# $$
# x = E(token) + PE(pos)
# $$
# 而是先得到：
# $$
# Q,\ K \in \mathbb{R}^{B \times h \times L \times d_k}
# $$
# 再对每个位置的 $Q/K$ 做旋转。
#
# 先定义每一对二维分量对应的频率：
# $$
# \theta_m = base^{-2m/d_k}
# $$
# 其中：
# - $m$ 是二维通道对编号
# - $d_k$ 是单个 head 的维度
# - $base$ 通常取 $10000$
#
# 对位置 $p$，第 $m$ 对二维子空间要旋转的角度是：
# $$
# \phi_{p,m} = p \cdot \theta_m
# $$
# 所以代码里先构造位置索引和频率：
# ```python
# positions = torch.arange(max_len).float()
# inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
# freqs = torch.outer(positions, inv_freq)
# ```
# 这里：
# $$
# freqs[p,m] = p \theta_m
# $$
# 所以：
# $$
# freqs \in \mathbb{R}^{max\_len \times d_k/2}
# $$
# 它就是“每个位置、每组二维分量”的角度表。
#
# 再看这句：
# ```python
# emb = torch.cat((freqs, freqs), dim=-1)
# ```
# 在当前这版实现里，采用的是 LLaMA-style 的前半段 / 后半段配对。
# 例如若 $d_k = 8$，会把最后一维拆成：
# $$
# [x_0, x_1, x_2, x_3 \mid x_4, x_5, x_6, x_7]
# $$
# 对应的二维配对是：
# $$
# (x_0, x_4),\ (x_1, x_5),\ (x_2, x_6),\ (x_3, x_7)
# $$
# 如果某个位置上的四组角度是：
# $$
# [\phi_{p,0}, \phi_{p,1}, \phi_{p,2}, \phi_{p,3}]
# $$
# 那么扩展到完整最后一维后会变成：
# $$
# [\phi_{p,0}, \phi_{p,1}, \phi_{p,2}, \phi_{p,3}, \phi_{p,0}, \phi_{p,1}, \phi_{p,2}, \phi_{p,3}]
# $$
# 所以：
# $$
# emb \in \mathbb{R}^{max\_len \times d_k}
# $$
# 它不是 token embedding，而是“扩展到完整维度后的旋转角度表”。
#
# 再对这个角度表分别取 $\cos$ 和 $\sin$：
# ```python
# self.register_buffer("cos_cached", emb.cos(), persistent=False)
# self.register_buffer("sin_cached", emb.sin(), persistent=False)
# ```
# 对应：
# $$
# cos\_cached[p,:] = \cos(emb[p,:]), \quad sin\_cached[p,:] = \sin(emb[p,:])
# $$
# 它们被缓存起来，是因为这些值只由 $max\_len,\ d_k,\ base$ 决定，和具体输入内容无关，不需要每次 forward 重算。

# %% [markdown]
# ## 为什么 `rotate_half` 就是在做旋转
# 对某一对二维分量，设：
# $$
# x =
# \begin{bmatrix}
# x_1 \\
# x_2
# \end{bmatrix}
# $$
# 二维旋转矩阵为：
# $$
# R(\phi)=
# \begin{bmatrix}
# \cos\phi & -\sin\phi \\
# \sin\phi & \cos\phi
# \end{bmatrix}
# $$
# 所以旋转后的结果是：
# $$
# R(\phi)x =
# \begin{bmatrix}
# x_1\cos\phi - x_2\sin\phi \\
# x_1\sin\phi + x_2\cos\phi
# \end{bmatrix}
# $$
# 改写一下：
# $$
# R(\phi)x =
# x \cos\phi +
# \begin{bmatrix}
# -x_2 \\
# x_1
# \end{bmatrix}
# \sin\phi
# $$
# 在当前实现里，若把一个向量写成前后两半：
# $$
# x =
# \begin{bmatrix}
# x_{\text{front}} \\
# x_{\text{back}}
# \end{bmatrix}
# $$
# 那么 `rotate_half` 做的是：
# $$
# \begin{bmatrix}
# -x_{\text{back}} \\
# x_{\text{front}}
# \end{bmatrix}
# $$
# 也就是把前半段和后半段按位置一一配对后，完成二维旋转里的 $(-b, a)$ 结构。
#
# 所以最终代码：
# ```python
# q = (q * cos) + (rotate_half(q) * sin)
# k = (k * cos) + (rotate_half(k) * sin)
# ```
# 就对应于：
# $$
# q_{\text{rot}} = R(\phi) q,\qquad k_{\text{rot}} = R(\phi) k
# $$
# 只是这里不是显式构造每个二维旋转矩阵，而是用逐元素乘法高效实现同样的结果。
#
# 如果：
# $$
# Q,\ K \in \mathbb{R}^{B \times h \times L \times d_k}
# $$
# 那么从缓存里取出的：
# $$
# cos,\ sin \in \mathbb{R}^{1 \times 1 \times L \times d_k}
# $$
# 会自动广播到 batch 和 head 维，最后直接进入 attention：
# $$
# \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
# $$
# 也就是说，RoPE 不是把位置“加”到输入上，而是把位置“转”进了 attention 点积本身。

# %%
from __future__ import annotations # 方便类型提示

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int, # 每个head的维度
        max_len: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.dim = dim
        self.max_len = max_len
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end = offset + seq_len
        if end > self.max_len:
            raise ValueError(f"seq_len={seq_len} with offset={offset} exceeds RoPE max_len={self.max_len}")
        cos = self.cos_cached[offset:end].to(device).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:end].to(device).unsqueeze(0).unsqueeze(0)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (rotate_half(q) * sin) # 逐元素乘法，自动广播到 batch 和 head 维
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k
