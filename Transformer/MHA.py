# %% [markdown]
# # Transformer Multi-Head Attention
# 多头注意力（Multi-Head Attention, MHA）是 Transformer 的核心计算模块。
# 它的基本公式是：
# $$
# \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
# $$
# 其中：
# - $Q$ 是 Query，表示“我想找什么信息”
# - $K$ 是 Key，表示“我这里有什么信息”
# - $V$ 是 Value，表示“真正要被聚合的内容”
# 计算流程可以理解为：
# 1. 先用 $QK^T$ 计算每个位置之间的相关性分数
# 2. 再经过 softmax 变成注意力权重
# 3. 最后用这些权重对 $V$ 做加权求和
# <img src="./Transformer/Multi_Head_Attention.png" height="320">

# %% [markdown]
# ## 为什么要除以 $\sqrt{d_k}$
# 如果 $Q$ 和 $K$ 的维度 $d_k$ 很大，点积 $QK^T$ 的数值也会变大。
# 这样 softmax 会变得过于尖锐，导致梯度变小，训练不稳定。
# 所以论文里做了一个缩放：
# $$
# \frac{QK^T}{\sqrt{d_k}}
# $$
# 这相当于把分数拉回到更合适的数值范围，让 softmax 更稳定。
# 一个常见的近似推导是：
# $$
# X_i = q_i k_i
# $$
# 如果假设每个分量近似满足：
# - 均值为 0
# - 方差为 1
# - 各维之间近似独立
# 那么有：
# $$
# \mathbb{E}[X_i] = 0
# $$
# $$
# \mathrm{Var}(X_i) = 1
# $$
# 而点积可以写成：
# $$
# q \cdot k = \sum_{i=1}^{d_k} X_i
# $$
# 因此：
# $$
# \mathrm{Var}(q \cdot k) \approx d_k
# $$
# 所以它的标准差大约是：
# $$
# \sqrt{d_k}
# $$
# 缩放之后就有：
# $$
# \mathrm{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) \approx 1
# $$
# 这就是除以 $\sqrt{d_k}$ 的核心理由：让 attention score 的尺度在不同维度下保持稳定。
# <img src="./Transformer/Scaled_Dot_Product_Attention.png" height="320">

# %% [markdown]
# ## 为什么叫Multi-Head
# 单头注意力只是在一个子空间里算一次 attention。
# 多头注意力会把 $d_{model}$ 拆成 $h$ 个头，每个头独立计算，再拼回来：
# $$
# \mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)W^O
# $$
# 其中：
# $$
# \mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
# $$
# 直觉上可以理解为：
# - 有的头更关注局部关系
# - 有的头更关注长距离依赖
# - 有的头更关注语义匹配
# 多头并行计算后，表达能力会比单头更强。

# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor, # 查询向量，表示“当前这个位置想找什么信息”
    key: torch.Tensor,   # 键向量，表示“每个位置能提供什么信息”
    value: torch.Tensor, # 值向量，表示”真正被聚合输出的内容“
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None, # 方便传参和调用
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    query, key, value shape: (batch_size, num_heads, seq_len, d_k)
    batch_size: 批量大小
    num_heads:  头数
    seq_len:    序列长度
    d_k:        每个头的维度,等于 d_model // num_heads
    mask shape: broadcast 到 (batch_size, num_heads, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # scores shape: (batch_size, num_heads, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 第 b 个样本，第 head 个头中，第 i 个 token 对第 j 个 token 有多关注 
    # q_i 和 k_j 的点积越大，说明它们越相关，第 i 个 token 就越关注第 j 个 token

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights
    # ouput.shape: (batch_size, num_heads, seq_len, d_k)
    # attn_weights.shape: (batch_size, num_heads, seq_len, seq_len)


def make_causal_mask(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    返回上三角 mask。
    True 表示这个位置要被屏蔽掉。
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    ) # 返回一个上三角矩阵，主对角线以上的元素为 True，其他为 False，表示未来位置要被屏蔽掉


class MultiHeadAttention(nn.Module):
    """
    输入和输出都使用 batch_first:
    (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        # d_model: token 向量维度
        # num_heads: 头数
        # dropout: attention 权重上的 dropout 概率
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除。")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 四个线性层：
        # W_q, W_k, W_v 用于生成 Q/K/V
        # W_o 用于多头 concat 之后再映射回 d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False) # 参数少一点，对 Q/K/V 投影通常够用了
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        return : (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现了Concat(head_1,..., head_h)的张良重排与拼接
        x shape: (batch_size, num_heads, seq_len, d_k)
        return : (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous() # contiguous 确保内存连续，方便后续的 view 操作
        x = x.view(batch_size, seq_len, self.d_model) # (B, S, h, d_k) -> (B, S, d_model)
        # x = x.reshape(batch_size, seq_len, self.d_model) # reshape 也可以，但 view 更高效，因为它不需要复制数据
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query, key, value shape: (batch_size, seq_len, d_model)
        mask shape:
        - (seq_len, seq_len)
        - (batch_size, seq_len, seq_len)
        - (batch_size, 1, seq_len, seq_len)
        """
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        # Q = xW^Q, K = xW^K, V = xW^V

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        # (B, S, d_model) -> (B, num_heads, S, d_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        out, attn_weights = scaled_dot_product_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        out = self._merge_heads(out)
        out = self.W_o(out)
        return out, attn_weights

# %% [markdown]
# ## 代码里的形状变化
# 假设输入：
# $$
# x \in \mathbb{R}^{B \times S \times d_{model}}
# $$
# 经过线性变换得到：
# $$
# Q, K, V \in \mathbb{R}^{B \times S \times d_{model}}
# $$
# 再拆头：
# $$
# (B, S, d_{model}) \rightarrow (B, S, h, d_k) \rightarrow (B, h, S, d_k)
# $$
# 其中：
# $$
# d_k = \frac{d_{model}}{h}
# $$
# 然后每个头独立做 attention，输出仍然是：
# $$
# (B, h, S, d_k)
# $$
# 最后合并回去：
# $$
# (B, h, S, d_k) \rightarrow (B, S, h, d_k) \rightarrow (B, S, d_{model})
# $$

# %% [markdown]
# ## 因果 Mask 是什么
# 在 GPT 这类自回归模型里，第 $i$ 个 token 不能看到未来位置。
# 所以要用一个上三角 mask，把未来位置的分数置为 $-\infty$：
# $$
# \mathrm{softmax}(-\infty) = 0
# $$
# 这样未来位置的注意力权重就会变成 0。

# %%
def demo_mha_shapes():
    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)

    out, attn = mha(x, x, x)
    print("=== 无 mask 的自注意力 ===")
    print("input shape :", x.shape)
    print("output shape:", out.shape)
    print("attn shape  :", attn.shape)
    print()

    causal_mask = make_causal_mask(seq_len)
    out_masked, attn_masked = mha(x, x, x, mask=causal_mask)
    print("=== 带 causal mask 的自注意力 ===")
    print("mask shape  :", causal_mask.shape)
    print("output shape:", out_masked.shape)
    print("attn shape  :", attn_masked.shape)
    print()
    print("第一个样本第一个头的注意力矩阵：")
    print(attn_masked[0, 0])


# %% [markdown]
# ## 一个直观理解
# 可以把多头注意力理解成“多组不同观察角度的检索器”：
# - 每个头都有自己的一套 $W^Q, W^K, W^V$
# - 所以每个头都能把同一串 token 投影到不同子空间里
# - 然后各自判断“谁该关注谁”
# 最后把多个头的结果拼起来，再线性融合，就得到了更丰富的表示。

# %%
if __name__ == "__main__":
    demo_mha_shapes()
