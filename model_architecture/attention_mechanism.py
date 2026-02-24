# attention_mechanism.py
# 注意力机制：从 Scaled Dot-Product Attention 到 Multi-Head Attention
#
# Transformer 的核心计算单元。理解这里的矩阵维度变换
# 是后续阅读 BERT/GPT/LLaVA 代码的基础。
#
# 参考：Vaswani et al., "Attention Is All You Need", NeurIPS 2017.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== 1. Scaled Dot-Product Attention ==========
print("=== Scaled Dot-Product Attention ===")
#
# Attention(Q, K, V) = softmax(QK^T / √d_k) V
#
# Q: 查询向量（Query）  —— "我想要什么信息"
# K: 键向量（Key）      —— "我能提供什么信息"
# V: 值向量（Value）    —— "实际要传递的内容"
#
# QK^T / √d_k 计算"相关性分数"，softmax 归一化为注意力权重，
# 再加权求和 V 得到输出。
#
# 为什么除以 √d_k？
#   d_k 维的随机向量点积方差为 d_k（均值0，方差1的每维），
#   不缩放时 d_k 大会导致 softmax 梯度消失（输入过大 → 极端输出 → 梯度趋近 0）。
#   除以 √d_k 将方差归一化到 1，使 softmax 工作在合适的区间。

def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """
    Q, K, V: (..., seq_len, d_k) 或 (batch, heads, seq, d_k)
    mask   : (..., seq_len, seq_len)，需要被 mask 的位置设为 True
    返回   : (output, attention_weights)
    """
    d_k     = Q.size(-1)
    scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (..., seq, seq)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))  # -inf 在 softmax 后变为 0

    attn_weights = F.softmax(scores, dim=-1)   # 沿 key 维度归一化

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, V)     # (..., seq, d_k)
    return output, attn_weights

# 单头 attention 示例
B, S, d_k = 2, 8, 64
Q = torch.randn(B, S, d_k)
K = torch.randn(B, S, d_k)
V = torch.randn(B, S, d_k)

out, attn = scaled_dot_product_attention(Q, K, V)
print(f"Q/K/V: ({B}, {S}, {d_k})")
print(f"输出:  {out.shape}  （与 Q 同形）")
print(f"注意力矩阵: {attn.shape}  （seq×seq，行和为1）")
print(f"注意力权重行和验证: {attn[0].sum(dim=-1)}  （每行 sum=1）")
print()

# ========== 2. 因果掩码（Causal Mask）==========
print("=== 因果掩码（Decoder Self-Attention）===")
#
# GPT 等自回归模型中，第 i 个 token 只能 attend 到位置 ≤ i 的 token。
# 通过上三角掩码实现（future tokens 的 score 置为 -inf）。

def make_causal_mask(seq_len, device='cpu'):
    """上三角矩阵（对角线以上为 True = 被 mask）"""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask

mask = make_causal_mask(5)
print(f"因果掩码 (seq=5):\n{mask.int()}")
print("1=被屏蔽（未来位置），0=可 attend（当前及过去位置）")
print()

# 带因果掩码的 attention
Q2 = torch.randn(1, 5, 32)
K2 = torch.randn(1, 5, 32)
V2 = torch.randn(1, 5, 32)
out2, attn2 = scaled_dot_product_attention(Q2, K2, V2, mask=mask.unsqueeze(0))
print(f"因果 attention 权重矩阵（第一层，上三角应为0）:")
print(attn2[0].data.round(3))
print()

# ========== 3. Multi-Head Attention ==========
print("=== Multi-Head Attention ===")
#
# 单头 attention：d_model 维的 Q/K/V，一次计算
# 多头 attention：将 d_model 分成 h 个"头"，每头维度 d_k = d_model / h，
#                 并行地从不同角度捕捉注意力模式，再 concat + 线性投影。
#
# 直觉：不同的头可以专注于不同的子空间关系，
#   例如一头关注句法依存，另一头关注指代关系。
#   实验验证（Voita et al. 2019）表明许多头确实有功能分工。
#
# 数学形式：
#   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
#   head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
#   W_i^Q ∈ R^{d_model × d_k},  d_k = d_model / h

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # 合并 W^Q, W^K, W^V 为一个大矩阵（实现更高效）
        self.W_q  = nn.Linear(d_model, d_model, bias=False)
        self.W_k  = nn.Linear(d_model, d_model, bias=False)
        self.W_v  = nn.Linear(d_model, d_model, bias=False)
        self.W_o  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x):
        """(B, S, d_model) → (B, h, S, d_k)"""
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, h, S, d_k)

    def _merge_heads(self, x):
        """(B, h, S, d_k) → (B, S, d_model)"""
        B, h, S, d_k = x.shape
        x = x.transpose(1, 2).contiguous()  # transpose 后需 contiguous 才能 view
        return x.view(B, S, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (B, S, d_model)
        自注意力时 query=key=value=x
        交叉注意力时 query 来自解码器，key/value 来自编码器
        """
        Q = self._split_heads(self.W_q(query))  # (B, h, S, d_k)
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.drop)

        out = self._merge_heads(out)  # (B, S, d_model)
        out = self.W_o(out)
        return out, attn

# 测试
B, S, d_model, n_heads = 4, 16, 256, 8
mha = MultiHeadAttention(d_model, n_heads)
x   = torch.randn(B, S, d_model)

out_mha, attn_mha = mha(x, x, x)  # 自注意力
print(f"输入: ({B}, {S}, {d_model})")
print(f"MHA 输出: {out_mha.shape}  （形状不变）")
print(f"注意力矩阵: {attn_mha.shape}  （batch, heads, seq, seq）")
print()

# ========== 4. 与 PyTorch 内置 nn.MultiheadAttention 对比 ==========
print("=== 与 nn.MultiheadAttention 对比 ===")
#
# PyTorch 内置版本接口略有不同：
#   ① 输入默认期望 (S, B, d_model)，batch_first=True 改为 (B, S, d_model)
#   ② 内部有 bias，与我们手写版的 bias=False 略有差异

mha_torch = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
out_pt, attn_pt = mha_torch(x, x, x)  # 自注意力

print(f"nn.MHA 输出: {out_pt.shape}  （需要 batch_first=True）")
print(f"nn.MHA attn_weights: {attn_pt.shape}  （默认对所有 heads 平均）")
print()

print("=== 总结 ===")
print("""
1. Attention(Q,K,V) = softmax(QK^T/√d_k)V，除以√d_k 防止 softmax 梯度消失
2. 因果掩码用上三角矩阵实现，-inf 在 softmax 后变为 0 权重
3. 多头：将 d_model 拆成 h 头，各头独立 attention 后 concat，增加表达多样性
4. 实现细节：_split_heads 后 transpose；_merge_heads 前必须 contiguous()
5. 自注意力 query=key=value=x；交叉注意力 query≠key=value（Encoder-Decoder 结构）
""")
