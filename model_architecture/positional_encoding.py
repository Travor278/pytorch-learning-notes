# positional_encoding.py
# 位置编码：Sinusoidal PE、可学习 PE 与 RoPE
#
# Transformer 的自注意力是排列不变的（permutation-invariant）——
# 打乱输入顺序不影响输出（忽略位置）。位置编码注入序列中 token 的位置信息。
#
# 参考：
#   - Vaswani et al., "Attention Is All You Need", NeurIPS 2017（正弦 PE）
#   - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", NLP 2021（RoPE）
#   - Press et al., "Train Short, Test Long: ALiBi", ICLR 2022

import torch
import torch.nn as nn
import math

# ========== 1. 正弦位置编码（Sinusoidal PE）==========
print("=== 正弦位置编码 ===")
#
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# 设计思路：
#   - 每个位置用 d_model 维向量表示，不同维度的频率不同（高维频率低）
#   - 类比二进制计数：低位快速翻转，高位缓慢翻转，合起来唯一标识每个位置
#
# 关键性质：
#   PE(pos+k) 可以表示为 PE(pos) 的线性函数（旋转矩阵）——
#   这意味着自注意力中 PE(pos)·PE(pos+k) 只依赖于相对距离 k，
#   给模型提供了相对位置感知的潜力。
#
# 优点：不需要学习，推理时可以外推到训练长度之外（一定范围内）。
# 缺点：绝对位置感知弱，实践中大模型多换用可学习 PE 或 RoPE。

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算整个 PE 矩阵，注册为 buffer（不参与梯度更新但随模型保存）
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()         # (max_len, 1)
        # 频率项：div_term[i] = 1 / 10000^(2i/d_model)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)   # 偶数维：sin
        pe[:, 1::2] = torch.cos(pos * div)   # 奇数维：cos

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)：broadcast 到 batch
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (B, S, d_model)"""
        x = x + self.pe[:, :x.size(1), :]   # 加法注入位置信息
        return self.dropout(x)

d_model, S = 64, 20
spe = SinusoidalPE(d_model, max_len=100)
x   = torch.randn(2, S, d_model)
out = spe(x)
print(f"输入: {x.shape} → 加 PE 后: {out.shape}  （形状不变）")

# 观察 PE 矩阵的前几行
pe_matrix = spe.pe[0, :6, :8]  # 前 6 个位置，前 8 维
print(f"PE 矩阵（前6位置，前8维）:")
for i, row in enumerate(pe_matrix):
    print(f"  pos={i}: {row.data.numpy().round(3).tolist()}")
print()

# ========== 2. 可学习位置编码 ==========
print("=== 可学习位置编码 ===")
#
# 将位置编码作为模型参数直接学习，比正弦 PE 更灵活。
# BERT、ViT、GPT-2 均采用可学习 PE。
#
# 缺点：
#   最大序列长度在初始化时固定（max_position_embeddings），
#   推理时无法泛化到超出训练长度的序列。

class LearnablePE(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.pe      = nn.Embedding(max_len, d_model)  # 可学习的位置嵌入
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pe.weight, std=0.02)      # BERT 初始化方式

    def forward(self, x):
        """x: (B, S, d_model)"""
        S        = x.size(1)
        pos_ids  = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)
        x        = x + self.pe(pos_ids)
        return self.dropout(x)

lpe = LearnablePE(d_model)
print(f"可学习 PE 参数量: {sum(p.numel() for p in lpe.parameters())}  （max_len × d_model）")
out_lpe = lpe(x)
print(f"输出形状: {out_lpe.shape}")
print()

# ========== 3. 旋转位置编码 RoPE ==========
print("=== RoPE（Rotary Position Embedding）===")
#
# Su et al. 2021，被 LLaMA / LLaVA / Qwen 等主流大模型广泛采用。
#
# 核心思想：
#   不是把 PE 加到 token embedding 上（绝对位置），
#   而是对 Q 和 K 向量旋转，使得 Q_m · K_n 只依赖于 (m-n)（相对位置）。
#
# 数学形式（以 2D 为例）：
#   将 d_k 维向量分成 d_k/2 对，每对对应一个旋转角 θ_i = pos · 10000^(-2i/d_k)：
#   [q_{2i}, q_{2i+1}] → [q_{2i}·cos θ - q_{2i+1}·sin θ,
#                          q_{2i}·sin θ + q_{2i+1}·cos θ]
#
# 关键性质：
#   (R_m q)^T (R_n k) = q^T R_{m-n}^T R_{m-n} k ... 实际上只依赖 m-n
#   → 内积自然编码相对位置距离，无需修改 attention 公式
#
# 优点：
#   ① 天然支持相对位置（不需要额外的相对位置 bias）
#   ② 支持序列长度外推（配合 YaRN / 线性插值技术）
#   ③ 实现简单，只在 Q/K 上操作，V 不参与

def get_rope_freqs(d_k, base=10000):
    """计算 RoPE 的频率 θ，每对维度一个值"""
    i     = torch.arange(0, d_k, 2).float()  # (d_k/2,)
    theta = 1.0 / (base ** (i / d_k))        # (d_k/2,)
    return theta  # θ_i = 1 / 10000^(2i/d_k)

def apply_rope(x, seq_len, theta):
    """
    x    : (B, h, S, d_k)
    theta: (d_k/2,)
    """
    d_k   = x.size(-1)
    pos   = torch.arange(seq_len, device=x.device).float()  # (S,)
    # 角度矩阵：(S, d_k/2)，每个位置的每对维度对应一个旋转角
    angles = torch.outer(pos, theta)               # (S, d_k/2)
    cos    = angles.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, S, d_k/2)
    sin    = angles.sin().unsqueeze(0).unsqueeze(0)

    # 将 x 拆成相邻两两一对：[x0,x1, x2,x3, ...]
    x_even = x[..., 0::2]   # (B, h, S, d_k/2)
    x_odd  = x[..., 1::2]

    # 旋转：[x_even, x_odd] → [x_even·cos - x_odd·sin, x_even·sin + x_odd·cos]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    # 交错合并回原始维度顺序
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
    return x_rot

B, h, S, d_k = 2, 8, 16, 64
theta = get_rope_freqs(d_k)
Q_test = torch.randn(B, h, S, d_k)

Q_rope = apply_rope(Q_test, S, theta)
print(f"RoPE 输入 Q: {Q_test.shape}")
print(f"RoPE 旋转后: {Q_rope.shape}  （形状不变，但包含旋转后的位置信息）")

# 验证性质：相对位置 m-n 的内积
q_pos2 = apply_rope(Q_test[:, :, [2], :], 2, theta)[:, :, 0, :]  # pos=2 的 Q
q_pos5 = apply_rope(Q_test[:, :, [5], :], 5, theta)[:, :, 0, :]  # pos=5 的 Q
k_pos3 = apply_rope(Q_test[:, :, [3], :], 3, theta)[:, :, 0, :]  # pos=3 的 K（相对距离 2-3=-1，5-3=2）
dot_25 = (q_pos2 * k_pos3).sum(-1).mean().item()
dot_55 = (q_pos5 * k_pos3).sum(-1).mean().item()
print(f"内积 Q(pos=2)·K(pos=3)（距离-1）: {dot_25:.4f}")
print(f"内积 Q(pos=5)·K(pos=3)（距离+2）: {dot_55:.4f}  （不同距离有不同内积，编码了相对位置）")
print()

print("=== 总结 ===")
print("""
1. Sinusoidal PE：固定，可理论外推，Transformer 原始方案
2. 可学习 PE：BERT/ViT 用，效果更好但不能泛化到超出 max_len 的序列
3. RoPE：旋转 Q/K 向量，天然编码相对位置，LLaMA/LLaVA 主流选择
4. RoPE 不修改 V 向量；旋转操作 O(d_k)，几乎无额外计算开销
5. ALiBi：在 attention score 上直接加线性偏置 -|m-n|，外推性最好但性能略弱
""")
