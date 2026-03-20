# %% [markdown]
# # Transformer Positional Encoding
# Transformer 的自注意力本身不关心输入顺序。
# 如果只看 token embedding，模型知道“有哪些词”，但不知道“谁在前、谁在后”。
# 所以原始 Transformer 会把一个位置向量加到词向量上：
# $$
# x_{input} = x_{token} + PE
# $$
# 原始论文使用的是固定的正弦位置编码（sinusoidal positional encoding）：
# $$
# PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)
# $$
# $$
# PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)
# $$
# 其中：
# - $pos$ 是位置索引
# - $i$ 是维度对编号
# - $d_{model}$ 是 embedding 维度
# 可以把每两维看成一组：
# $$
# \left[\sin(pos \cdot \omega_i), \cos(pos \cdot \omega_i)\right]
# $$
# 其中：
# $$
# \omega_i = 10000^{-2i/d_{model}}
# $$
# 这相当于给不同维度分配不同频率。
# 低维变化快，高维变化慢，多种频率叠加后，每个位置都会得到一个独特的“波形签名”。

# %% [markdown]
# ## 为什么它能表达相对位置
# 这是这个设计最漂亮的地方。
# 对某一组二维向量，设：
# $$
# v_{pos}^{(i)} =
# \begin{bmatrix}
# \sin(pos \omega_i) \\
# \cos(pos \omega_i)
# \end{bmatrix}
# $$
# 那么偏移 $k$ 之后的位置为：
# $$
# v_{pos+k}^{(i)} =
# \begin{bmatrix}
# \sin((pos+k)\omega_i) \\
# \cos((pos+k)\omega_i)
# \end{bmatrix}
# $$
# 用三角恒等式展开：
# $$
# \sin(a+b) = \sin a \cos b + \cos a \sin b
# $$
# $$
# \cos(a+b) = \cos a \cos b - \sin a \sin b
# $$
# 可得：
# $$
# v_{pos+k}^{(i)} =
# \begin{bmatrix}
# \cos(k\omega_i) & \sin(k\omega_i) \\
# -\sin(k\omega_i) & \cos(k\omega_i)
# \end{bmatrix}
# v_{pos}^{(i)}
# $$
# 注意这个矩阵只依赖于 $k$，不依赖于 $pos$。
# 所以“向后移动 k 个位置”在每一组二维空间里都是一个固定线性变换。
# 这让模型更容易从编码中学到相对位置信息。

# %%
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    原始 Transformer 使用的固定正弦位置编码。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        # d_model: embedding 维度,模型里每个 token 向量的维度
        # dropout: 位置编码的 dropout 概率,默认0.1,防止过拟合
        # max_len: 最长序列长度,表示预先最多生成多少个位置的编码
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) # 创建一个 dropout 层。以后前向传播最后会调用它。训练时生效，推理时自动关闭。

        # pe shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        # position shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # div_term 对应公式中的 10000^{-2i/d_model}
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 变成 (1, max_len, d_model)，自动 broadcast 到 batch,便于和 (batch, seq_len, d_model) 相加
        pe = pe.unsqueeze(0) # 在第 dim 个位置插入一个长度为 1 的新维度

        # buffer 会随着模型保存，但不会参与梯度更新
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
    
# %% [markdown]
# ## 公式和代码如何一一对应
# 论文公式：
# $$
# PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
# $$
# 因为：
# $$
# \frac{pos}{10000^{2i/d_{model}}} = pos \cdot 10000^{-2i/d_{model}}
# $$
# 所以代码里先构造：
# ```python
# div_term = exp(arange(0, d_model, 2) * (-log(10000.0) / d_model))
# ```
# 它得到的正是：
# $$
# 10000^{-2i/d_{model}}
# $$
# 然后：
# ```python
# pe[:, 0::2] = sin(position * div_term)
# pe[:, 1::2] = cos(position * div_term)
# ```
# 就对应偶数维和奇数维的两条公式。

# %%
def demo_shapes():
    d_model = 8
    seq_len = 6
    batch_size = 2

    x = torch.zeros(batch_size, seq_len, d_model)
    pe = PositionalEncoding(d_model=d_model, dropout=0.0, max_len=20)
    y = pe(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)
    print()
    print("位置编码前 6 个位置、前 8 维：")
    print(y[0])

# %% [markdown]
# ## 一个直观理解
# 可以把它类比成“多组速度不同的时钟”：
# - 有的维度转得很快
# - 有的维度转得很慢
# - 所有时钟的状态拼起来，就能区分当前位置
# 这比只用一个位置数字更适合神经网络，因为：
# - 数值范围更平滑
# - 不同维度有不同尺度
# - 相对位移会对应到稳定的三角关系

# %%
if __name__ == "__main__":
    demo_shapes()
    