# %% [markdown]
# # Transformer Encoder
# Transformer Encoder 的职责可以粗略理解成：
#
# “读完整个输入序列，并把每个 token 编码成带上下文的信息表示。”
#
# 原始 Transformer 的一个 Encoder Layer 包含两大模块：
# 1. Multi-Head Self-Attention
# 2. Feed Forward Network
#
# 每个子层外面还会配：
# - Residual Connection（残差连接）
# - LayerNorm（层归一化）
#
# 原始论文中的结构可以写成：
# $$
# \mathrm{EncoderLayer}(x)=\mathrm{LayerNorm}(x+\mathrm{SelfAttention}(x))
# $$
# $$
# \mathrm{Output}=\mathrm{LayerNorm}(x+\mathrm{FFN}(x))
# $$
# 也就是常说的 Add & Norm。

# %% [markdown]
# ## Encoder 在做什么
# 和 Decoder 不同，Encoder 处理的是“完整给定的输入”。
# 所以 Encoder Self-Attention 通常是双向的：
# - 每个 token 都可以看到整句里的其他 token
# - 不需要 causal mask
#
# 例如输入一句话：
# - “Trvor like Aurora.”
#
# Encoder 会把每个词都编码成带上下文的表示，例如：
# - “Travor” 不再只是“Travor”的词向量
# - “like” 也不再只是“like”的词向量
# - 它们都会融合彼此的信息

# %%
import torch
import torch.nn as nn

try:
    from MHA import MultiHeadAttention
    from FFN import PositionwiseFeedForward
    from PostionalEncoding import PositionalEncoding
except ImportError:
    from Transformer.MHA import MultiHeadAttention
    from Transformer.FFN import PositionwiseFeedForward
    from Transformer.PostionalEncoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """
    原始 Transformer 风格的 Encoder Layer（Post-LN）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        # d_model: token 表示维度
        # num_heads: 多头注意力中的头数
        # d_ff: FFN 隐藏层维度
        # dropout: dropout 概率
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x shape       : (batch_size, seq_len, d_model)
        src_mask shape:
        - (seq_len, seq_len)
        - (batch_size, seq_len, seq_len)
        - (batch_size, 1, seq_len, seq_len)
        """
        attn_out, attn_weights = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_weights


# %% [markdown]
# ## 一个 Encoder Layer 的数据流
# 如果输入是：
# $$
# x \in \mathbb{R}^{B \times S \times d_{model}}
# $$
# 那么 Self-Attention 部分会输出：
# $$
# \mathrm{MHA}(x) \in \mathbb{R}^{B \times S \times d_{model}}
# $$
# 然后做残差和归一化：
# $$
# x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Dropout}(\mathrm{MHA}(x)))
# $$
# 接着进入 FFN：
# $$
# \mathrm{FFN}(x) \in \mathbb{R}^{B \times S \times d_{model}}
# $$
# 再做一次残差和归一化：
# $$
# x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Dropout}(\mathrm{FFN}(x)))
# $$
# 最终输出 shape 仍然不变：
# $$
# (B, S, d_{model})
# $$

# %%
class TransformerEncoder(nn.Module):
    """
    原始 Transformer 论文主干结构实现的 Encoder（Post-LN 风格）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        x = self.pos_encoding(x)

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask=src_mask)
            all_attn_weights.append(attn_weights)

        return x, all_attn_weights


# %% [markdown]
# ## 为什么要堆很多层 Encoder
# 一个 Encoder Layer 只能做一轮“信息交互 + 特征加工”。
# 堆多层之后，模型就能逐层构建更高级的表示：
#
# - 浅层可能更关注局部关系
# - 中层可能学习语法和搭配
# - 深层可能学习更抽象的语义关系
#
# 所以完整 Encoder 往往不是一层，而是很多层堆叠。

# %%
def demo_encoder_shapes():
    batch_size = 8
    seq_len = 128
    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 1024

    x = torch.randn(batch_size, seq_len, d_model)

    encoder = TransformerEncoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0,
        max_len=128,
    )

    out, attn_list = encoder(x)

    print("input shape :", x.shape)
    print("output shape:", out.shape)
    print("num layers  :", len(attn_list))
    print("layer1 attn :", attn_list[0].shape)
    print()
    print("第一个样本第一个 token 编码前：")
    print(x[0, 0])
    print()
    print("第一个样本第一个 token 编码后：")
    print(out[0, 0])


# %% [markdown]
# ## 一个直观理解
# 可以把 Encoder 看成一个“多轮读句子并做笔记”的过程：
#
# - Position Encoding：先告诉模型每个词在什么位置
# - Self-Attention：让每个词去看整句里和自己相关的其他词
# - FFN：每个词在融合上下文后，再单独做一轮深加工
# - 多层堆叠：一层层把表示变得更抽象、更有语义
#
# 所以 Encoder 的最终输出，不再是“原始词向量”，
# 而是“已经带上下文理解的表示”。

# %%
if __name__ == "__main__":
    demo_encoder_shapes()
