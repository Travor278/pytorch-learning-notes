# %% [markdown]
# # Transformer Decoder
# Transformer Decoder 的职责可以粗略理解成：
#
# “在已经生成的前缀基础上，一边参考 Encoder 输出，一边逐步生成后续 token。”
#
# 和 Encoder 不同，Decoder Layer 通常包含三大模块：
# 1. Masked Multi-Head Self-Attention
# 2. Cross-Attention（和 Encoder 输出交互）
# 3. Feed Forward Network
#
# 每个子层外面同样会配：
# - Residual Connection
# - LayerNorm
#
# 所以一个 Decoder Layer 比 Encoder Layer 多了一步：
# “拿当前解码状态去看 Encoder 的输出”。

# %% [markdown]
# ## Decoder 在做什么
# Decoder 和 Encoder 的关键区别在于：
#
# - Encoder 处理完整输入，可以双向看全句
# - Decoder 是逐步生成，因此 self-attention 必须加 causal mask
#
# 例如生成句子时：
# - 当前已经生成了 “Travor like”
# - Decoder 要预测下一个 token
#
# 这时它：
# 1. 只能看自己和前面已经生成的 token
# 2. 还可以去参考 Encoder 输出的输入语义表示
#
# 所以 Decoder 兼具：
# - 自回归生成能力
# - 对输入信息的条件化能力

# %%
import torch
import torch.nn as nn

try:
    from MHA import MultiHeadAttention, make_causal_mask
    from FFN import PositionwiseFeedForward
    from PostionalEncoding import PositionalEncoding
except ImportError:
    from Transformer.MHA import MultiHeadAttention, make_causal_mask
    from Transformer.FFN import PositionwiseFeedForward
    from Transformer.PostionalEncoding import PositionalEncoding


class DecoderLayer(nn.Module):
    """
    原始 Transformer 风格的 Decoder Layer（Post-LN）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_attn = MultiHeadAttention(
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
        self.dropout3 = nn.Dropout(p=dropout)

        # LayerNorm 对每个 token 的最后一维做归一化：
        # miu = (1/d) * sum(x_i)
        # sigma^2 = (1/d) * sum((x_i - mu)^2)
        # y_i = gamma_i * (x_i - miu) / sqrt(sigma^2 + eps) + beta_i
        # 这里 d = d_model，所以它能让 x + 子层输出 之后的数值尺度更稳定。
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None, # mask out future tokens in the target sequence
        memory_mask: torch.Tensor | None = None, # mask out the source tokens that shouldn't be attended to
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x shape          : (batch_size, tgt_len, d_model)
        memory shape     : (batch_size, src_len, d_model)
        tgt_mask shape   :
        - (tgt_len, tgt_len)
        - (batch_size, tgt_len, tgt_len)
        - (batch_size, 1, tgt_len, tgt_len)
        memory_mask shape:
        - (tgt_len, src_len)
        - (batch_size, tgt_len, src_len)
        - (batch_size, 1, tgt_len, src_len)
        """
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        cross_attn_out, cross_attn_weights = self.cross_attn(
            x, memory, memory, mask=memory_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x, self_attn_weights, cross_attn_weights


# %% [markdown]
# ## Decoder Layer 的数据流
# 如果输入是：
# $$
# x \in \mathbb{R}^{B \times T \times d_{model}}
# $$
# 而 Encoder 输出是：
# $$
# memory \in \mathbb{R}^{B \times S \times d_{model}}
# $$
# 那么 Decoder Layer 会经历三步：
# ### 1. Masked Self-Attention
# $$
# x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Dropout}(\mathrm{MaskedMHA}(x)))
# $$
# ### 2. Cross-Attention
# $$
# x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Dropout}(\mathrm{CrossAttention}(x, memory)))
# $$
# 这里：
# - Query 来自当前 Decoder 状态
# - Key / Value 来自 Encoder 输出
# ### 3. FFN
# $$
# x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Dropout}(\mathrm{FFN}(x)))
# $$
# 最终输出 shape 仍然保持：
# $$
# (B, T, d_{model})
# $$

# %%
class TransformerDecoder(nn.Module):
    """
    原始 Transformer 论文主干结构实现的 Decoder（Post-LN 风格）。
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
                DecoderLayer(
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
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        x shape      : (batch_size, tgt_len, d_model)
        memory shape : (batch_size, src_len, d_model)
        """
        x = self.pos_encoding(x)

        if tgt_mask is None:
            tgt_mask = make_causal_mask(x.size(1), device=x.device)

        all_self_attn_weights = []
        all_cross_attn_weights = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)

        return x, all_self_attn_weights, all_cross_attn_weights


# %% [markdown]
# ## 为什么 Decoder 比 Encoder 多一个 Cross-Attention
# Encoder 只需要“理解输入”。
# Decoder 不一样，它不仅要看自己已经生成的前缀，还要参考输入语义。
#
# 所以 Decoder 多出来的 Cross-Attention 本质上是在做：
#
# “当前解码状态应该去 Encoder 输出里检索哪些信息。”
#
# 这也是为什么：
# - Encoder 输出通常叫 `memory`
# - Decoder 会把 `memory` 当作 Key / Value
# - 自己当前状态当作 Query

# %%
def demo_decoder_shapes():
    batch_size = 4
    src_len = 32
    tgt_len = 24
    d_model = 64
    num_heads = 4
    num_layers = 3
    d_ff = 256

    memory = torch.randn(batch_size, src_len, d_model)
    x = torch.randn(batch_size, tgt_len, d_model)

    decoder = TransformerDecoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0,
        max_len=64,
    )

    out, self_attn_list, cross_attn_list = decoder(x, memory)

    print("decoder input shape :", x.shape)
    print("encoder memory shape:", memory.shape)
    print("decoder output shape:", out.shape)
    print("num layers          :", len(self_attn_list))
    print("layer1 self-attn    :", self_attn_list[0].shape)
    print("layer1 cross-attn   :", cross_attn_list[0].shape)
    print()
    print("第一个样本第一个 token 解码前：")
    print(x[0, 0])
    print()
    print("第一个样本第一个 token 解码后：")
    print(out[0, 0])


# %% [markdown]
# ## 一个直观理解
# 可以把 Decoder 看成一个“边写边查资料”的过程：
#
# - Masked Self-Attention：只能回看自己已经写过的内容
# - Cross-Attention：去查 Encoder 已经整理好的输入信息
# - FFN：对当前 token 表示做进一步加工
#
# 所以 Decoder 的输出，不只是“基于前缀生成”，
# 还是“带条件输入约束的生成表示”。

# %%
if __name__ == "__main__":
    demo_decoder_shapes()
