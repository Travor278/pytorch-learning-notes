# %% [markdown]
# # Transformer
# 现在把前面已经实现的模块拼起来：
#
# - Token Embedding
# - Positional Encoding（由 Encoder / Decoder 内部完成）
# - Transformer Encoder
# - Transformer Decoder
# - Generator（输出到词表）
#
# 这样我们就得到一个按原始论文主干结构拼装的完整 Transformer。

# %% [markdown]
# ## 整体数据流
# 设：
# - 源序列 token id：
# $$
# src \in \mathbb{R}^{B \times S}
# $$
# - 目标序列 token id：
# $$
# tgt \in \mathbb{R}^{B \times T}
# $$
# 那么完整流程可以写成：
# 1. `src -> src_embedding -> encoder -> memory`
# 2. `tgt -> tgt_embedding -> decoder(memory) -> decoder_out`
# 3. `decoder_out -> generator -> logits`
#
# 最终：
# $$
# logits \in \mathbb{R}^{B \times T \times vocab\_size}
# $$
# 这表示：每个目标位置在整个词表上的分数。

# %% [markdown]
# ## 为什么 Embedding 后要乘 $\sqrt{d_{model}}$
# 在原始 Transformer 论文里，token embedding 常写成：
#
# $$
# x = \sqrt{d_{model}} \cdot E(token)
# $$
#
# 直觉上，这不是像 attention 里除以 $\sqrt{d_k}$ 那样为了把方差压回常数量级，
# 而是为了把 embedding 的整体幅度抬高一些，让它和位置编码、后续隐藏表示的尺度更匹配。
#
# 一个常见的量级分析是：
# 如果 embedding 每一维初始化后近似满足
#
# $$
# E_i \sim \text{mean } 0,\ \text{var } \sigma^2
# $$
#
# 那么一个 $d_{model}$ 维 embedding 向量的期望平方范数大致为：
#
# $$
# \mathbb{E}\|E\|^2
# = \sum_{i=1}^{d_{model}} \mathbb{E}[E_i^2]
# \approx d_{model}\sigma^2
# $$
#
# 如果再乘上 $\sqrt{d_{model}}$，则有：
#
# $$
# \mathbb{E}\|\sqrt{d_{model}}E\|^2
# = d_{model}\,\mathbb{E}\|E\|^2
# \approx d_{model}^2\sigma^2
# $$
#
# 所以这一步可以理解成：
# - token embedding 的幅度被整体放大
# - 输入到 Transformer 时，词语义信号不会显得太弱
# - 与位置编码相加时，量级更协调
#
# 一句话说：
#
# $$
# \text{embedding} \times \sqrt{d_{model}}
# $$
#
# 更像是一种输入尺度设计，而不是像
#
# $$
# \frac{QK^T}{\sqrt{d_k}}
# $$
#
# 那样的 attention score 方差稳定化。

# %%
import math

import torch
import torch.nn as nn

try:
    from Encoder import TransformerEncoder
    from Decoder import TransformerDecoder
    from create_mask import (
        create_memory_mask,
        create_src_padding_mask,
        create_tgt_mask,
    )
except ImportError:
    from Transformer.Encoder import TransformerEncoder
    from Transformer.Decoder import TransformerDecoder
    from Transformer.create_mask import (
        create_memory_mask,
        create_src_padding_mask,
        create_tgt_mask,
    )


class Transformer(nn.Module):
    """
    原始 Transformer 论文主干结构拼装的完整模型。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )
        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        # Generator: 把 Decoder hidden state 映射到词表维度。
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def encode(
        self,
        src_tokens: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        把源 token id 序列编码成 可供 Decoder 查询的 memory
        src_tokens shape: (batch_size, src_len)
        """
        # 这里乘 sqrt(d_model) 是原论文里的 embedding 缩放。
        # 如果 embedding 每一维初始近似满足：
        # E_i ~ mean 0, var sigma^2
        # 那么一个 d_model 维向量的期望平方范数大致是：
        # E[||E||^2] = sum(E[E_i^2]) ≈ d_model * sigma^2
        # 乘上 sqrt(d_model) 后：
        # E[||sqrt(d_model) * E||^2]
        # = d_model * E[||E||^2]
        # ≈ d_model^2 * sigma^2
        # 直觉上就是把 token embedding 的整体量级抬高一些，
        # 让它和位置编码、后续 Transformer 表示更匹配。
        src_embed = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        memory, enc_self_attn_list = self.encoder(src_embed, src_mask=src_mask)
        return memory, enc_self_attn_list

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        把目标 token 前缀和源端 memory 结合起来，算出 Decoder 的输出表示
        tgt_tokens shape: (batch_size, tgt_len)
        memory shape    : (batch_size, src_len, d_model)
        """
        # 与 source embedding 同理，这里也做 sqrt(d_model) 缩放，
        # 让 target token embedding 的输入幅度更合理。
        tgt_embed = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)
        decoder_out, dec_self_attn_list, dec_cross_attn_list = self.decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return decoder_out, dec_self_attn_list, dec_cross_attn_list

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        src_tokens shape: (batch_size, src_len)
        tgt_tokens shape: (batch_size, tgt_len)
        """
        memory, enc_self_attn_list = self.encode(src_tokens, src_mask=src_mask)
        decoder_out, dec_self_attn_list, dec_cross_attn_list = self.decode(
            tgt_tokens,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        logits = self.generator(decoder_out)
        return logits, enc_self_attn_list, dec_self_attn_list, dec_cross_attn_list


# %% [markdown]
# ## 为什么还要有 Generator
# Decoder 输出的还是 hidden states：
#
# $$
# (B, T, d_{model})
# $$
#
# 但真正训练和预测时，我们要的是：
#
# “每个目标位置对词表中每个词的分数”
#
# 所以需要最后一层：
#
# $$
# \text{logits} = W_{out} \cdot decoder\_out + b
# $$
#
# 也就是 `Linear(d_model, tgt_vocab_size)`。
#
# 如果要把 logits 变成词表概率分布，可以再做：
#
# $$
# p_i = \mathrm{softmax}(z_i)
# $$
#
# 其中 $z_i$ 是第 $i$ 个词的 logit。
#
# 推理时最简单的 greedy decoding 是：
#
# $$
# \operatorname*{argmax}_i z_i
# $$
#
# 因为：
#
# $$
# \operatorname*{argmax}_i z_i
# =
# \operatorname*{argmax}_i \mathrm{softmax}(z_i)
# $$
#
# 所以如果只是取最大值，其实不一定要显式先做 softmax。
#
# 但如果想采样，就常会用 temperature sampling：
#
# $$
# p_i = \mathrm{softmax}\left(\frac{z_i}{T}\right)
# $$
#
# 其中：
# - $T < 1$：分布更尖锐，更保守
# - $T = 1$：就是普通 softmax
# - $T > 1$：分布更平缓，更随机
#
# 所以 temperature 的作用，本质上是调节“生成更像 greedy，还是更有随机性”。

# %%
def demo_transformer_shapes():
    batch_size = 2
    src_len = 12
    tgt_len = 10

    src_vocab_size = 100
    tgt_vocab_size = 120
    pad_idx = 0

    d_model = 64
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 256

    src_tokens = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt_tokens = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    src_tokens[0, -3:] = pad_idx
    tgt_tokens[1, -2:] = pad_idx

    src_mask = create_src_padding_mask(src_tokens, pad_idx)
    tgt_mask = create_tgt_mask(tgt_tokens, pad_idx)
    memory_mask = create_memory_mask(tgt_tokens, src_tokens, pad_idx)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=0.0,
        max_len=128,
    )

    logits, enc_attn_list, dec_self_attn_list, dec_cross_attn_list = model(
        src_tokens,
        tgt_tokens,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
    )

    print("src_tokens shape        :", src_tokens.shape)
    print("tgt_tokens shape        :", tgt_tokens.shape)
    print("logits shape            :", logits.shape)
    print("encoder layers          :", len(enc_attn_list))
    print("decoder layers          :", len(dec_self_attn_list))
    print("layer1 enc self-attn    :", enc_attn_list[0].shape)
    print("layer1 dec self-attn    :", dec_self_attn_list[0].shape)
    print("layer1 dec cross-attn   :", dec_cross_attn_list[0].shape)
    print()
    print("第一个样本第一个目标位置的 logits 前 10 维：")
    print(logits[0, 0, :10])


# %%
if __name__ == "__main__":
    demo_transformer_shapes()
    