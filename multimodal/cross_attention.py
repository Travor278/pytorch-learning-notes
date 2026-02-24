# cross_attention.py
# 跨模态交叉注意力：多模态融合的核心机制
#
# 交叉注意力（Cross-Attention）允许一种模态的信息"查询"另一种模态，
# 是 Transformer Encoder-Decoder 架构的核心，也是多模态融合的主要手段。
# 在 Flamingo / IDEFICS / LLaVA 等多模态模型中被广泛使用。
#
# 参考：
#   - Vaswani et al., "Attention Is All You Need", NeurIPS 2017（原始 cross-attn）
#   - Alayrac et al., "Flamingo: a Visual Language Model", NeurIPS 2022
#   - Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training", ICML 2023（Q-Former）

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== 1. Cross-Attention 与 Self-Attention 的区别 ==========
print("=== Cross-Attention 结构 ===")
print("""
Self-Attention：Query、Key、Value 均来自同一序列
  Q = K = V = X_text  → 文本内部关系建模

Cross-Attention：Query 来自一种模态，Key/Value 来自另一种模态
  Q = W_Q · X_text    ← 来自语言 token（"我想要什么信息"）
  K = W_K · X_vision  ← 来自图像 token（"图像能提供什么"）
  V = W_V · X_vision  ← 来自图像 token（"实际要传递的内容"）

  输出形状与 Q 相同（语言序列长度），但内容融合了视觉信息。
  直觉：文本 token 通过 attention 机制"看"图像的不同区域，
        动态地从图像中提取与当前文本上下文相关的视觉特征。
""")

# ========== 2. Cross-Attention 实现 ==========
print("=== Cross-Attention 实现 ===")

class CrossAttention(nn.Module):
    """
    多头跨模态注意力层
    Q 来自查询序列（如语言），K/V 来自上下文序列（如图像）
    """
    def __init__(self, d_query, d_context, d_model, num_heads, dropout=0.1):
        """
        d_query  : Query 序列的特征维度（语言 token 维度）
        d_context: K/V 序列的特征维度（图像 token 维度）
        d_model  : 内部注意力维度（通常与 d_query 相同）
        num_heads: 注意力头数
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # 三个投影：Q 来自查询序列，K/V 来自上下文
        self.W_q = nn.Linear(d_query,   d_model, bias=False)
        self.W_k = nn.Linear(d_context, d_model, bias=False)
        self.W_v = nn.Linear(d_context, d_model, bias=False)
        self.W_o = nn.Linear(d_model,   d_query, bias=False)  # 输出回到查询维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, mask=None):
        """
        query  : (B, S_q, d_query)   查询序列（语言 token）
        context: (B, S_c, d_context) 上下文序列（图像 token）
        mask   : (B, S_q, S_c) 可选的注意力掩码
        返回   : (B, S_q, d_query)  与 query 同形
        """
        B, S_q, _ = query.shape
        B, S_c, _ = context.shape

        # 投影并分头
        Q = self.W_q(query).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(context).view(B, S_c, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(context).view(B, S_c, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (B, h, S_q, d_k)   K/V: (B, h, S_c, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (B, h, S_q, S_c) —— 每个语言 token 对所有图像 token 的相关性

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn    = F.softmax(scores, dim=-1)   # 沿图像 token 维度归一化
        attn    = self.dropout(attn)
        context_out = torch.matmul(attn, V)   # (B, h, S_q, d_k)

        # 合并多头，投影回输出维度
        context_out = context_out.transpose(1, 2).contiguous().view(B, S_q, self.d_model)
        output = self.W_o(context_out)        # (B, S_q, d_query)
        return output, attn

# 测试：语言 token 查询图像 token
B        = 4
S_lang   = 32   # 语言序列长度
S_vis    = 196  # 图像 token 数量（ViT-B/16 在 224×224 上产生 196 个 patch）
d_lang   = 768  # 语言模型隐层维度（BERT-base）
d_vis    = 768  # CLIP ViT 特征维度
d_model  = 512
n_heads  = 8

cross_attn = CrossAttention(d_lang, d_vis, d_model, n_heads)

lang_tokens = torch.randn(B, S_lang, d_lang)   # 文本 token 序列
vis_tokens  = torch.randn(B, S_vis,  d_vis)    # 图像 patch 序列

out, attn_weights = cross_attn(lang_tokens, vis_tokens)
print(f"语言 token: ({B}, {S_lang}, {d_lang})")
print(f"图像 token: ({B}, {S_vis}, {d_vis})")
print(f"Cross-Attn 输出: {out.shape}  （形状与语言序列一致）")
print(f"注意力权重: {attn_weights.shape}  （每个语言 token 关注所有图像 patch）")
print()

# ========== 3. Flamingo 式的门控交叉注意力 ==========
print("=== 门控交叉注意力（Gated Cross-Attention，Flamingo 2022）===")
#
# Flamingo 的设计：
#   在预训练语言模型的每隔一层插入门控交叉注意力层，
#   语言模型本身（self-attention + FFN）完全冻结。
#   只训练新插入的交叉注意力层。
#
# 门控机制（Gating）：
#   output = x + tanh(α) * cross_attn(x, visual_tokens)
#   α 是可学习标量，初始化为 0 → tanh(0) = 0
#   训练开始时交叉注意力贡献为 0，模型行为与原始 LM 完全一致，
#   随着 α 的学习，视觉信息逐渐融入。
#   这保证了初始化时模型的稳定性（类似 Adapter 的零初始化技巧）。

class GatedCrossAttentionBlock(nn.Module):
    """Flamingo 风格的门控交叉注意力块"""
    def __init__(self, d_lang, d_vis, d_model, num_heads):
        super().__init__()
        self.cross_attn = CrossAttention(d_lang, d_vis, d_model, num_heads)
        self.norm       = nn.LayerNorm(d_lang)
        self.alpha_attn = nn.Parameter(torch.tensor(0.0))  # 门控参数，初始化为 0

        # FFN（可选，Flamingo 还有一个门控 FFN）
        self.ffn   = nn.Sequential(
            nn.Linear(d_lang, d_lang * 4),
            nn.GELU(),
            nn.Linear(d_lang * 4, d_lang)
        )
        self.alpha_ffn = nn.Parameter(torch.tensor(0.0))
        self.norm2     = nn.LayerNorm(d_lang)

    def forward(self, x, visual_tokens):
        """
        x           : (B, S_lang, d_lang) 语言 token（来自冻结 LM）
        visual_tokens: (B, S_vis, d_vis)  图像 token（来自 CLIP ViT）
        """
        # 门控交叉注意力（α 初始为 0，tanh(0) = 0）
        attn_out, _ = self.cross_attn(self.norm(x), visual_tokens)
        x = x + torch.tanh(self.alpha_attn) * attn_out

        # 门控 FFN
        x = x + torch.tanh(self.alpha_ffn) * self.ffn(self.norm2(x))
        return x

gated_block = GatedCrossAttentionBlock(d_lang, d_vis, d_model=d_lang, num_heads=8)

# 验证初始化：gate=0 时输出应与输入相同
with torch.no_grad():
    out_gated = gated_block(lang_tokens, vis_tokens)
    diff = (out_gated - lang_tokens).abs().max().item()
    print(f"门控初始化验证（alpha=0 时输出≈输入）: 最大差异={diff:.6f}")
    print(f"  alpha_attn: {gated_block.alpha_attn.item():.2f}，"
          f"tanh(alpha): {gated_block.alpha_attn.tanh().item():.6f}")
print()

# ========== 4. Q-Former（BLIP-2 的连接模块）==========
print("=== Q-Former（BLIP-2 2023）===")
print("""
Q-Former（Querying Transformer）是 BLIP-2 提出的图文接口模块，
解决 CLIP 特征维度（256 个 patch token）与 LLM 序列长度不兼容的问题：

  图像 → CLIP ViT (冻结) → 256 个 vision token (d=1408)
                                      ↓
                               Q-Former
                              (可学习的 32 个 query token)
                              ↑ self-attn: query 之间交互
                              ↑ cross-attn: query 查询 vision token
                                      ↓
                               32 个 query 输出 (d=768)
                                      ↓
                            Linear Projection
                                      ↓
                               LLM（OPT/LLaMA）

Q-Former 的核心思路：
  用 32 个可学习的 query 向量（类似 Perceiver IO 的 latent array），
  通过交叉注意力从 256 个图像 patch 中"提取"最相关的视觉信息，
  将图像信息压缩到固定的 32 个 token 再输入 LLM。

优点：
  ① 图像 token 数量与分辨率无关（总是 32 个）
  ② Q-Former 可以在图文对上做预训练（图文匹配、图文生成等任务）
  ③ 对 LLM 的输入长度更友好，推理效率高
""")

# 模拟 Q-Former 的核心结构
class QFormerLayer(nn.Module):
    """单层 Q-Former（简化版）"""
    def __init__(self, d_query, d_context, num_heads):
        super().__init__()
        # Query 之间的 Self-Attention
        self.self_attn  = nn.MultiheadAttention(d_query, num_heads, batch_first=True)
        self.norm1      = nn.LayerNorm(d_query)
        # Query 对 Vision Token 的 Cross-Attention
        self.cross_attn = CrossAttention(d_query, d_context, d_query, num_heads)
        self.norm2      = nn.LayerNorm(d_query)
        # FFN
        self.ffn  = nn.Sequential(nn.Linear(d_query, d_query*4), nn.GELU(), nn.Linear(d_query*4, d_query))
        self.norm3= nn.LayerNorm(d_query)

    def forward(self, queries, vision_tokens):
        # Self-attention（query 之间交互）
        h, _ = self.self_attn(self.norm1(queries), self.norm1(queries), self.norm1(queries))
        queries = queries + h
        # Cross-attention（query → vision）
        h, _ = self.cross_attn(self.norm2(queries), vision_tokens)
        queries = queries + h
        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries

B_q       = 4
N_queries = 32    # Q-Former 固定的 query 数量
d_q       = 768   # query 维度
d_vision  = 1024  # CLIP ViT-L 的特征维度
S_vision  = 256   # ViT-L/14@336 产生的 patch 数量

q_former_layer = QFormerLayer(d_q, d_vision, num_heads=8)

learnable_queries = torch.randn(B_q, N_queries, d_q)   # 可学习 query（类似 nn.Embedding）
vision_tokens_q   = torch.randn(B_q, S_vision, d_vision)

out_q = q_former_layer(learnable_queries, vision_tokens_q)
print(f"Q-Former 输入: 图像 ({B_q}, {S_vision}, {d_vision}) + query ({B_q}, {N_queries}, {d_q})")
print(f"Q-Former 输出: {out_q.shape}  ← 固定 {N_queries} 个 query，无论图像分辨率")
print()

print("=== 总结 ===")
print("""
1. Cross-Attention: Q 来自语言，K/V 来自图像；文本 token 动态聚焦图像区域
2. 输出形状与 Query 序列一致（语言维度），但内容融合了视觉信息
3. 门控 Cross-Attention（Flamingo）：alpha 初始为 0 保证训练稳定性
4. Q-Former（BLIP-2）：用 32 个可学习 query 压缩图像信息，桥接 ViT 和 LLM
5. LLaVA 更简单：直接用 Linear/MLP 将 CLIP 特征投影到 LLM 的 token 空间
""")
