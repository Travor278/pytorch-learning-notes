# adapter_layers.py
# Adapter 层：瓶颈结构的参数高效微调
#
# Adapter（Houlsby et al. 2019）是最早的 PEFT 方法之一。
# 在 Transformer 的 Attention 和 FFN 之后各插入一个小型瓶颈网络，
# 只训练 Adapter 参数（约 3~5% 的总参数量），冻结预训练权重。
#
# 相比 LoRA：
#   Adapter 修改前向计算路径（插入新模块），推理时有额外计算
#   LoRA 通过合并权重实现零推理开销
#
# 参考：
#   - Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019
#   - He et al., "Towards a Unified View of Parameter-Efficient Transfer Learning", ICLR 2022
#   - Hu et al., "LoRA", ICLR 2022

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. Adapter 瓶颈结构 ==========
print("=== Adapter 瓶颈结构 ===")
#
# 标准 Adapter（Houlsby 2019 的串行版本）：
#
#   x → LayerNorm → Down-projection (d_model → r) → Activation → Up-projection (r → d_model) → + x
#                                                                                                  ↑
#                                                                                          残差连接（确保初始化为恒等映射）
#
# 关键设计：
#   ① Down-projection（W_down）：将 d_model 维压缩到 r 维（r << d_model）
#   ② 非线性激活（GELU/ReLU）：引入非线性表达能力
#   ③ Up-projection（W_up）：从 r 维还原到 d_model 维
#   ④ 残差连接：确保初始化时 Adapter 输出 ≈ 输入（新能力增量式学习）
#
# 初始化策略：
#   W_up 初始化为近似 0（正态分布标准差 0.01），使得初始 Adapter 输出趋近于 0，
#   通过残差连接整体等价于恒等映射，保持预训练模型的初始行为。

class BotleneckAdapter(nn.Module):
    def __init__(self, d_model, r, activation=nn.GELU()):
        """
        d_model  : 隐层维度（与 Transformer 一致）
        r        : 瓶颈维度（通常 8~64，越小参数越少）
        """
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.down    = nn.Linear(d_model, r)
        self.act     = activation
        self.up      = nn.Linear(r, d_model)

        # 初始化：up-projection 接近 0，确保初始 Adapter ≈ 恒等映射
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Pre-LN + 瓶颈 + 残差
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.up(h)
        return x + h  # 残差：adapter 只学增量

d_model, r = 256, 16
adapter = BotleneckAdapter(d_model, r)

# 参数统计
params_adapter = sum(p.numel() for p in adapter.parameters())
params_linear  = d_model * d_model  # 等价的全量线性层
print(f"Adapter (r={r}) 参数量: {params_adapter}")
print(f"等价全量 Linear 参数量: {params_linear}")
print(f"参数压缩比: {params_linear / params_adapter:.1f}×")

# 验证初始化：输出应接近输入
x = torch.randn(4, 32, d_model)
out = adapter(x)
delta = (out - x).abs().mean().item()
print(f"\n初始 Adapter 输出与输入差异（期望≈0）: {delta:.6f}")
print()

# ========== 2. 将 Adapter 插入 Transformer ==========
print("=== Adapter 插入 Transformer Block ===")
#
# Houlsby 2019 原版：在 Attention 和 FFN 各加一个 Adapter（串行）
# 后续工作 Pfeiffer 2021：只在 FFN 后加一个 Adapter（效果相近，参数更少）
#
# 串行 Adapter（Houlsby）：
#   x → MultiHeadAttention → Adapter1 → Add&Norm → FFN → Adapter2 → Add&Norm
#
# 并行 Adapter（MAM Adapter 等）：
#   x → MultiHeadAttention → Add&Norm
#       ↘ Adapter（并行）  ↗

class TransformerBlockWithAdapter(nn.Module):
    """带 Adapter 的 Transformer Block（串行，Houlsby 2019 风格）"""
    def __init__(self, d_model=256, ffn_dim=1024, adapter_r=16):
        super().__init__()
        # 原始预训练层（冻结）
        self.attn_proj = nn.Linear(d_model, d_model)  # 简化 attention
        self.ffn_fc1   = nn.Linear(d_model, ffn_dim)
        self.ffn_fc2   = nn.Linear(ffn_dim, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

        # Adapter 层（可训练）
        self.adapter_attn = BotleneckAdapter(d_model, adapter_r)
        self.adapter_ffn  = BotleneckAdapter(d_model, adapter_r)

        # 冻结预训练参数
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

    def forward(self, x):
        # Attention sub-layer
        h    = self.attn_proj(self.norm1(x))
        h    = self.adapter_attn(h)           # Adapter 在 attn 后（串行）
        x    = x + h

        # FFN sub-layer
        h    = self.ffn_fc2(F.gelu(self.ffn_fc1(self.norm2(x))))
        h    = self.adapter_ffn(h)             # Adapter 在 FFN 后（串行）
        x    = x + h
        return x

block = TransformerBlockWithAdapter(d_model=256, ffn_dim=1024, adapter_r=16)

total     = sum(p.numel() for p in block.parameters())
trainable = sum(p.numel() for p in block.parameters() if p.requires_grad)
print(f"总参数量: {total:,}")
print(f"可训练（Adapter）: {trainable:,}  （{trainable/total*100:.1f}%）")

x_in  = torch.randn(2, 16, 256)
out_b = block(x_in)
print(f"前向传播正常: {out_b.shape}")
print()

# ========== 3. PEFT 方法横向对比 ==========
print("=== PEFT 方法对比 ===")
print("""
方法              参数位置            推理开销    典型参数比     主要优势
────────────────────────────────────────────────────────────────────────
Full Fine-tuning  所有参数            无          100%          效果上界
Adapter           插入新瓶颈模块      有（小）    1~5%          模块化，易组合
LoRA              分解权重矩阵        无（合并后）0.1~1%        推理无开销，参数少
Prefix Tuning     在 K/V 前拼接可训练 有（每层）  0.1~1%        不修改权重结构
Prompt Tuning     只训练 soft prompt  有（输入端）< 0.1%        最少参数
IA³               缩放 K/V/FFN       无          < 0.1%        极少参数

推荐：
  - 参数效率最优：LoRA（推理零开销 + 效果最接近全量）
  - 多任务切换：Adapter（不同任务的 Adapter 可热插拔，主干共享）
  - 极小参数：Prompt Tuning / IA³（但效果不稳定）
""")

# ========== 4. 参数统计工具函数 ==========
print("=== 参数统计工具 ===")

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数:    {total:>10,}")
    print(f"  可训练:    {trainable:>10,}  ({trainable/total*100:.2f}%)")
    print(f"  冻结:      {total-trainable:>10,}  ({(total-trainable)/total*100:.2f}%)")
    return trainable, total

print("TransformerBlockWithAdapter（含 Adapter）:")
count_parameters(block)
