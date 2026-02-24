# normalization.py
# 归一化层全景：BatchNorm、LayerNorm、GroupNorm、InstanceNorm
#
# 归一化层的共同目标：稳定各层输入分布，加速收敛，允许使用更大学习率。
# 不同场景对归一化统计维度的要求不同，选错会严重影响效果。
#
# 参考：
#   - Ioffe & Szegedy, "Batch Normalization", ICML 2015
#   - Ba et al., "Layer Normalization", arXiv 2016
#   - Wu & He, "Group Normalization", ECCV 2018

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. 各归一化方式的统计维度 ==========
print("=== 统计维度对比（CNN 特征图 (B, C, H, W)）===")
#
# 以 (B=4, C=8, H=32, W=32) 的特征图为例：
#
#  BatchNorm  ：对每个 channel，跨整个 batch + 空间统计均值和方差
#               统计维度 (B, H, W)，每个 channel 一组 (μ, σ)
#               计算 μ 时平均 B×H×W = 4×32×32 = 4096 个值 → 统计量稳定
#
#  LayerNorm  ：对每个样本，跨 channel + 空间统计（或仅 channel 维度）
#               统计维度 (C, H, W)，每个样本一组 (μ, σ)
#               不依赖 batch，适合 NLP（序列长度可变）和小 batch 训练
#
#  GroupNorm  ：将 channel 分成 G 组，每组内 + 空间统计
#               介于 BN（G=1 ≡ IN）和 LN（G=C ≡ LN）之间
#               batch=1 时依然有效（检测/分割小 batch 场景）
#
#  InstanceNorm：每个样本每个 channel 独立统计（统计维度 (H, W)）
#               风格迁移（Style Transfer）中用于去除内容信息保留风格
#
# 核心公式（对所有变体相同）：
#   y = γ · (x - μ) / √(σ² + ε) + β
#   γ（scale）和 β（bias）是可学习参数，初始化为 1 和 0

B, C, H, W = 4, 8, 32, 32
x = torch.randn(B, C, H, W)

bn = nn.BatchNorm2d(C)
ln = nn.LayerNorm([C, H, W])   # NLP 中通常是 nn.LayerNorm(d_model)
gn = nn.GroupNorm(num_groups=4, num_channels=C)  # 分 4 组，每组 2 channels
inst = nn.InstanceNorm2d(C, affine=True)

out_bn   = bn(x)
out_ln   = ln(x)
out_gn   = gn(x)
out_inst = inst(x)

for name, out in [("BatchNorm2d", out_bn), ("LayerNorm", out_ln),
                  ("GroupNorm(G=4)", out_gn), ("InstanceNorm2d", out_inst)]:
    print(f"{name:20s}: output {out.shape}, "
          f"mean={out.mean().item():.4f}, std={out.std().item():.4f}")
print()

# ========== 2. BatchNorm 详解 ==========
print("=== BatchNorm 详解 ===")
#
# 训练阶段：用当前 mini-batch 的 (μ_B, σ²_B) 归一化
# 推理阶段：用训练期间积累的指数移动平均 (running_mean, running_var) 归一化
#
# running_mean ← (1 - momentum) * running_mean + momentum * μ_B
# 推理时 running_mean/var 已固定，BN 退化为简单的线性缩放。
#
# BN 的缺陷：
#   ① batch size 小时统计量噪声大（batch=1 时退化为 InstanceNorm，效果极差）
#   ② 不适合序列模型（不同序列的同一 token 位置语义差异大，跨 batch 归一化无意义）
#   ③ 分布式训练时各卡独立统计与全局统计有偏差（SyncBN 可解决但有通信开销）

bn_test = nn.BatchNorm2d(C, momentum=0.1)  # momentum=0.1 是 PyTorch 默认值
x_train = torch.randn(16, C, H, W)

# 训练模式：running stats 被更新
bn_test.train()
_ = bn_test(x_train)
print(f"训练后 running_mean 均值: {bn_test.running_mean.mean().item():.4f}  （接近0）")
print(f"训练后 running_var  均值: {bn_test.running_var.mean().item():.4f}   （接近1）")

# 推理模式：使用 running stats，不更新
bn_test.eval()
x_eval = torch.randn(1, C, H, W)  # batch=1 在 eval 下正常工作（用 running stats）
out_eval = bn_test(x_eval)
print(f"eval 模式 batch=1 输出 shape: {out_eval.shape}")
print()

# ========== 3. LayerNorm 在 NLP 中的使用 ==========
print("=== LayerNorm（NLP 场景）===")
#
# NLP 中的输入形状：(B, S, d_model)，即 batch × seq_len × hidden_dim
# LayerNorm 归一化维度 d_model（每个 token 独立归一化）：
#   对每个 (B, S) 位置，跨 d_model 维度计算统计量
#   批大小和序列长度不影响统计量，天然适合可变长序列

d_model = 512
ln_nlp  = nn.LayerNorm(d_model)

x_nlp = torch.randn(8, 64, d_model)   # (B=8, S=64, d=512)
out_nlp = ln_nlp(x_nlp)

# 验证：每个 token 的输出应接近标准正态
# 选第一个 batch、第一个 token 位置
first_token = out_nlp[0, 0, :]
print(f"LN 后第一个 token: mean={first_token.mean().item():.4f}, "
      f"std={first_token.std().item():.4f}  （理论上 mean≈0, std≈1/√(1-1/d)≈1）")
print(f"实际 std（含 γ=1 缩放）: {first_token.std().item():.4f}")
print()

# ========== 4. GroupNorm 与小 batch 训练 ==========
print("=== GroupNorm（小 batch / 检测分割）===")
#
# GroupNorm 与 batch size 无关，batch=1 时效果与 batch=32 相同。
# 在目标检测（Faster R-CNN）和实例分割（Mask R-CNN）中：
#   由于高分辨率输入，batch size 通常只有 1~2，BN 效果差。
#   GN 成为标准选择（Facebook Research 原文建议 G=32）。
#
# 调参建议：
#   num_groups 选择使每组至少有 16~32 个 channel
#   C=64 → G=4（每组16），C=256 → G=32（每组8）

gn_variants = {
    "G=1 (InstanceNorm等价)": nn.GroupNorm(1, C),
    "G=4 (推荐)":              nn.GroupNorm(4, C),
    "G=8 (LayerNorm-like)":    nn.GroupNorm(8, C),  # G=C 等价 LN（沿空间维）
}

x_small = torch.randn(1, C, H, W)  # 极小 batch
print(f"batch=1 时各归一化结果 (C={C}):")
for name, norm in gn_variants.items():
    out = norm(x_small)
    print(f"  {name:30s}: mean={out.mean().item():.4f}, std={out.std().item():.4f}")
print()

# ========== 5. 快速选型指南 ==========
print("=== 归一化选型指南 ===")
print("""
场景                      推荐归一化
─────────────────────────────────────────────────────
CV（大 batch，batch≥16）  BatchNorm2d（收敛最快）
CV（小 batch，batch<4）   GroupNorm（G=32）
NLP / Transformer         LayerNorm（在 d_model 维）
大模型（LLaMA 等）        RMSNorm（去掉中心化，更高效）
风格迁移 / 图像生成       InstanceNorm2d（逐通道归一化）
GAN 生成器                GroupNorm 或 InstanceNorm（BN 训练不稳定）
""")

print("=== 总结 ===")
print("""
1. BN 跨 batch 统计，eval 用 running stats；小 batch 时统计噪声大
2. LN 跨 feature（d_model）统计，batch 无关；NLP 标准，Transformer 必备
3. GN 跨 channel 分组统计；batch 无关，小 batch 检测分割的救星
4. IN 每样本每 channel 独立统计；风格迁移去除内容信息
5. RMSNorm = LN 去掉均值中心化 + 去掉 β，更快，LLaMA/Mistral 的选择
""")
