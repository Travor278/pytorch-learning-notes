# contrastive_loss.py
# 对比损失：InfoNCE、NT-Xent（SimCLR）与 Triplet Loss
#
# 对比学习的核心思想：
#   不需要人工标注，通过"拉近相似样本、推远不相似样本"来学习表征。
#   自然产生的"正样本对"（同一图片的两种增强）作为监督信号。
#
# 参考：
#   - Gutmann & Hyvärinen, "Noise-Contrastive Estimation", AISTATS 2010（NCE 原型）
#   - Chen et al., "A Simple Framework for Contrastive Learning", ICML 2020（SimCLR）
#   - Oord et al., "Representation Learning with Contrastive Predictive Coding", NeurIPS 2018（InfoNCE）
#   - Schroff et al., "FaceNet: A Unified Embedding for Face Recognition", CVPR 2015（Triplet）

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. InfoNCE 损失 ==========
print("=== InfoNCE 损失 ===")
#
# 设 batch 内有 N 个样本，每个样本有一个对应的正样本（positive）：
#   query z_i 与 key z_i⁺（正样本）相似，与 z_j（j≠i，负样本）不相似。
#
# InfoNCE(z_i, z_i⁺) = -log [ exp(sim(z_i, z_i⁺)/τ) / Σ_j exp(sim(z_i, z_j)/τ) ]
#
# 其中 sim 通常是余弦相似度，τ 是温度超参数（类似 KL 中的 T）。
# 可以理解为一个 N 类分类问题：给定 query，在 batch 内找出正样本。
# 最小化 InfoNCE 等价于最大化 z_i 与 z_i⁺ 的互信息下界（Oord 2018）。
#
# 温度 τ 的作用：
#   τ 小（如 0.07）：对困难负样本（hard negative）的梯度更大，判别力强，但训练更不稳定
#   τ 大（如 0.5）：梯度更均匀，训练稳定但表征区分度可能不足
#   SimCLR 用 τ=0.1，CLIP 用可学习的 τ（初始化为 log(1/0.07)）

def info_nce_loss(z1, z2, temperature=0.1):
    """
    z1, z2: (N, D) 已 L2 归一化的嵌入向量
    z1[i] 和 z2[i] 是同一样本的两个视角（正样本对）
    其余 j≠i 的 z2[j] 作为负样本
    """
    N = z1.size(0)

    # 计算所有 z1[i] 与 z2[j] 的余弦相似度矩阵 (N, N)
    # 由于已 L2 归一化，内积 = 余弦相似度
    sim_matrix = torch.mm(z1, z2.T) / temperature  # (N, N)

    # 正样本对角线上：z1[i] 对应 z2[i]
    # InfoNCE 是 N 个 N 类分类问题，标签为 [0, 1, 2, ..., N-1]
    labels = torch.arange(N, device=z1.device)
    loss   = F.cross_entropy(sim_matrix, labels)
    return loss

# 模拟
N, D = 32, 128
torch.manual_seed(42)
z1 = F.normalize(torch.randn(N, D), dim=-1)
# z2 是 z1 加上轻微噪声（模拟同一图片的两个增强视角）
z2 = F.normalize(z1 + 0.1 * torch.randn(N, D), dim=-1)

loss_nce = info_nce_loss(z1, z2, temperature=0.1)
print(f"InfoNCE loss (τ=0.1): {loss_nce.item():.4f}  （完美对齐时理论值 -log(1/N) = {-(-torch.log(torch.tensor(1.0/N)).item()):.4f}）")

# 温度对损失的影响
for tau in [0.05, 0.1, 0.3, 0.5]:
    l = info_nce_loss(z1, z2, temperature=tau)
    print(f"  τ={tau}: loss={l.item():.4f}")
print()

# ========== 2. NT-Xent 损失（SimCLR）==========
print("=== NT-Xent 损失（SimCLR）===")
#
# SimCLR（Chen et al. 2020）的双向 InfoNCE：
#   一个 batch 有 N 个原始样本，每个做 2 种增强，共 2N 个视角。
#   正样本对：(z_i, z_{i+N})（同一样本的两种增强）
#   负样本：batch 内其他 2(N-1) 个样本
#
# NT-Xent = (1/2N) Σ_{i=1}^{2N} -log[ exp(sim(z_i,z_j)/τ) / Σ_{k≠i} exp(sim(z_i,z_k)/τ) ]
# 对称地对两个方向（z_i→z_j 和 z_j→z_i）都计算，再平均。
#
# NT-Xent 的关键工程细节：
#   同一样本的两个增强互为正对，但不能把自己（z_i → z_i）计入分母，
#   需要 mask 掉对角线。

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (N, D) 各自 L2 归一化后的嵌入
        """
        N = z1.size(0)
        # 合并为 2N
        z  = torch.cat([z1, z2], dim=0)  # (2N, D)
        sim = torch.mm(z, z.T) / self.T  # (2N, 2N)

        # 正样本位置：i 对应 i+N，i+N 对应 i
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)

        # 排除自身（对角线），避免计入分母
        mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        sim  = sim.masked_fill(mask, -1e9)

        loss = F.cross_entropy(sim, labels)
        return loss

nt_xent = NTXentLoss(temperature=0.1)
loss_ntx = nt_xent(z1, z2)
print(f"NT-Xent loss: {loss_ntx.item():.4f}")
print()

# ========== 3. Triplet Loss ==========
print("=== Triplet Loss ===")
#
# Triplet Loss（Schroff et al. 2015，FaceNet）：
#   给定 (anchor, positive, negative) 三元组：
#   L = max(0, ‖a - p‖² - ‖a - n‖² + margin)
#
#   目标：anchor 与 positive 的距离 < anchor 与 negative 的距离 - margin
#   margin 防止模型坍塌为所有嵌入相同的平凡解
#
# Hard Mining 策略（FaceNet 的关键）：
#   Semi-hard negative：选择满足 ‖a-p‖ < ‖a-n‖ < ‖a-p‖ + margin 的负样本
#   Hard negative：‖a-n‖ < ‖a-p‖ 的负样本（梯度信号最强，但训练不稳定）
#   Easy negative：‖a-n‖ >> ‖a-p‖ + margin，损失为 0，梯度消失

# PyTorch 内置实现
triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2)

torch.manual_seed(0)
anchor   = F.normalize(torch.randn(8, 64), dim=-1)
positive = F.normalize(anchor + 0.1 * torch.randn(8, 64), dim=-1)  # 接近 anchor
negative = F.normalize(torch.randn(8, 64), dim=-1)                  # 随机负样本

loss_triplet = triplet_loss(anchor, positive, negative)
d_pos = F.pairwise_distance(anchor, positive).mean().item()
d_neg = F.pairwise_distance(anchor, negative).mean().item()
print(f"Triplet loss: {loss_triplet.item():.4f}")
print(f"d(a,p) mean: {d_pos:.4f}  d(a,n) mean: {d_neg:.4f}")
print(f"margin 满足: {d_neg > d_pos:.1f}")
print()

# ========== 4. In-batch Negatives 的效率优势 ==========
print("=== In-batch Negatives vs Explicit Negatives ===")
print("""
显式负样本（Triplet / Contrastive Loss）：
  需要事先构建负样本，mining 成本高，每次更新只利用 1 个负样本。

In-batch Negatives（InfoNCE / NT-Xent）：
  同一 batch 内所有其他样本自动成为负样本。
  batch size=256 → 每个 query 有 255 个负样本，信息利用率极高。
  大 batch size 通常使对比学习性能更好（更多 hard negative）。
  SimCLR 用 batch_size=4096 才达到最佳效果。

对比学习中 batch size 的"免费午餐"：
  更大 batch = 更多负样本 = 更准确的噪声对比估计 = 更好的表征。
  这是 SimCLR/CLIP 需要大量 GPU 训练的根本原因。
""")

print("=== 总结 ===")
print("""
1. InfoNCE = N 类分类，正样本为目标，batch 内其余为负样本
2. NT-Xent 是 InfoNCE 的双向对称版本，SimCLR 的核心损失
3. 温度 τ 控制难负样本的梯度权重：小 τ 聚焦 hard negative，大 τ 训练稳定
4. Triplet loss 需要显式 mining；InfoNCE 利用 in-batch negatives 更高效
5. 对比学习中 batch size 越大越好，是需要大算力的本质原因
""")
