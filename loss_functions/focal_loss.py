# focal_loss.py
# Focal Loss：解决类别不平衡的损失函数
#
# 来源：Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017（RetinaNet 论文）
# 问题背景：目标检测中前景/背景极度不平衡（1:1000 量级），
# 大量"容易"的负样本主导了梯度，抑制了对困难样本的学习。
#
# Focal Loss 通过降低"容易"样本的损失权重，让模型专注于"困难"样本。

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. 从交叉熵到 Focal Loss ==========
print("=== Cross-Entropy → Focal Loss ===")
#
# 标准交叉熵（二分类）：
#   CE(p, y) = -y·log(p) - (1-y)·log(1-p)
#   简写 p_t = p if y=1 else (1-p)
#   CE = -log(p_t)
#
# 问题：模型对"容易"样本预测置信度很高（p_t ≈ 0.9），
#        此时 CE = -log(0.9) ≈ 0.1 虽然单个损失不大，
#        但背景样本数量极多（数万个），累积梯度远超前景。
#
# α-balanced CE（第一步）：
#   CE_α = -α_t · log(p_t)    （α 直接加权不同类别）
#   问题：仍然无法区分"容易"和"困难"样本
#
# Focal Loss（第二步）：
#   FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
#
#   调制因子 (1 - p_t)^γ 的作用：
#     - p_t → 1（容易样本）：(1-p_t)^γ → 0，损失几乎为 0（被抑制）
#     - p_t → 0（困难样本）：(1-p_t)^γ → 1，损失等于标准 CE（不受影响）
#   γ=0 时退化为标准 α-balanced CE。
#   Lin et al. 实验表明 γ=2, α=0.25 在 COCO 检测上效果最优。

def focal_loss_manual(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    logits : (N,) 二分类的原始 logit（未经过 sigmoid）
    targets: (N,) 二值标签 {0, 1}
    alpha  : 正类的权重（通常 < 0.5，因为正类困难但数量少）
    gamma  : 聚焦参数（越大越聚焦于困难样本）
    """
    p    = torch.sigmoid(logits)
    # p_t：正类时用 p，负类时用 1-p
    p_t  = p * targets + (1 - p) * (1 - targets)
    # α_t：正类用 alpha，负类用 1-alpha
    a_t  = alpha * targets + (1 - alpha) * (1 - targets)

    # 调制因子
    focal_weight = (1 - p_t) ** gamma

    # 二分类交叉熵（F.binary_cross_entropy_with_logits 数值更稳定）
    bce  = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')

    loss = a_t * focal_weight * bce

    if   reduction == 'mean': return loss.mean()
    elif reduction == 'sum' : return loss.sum()
    return loss

# 演示：容易样本 vs 困难样本的损失差异
easy_logits = torch.tensor([5.0, -5.0])   # 模型很确定的预测
hard_logits = torch.tensor([0.1, -0.1])   # 模型接近随机猜测
targets_e   = torch.tensor([1, 0])
targets_h   = torch.tensor([1, 0])

bce_easy  = F.binary_cross_entropy_with_logits(easy_logits, targets_e.float(), reduction='mean')
bce_hard  = F.binary_cross_entropy_with_logits(hard_logits, targets_h.float(), reduction='mean')
fl_easy   = focal_loss_manual(easy_logits, targets_e, gamma=2.0, reduction='mean')
fl_hard   = focal_loss_manual(hard_logits, targets_h, gamma=2.0, reduction='mean')

print(f"容易样本（确信预测）: CE={bce_easy.item():.4f},  FL(γ=2)={fl_easy.item():.4f}  ← FL 大幅抑制")
print(f"困难样本（随机猜测）: CE={bce_hard.item():.4f},  FL(γ=2)={fl_hard.item():.4f}  ← FL 基本不变")
print(f"FL/CE 比率（容易）: {fl_easy.item()/bce_easy.item():.3f}")
print(f"FL/CE 比率（困难）: {fl_hard.item()/bce_hard.item():.3f}")
print()

# ========== 2. γ 参数对损失的影响 ==========
print("=== γ 参数的影响 ===")
#
# γ 越大，对容易样本（p_t 接近 1）的抑制越强
# γ=0：标准 CE；γ=2：RetinaNet 默认；γ=5：极度聚焦困难样本

import numpy as np

# p_t 范围：0.01 到 0.99
p_t_vals = torch.linspace(0.01, 0.99, 20)

print(f"{'p_t':>8} {'γ=0':>10} {'γ=1':>10} {'γ=2':>10} {'γ=5':>10}")
for p_t in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    p_t_t = torch.tensor(p_t)
    ce    = -torch.log(p_t_t)
    row   = f"{p_t:>8.2f}"
    for gamma in [0, 1, 2, 5]:
        fl = (1 - p_t_t) ** gamma * ce
        row += f" {fl.item():>10.4f}"
    print(row)
print()

# ========== 3. 多分类 Focal Loss ==========
print("=== 多分类 Focal Loss ===")
#
# 将思路推广到多分类（与 CrossEntropyLoss 对应）：
# FL_multi(z, y) = -(1 - softmax(z)[y])^γ · log(softmax(z)[y])
#
# 注意：多分类场景下通常不需要 α（类别不平衡可通过 class_weight 处理），
#       单纯使用 γ 来聚焦困难样本。

class FocalLossMultiClass(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits : (N, C) 多分类 logits
        targets: (N,)   类别标签
        """
        log_prob = F.log_softmax(logits, dim=-1)          # (N, C)
        prob     = log_prob.exp()                          # (N, C)

        # 取出目标类别的 log_prob 和 prob
        log_prob_t = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        prob_t     = prob.gather(1, targets.unsqueeze(1)).squeeze(1)      # (N,)

        focal_weight = (1 - prob_t) ** self.gamma
        loss = -focal_weight * log_prob_t

        if   self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum' : return loss.sum()
        return loss

N, C = 64, 10
torch.manual_seed(42)
logits  = torch.randn(N, C)
targets = torch.randint(0, C, (N,))

fl_loss = FocalLossMultiClass(gamma=2.0)(logits, targets)
ce_loss = F.cross_entropy(logits, targets)

print(f"CE loss:       {ce_loss.item():.4f}")
print(f"Focal loss (γ=2): {fl_loss.item():.4f}  （通常 < CE，因为容易样本被抑制）")
print()

# ========== 4. 类别权重 vs Focal Loss 的对比 ==========
print("=== 类别权重 vs Focal Loss ===")
print("""
类别权重（class_weight）：
  针对不同类别施加固定权重，解决"类别频率"不平衡。
  缺点：无法区分同类别内的"容易"和"困难"样本。

Focal Loss：
  针对每个"样本"的预测置信度动态调整权重。
  容易正例（已学会）和容易负例（明显背景）都被抑制，
  训练专注于当前模型还没学好的"边缘"样本。

实际使用：
  - 极度不平衡（检测 1:1000）：Focal Loss 效果显著
  - 中等不平衡（分类 1:10）：class_weight 通常足够
  - 语义分割：Focal Loss 在小目标上有优势
  - DETR 等新框架：因匈牙利匹配保证 1:1，Focal Loss 效果有限
""")
