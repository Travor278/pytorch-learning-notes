# label_smoothing.py
# 标签平滑（Label Smoothing）：正则化与校准
#
# 来源：Szegedy et al., "Rethinking the Inception Architecture...", CVPR 2016
# 后续理论分析：Müller et al., "When Does Label Smoothing Help?", NeurIPS 2019
#
# 标签平滑是一种简单有效的正则化手段，被 BERT、ViT 等大模型广泛采用。
# 它通过软化 one-hot 标签防止模型过度自信（logit 无限增大），同时改善校准度。

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. 硬标签（Hard Label）的问题 ==========
print("=== 硬标签的问题 ===")
#
# 标准 one-hot 标签驱动模型将正确类的 logit 推向 +∞，其余推向 -∞。
# 极端的 logit 意味着模型"过于自信"：
#   ① 过拟合：在训练集上完全确定，泛化能力下降
#   ② 校准差（Calibration）：softmax 输出概率远高于实际准确率
#   ③ 特征退化（Müller 2019）：最后一层 embedding 趋向聚类，
#      类间距离不再反映语义相似性

# 模拟：固定 logit，观察 CE 损失对越来越自信预测的行为
target = torch.tensor([0])  # 真实类别为 0
for scale in [1.0, 2.0, 5.0, 10.0]:
    logit = torch.tensor([[scale, -scale, -scale]])  # 越来越确信 class 0
    loss  = F.cross_entropy(logit, target)
    prob  = F.softmax(logit, dim=-1)[0, 0].item()
    print(f"  logit={scale:.1f}: 预测概率={prob:.4f}, CE={loss.item():.4f}")

print("→ 标准 CE 驱动 logit 无限增大（损失单调递减）")
print()

# ========== 2. 标签平滑的原理 ==========
print("=== 标签平滑原理 ===")
#
# 将 one-hot 硬标签替换为"软"分布：
#   y_smooth[k] = (1 - ε) · y_hard[k] + ε / C
#
# 对正确类 k*：    y_smooth[k*] = 1 - ε + ε/C  ≈ 1 - ε(C-1)/C
# 对其他类 k≠k*： y_smooth[k]  = ε / C
#
# 等价形式：
#   L_LS = (1 - ε) · CE(y_hard) + ε · CE(uniform)
#        = CE(y_hard) - ε · [CE(y_hard) - H(uniform)]
#        = CE(y_hard) + ε · KL(uniform ‖ model) - const
#
# 直觉：标签平滑 = 标准 CE + 一个"迫使分布接近均匀"的 KL 惩罚项。
# 这个惩罚项防止模型为了某个固定的真实标签把 logit 推到无穷大。

C   = 10
eps = 0.1
one_hot = torch.zeros(C); one_hot[0] = 1.0
smooth  = (1 - eps) * one_hot + eps / C

print(f"硬标签 (C=10):         {one_hot.numpy().round(2).tolist()}")
print(f"平滑标签 (ε={eps}): {smooth.numpy().round(3).tolist()}")
print(f"平滑后最大值: {smooth.max().item():.3f}  （最小值: {smooth.min().item():.3f}）")
print()

# ========== 3. PyTorch 内置实现 ==========
print("=== PyTorch 内置 Label Smoothing ===")
#
# PyTorch 1.10 起，CrossEntropyLoss 直接支持 label_smoothing 参数。
# 内部等价于：L_LS = (1-ε)·CE + ε·（-log_softmax 所有类的均值）

N, C = 32, 10
torch.manual_seed(42)
logits  = torch.randn(N, C)
targets = torch.randint(0, C, (N,))

# 标准 CE
loss_hard = F.cross_entropy(logits, targets)

# Label Smoothing（ε=0.1，常见设置）
loss_ls_01 = F.cross_entropy(logits, targets, label_smoothing=0.1)
loss_ls_02 = F.cross_entropy(logits, targets, label_smoothing=0.2)

print(f"CE (ε=0):   {loss_hard.item():.4f}")
print(f"CE (ε=0.1): {loss_ls_01.item():.4f}  ← 通常略高于标准 CE（正则效果）")
print(f"CE (ε=0.2): {loss_ls_02.item():.4f}")
print()

# ========== 4. 手动实现验证 ==========
print("=== 手动实现验证 ===")

def label_smoothing_ce_manual(logits, targets, eps=0.1):
    C        = logits.size(-1)
    log_prob = F.log_softmax(logits, dim=-1)                    # (N, C)

    # 硬标签部分：取目标类的 log_prob
    loss_hard = -log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)   # (N,)

    # 均匀分布部分：所有类 log_prob 的均值（负号）
    loss_uni  = -log_prob.mean(dim=-1)                          # (N,)

    loss = (1 - eps) * loss_hard + eps * loss_uni
    return loss.mean()

loss_manual = label_smoothing_ce_manual(logits, targets, eps=0.1)
print(f"手动实现 (ε=0.1):     {loss_manual.item():.6f}")
print(f"PyTorch 内置 (ε=0.1): {loss_ls_01.item():.6f}")
print(f"数值一致: {torch.isclose(loss_manual, loss_ls_01, atol=1e-5).item()}")
print()

# ========== 5. 对模型置信度的影响 ==========
print("=== 对模型置信度的影响 ===")
#
# 标签平滑改善模型校准（Calibration）：
#   校准差的模型：预测概率 0.95 时实际准确率只有 0.7（过度自信）
#   标签平滑后：模型对自己不确定的样本预测概率更接近实际准确率
#
# 但有一个权衡（Müller 2019）：
#   标签平滑提高校准度 + 测试准确率，但损害知识蒸馏！
#   平滑后的教师模型，其 logit 的类间差异被压缩，
#   学生模型蒸馏时获得的暗知识（dark knowledge）反而减少。
#   因此，蒸馏场景下建议教师模型不使用标签平滑（或使用较小的 ε）。

# 训练一个小模型对比有无标签平滑的置信度分布
class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
    def forward(self, x):
        return self.net(x)

torch.manual_seed(42)
# 生成不平衡的三分类玩具数据
X = torch.randn(200, 4); Y = torch.randint(0, 3, (200,))

def train_small(label_smooth=0.0):
    m   = TinyClassifier()
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(300):
        loss = F.cross_entropy(m(X), Y, label_smoothing=label_smooth)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        probs = F.softmax(m(X), dim=-1)
        max_conf = probs.max(dim=-1).values.mean().item()
    return max_conf

conf_hard   = train_small(0.0)
conf_smooth = train_small(0.1)

print(f"无标签平滑 — 平均最大置信度: {conf_hard:.4f}   ← 过度自信")
print(f"标签平滑ε=0.1 — 平均最大置信度: {conf_smooth:.4f} ← 校准更好")
print()

print("=== 总结 ===")
print("""
1. 标签平滑将 one-hot 软化：y_smooth = (1-ε)·y_hard + ε/C
2. 等价于在 CE 上加均匀分布 KL 惩罚，防止 logit 无限增大
3. PyTorch: F.cross_entropy(logits, targets, label_smoothing=0.1)
4. ε=0.1 是主流默认值（BERT、ViT、Inception 等均使用）
5. 注意：标签平滑损害知识蒸馏效果（Müller 2019），蒸馏教师模型慎用
""")
