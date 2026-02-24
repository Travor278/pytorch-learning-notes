# kl_divergence.py
# KL 散度、温度缩放与知识蒸馏
#
# 这是理解 SEED 等知识蒸馏方法的核心数学基础。
# KL 散度衡量两个概率分布之间的"信息差"，在蒸馏中用于
# 将教师模型的"软标签"知识迁移到学生模型。
#
# 参考：
#   - Hinton et al., "Distilling the Knowledge in a Neural Network", arXiv 2015
#   - SEED: Li et al., "SEED-LLaVA: Knowledge Distillation...", arXiv 2024

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. KL 散度的定义 ==========
print("=== KL 散度定义 ===")
#
# KL(P‖Q) = Σ_x P(x) · log[P(x) / Q(x)]
#          = E_P[log P - log Q]
#          = H(P, Q) - H(P)     （交叉熵 - 信息熵）
#
# 性质：
#   - 非负：KL(P‖Q) ≥ 0，等号当且仅当 P = Q（Gibbs 不等式）
#   - 非对称：KL(P‖Q) ≠ KL(Q‖P)
#   - 不是距离度量（不满足三角不等式）
#
# 直觉：用分布 Q 近似 P 时，额外付出的 bits 数量。

P = torch.tensor([0.7, 0.2, 0.1])
Q = torch.tensor([0.3, 0.4, 0.3])

# 手动计算
kl_manual = (P * (P.log() - Q.log())).sum()
print(f"KL(P‖Q) 手动: {kl_manual.item():.4f}")

# PyTorch 实现注意：nn.KLDivLoss 的 input 期望的是 log 概率（log Q），target 是概率（P）
# KL(P‖Q) = Σ P · (log P - log Q) = Σ P · log Q 的负值 + 常数（H(P)）
# 在优化时 H(P) 是常数，所以最小化 KL 等价于最小化交叉熵
kl_loss = nn.KLDivLoss(reduction='sum')
kl_torch = kl_loss(Q.log(), P)
print(f"KL(P‖Q) PyTorch (reduction=sum): {kl_torch.item():.4f}  （应与手动一致）")

# batchmean 是推荐的 reduction 方式（除以 batch size，更稳定的梯度幅度）
kl_bm = nn.KLDivLoss(reduction='batchmean')
P_batch = P.unsqueeze(0)  # (1, 3)
Q_batch = Q.unsqueeze(0)
kl_bm_val = kl_bm(Q_batch.log(), P_batch)
print(f"KL(P‖Q) batchmean: {kl_bm_val.item():.4f}")
print()

# ========== 2. Forward KL vs Reverse KL ==========
print("=== Forward KL vs Reverse KL ===")
#
# Forward KL：KL(P_data ‖ Q_model) —— "均值寻找"（mean-seeking）
#   Q 必须覆盖 P 有概率的所有地方（零概率处 P·log∞ = ∞），
#   因此 Q 会扩散、平均，适合 MLE 训练（最大化 log 似然等价于最小化 forward KL）
#
# Reverse KL：KL(Q_model ‖ P_data) —— "模式寻找"（mode-seeking）
#   Q 只需匹配 P 的一个模式，在 P 低概率处 Q 也会趋近于 0，
#   导致 Q 集中在 P 的某个峰值附近（mode collapse 倾向）
#   变分推断（VAE）优化的是 KL(Q_posterior ‖ P_prior)
#
# 蒸馏中的选择：
#   SEED 使用 reverse KL（KL(Q_student ‖ P_teacher)），
#   强迫学生模型精确匹配教师的高置信度区域，
#   避免学生在教师不确定的位置"散开"。

print("Forward KL（均值寻找）：学生必须覆盖教师的所有峰值")
print("Reverse KL（模式寻找）：学生聚焦于教师最确定的预测")
print()

# ========== 3. 温度缩放（Temperature Scaling）==========
print("=== 温度缩放 ===")
#
# 原始 logit z = [2.0, 5.0, 1.0, 0.5]
# 标准 softmax：温度 T=1，输出是"硬"分布，接近 one-hot
# 软化版：将 logit 除以 T > 1，使分布更平滑，保留更多类间关系信息
#
# Hinton et al. 2015 的关键洞见：
#   模型对"错误"类别的软标签概率（如狗被预测为猫 p=0.01 vs 卡车 p=0.0001）
#   包含了类别间相似性信息（"暗知识"，dark knowledge），
#   用高温度提取这些细节，然后传授给学生模型。

def softmax_temperature(logits, T=1.0):
    return F.softmax(logits / T, dim=-1)

logits = torch.tensor([2.0, 5.0, 1.0, 0.5])

for T in [0.5, 1.0, 2.0, 5.0]:
    probs = softmax_temperature(logits, T)
    print(f"T={T:.1f}: {probs.numpy().round(3)}  (最大值={probs.max().item():.3f})")

print()
print("T→0：接近 argmax（one-hot），信息量最少")
print("T→∞：趋近于均匀分布，信息量最多但信号噪声也高")
print("蒸馏推荐 T=4~10：软化但仍保留教师的相对排序信息")
print()

# ========== 4. 知识蒸馏损失（Hinton 2015 版）==========
print("=== 知识蒸馏损失 ===")
#
# L_total = α · L_CE(student, hard_label) + (1-α) · T² · L_KD(student_T, teacher_T)
#
# 其中：
#   L_CE：学生对真实标签的交叉熵（保证任务准确率）
#   L_KD：学生软输出与教师软输出之间的 KL 散度（知识迁移）
#   T²  ：梯度缩放因子——用 T 缩放 logit 后梯度幅度减小了 1/T²，
#          乘以 T² 恢复到与 T=1 时相当的梯度幅度，使 α 的含义不随 T 变化

class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5):
        """
        T    : 蒸馏温度（软化 logit 的温度）
        alpha: 硬标签损失的权重（1-alpha 为蒸馏损失权重）
        """
        super().__init__()
        self.T     = T
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss()
        self.kl    = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # 硬标签损失（T=1，使用真实类别）
        loss_ce = self.ce(student_logits, labels)

        # 软标签蒸馏损失（T > 1，传递教师的类间关系）
        # student 用 log_softmax（KLDivLoss 期望 log 概率）
        # teacher 用 softmax（KLDivLoss 期望概率）
        s_soft = F.log_softmax(student_logits / self.T, dim=-1)
        t_soft = F.softmax(teacher_logits / self.T, dim=-1)

        loss_kd = self.kl(s_soft, t_soft)

        # T² 补偿梯度缩放（Hinton 2015 公式推导）
        return self.alpha * loss_ce + (1 - self.alpha) * (self.T ** 2) * loss_kd

# 模拟一个 batch
B, C = 16, 10
torch.manual_seed(42)
student_logits  = torch.randn(B, C)
teacher_logits  = torch.randn(B, C) * 2  # 教师通常更自信（幅度更大）
labels          = torch.randint(0, C, (B,))

distill = DistillationLoss(T=4.0, alpha=0.5)
loss    = distill(student_logits, teacher_logits, labels)
print(f"蒸馏总损失: {loss.item():.4f}")

# 单独查看各损失分量
ce_only = nn.CrossEntropyLoss()(student_logits, labels)
kl_only = nn.KLDivLoss(reduction='batchmean')(
    F.log_softmax(student_logits / 4.0, dim=-1),
    F.softmax(teacher_logits / 4.0, dim=-1)
)
print(f"  CE 分量: {ce_only.item():.4f}")
print(f"  KL 分量（×T²）: {(4.0**2 * kl_only).item():.4f}")
print()

# ========== 5. Forward vs Reverse KL 的实验对比 ==========
print("=== Forward vs Reverse KL 优化行为对比 ===")
#
# 用一个简单的双峰分布演示两种 KL 的优化差异

# 教师分布：双峰（类 0 和类 2 都有较高概率）
teacher_bimodal = torch.tensor([[0.45, 0.05, 0.45, 0.05]])  # shape (1, 4)

# 学生初始化：均匀分布
student_logits_fw = torch.zeros(1, 4, requires_grad=True)
student_logits_rv = torch.zeros(1, 4, requires_grad=True)

opt_fw = torch.optim.Adam([student_logits_fw], lr=0.1)
opt_rv = torch.optim.Adam([student_logits_rv], lr=0.1)

kl_div = nn.KLDivLoss(reduction='batchmean')

for _ in range(200):
    # Forward KL: min KL(teacher || student)
    opt_fw.zero_grad()
    s_fw   = F.log_softmax(student_logits_fw, dim=-1)
    # KL(teacher || student) = sum teacher * (log teacher - log student)
    loss_fw = (teacher_bimodal * (teacher_bimodal.log() - s_fw)).sum()
    loss_fw.backward()
    opt_fw.step()

    # Reverse KL: min KL(student || teacher)
    opt_rv.zero_grad()
    s_rv      = F.softmax(student_logits_rv, dim=-1)
    s_rv_log  = F.log_softmax(student_logits_rv, dim=-1)
    loss_rv   = kl_div(s_rv_log, teacher_bimodal)
    loss_rv.backward()
    opt_rv.step()

print("教师分布（双峰）:", teacher_bimodal.squeeze().numpy().round(3).tolist())
print("Forward KL 拟合：", F.softmax(student_logits_fw, dim=-1).detach().squeeze().numpy().round(3).tolist(),
      "← 覆盖两个峰")
print("Reverse KL 拟合：", F.softmax(student_logits_rv, dim=-1).detach().squeeze().numpy().round(3).tolist(),
      "← 集中于一个峰（mode-seeking）")
print()

print("=== 总结 ===")
print("""
1. KL(P‖Q) = H(P,Q) - H(P)，非对称非负，最小化等价于最大化对数似然
2. Forward KL：均值寻找，学生覆盖教师所有模式；Reverse KL：模式寻找，聚焦主峰
3. 温度 T 软化 logit，提取类间暗知识；T² 因子补偿梯度幅度缩减
4. 蒸馏损失 = α·CE(硬标签) + (1-α)·T²·KL(软标签)
5. SEED 使用 reverse KL 让学生精确匹配教师的高置信预测，抑制幻觉传播
""")
