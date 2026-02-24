# optimizer_scheduler.py
# 优化器与学习率调度：AdamW、参数组、各类 Scheduler 及 Warmup
#
# 关注重点：
#   - 为什么 Adam + L2 != weight decay，AdamW 如何修复这个问题
#   - Bias / LayerNorm 参数为什么不加 weight decay
#   - Warmup 的必要性及 LambdaLR 的灵活实现
#   - 各调度器适用场景与调用时机（step-level vs epoch-level）

import torch
import torch.nn as nn
import math

# ========== 1. 优化器演进：SGD → Adam → AdamW ==========
print("=== 优化器演进 ===")
#
# SGD 的问题：所有维度共享同一学习率，面对病态曲率（ill-conditioned）的
# 损失面时收敛缓慢——曲率大的方向震荡，曲率小的方向爬行。
#
# Adam（Kingma & Ba, ICLR 2015）：
#   维护一阶矩 m（梯度指数平均）和二阶矩 v（梯度平方指数平均），
#   为每个参数自适应地计算独立步长。
#
#   m_t = β₁ m_{t-1} + (1-β₁) g_t         一阶矩（动量项）
#   v_t = β₂ v_{t-1} + (1-β₂) g_t²        二阶矩（自适应缩放）
#   m̂_t = m_t / (1-β₁ᵗ)                   偏差修正（早期 m₀=0 会低估）
#   v̂_t = v_t / (1-β₂ᵗ)
#   θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
#
# Adam + L2 ≠ weight decay 的根本原因：
#   若在损失上加 λ/2·‖θ‖²，梯度变为 g_t + λθ。
#   这个正则项会被 v_t 自适应缩放，对更新幅度大的参数实际正则效果更弱，
#   导致 L2 正则在 Adam 下行为异常（参数越活跃，正则越弱）。
#
# AdamW（Loshchilov & Hutter, ICLR 2019）：
#   将 weight decay 从梯度中剥离，直接施加在参数更新步上：
#   θ_t = θ_{t-1} - α · [m̂_t / (√v̂_t + ε) + λ · θ_{t-1}]
#   weight decay 不受自适应缩放影响，行为可预期。
#   几乎所有现代 Transformer 微调均使用 AdamW。

model = nn.Linear(10, 2)

# SGD with Nesterov momentum（视觉任务 backbone 训练的首选）
opt_sgd = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True       # 先沿动量方向预走一步再算梯度，收敛更快
)

# AdamW（Transformer 微调标配）
opt_adamw = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01   # 解耦的 weight decay，值通常 0.01~0.1
)

print(f"SGD   lr={opt_sgd.defaults['lr']},   wd={opt_sgd.defaults['weight_decay']}")
print(f"AdamW lr={opt_adamw.defaults['lr']}, wd={opt_adamw.defaults['weight_decay']}")
print()

# ========== 2. 参数组：差异化学习率与 weight decay ==========
print("=== 参数组（Parameter Groups）===")
#
# 两个常见需求：
#   ① Backbone 用小 lr，Head 用大 lr（迁移学习标准做法）
#   ② Bias 和 LayerNorm/BatchNorm 参数不施加 weight decay
#
# 为什么 LayerNorm 参数不加 weight decay？
#   LN 的 γ（scale）和 β（bias）是归一化后的仿射变换参数，
#   其绝对幅度没有过拟合意义，强行压缩会干扰归一化效果。
#   BERT、GPT 等预训练模型的复现实现均遵循这一约定。

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.fc1   = nn.Linear(32, 64)
        self.norm  = nn.LayerNorm(64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(self.embed(x))))

model2 = TinyTransformer()

# 区分 decay / no_decay 参数
decay_params    = []
no_decay_params = []

for name, param in model2.named_parameters():
    # LayerNorm 参数（weight/bias）和所有 bias 均不加 weight decay
    if 'bias' in name or 'norm' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay_params,    'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=1e-3)

print(f"decay 参数数量:    {len(decay_params)}")
print(f"no_decay 参数数量: {len(no_decay_params)}")
print()

# ========== 3. 学习率调度器 ==========
print("=== 学习率调度器 ===")

model3 = nn.Linear(10, 2)
opt3   = torch.optim.AdamW(model3.parameters(), lr=1e-3)

# ---- StepLR：每 step_size 个 epoch，lr *= gamma ----
# 简单直观，适合有明显阶段性的训练（如 ResNet ImageNet 训练）
sched_step = torch.optim.lr_scheduler.StepLR(opt3, step_size=30, gamma=0.1)

# ---- CosineAnnealingLR：按余弦曲线退火 ----
# η_t = η_min + 0.5(η_max - η_min)(1 + cos(π · t / T_max))
# 优点：末期 lr 平滑趋近 η_min，避免 StepLR 的突变；
#       配合 warm restart（SGDR）可跳出局部极小
sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt3, T_max=100, eta_min=1e-6
)

# ---- OneCycleLR：warmup + cosine annealing（Smith 2018 Super-Convergence）----
# 按 batch 调用（step-level），先升至 max_lr 再降，训练更快收敛
# pct_start=0.3 表示前 30% 步用于 warmup
sched_onecycle = torch.optim.lr_scheduler.OneCycleLR(
    opt3,
    max_lr      = 1e-2,
    total_steps = 100,
    pct_start   = 0.3,
    anneal_strategy = 'cos',
    div_factor      = 25.0,   # init_lr = max_lr / div_factor = 4e-4
    final_div_factor= 1e4     # final_lr = init_lr / final_div_factor
)

# 打印不同步数的 lr 变化
print("OneCycleLR lr 采样（步 0/10/30/50/99）:")
lrs = []
for step in range(100):
    lrs.append(opt3.param_groups[0]['lr'])
    opt3.step()
    sched_onecycle.step()

for i in [0, 10, 30, 50, 99]:
    print(f"  step={i:3d}: lr={lrs[i]:.2e}")
print()

# ========== 4. Linear Warmup + Cosine Decay（LambdaLR 实现）==========
print("=== Linear Warmup + Cosine Decay ===")
#
# BERT/GPT 等 Transformer 的标配调度策略：
#   前 warmup_steps 步 lr 从 0 线性升至 peak_lr，
#   之后按余弦曲线降至 0（或 peak 的某个小比例）。
#
# 为什么需要 warmup？
#   训练初期参数随机，各层梯度方差差异极大。
#   若一开始就用大 lr，Adam 的 v_t 估计不稳定（初始为 0，偏差修正不足），
#   步长可能非常大，导致模型在训练初期发散。
#   warmup 让优化器先积累足够的梯度统计再加速。

def get_linear_warmup_cosine_decay(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

model4 = nn.Linear(10, 2)
opt4   = torch.optim.AdamW(model4.parameters(), lr=1e-3)
sched4 = get_linear_warmup_cosine_decay(opt4, warmup_steps=10, total_steps=100)

lrs4 = []
for step in range(100):
    lrs4.append(opt4.param_groups[0]['lr'])
    opt4.step()
    sched4.step()

print(f"step  0（起始）: {lrs4[0]:.2e}")
print(f"step  5（warmup 中）: {lrs4[5]:.2e}")
print(f"step 10（warmup 结束，峰值）: {lrs4[10]:.2e}")
print(f"step 50（余弦中段）: {lrs4[50]:.2e}")
print(f"step 99（末尾）: {lrs4[99]:.2e}")
print()

# ========== 5. 完整训练循环示例 ==========
print("=== 完整训练循环（含调度器规范用法）===")
#
# 调用时机：
#   step-level scheduler（OneCycleLR / LambdaLR）：每个 batch 后调用
#   epoch-level scheduler（StepLR / CosineAnnealing）：每个 epoch 后调用
#   混淆调用时机会导致 lr 变化速率错误。

torch.manual_seed(42)

model5    = nn.Linear(10, 1)
criterion = nn.MSELoss()
opt5      = torch.optim.AdamW(model5.parameters(), lr=1e-3, weight_decay=0.01)
EPOCHS, STEPS = 5, 8
sched5 = get_linear_warmup_cosine_decay(opt5, warmup_steps=4, total_steps=EPOCHS * STEPS)

for epoch in range(EPOCHS):
    model5.train()
    epoch_loss = 0.0
    for _ in range(STEPS):
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        loss = criterion(model5(x), y)
        opt5.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model5.parameters(), max_norm=1.0)
        opt5.step()
        sched5.step()   # step-level：每 batch 调用
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}: "
          f"loss={epoch_loss/STEPS:.4f}, "
          f"lr={opt5.param_groups[0]['lr']:.2e}")

print()
print("=== 总结 ===")
print("""
1. AdamW = Adam + 解耦 weight decay，L2 正则不经过自适应缩放，行为可预期
2. Bias / Norm 参数设 weight_decay=0，避免对归一化仿射参数不必要的约束
3. step-level scheduler 每 batch 调用；epoch-level 每 epoch 调用，不可混淆
4. warmup 让 Adam 的二阶矩估计先稳定再放开步长，防止早期发散
5. 完整顺序：zero_grad → forward → loss → backward → clip → step → scheduler.step
""")
