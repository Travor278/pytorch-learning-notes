# diffusion_basics.py
# 扩散模型（DDPM）：前向过程、反向过程与噪声预测
#
# 扩散模型从热力学中"扩散"（diffusion）过程得名：
#   前向：向数据逐步加噪，最终变为纯高斯噪声
#   反向：训练神经网络学习去噪，从噪声中还原数据
# 数学上是马尔可夫链上的变分推断问题（与 VAE 有深刻联系）。
#
# 参考：Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020（DDPM）

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== 1. 前向过程（加噪）==========
print("=== 前向过程（Forward Process）===")
#
# 前向过程定义为马尔可夫链：
#   q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
#
# 即每步以 β_t 的比例加入高斯噪声，同时将原始信号缩放 √(1-β_t)。
#
# 关键性质（DDPM 的核心推导）：
#   令 ᾱ_t = Π_{s=1}^{t} (1-β_s)（累积乘积），则：
#   q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) I)
#
# 这意味着任意时间步 t 的加噪样本可以直接从 x_0 一步计算：
#   x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0, I)
# 不需要逐步迭代，训练非常高效（每步只需前向一次）。

class LinearNoiseSchedule:
    """
    线性噪声调度（DDPM 原文）：β 从 β_start 到 β_end 线性增大
    这意味着早期加噪少（保留更多信号），晚期加噪多
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T      = T
        self.betas  = torch.linspace(beta_start, beta_end, T)     # β_t
        self.alphas = 1.0 - self.betas                             # α_t = 1 - β_t
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)        # ᾱ_t = Π α_s

    def q_sample(self, x0, t):
        """
        一步加噪：x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
        x0: (B, ...) 原始数据
        t : (B,) 时间步索引（整数）
        返回 (x_t, ε)
        """
        alpha_bar = self.alpha_bars[t].float()

        # 形状广播：alpha_bar (B,) 需要与 x0 的额外维度对齐
        while alpha_bar.ndim < x0.ndim:
            alpha_bar = alpha_bar.unsqueeze(-1)

        eps    = torch.randn_like(x0)
        x_t    = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
        return x_t, eps

schedule = LinearNoiseSchedule(T=1000)
print(f"β_1={schedule.betas[0]:.5f}, β_500={schedule.betas[499]:.5f}, β_T={schedule.betas[-1]:.5f}")
print(f"ᾱ_1={schedule.alpha_bars[0]:.4f}  （几乎是原始信号）")
print(f"ᾱ_500={schedule.alpha_bars[499]:.4f}  （信号和噪声各半）")
print(f"ᾱ_T={schedule.alpha_bars[-1]:.4f}  （接近纯噪声）")

# 可视化加噪过程
B, C, H = 2, 1, 16  # (batch, channel, height) 模拟一维信号
x0 = torch.randn(B, C, H)
for t_val in [0, 100, 500, 999]:
    t = torch.full((B,), t_val, dtype=torch.long)
    xt, eps = schedule.q_sample(x0, t)
    signal_ratio = schedule.alpha_bars[t_val].item()
    print(f"  t={t_val:4d}: ᾱ={signal_ratio:.4f}，x_t std={xt.std().item():.3f}")
print()

# ========== 2. 反向过程（去噪）==========
print("=== 反向过程（Reverse Process）===")
#
# 目标：学习真实的反向分布 p(x_{t-1} | x_t)（从噪声一步步去噪）
# 由于真实反向分布不可知，DDPM 用神经网络 ε_θ(x_t, t) 参数化：
#
# DDPM 的关键简化（Ho et al. 2020 的推导）：
#   可以证明，最优的 L_simple = E_{x0,ε,t}[‖ε - ε_θ(x_t, t)‖²]
#   即：训练网络预测前向过程中加入的噪声 ε，而不是直接预测 x_{t-1}
#
# 为什么预测 ε（噪声）比预测 x_0 更好？
#   预测 ε 在所有时间步的 scale 一致（ε ~ N(0,I) 始终方差为 1），
#   而预测 x_0 在 t 大时信噪比极低，导致目标方差非常大，训练困难。
#   当然，预测 ε_θ 后可以直接推算出 x_0 的估计。

class SimpleUNet(nn.Module):
    """
    简化的 U-Net（实际 DDPM 使用更复杂的 U-Net with attention）
    这里用全连接层演示核心接口：输入 (x_t, t)，输出预测的噪声 ε_θ
    """
    def __init__(self, data_dim=64, time_emb_dim=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, data_dim)
        )
        self.time_emb_dim = time_emb_dim

    def sinusoidal_time_emb(self, t):
        """时间步嵌入（类比 Transformer 位置编码）"""
        d = self.time_emb_dim
        half = d // 2
        freqs  = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb    = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb

    def forward(self, x_t, t):
        """x_t: (B, data_dim), t: (B,) 时间步整数"""
        t_emb = self.time_embed(self.sinusoidal_time_emb(t))  # (B, time_emb_dim)
        inp   = torch.cat([x_t, t_emb], dim=-1)               # (B, data_dim + time_emb_dim)
        return self.net(inp)

# ========== 3. 训练过程 ==========
print("=== DDPM 训练循环 ===")

data_dim = 64
model    = SimpleUNet(data_dim=data_dim)
optimizer= torch.optim.Adam(model.parameters(), lr=1e-3)

T = 1000
# 模拟若干训练步
for step in range(5):
    # 真实数据（这里用随机数模拟）
    x0 = torch.randn(16, data_dim)

    # 随机采样时间步 t ~ U[1, T]
    t = torch.randint(1, T, (16,))

    # 前向加噪：得到 x_t 和真实噪声 ε
    x_t, eps_true = schedule.q_sample(x0.unsqueeze(1).unsqueeze(1), t)
    x_t   = x_t.squeeze(1).squeeze(1)  # 恢复 (B, data_dim)

    # 预测噪声
    eps_pred = model(x_t, t)

    # 简单 MSE 损失（L_simple，Ho et al. 2020）
    loss = F.mse_loss(eps_pred, eps_true.squeeze(1).squeeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  step={step+1}: loss={loss.item():.4f}")
print()

# ========== 4. DDPM 采样（推理）==========
print("=== DDPM 采样（Ancestral Sampling）===")
#
# 反向采样：从 x_T ~ N(0,I) 出发，逐步去噪 T → T-1 → ... → 0
# 每步去噪公式（DDPM 推导）：
#   x_{t-1} = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
#   其中 z ~ N(0,I)（t>0 时），σ_t = √β_t（DDPM 取）

@torch.no_grad()
def ddpm_sample(model, schedule, n_samples, data_dim, n_steps=20):
    """简化的 DDPM 采样（只运行 n_steps 步，非完整 1000 步）"""
    model.eval()
    x = torch.randn(n_samples, data_dim)  # x_T ~ N(0, I)

    # 逆序时间步（T → 0）
    timesteps = torch.linspace(schedule.T - 1, 0, n_steps).long()

    for i, t_val in enumerate(timesteps):
        t = torch.full((n_samples,), t_val, dtype=torch.long)

        # 预测噪声
        eps_pred = model(x, t)

        alpha     = schedule.alphas[t_val]
        alpha_bar = schedule.alpha_bars[t_val]
        beta      = schedule.betas[t_val]

        # 去噪一步（DDPM 公式）
        coeff = beta / (1 - alpha_bar).sqrt()
        x     = (1.0 / alpha.sqrt()) * (x - coeff * eps_pred)

        # 添加噪声（最后一步除外）
        if i < len(timesteps) - 1:
            x = x + beta.sqrt() * torch.randn_like(x)

    return x

samples = ddpm_sample(model, schedule, n_samples=4, data_dim=data_dim, n_steps=20)
print(f"生成样本 shape: {samples.shape}")
print(f"样本均值: {samples.mean().item():.3f}  （理想情况接近真实数据分布均值 0）")
print(f"样本标准差: {samples.std().item():.3f}")
print()

print("=== 总结 ===")
print("""
1. 前向：q(x_t|x_0) = N(√ᾱ_t·x_0, (1-ᾱ_t)I)，任意 t 步加噪可一步计算
2. 训练目标：最小化 E[‖ε - ε_θ(x_t, t)‖²]（预测加入的高斯噪声）
3. 时间步 t 以正弦位置编码注入模型，使网络区分不同噪声级别
4. 采样：从 N(0,I) 出发，逐步调用 ε_θ 去噪，共 T 步（通常 T=1000）
5. DDIM（Song 2020）：确定性采样，仅需 50~100 步即可生成高质量图像
""")
