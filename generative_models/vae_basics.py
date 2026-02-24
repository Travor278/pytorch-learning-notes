# vae_basics.py
# 变分自编码器（VAE）：ELBO 推导与 Reparameterization Trick
#
# VAE 是生成模型的基础范式，也是理解扩散模型（DDPM）的前置知识。
# 核心思想：将生成建模转化为变分推断问题，通过可微的方式优化下界（ELBO）。
#
# 参考：Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. VAE 的生成模型假设 ==========
print("=== VAE 的生成模型假设 ===")
print("""
生成过程（从隐变量 z 生成数据 x）：
  z ~ p(z) = N(0, I)              隐变量的先验（各向同性高斯）
  x ~ p_θ(x|z) = 解码器(z)       给定 z，解码器生成 x 的分布

目标：学习 θ 使得 p_θ(x) = ∫ p_θ(x|z) p(z) dz 最大化（最大似然）
问题：积分对高维 z 无法直接计算（维数灾难）

解决方案（变分推断）：
  引入近似后验 q_φ(z|x) = 编码器(x)（用 N(μ(x), σ²(x)) 近似真实后验 p(z|x)）
  最大化对数似然的下界（ELBO）：

  log p_θ(x) ≥ ELBO = E_{q_φ(z|x)}[log p_θ(x|z)] - KL(q_φ(z|x) ‖ p(z))
                       ↑重建项                          ↑正则项
  重建项：编码 x 得到 z，再解码 z 重建 x，最大化重建质量
  KL 项：强迫后验 q_φ(z|x) 接近先验 N(0,I)，避免 z 空间塌缩

KL(N(μ,σ²) ‖ N(0,I)) 的解析解（高斯时可以直接计算，不需要采样）：
  KL = 0.5 · Σ_i (μ_i² + σ_i² - 1 - log σ_i²)
""")

# ========== 2. Reparameterization Trick ==========
print("=== Reparameterization Trick ===")
#
# 问题：ELBO 中 E_{q_φ(z|x)}[log p_θ(x|z)] 涉及对 z 的期望，
#       z 的采样过程（z ~ N(μ, σ²)）不可微，梯度无法流过采样操作。
#
# 解决方案：将随机性从参数中分离出来：
#   z = μ(x) + σ(x) ⊙ ε,    ε ~ N(0, I)
#
# 现在 z 是 (μ, σ) 的确定性函数，加上与参数无关的随机噪声 ε。
# 梯度可以流过 μ 和 σ，而 ε 的随机性不影响梯度计算。
# 这个技巧是 VAE 训练的关键。

def reparameterize(mu, log_var):
    """
    mu, log_var: (B, latent_dim)
    log_var 代替 log(σ²) 以数值稳定（exp 可能上溢）
    返回 z = μ + σ·ε
    """
    std = torch.exp(0.5 * log_var)  # σ = exp(log σ) = exp(0.5 · log σ²)
    eps = torch.randn_like(std)     # ε ~ N(0, I)，形状与 std 相同
    return mu + std * eps

B, latent_dim = 8, 16
mu      = torch.randn(B, latent_dim)
log_var = torch.zeros(B, latent_dim)   # σ=1，此时 q=p=N(0,I)

z = reparameterize(mu, log_var)
print(f"μ shape: {mu.shape},  z shape: {z.shape}")
print(f"重参数化使 z 是 (μ, log_var) 的可微函数，梯度可以回传到编码器")
print()

# ========== 3. KL 散度的解析计算 ==========
print("=== KL 散度（高斯对高斯）解析解 ===")

def kl_divergence_gaussian(mu, log_var):
    """
    KL(N(μ,σ²) ‖ N(0,I)) 的解析解
    = 0.5 · Σ (μ² + σ² - 1 - log σ²)
    = 0.5 · Σ (μ² + exp(log_var) - 1 - log_var)
    """
    # 对每个 latent 维求和（dim=1），然后对 batch 求平均
    kl = 0.5 * (mu.pow(2) + log_var.exp() - 1 - log_var).sum(dim=1)
    return kl.mean()   # batch 均值

kl = kl_divergence_gaussian(mu, log_var)
print(f"KL 散度（μ随机, log_var=0）: {kl.item():.4f}  （σ=1 时 KL = 0.5·Σμ²）")

# 验证：当 μ=0, σ=1 时 KL = 0
mu_zero = torch.zeros(B, latent_dim)
kl_zero = kl_divergence_gaussian(mu_zero, log_var)
print(f"KL 散度（μ=0, σ=1）: {kl_zero.item():.6f}  （期望为 0）")
print()

# ========== 4. 完整 VAE 模型 ==========
print("=== 完整 VAE（MNIST 风格）===")

class Encoder(nn.Module):
    """将输入 x 编码为高斯分布的参数 (μ, log_var)"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net     = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h       = self.net(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

class Decoder(nn.Module):
    """从隐变量 z 解码为 x 的重建"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()   # 输出范围 [0,1]（对应像素归一化后的值）
        )

    def forward(self, z):
        return self.net(z)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=16):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = reparameterize(mu, log_var)
        x_recon     = self.decoder(z)
        return x_recon, mu, log_var

    def elbo_loss(self, x, x_recon, mu, log_var, beta=1.0):
        """
        ELBO = 重建损失 + β·KL 散度（取负值，因为我们最小化损失）
        重建损失：BCE（像素级二值交叉熵），对应 p_θ(x|z) = Bernoulli
        β-VAE：β > 1 时更强的解耦压力，学到更独立的潜变量维度（Higgins 2017）
        """
        # 重建损失（BCE 对每个像素计算，然后求和）
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)

        # KL 散度（解析解）
        kl_loss = kl_divergence_gaussian(mu, log_var)

        return recon_loss + beta * kl_loss, recon_loss, kl_loss

# 模拟训练一个 batch
torch.manual_seed(42)
vae = VAE(input_dim=784, hidden_dim=256, latent_dim=16)
optimizer = torch.optim.Adam(vae.parameters(), lr=3e-4)

x_batch = torch.rand(32, 784)  # 模拟 MNIST（像素值 0~1）

for step in range(5):
    x_recon, mu, log_var = vae(x_batch)
    total_loss, recon, kl = vae.elbo_loss(x_batch, x_recon, mu, log_var, beta=1.0)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f"  step={step+1}: total={total_loss.item():.2f}, "
          f"recon={recon.item():.2f}, KL={kl.item():.4f}")

print()

# ========== 5. 采样生成 ==========
print("=== 从 VAE 采样生成新图像 ===")
vae.eval()
with torch.no_grad():
    # 直接从先验 p(z)=N(0,I) 采样，然后解码
    z_sample = torch.randn(4, 16)  # latent_dim=16
    x_gen    = vae.decoder(z_sample)
    print(f"生成样本 shape: {x_gen.shape}  （4 张 784 维的虚拟图像）")
    print(f"像素值范围: [{x_gen.min().item():.3f}, {x_gen.max().item():.3f}]")

print()
print("=== 总结 ===")
print("""
1. ELBO = E[log p(x|z)] - KL(q(z|x) ‖ p(z))：重建质量 - 后验与先验的差距
2. Reparameterization: z = μ + σ·ε，将随机性与参数分离，使梯度可流过采样
3. KL 高斯解析解: 0.5·Σ(μ² + exp(log_var) - 1 - log_var)，不需要蒙特卡洛估计
4. β-VAE (β>1)：更强 KL 约束，学到更解耦的隐变量空间，代价是重建质量略降
5. 生成：从 N(0,I) 采样 z，通过解码器生成新样本（无需真实图像输入）
""")
