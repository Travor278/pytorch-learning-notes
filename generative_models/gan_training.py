# gan_training.py
# 生成对抗网络（GAN）：原始 GAN 与训练技巧
#
# GAN（Goodfellow et al. 2014）通过生成器（G）与判别器（D）的对抗博弈
# 学习数据分布，无需显式建模密度函数。
#
# 参考：
#   - Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014
#   - Arjovsky et al., "Wasserstein GAN", ICML 2017（WGAN）
#   - Miyato et al., "Spectral Normalization for GANs", ICLR 2018

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. GAN 的极大极小博弈 ==========
print("=== GAN 的极大极小博弈 ===")
print("""
原始目标（minimax game）：
  min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

直观含义：
  判别器 D：最大化"真实样本判为真"和"生成样本判为假"的概率
  生成器 G：最小化"生成样本被判别器识别为假"的概率

理论最优：当 G 完美拟合 p_data 时，D(x) = 0.5（无法区分真假）
等价于最小化 Jensen-Shannon 散度：
  min_G JSD(p_data ‖ p_G) = V(D*, G) - log(4)

实践问题：
  早期训练时 D 很强，G 很弱，log(1-D(G(z))) 接近 0，梯度消失
  解决：非饱和损失（Non-saturating loss）= 最大化 log D(G(z))（等价目标，更强梯度）
""")

# ========== 2. 生成器与判别器网络 ==========
print("=== 生成器与判别器网络 ===")

latent_dim = 32
data_dim   = 64   # 模拟低维数据（图像扁平化等）

class Generator(nn.Module):
    """从噪声 z 生成数据 x"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()   # 输出归一化到 [-1, 1]（对应像素归一化后的范围）
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """判断输入样本是真实数据还是生成数据"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),    # LeakyReLU：负值时有 0.2 的梯度，避免死神经元
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)     # 输出 logit（不用 sigmoid，配合 BCEWithLogitsLoss）
        )

    def forward(self, x):
        return self.net(x)

G = Generator(latent_dim, data_dim)
D = Discriminator(data_dim)

params_G = sum(p.numel() for p in G.parameters())
params_D = sum(p.numel() for p in D.parameters())
print(f"Generator 参数量: {params_G:,}")
print(f"Discriminator 参数量: {params_D:,}")
print()

# ========== 3. GAN 训练循环 ==========
print("=== GAN 训练循环 ===")
#
# 标准 GAN 训练：每步先更新 D 若干次，再更新 G 一次
# 原因：D 需要比 G 更强才能提供有效的梯度信号
#
# 训练技巧：
#   ① 标签平滑（label smoothing）：真实标签用 0.9 而非 1，增加 D 的不确定性
#   ② 标签翻转（label flip）：偶尔翻转标签，防止 D 过度自信
#   ③ 独立优化器：G 和 D 用不同学习率（D 通常更小）

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))  # β₁=0.5（GAN 经验值）
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss()

def train_step(real_data):
    B = real_data.size(0)

    # ---- 更新判别器 ----
    # 真实样本：标签 1（用 0.9 做标签平滑）
    real_labels = torch.full((B, 1), 0.9)
    # 生成样本：标签 0
    fake_labels = torch.zeros(B, 1)

    # 生成假样本（detach：不传梯度给 G）
    z         = torch.randn(B, latent_dim)
    fake_data = G(z).detach()

    loss_D_real = criterion(D(real_data), real_labels)
    loss_D_fake = criterion(D(fake_data), fake_labels)
    loss_D      = (loss_D_real + loss_D_fake) / 2

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # ---- 更新生成器 ----
    # G 希望 D 把生成样本判为真（标签 1）
    z          = torch.randn(B, latent_dim)
    fake_data  = G(z)   # 不 detach，梯度需要流过 G
    real_labels_G = torch.ones(B, 1)  # G 期望 D 输出 1

    # 非饱和损失：max log D(G(z)) ↔ min -log D(G(z)) ↔ min BCE(D(G(z)), 1)
    loss_G = criterion(D(fake_data), real_labels_G)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    return loss_D.item(), loss_G.item()

torch.manual_seed(42)
# 模拟真实数据（混合高斯分布）
def sample_real_data(n):
    mix = torch.randint(0, 3, (n,))
    centers = torch.tensor([[-2.0], [0.0], [2.0]]).expand(3, data_dim // 2)
    x = torch.randn(n, data_dim) * 0.3
    for i in range(3):
        mask = (mix == i)
        x[mask, :data_dim//2] += centers[i]
    return x.tanh()

print("step  loss_D    loss_G    D(real)   D(fake)")
for step in range(8):
    real = sample_real_data(64)
    loss_d, loss_g = train_step(real)

    with torch.no_grad():
        z_test    = torch.randn(64, latent_dim)
        fake_test = G(z_test)
        d_real    = torch.sigmoid(D(real)).mean().item()
        d_fake    = torch.sigmoid(D(fake_test)).mean().item()

    print(f"  {step+1:3d}  {loss_d:.4f}    {loss_g:.4f}    {d_real:.3f}     {d_fake:.3f}")

print()

# ========== 4. 训练稳定性：WGAN 的思路 ==========
print("=== 训练稳定性问题与 WGAN ===")
print("""
原始 GAN 的主要不稳定因素：

1. 模式崩塌（Mode Collapse）：
   G 学会只生成少数几种样本（能骗过 D 的固定模式），
   忽略数据分布的其他模式。
   D 的梯度饱和时，G 缺乏多样性探索的动力。

2. 判别器过强：
   D 很快完美区分真假，log D(G(z)) 接近 0，G 的梯度消失。
   原文建议每更新 G 一次，D 更新 k 次（通常 k=1~5）。

3. JS 散度的不连续性（Arjovsky 2017 的分析）：
   当 p_data 和 p_G 支撑集不重叠时（高维中常见），JSD = log 2（常数），
   梯度为 0。GAN 在高维空间难以收敛的根本原因。

WGAN（Wasserstein GAN，Arjovsky 2017）的修复：
   用 Wasserstein-1 距离（Earth Mover's Distance）替代 JSD：
   W(p, q) = sup_{‖f‖_L ≤ 1} E_{p}[f(x)] - E_{q}[f(x)]
   其中 f 是 1-Lipschitz 函数（由判别器参数化）

   WGAN 的损失：
     D：max E_{x~real}[D(x)] - E_{x~fake}[D(x)]（线性，不是 log）
     G：min -E_{z}[D(G(z))]

   实现：去掉 D 最后的 sigmoid；权重裁剪（clipping）强制 1-Lipschitz
   改进：WGAN-GP（Gulrajani 2017）用梯度惩罚（gradient penalty）替代权重裁剪
""")

# WGAN-GP 梯度惩罚示意
def gradient_penalty(D, real, fake, device='cpu'):
    """
    WGAN-GP：在真实和生成样本的插值上惩罚判别器梯度的 L2 范数偏离 1
    GP = λ · E[(‖∇_x̂ D(x̂)‖₂ - 1)²],  x̂ = εx_real + (1-ε)x_fake
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    d_interp = D(interpolated)
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]

    grad_norm = grad.view(B, -1).norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp

real_sample = sample_real_data(16)
z_gp        = torch.randn(16, latent_dim)
fake_sample = G(z_gp).detach()

gp = gradient_penalty(D, real_sample, fake_sample)
print(f"WGAN-GP 梯度惩罚示例: {gp.item():.4f}")
print()

print("=== 总结 ===")
print("""
1. GAN = 生成器 G 与判别器 D 的极大极小博弈，等价于最小化 JSD
2. 非饱和损失（max log D(G(z))）比原始 min log(1-D(G(z))) 梯度更强
3. 训练技巧：G/D 交替更新、标签平滑（0.9）、LeakyReLU、β₁=0.5
4. 模式崩塌和梯度消失是 GAN 训练的两大核心问题
5. WGAN 用 Wasserstein 距离解决理论问题；WGAN-GP 用梯度惩罚改进 Lipschitz 约束
""")
