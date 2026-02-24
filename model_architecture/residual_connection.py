# residual_connection.py
# 残差连接：从 ResNet 到 Transformer 的 Pre-LN vs Post-LN
#
# 残差连接（Residual Connection / Skip Connection）是深层网络训练的关键。
# 它解决了梯度消失问题，使得训练 100+ 层的网络成为可能。
#
# 参考：
#   - He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
#   - Wang et al., "Learning Deep Transformer Models for Machine Translation"
#     ACL 2019（Pre-LN 的系统分析）

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. 残差连接的梯度分析 ==========
print("=== 残差连接的梯度分析 ===")
#
# 普通深层网络（无残差）：
#   y = f_L(f_{L-1}(...f_1(x)...))
#   ∂L/∂x = ∂L/∂y · Π_{i=1}^{L} J_i    ← 所有 Jacobian 连乘
#   若每层 Jacobian 谱范数 < 1，乘积指数衰减 → 梯度消失
#   若每层 Jacobian 谱范数 > 1，乘积指数增长 → 梯度爆炸
#
# 残差网络：
#   y = x + f(x)
#   ∂L/∂x = ∂L/∂y · (1 + ∂f/∂x)
#
#   关键：梯度路径中始终存在"1"这个分量（恒等映射的贡献），
#   保证即使 ∂f/∂x 很小，梯度也能无衰减地回传到早期层。
#   这是 ResNet 能训练 1000 层的根本原因。

# 演示：50 层无残差 vs 有残差网络的梯度范数
def check_grad_flow(use_residual=False, n_layers=50, dim=64):
    layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
    x = torch.randn(4, dim, requires_grad=True)

    h = x
    for layer in layers:
        out = F.relu(layer(h))
        h   = out + h if use_residual else out

    loss = h.sum()
    loss.backward()
    return x.grad.abs().mean().item()

grad_plain = check_grad_flow(use_residual=False)
grad_resid = check_grad_flow(use_residual=True)

print(f"50 层无残差网络 — 输入梯度 L1 均值: {grad_plain:.6f}  ← 接近 0（梯度消失）")
print(f"50 层残差网络   — 输入梯度 L1 均值: {grad_resid:.6f}  ← 正常量级")
print()

# ========== 2. ResNet 基本残差块 ==========
print("=== ResNet 残差块 ===")
#
# ResNet 的 BasicBlock（用于 ResNet-18/34）：
#   out = F(x, {W_i}) + x
#   F(x) = BN → Conv → ReLU → BN → Conv
#
# Bottleneck Block（用于 ResNet-50/101/152）：
#   1×1 Conv（降维）→ 3×3 Conv → 1×1 Conv（升维）
#   参数量减少但感受野相同
#
# 维度不匹配时的处理：当输入/输出通道数不同或 stride≠1 时，
# shortcut 需要用 1×1 Conv 投影（projection shortcut）

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 当 stride≠1 或通道数变化时，shortcut 需投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))   # 残差相加后再 ReLU
        return out

# 测试
blk = BasicBlock(64, 128, stride=2)
x_test = torch.randn(2, 64, 32, 32)
out_blk = blk(x_test)
print(f"BasicBlock: ({x_test.shape}) → ({out_blk.shape})")
print(f"  stride=2, 通道 64→128，shortcut 用 1×1 Conv 投影")
print()

# ========== 3. Post-LN vs Pre-LN ==========
print("=== Post-LN vs Pre-LN ===")
#
# 原始 Transformer（Vaswani 2017）使用 Post-LN：
#   out = LayerNorm(x + Sublayer(x))
#
# Post-LN 训练不稳定：
#   早期 x 是随机初始化，Sublayer(x) 也是噪声，
#   LayerNorm 在两者之和上归一化，梯度流经 LN 容易不稳定。
#   通常需要复杂的 warmup 才能收敛（尤其是深层 Transformer）。
#
# Pre-LN（BERT 2 / GPT-2 实际实现）：
#   out = x + Sublayer(LayerNorm(x))
#
# Pre-LN 更稳定：
#   残差路径上（x 的分支）始终没有 LayerNorm，
#   梯度可以无阻碍地流回早期层，不需要精细的 warmup。
#   代价是模型最后一层需要额外加一个 LN（final LN）。
#   现代大模型（LLaMA / GPT-4 / LLaVA）几乎全用 Pre-LN。

class PostLNBlock(nn.Module):
    """原始 Transformer 风格：Sublayer → Add → LN"""
    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()
        self.ffn  = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.drop(self.ffn(x)))   # Post-LN

class PreLNBlock(nn.Module):
    """现代 Transformer 风格：LN → Sublayer → Add"""
    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()
        self.ffn  = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.ffn(self.norm(x)))   # Pre-LN

d_model, ffn_dim = 256, 1024
post_ln = PostLNBlock(d_model, ffn_dim)
pre_ln  = PreLNBlock(d_model, ffn_dim)

x_in = torch.randn(4, 16, d_model, requires_grad=True)

# 梯度流对比（简化：10 层叠加）
def stack_blocks(block_class, n=10, **kwargs):
    blocks = nn.ModuleList([block_class(**kwargs) for _ in range(n)])
    x = torch.randn(4, 16, d_model, requires_grad=True)
    for b in blocks:
        x = b(x)
    x.sum().backward()
    return x.grad

post_grad = stack_blocks(PostLNBlock, 10, d_model=d_model, ffn_dim=ffn_dim)
pre_grad  = stack_blocks(PreLNBlock,  10, d_model=d_model, ffn_dim=ffn_dim)

print(f"Post-LN 10 层梯度范数: {post_grad.norm().item():.4f}")
print(f"Pre-LN  10 层梯度范数: {pre_grad.norm().item():.4f}")
print("Pre-LN 梯度更稳定（不依赖精细 warmup 即可训练）")
print()

# ========== 4. RMSNorm（LLaMA 的选择）==========
print("=== RMSNorm vs LayerNorm ===")
#
# LayerNorm：y = (x - μ) / √(σ² + ε) * γ + β
#   需要计算均值 μ（中心化）和方差 σ²
#
# RMSNorm（Zhang & Sennrich 2019）：
#   y = x / √(mean(x²) + ε) * γ
#   去掉中心化步骤（无减均值），去掉偏置 β，
#   理论上只需均方根归一化，计算更快，效果与 LayerNorm 相当。
#   LLaMA、Mistral、Qwen 等开源大模型均使用 RMSNorm。

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))   # γ，初始化为 1
        self.eps    = eps

    def forward(self, x):
        # x: (B, S, d_model)
        rms  = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x    = x.float() / rms
        return (self.weight * x).to(x.dtype)

rms_norm  = RMSNorm(d_model)
layer_norm = nn.LayerNorm(d_model)

x_norm_test = torch.randn(4, 16, d_model)
out_rms = rms_norm(x_norm_test)
out_ln  = layer_norm(x_norm_test)

print(f"RMSNorm 参数量: {sum(p.numel() for p in rms_norm.parameters())}  （只有 γ）")
print(f"LayerNorm 参数量: {sum(p.numel() for p in layer_norm.parameters())}  （γ + β）")
print(f"RMSNorm 输出均值: {out_rms.mean().item():.4f}  （不保证零均值）")
print(f"LayerNorm 输出均值: {out_ln.mean().item():.6f}  （零均值）")
print()

print("=== 总结 ===")
print("""
1. 残差连接在梯度中保留恒等分量（梯度=1+∂f/∂x），解决梯度消失
2. ResNet Bottleneck：1×1降维→3×3卷积→1×1升维，减少参数量
3. 通道/stride 不匹配时用 1×1 projection shortcut 对齐维度
4. Pre-LN > Post-LN：梯度更稳定，不需要精细 warmup，现代大模型标准
5. RMSNorm 去掉中心化，计算更快，LLaMA/Mistral 的选择
""")
