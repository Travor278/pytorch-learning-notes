# 01_manual_gradient.py
# 手动梯度计算 —— 在引入 autograd 之前，先从第一性原理推导梯度。
#
# 目标：用 MSE 损失 + 梯度下降拟合线性模型 y = 2x + 1。
# 本文件不依赖 autograd，所有梯度由解析推导得出。
#
# 核心数学：
#   模型：  ŷ = wx + b
#   损失：  L = (1/N) * Σᵢ (ŷᵢ - yᵢ)²           （均方误差，MSE）
#
#   令 eᵢ = ŷᵢ - yᵢ（残差），则：
#
#   ∂L/∂w = (1/N) * Σᵢ ∂(eᵢ²)/∂w
#           = (1/N) * Σᵢ 2eᵢ · ∂eᵢ/∂w
#           = (2/N) * Σᵢ eᵢ · xᵢ                （因为 ∂eᵢ/∂w = xᵢ）
#
#   ∂L/∂b = (2/N) * Σᵢ eᵢ                        （因为 ∂eᵢ/∂b = 1）
#
#   参数更新（梯度下降）：
#   w ← w - η · ∂L/∂w
#   b ← b - η · ∂L/∂b
#
# 注意：此问题是凸二次规划，任意学习率 η ∈ (0, 2/λ_max) 均保证收敛，
# 其中 λ_max 是 Hessian 矩阵 ∂²L/∂θ² 的最大特征值。
# 对 MSE + 线性模型，存在解析解（正规方程），但梯度下降更具一般性。

import torch

# ---- 数据 ----
x      = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([3.0, 5.0, 7.0])   # 真实关系：y = 2x + 1

# 参数初始化为零（不用 requires_grad，因为梯度由我们自己计算）
w      = torch.tensor(0.0)
b      = torch.tensor(0.0)
lr     = 0.1    # 学习率
epochs = 20

print("手动梯度下降：拟合 y = 2x + 1")
print(f"初始: w={w:.4f}, b={b:.4f}")
print("-" * 55)

# ---- 训练循环 ----
for epoch in range(epochs):

    # 前向传播
    y_pred = w * x + b

    # MSE 损失
    loss = ((y_pred - y_true) ** 2).mean()

    # 手动链式法则
    N     = x.shape[0]
    error = y_pred - y_true          # 残差向量 e ∈ R^N
    grad_w = (2.0 / N) * torch.sum(error * x)   # ∂L/∂w
    grad_b = (2.0 / N) * torch.sum(error)        # ∂L/∂b

    # 梯度下降步（in-place，因为 w, b 不在计算图中）
    w = w - lr * grad_w
    b = b - lr * grad_b

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}: Loss={loss:.6f}, w={w:.4f}, b={b:.4f}, "
              f"grad_w={grad_w:.4f}")

print("-" * 55)
print(f"收敛结果: w={w:.4f}（期望 2.0），b={b:.4f}（期望 1.0）")
print(f"泛化验证: x=4 → ŷ={w*4+b:.4f}（期望 9.0）")

# ---- 本文件的工程意义 ----
# 线性模型只有两个参数，手推梯度仅需两行。
# 然而对一个 L 层的 MLP，参数量为 Σₗ (dₗ × dₗ₋₁ + dₗ)；
# 对 ResNet-50，参数量 ≈ 2.5×10⁷。逐参数手写导数在工程上根本不可行。
# 这正是 autograd 存在的根本动机——见 02_autograd_basics.py。
