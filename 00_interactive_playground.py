#%%
# 00_interactive_playground.py
# 快速验证环境，并以最小示例感知 autograd 的"形"与 Jacobian 的"意"。
#
# PyTorch 的自动求导基于反向模式自动微分（Reverse-Mode Automatic Differentiation，亦称 Backpropagation）。
# 与前向模式 AD 相比，反向模式在"输出维度远小于输入维度"时（神经网络的标量损失对数百万参数求导恰好符合此条件）计算复杂度更低。
# 参考：Baydin et al., "Automatic Differentiation in Machine Learning: a Survey", JMLR 2018.

import torch

print(f"PyTorch:          {torch.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
# torch.version.cuda 是编译 PyTorch 时链接的 CUDA 工具链版本；
# torch.cuda.is_available() 则反映运行时驱动是否就绪，两者不一致时需重新安装。
print(f"CUDA build ver:   {torch.version.cuda}")

#%%
# ---- 最小反向传播示例 ----
# 函数 y = sum((2x + 1)^2)，标量输出 → 可直接调 backward()。
# 分析解：∂y/∂x_i = 2 * 2 * (2x_i + 1) = 4(2x_i + 1)
# x = [1, 2, 3] → 期望梯度 = [4*(3), 4*(5), 4*(7)] = [12, 20, 28]

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True) # requires_grad=True 使得 PyTorch 追踪对 x 的操作以便后续求导
y = (x * 2 + 1).pow(2).sum()
y.backward()

print("x:      ", x)
print("y:      ", y)
print("x.grad: ", x.grad)   # 应为 [12., 20., 28.]

#%%
# ---- Jacobian 矩阵示例 ----
# 当 f: R^n → R^m（m > 1）时，∂f/∂x 是 m×n 的 Jacobian 矩阵 J。
# 深度学习中损失通常为标量（m=1），使得反向传播只需计算一次矩阵-向量积（VJP），
# 代价 O(n) 而非 O(mn)。这是反向模式 AD 相对于数值微分在参数规模上高效的根本原因。

def f(v):
    return torch.stack([
        v[0] ** 2 + v[1],   # ∂/∂v = [2v0, 1, 0]
        v[1] ** 2 + v[2],   # ∂/∂v = [0, 2v1, 1]
        v[2] * v[0],         # ∂/∂v = [v2, 0, v0]
    ])

from torch.autograd.functional import jacobian

v = torch.tensor([1.0, 2.0, 3.0])
J = jacobian(f, v)
# 期望：[[2,1,0],[0,4,1],[3,0,1]]（在 v=[1,2,3] 处求值）
print("f(v):\n", f(v))
print("Jacobian:\n", J)