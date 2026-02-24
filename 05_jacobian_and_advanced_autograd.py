# 05_jacobian_and_advanced_autograd.py
# Jacobian 矩阵、高阶导数与 Hessian
#
# 本文件回答三个递进的问题：
#   Q1：输出是向量时，梯度是什么形状？（Jacobian 矩阵）
#   Q2：PyTorch 的 backward(v) 究竟算的是什么？（VJP）
#   Q3：如何获取二阶及更高阶导数？（create_graph + 递归 grad()）
#
# 背景：深度学习训练几乎只需要 ∂L/∂θ（标量对向量的梯度），
# 这恰好是反向模式 AD（VJP）最擅长的情形，复杂度为 O(n)（n=参数数）。
# 而计算完整 Jacobian（m×n 矩阵）需调用 m 次反向传播，
# 或切换为前向模式 AD（JVP），复杂度 O(m)（m=输出维度）。
# 当 m << n 时（神经网络标量损失的典型情形），反向模式更优。

import torch
from torch.autograd.functional import jacobian, hessian

# ========== Jacobian 矩阵 ==========
print("=== Jacobian 矩阵 ===")

# f: R³ → R³
# f₀(x) = x₀² + x₁
# f₁(x) = x₁² + x₂
# f₂(x) = x₂ · x₀
#
# 解析 Jacobian J = ∂f/∂x（行 i 对应 fᵢ 对所有 xⱼ 的偏导）：
#     ∂f₀/∂x = [2x₀, 1,    0  ]
#     ∂f₁/∂x = [0,   2x₁,  1  ]
#     ∂f₂/∂x = [x₂,  0,    x₀ ]
# 在 x=[1,2,3] 处：
#   J = [[2, 1, 0],
#        [0, 4, 1],
#        [3, 0, 1]]

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y = torch.stack([
    x[0]**2 + x[1],
    x[1]**2 + x[2],
    x[2] * x[0]
])

print(f"x = {x.data}")
print(f"y = {y.data}")
print()
print("解析 Jacobian:")
print("J = [[2, 1, 0],")
print("     [0, 4, 1],")
print("     [3, 0, 1]]")
print()

# ---- 方法 1：用 backward + 单位向量逐行提取（手动 VJP）----
# backward(v) 计算的是 VJP：vᵀ·J（行向量与 Jacobian 的乘积）。
# 令 v = eᵢ（第 i 个标准基向量），则 eᵢᵀ·J = J 的第 i 行，
# 即 ∂yᵢ/∂x。循环 m 次即可重建完整 J，代价为 m 次反向传播。

def compute_jacobian_via_backward(func, x_val):
    """通过 m 次 VJP 重建 m×n Jacobian"""
    rows = []
    for i in range(x_val.shape[0]):
        xi = x_val.clone().requires_grad_(True)
        yi = func(xi)
        v  = torch.zeros_like(yi)
        v[i] = 1.0
        yi.backward(v)
        rows.append(xi.grad.clone())
    return torch.stack(rows)

def f(x):
    return torch.stack([x[0]**2 + x[1], x[1]**2 + x[2], x[2]*x[0]])

J_manual = compute_jacobian_via_backward(f, torch.tensor([1.0, 2.0, 3.0]))
print(f"方法 1（手动 VJP）:\n{J_manual}\n")

# ---- 方法 2：torch.autograd.functional.jacobian（推荐）----
# 内部同样调用 m 次 VJP（或切换到 forward-mode AD 使用 jvp），
# 但封装更简洁，支持更多高级选项（如 vectorize 加速）。

x_val = torch.tensor([1.0, 2.0, 3.0])
J_auto = jacobian(f, x_val)
print(f"方法 2（torch.autograd.functional.jacobian）:\n{J_auto}\n")
print(f"两种方法结果一致: {torch.allclose(J_manual.float(), J_auto.float())}\n")

# ========== VJP 的直观理解 ==========
print("=== VJP（Vector-Jacobian Product）深度理解 ===")
print("""
深度学习为何使用反向模式 AD（VJP）而非前向模式（JVP）：
  设模型有 n 个参数，损失 L 为标量（输出维度 m=1）。
  - 反向模式（VJP）：1 次反向遍历 → 得到 ∂L/∂θ（1×n 行向量）
                    代价：O(n) 或更精确地 O(FLOPs_forward)
  - 前向模式（JVP）：需 n 次前向遍历才能逐列构建 J
                    代价：O(n · FLOPs_forward)

当 n（参数量）远大于 m（输出维度）时，反向模式效率优势显著。
ResNet-50 约 2.5×10⁷ 参数但只有 1 个损失值，反向模式只需一次遍历。
""")

# ========== 高阶导数：create_graph=True ==========
print("=== 高阶导数 ===")
#
# 默认情况下，grad() / backward() 的输出（梯度 tensor）不在计算图中，
# 无法对其进一步求导。
# create_graph=True 使梯度本身也成为计算图的节点，从而允许链式调用 grad()。
# 应用：MAML（Model-Agnostic Meta-Learning, Finn et al. NeurIPS 2017）的
# 元梯度计算；物理模拟中的梯度惩罚；谱归一化的 Lipschitz 约束估计。

x = torch.tensor(3.0, requires_grad=True)
y = x ** 3   # y = x³

# 一阶导：dy/dx = 3x² = 27
g1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"y = x³，x = {x.item()}")
print(f"一阶导 dy/dx  = 3x²  = {g1.item():.1f}  （期望 27）")

# 二阶导：d²y/dx² = 6x = 18
g2 = torch.autograd.grad(g1, x, create_graph=True)[0]
print(f"二阶导 d²y/dx² = 6x  = {g2.item():.1f}  （期望 18）")

# 三阶导：d³y/dx³ = 6（常数，多项式的最高非零阶导）
g3 = torch.autograd.grad(g2, x)[0]
print(f"三阶导 d³y/dx³ = 6   = {g3.item():.1f}  （期望 6）")
print()

# ========== torch.autograd.grad vs .backward() ==========
print("=== grad() vs backward() 语义区别 ===")
print("""
backward()：梯度累加到目标 tensor 的 .grad 属性，是状态性的（有副作用）。
grad()    ：直接返回梯度 tensor，不修改任何 .grad，是函数式的（无副作用）。
             更适合高阶梯度计算和函数式编程风格。

grad() 的另一优势：可以一次性对多个输出关于多个输入求梯度，
并可配合 is_grads_batched=True 实现批量 Jacobian 计算。
""")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

g = torch.autograd.grad(y, x)[0]
print(f"grad()     返回: {g}，x.grad 未被修改: {x.grad}")   # x.grad 仍为 None

x2 = torch.tensor(2.0, requires_grad=True)
y2 = x2 ** 2
y2.backward()
print(f"backward() 后 x2.grad: {x2.grad}")
print()

# ========== Hessian 矩阵 ==========
print("=== Hessian 矩阵 ===")
#
# Hessian H[i][j] = ∂²f/(∂xᵢ ∂xⱼ)，描述曲率，用于：
#   ① 二阶优化方法（牛顿法、L-BFGS）；
#   ② 神经网络损失曲面分析；
#   ③ Fisher 信息矩阵的近似（与 Hessian 在极值点处相等）；
#   ④ 影响函数（Influence Function）计算，用于数据归因分析。
# 实践中全量 Hessian 对大模型不可行（O(n²) 存储），
# 通常使用 Hessian-vector product（无需显式构建 H）或 K-FAC 近似。

def g(x):
    # f(x₀, x₁) = x₀² + 3x₀x₁ + x₁³
    return x[0]**2 + 3*x[0]*x[1] + x[1]**3

x_val = torch.tensor([1.0, 2.0])
H = hessian(g, x_val)

print("f(x₀, x₁) = x₀² + 3x₀x₁ + x₁³")
print(f"在 x = {x_val.tolist()} 处的 Hessian 矩阵:")
print(H)
# ∂²f/∂x₀²     = 2
# ∂²f/(∂x₀∂x₁) = 3
# ∂²f/∂x₁²     = 6x₁ = 12（在 x₁=2 处）
# H 应为 [[2, 3], [3, 12]]
print()

# Hessian 正定 ↔ 当前点为严格局部极小值（second-order sufficient condition）
eigenvalues = torch.linalg.eigvalsh(H)
print(f"Hessian 特征值: {eigenvalues}")
print(f"Hessian 正定（所有特征值 > 0）: {(eigenvalues > 0).all().item()}")
print()

print("=== 总结 ===")
print("""
1. 向量输出对向量输入的导数是 Jacobian 矩阵 J（m×n）
2. backward(v) 计算 VJP = vᵀ·J，是标量损失反传的数学本质
3. 完整 Jacobian 需 m 次 VJP 或直接用 torch.autograd.functional.jacobian
4. create_graph=True 使梯度进入计算图，从而支持高阶导数
5. Hessian 描述曲率，完整计算代价 O(n²)；实践中通常使用 Hessian-vector product
6. 高阶导数在元学习（MAML）、物理模拟、影响函数中均有关键应用
""")
