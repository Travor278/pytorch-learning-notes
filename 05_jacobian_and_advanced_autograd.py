# 05_jacobian_and_advanced_autograd.py
# 用途：雅可比矩阵、高阶导数、向量对向量求导
#
# 这个文件展示 PyTorch 怎么处理"非标量输出"的求导，
# 以及怎么算二阶导（Hessian相关）。

import torch

# ========== 标量对向量求导（你已经会了）==========
# loss 是标量，w 是向量 → ∂loss/∂w 也是向量，跟 w 同形状
# 这就是前面几个文件一直在做的事

# ========== 向量对向量求导 → 雅可比矩阵 ==========
# 但如果输出不是标量呢？
# 比如 y = f(x)，其中 x 是 n 维，y 是 m 维
# 那么 ∂y/∂x 是一个 m×n 的矩阵 —— 雅可比矩阵

print("=== 雅可比矩阵（Jacobian）===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# y 是一个 3 维向量（不是标量！）
# y[0] = x[0]² + x[1]
# y[1] = x[1]² + x[2]
# y[2] = x[0] * x[2]
y = torch.stack([
    x[0]**2 + x[1],
    x[1]**2 + x[2],
    x[2] * x[0]
])

print(f"x = {x}")
print(f"y = {y}")
print()

# 直接调 y.backward() 会报错！因为 PyTorch 的 backward 只能处理标量输出。
# y.backward()  # RuntimeError: grad can be implicitly created only for scalar outputs

# 方法1：传入 gradient 参数（VJP，Vector-Jacobian Product）
# backward(v) 实际上算的是 v^T @ J，其中 J 是雅可比矩阵
# 如果 v 是单位向量 e_i，那 v^T @ J 就是 J 的第 i 行

# 先算完整的雅可比矩阵
# J = [[∂y0/∂x0, ∂y0/∂x1, ∂y0/∂x2],
#      [∂y1/∂x0, ∂y1/∂x1, ∂y1/∂x2],
#      [∂y2/∂x0, ∂y2/∂x1, ∂y2/∂x2]]
#
# 手算一下：
# ∂y0/∂x0 = 2*x0 = 2,  ∂y0/∂x1 = 1,     ∂y0/∂x2 = 0
# ∂y1/∂x0 = 0,          ∂y1/∂x1 = 2*x1=4, ∂y1/∂x2 = 1
# ∂y2/∂x0 = x2 = 3,     ∂y2/∂x1 = 0,      ∂y2/∂x2 = x0 = 1
#
# J = [[2, 1, 0],
#      [0, 4, 1],
#      [3, 0, 1]]

print("手算的雅可比矩阵:")
print("J = [[2, 1, 0],")
print("     [0, 4, 1],")
print("     [3, 0, 1]]")
print()

# 用 backward + 单位向量逐行提取
jacobian_rows = []
for i in range(3):
    # 每次提取一行，需要重新算 y（因为图被销毁了）
    x_copy = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y_copy = torch.stack([
        x_copy[0]**2 + x_copy[1],
        x_copy[1]**2 + x_copy[2],
        x_copy[2] * x_copy[0]
    ])
    
    # v = 第 i 个单位向量
    v = torch.zeros(3)
    v[i] = 1.0
    
    y_copy.backward(v)
    jacobian_rows.append(x_copy.grad.clone())

J = torch.stack(jacobian_rows)
print(f"用 backward 逐行提取的雅可比矩阵:\n{J}")
print()

# 方法2：用 torch.autograd.functional.jacobian（更方便）
from torch.autograd.functional import jacobian

def f(x):
    return torch.stack([
        x[0]**2 + x[1],
        x[1]**2 + x[2],
        x[2] * x[0]
    ])

x_val = torch.tensor([1.0, 2.0, 3.0])
J_auto = jacobian(f, x_val)
print(f"用 torch.autograd.functional.jacobian 计算:\n{J_auto}")
print()

# ========== 高阶导数：二阶导 ==========
# 有些优化方法需要 Hessian 矩阵（二阶导矩阵），
# 比如牛顿法、L-BFGS 等

print("=== 高阶导数（二阶导）===")

x = torch.tensor(3.0, requires_grad=True)
y = x ** 3  # y = x³

# 一阶导：dy/dx = 3x² = 27
# 关键是 create_graph=True：让一阶导也成为计算图的一部分，这样才能对它再求导
grad_1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"y = x³, x = {x.item()}")
print(f"一阶导 dy/dx = 3x² = {grad_1.item()}")  # 27

# 二阶导：d²y/dx² = 6x = 18
grad_2 = torch.autograd.grad(grad_1, x, create_graph=True)[0]
print(f"二阶导 d²y/dx² = 6x = {grad_2.item()}")  # 18

# 三阶导也行：d³y/dx³ = 6
grad_3 = torch.autograd.grad(grad_2, x)[0]
print(f"三阶导 d³y/dx³ = {grad_3.item()}")  # 6

print()

# ========== torch.autograd.grad vs .backward() ==========
# grad() 和 backward() 的区别：
# - backward() 把梯度累加到 .grad 属性上
# - grad() 直接返回梯度值，不修改任何属性

print("=== grad() vs backward() ===")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# 用 grad()
g = torch.autograd.grad(y, x)[0]
print(f"grad() 返回: {g}")
print(f"x.grad 没变: {x.grad}")  # None

# 用 backward()
x2 = torch.tensor(2.0, requires_grad=True)
y2 = x2 ** 2
y2.backward()
print(f"backward() 后 x.grad: {x2.grad}")

print()

# ========== Hessian 矩阵（多维的二阶导）==========
# Hessian 矩阵 H[i][j] = ∂²f / (∂x_i ∂x_j)
# 在优化理论中用来判断极值点的性质

print("=== Hessian 矩阵 ===")

from torch.autograd.functional import hessian

def g(x):
    # f(x0, x1) = x0² + 3*x0*x1 + x1³
    return x[0]**2 + 3*x[0]*x[1] + x[1]**3

x_val = torch.tensor([1.0, 2.0])
H = hessian(g, x_val)
print(f"f(x0, x1) = x0² + 3*x0*x1 + x1³")
print(f"在 x = {x_val.tolist()} 处的 Hessian 矩阵:")
print(H)
# H = [[∂²f/∂x0², ∂²f/(∂x0∂x1)],
#      [∂²f/(∂x1∂x0), ∂²f/∂x1²]]
# = [[2, 3],
#    [3, 6*x1]] = [[2, 3], [3, 12]]

print()
print("=== 总结 ===")
print("""
1. 向量对向量求导得到雅可比矩阵（Jacobian）
2. backward(v) 实际上计算的是 VJP（v^T @ J），不是完整的 J
3. 要算完整的 J，用 torch.autograd.functional.jacobian
4. create_graph=True 让梯度本身也参与计算图，从而能算高阶导
5. Hessian 矩阵用 torch.autograd.functional.hessian 直接算
6. 这些在物理模拟、最优控制、元学习（MAML）中都会用到
""")
