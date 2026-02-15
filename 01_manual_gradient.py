# 01_manual_gradient.py
# 用途：手动梯度计算 —— 不用 autograd，纯手推链式法则
#
# 如果没有 PyTorch，得自己写多少东西。
# 目标：拟合 y = 2x + 1，MSE 损失，手动实现梯度下降。

import torch

# --- 准备数据 ---
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([3.0, 5.0, 7.0])  # y = 2x + 1

# 初始化 w, b 为 0（不用 requires_grad，因为我们自己算梯度）
w = torch.tensor(0.0)
b = torch.tensor(0.0)
lr = 0.1    # 学习率
epochs = 20

print("手动梯度下降：拟合 y = 2x + 1")
print(f"初始: w={w:.4f}, b={b:.4f}")
print("-" * 50)

# --- 训练循环 ---
for epoch in range(epochs):
    # 前向传播：y_pred = wx + b
    y_pred = w * x + b
    
    # MSE 损失 = (1/N) * Σ(y_pred - y_true)²
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 反向传播（手动推导链式法则）
    #
    # 设 e = y_pred - y_true = wx + b - y
    # Loss = (1/N) * Σ(e²)
    #
    # ∂Loss/∂w = (2/N) * Σ(e * x)     ← 链式法则：外层导数 * 内层对w的导数
    # ∂Loss/∂b = (2/N) * Σ(e)         ← 内层对b的导数是1
    N = x.shape[0]
    error = y_pred - y_true
    grad_w = (2.0 / N) * torch.sum(error * x)
    grad_b = (2.0 / N) * torch.sum(error)
    
    # 梯度下降更新
    w = w - lr * grad_w
    b = b - lr * grad_b
    
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}: Loss={loss:.6f}, w={w:.4f}, b={b:.4f}, "
              f"grad_w={grad_w:.4f}")

print("-" * 50)
print(f"结果: w={w:.4f}(期望2.0), b={b:.4f}(期望1.0)")
print(f"验证: x=4 时, y={w*4+b:.4f}(期望9.0)")

# 思考：
# 线性回归手推导数两行就搞定了。
# 但如果是个 LSTM 或 100 层 ResNet？手写 grad_w 根本不可能。
# 这就是为什么需要 autograd —— 见 02。
