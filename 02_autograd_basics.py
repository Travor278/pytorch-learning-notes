# 02_autograd_basics.py
# 用途：引入 PyTorch 自动求导（autograd），用 1 行代码替代手动推导
#
# 对比 01，autograd 把手写链式法则这件事全自动化了。
# 核心概念：requires_grad, backward(), .grad, zero_(), no_grad()

import torch

# --- 准备数据（跟 01 一样）---
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([3.0, 5.0, 7.0])

# 关键区别：requires_grad=True
# 意思是"PyTorch，帮我追踪这个变量的所有运算，到时候自动算梯度"
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.1
epochs = 20

print("Autograd 自动求导：拟合 y = 2x + 1")
print(f"初始: w={w.data:.4f}, b={b.data:.4f}")
print("-" * 50)

for epoch in range(epochs):
    # 前向传播 —— PyTorch 在后台默默建计算图
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 反向传播 —— 就这一行，替代了 01 里所有的手动推导
    loss.backward()
    
    # 记录一下梯度值（后面要打印）
    gw = w.grad.item()
    gb = b.grad.item()
    
    # 更新参数
    # no_grad：告诉 PyTorch 这一步不要记在计算图里（更新操作不需要求导）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        
        # 清空梯度！PyTorch 默认累加梯度，不清零下一轮就错了
        # 为什么要累加？因为 RNN 的 BPTT 和梯度累积（大 batch 模拟）需要
        # 但普通训练必须清零
        w.grad.zero_()
        b.grad.zero_()
    
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}: Loss={loss.item():.6f}, "
              f"w={w.data:.4f}, b={b.data:.4f}, grad_w={gw:.4f}")

print("-" * 50)
print(f"结果: w={w.data:.4f}(期望2.0), b={b.data:.4f}(期望1.0)")

# --- 验证 autograd 和手动计算一致 ---
print("\n验证 autograd == 手动梯度:")

wa = torch.tensor(0.0, requires_grad=True)
ba = torch.tensor(0.0, requires_grad=True)
wm = torch.tensor(0.0)
bm = torch.tensor(0.0)

# autograd
ya = wa * x + ba
la = ((ya - y_true) ** 2).mean()
la.backward()

# 手动
ym = wm * x + bm
N = x.shape[0]
err = ym - y_true
gw_manual = (2.0 / N) * torch.sum(err * x)
gb_manual = (2.0 / N) * torch.sum(err)

print(f"  autograd: grad_w={wa.grad.item():.6f}, grad_b={ba.grad.item():.6f}")
print(f"  手动:     grad_w={gw_manual.item():.6f}, grad_b={gb_manual.item():.6f}")
print(f"  一致: {torch.allclose(wa.grad, gw_manual)}")

# --- 演示不清零的后果 ---
print("\n坑：不清零梯度会累加")

wt = torch.tensor(0.0, requires_grad=True)
bt = torch.tensor(0.0, requires_grad=True)

for i in range(3):
    yt = wt * x + bt
    lt = ((yt - y_true) ** 2).mean()
    lt.backward()
    print(f"  第{i+1}次: w.grad={wt.grad.item():.4f} (每次都在累加！)")
    # 不清零，不更新，只是展示累加现象

# --- .data vs .item() vs .detach() ---
print("\n三种取值方式:")
t = torch.tensor(3.14, requires_grad=True)
print(f"  .item() = {t.item()} → Python float")
print(f"  .data   = {t.data} → tensor，但脱离计算图（修改会出事）")
print(f"  .detach() = {t.detach()} → tensor，脱离计算图（推荐用这个）")
