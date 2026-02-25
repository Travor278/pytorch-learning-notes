# 02_autograd_basics.py
# PyTorch 自动求导（Autograd）入门
#
# PyTorch 的 autograd 基于 Wengert list（也称"磁带"，tape），即在前向传播时将所有运算
# 记录在一条有向无环图（DAG）上，随后在反向传播时沿图反向遍历，逐节点应用链式法则。
# 这与 Griewank & Walther (2008) "Evaluating Derivatives" 中描述的反向模式 AD 在算法上完全等价。
#
# 关键 API：
#   requires_grad=True  —— 将 tensor 标记为可微参数，令其入计算图
#   loss.backward()     —— 触发反向传播，将梯度累加到各叶子节点的 .grad
#   optimizer.zero_grad() / .grad.zero_() —— 在每步更新前清空累积梯度
#   torch.no_grad()     —— 上下文管理器，暂停梯度追踪（常用于参数更新、推理）

import torch

# ---- 数据（与 01 相同，便于对比）----
x      = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([3.0, 5.0, 7.0])

# requires_grad=True 使 PyTorch 在前向传播时将涉及 w, b 的所有运算写入计算图。
# 从 C++ 实现角度：每个运算节点会创建一个 grad_fn（继承自 Node），
# 反向传播时按拓扑逆序调用每个 grad_fn 的 call_function() 方法。
w  = torch.tensor(0.0, requires_grad=True)
b  = torch.tensor(0.0, requires_grad=True)
lr = 0.1
epochs = 20

print("Autograd 自动求导：拟合 y = 2x + 1")
print(f"初始: w={w.data:.4f}, b={b.data:.4f}")
print("-" * 55)

for epoch in range(epochs):
    # 前向传播——PyTorch 静默构建计算图
    y_pred = w * x + b
    loss   = ((y_pred - y_true) ** 2).mean()

    # backward() 遍历计算图，等价于 01 中所有手写偏导数推导
    loss.backward()

    gw = w.grad.item()
    gb = b.grad.item()

    # 参数更新必须在 no_grad 下进行：
    # 若不用 no_grad，w -= lr * w.grad 这个赋值本身也会被记入计算图，
    # 导致下一轮 backward 时图结构被污染，引发错误的梯度。
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        # .grad 默认累加（支持 BPTT 和梯度累积场景），普通训练必须手动清零。
        # 使用 zero_() 而非 = None 是因为前者复用已分配的内存，避免重复分配。
        w.grad.zero_()
        b.grad.zero_()

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}: Loss={loss.item():.6f}, "
              f"w={w.data:.4f}, b={b.data:.4f}, grad_w={gw:.4f}")

print("-" * 55)
print(f"收敛结果: w={w.data:.4f}（期望 2.0），b={b.data:.4f}（期望 1.0）")

# ---- 验证 autograd 与手动计算数值一致 ----
print("\n验证 autograd == 手动梯度:")

wa = torch.tensor(0.0, requires_grad=True)
ba = torch.tensor(0.0, requires_grad=True)
wm = torch.tensor(0.0)
bm = torch.tensor(0.0)

ya = wa * x + ba
la = ((ya - y_true) ** 2).mean()
la.backward()

ym = wm * x + bm
N        = x.shape[0]
err      = ym - y_true
gw_manual = (2.0 / N) * torch.sum(err * x)
gb_manual = (2.0 / N) * torch.sum(err)

print(f"  autograd: grad_w={wa.grad.item():.6f}, grad_b={ba.grad.item():.6f}")
print(f"  手动推导: grad_w={gw_manual.item():.6f}, grad_b={gb_manual.item():.6f}")
print(f"  数值一致: {torch.allclose(wa.grad, gw_manual)}")

# ---- 梯度累加陷阱 ----
# PyTorch 不在每次 backward 前自动清零是有意设计的：
# ① BPTT（Backpropagation Through Time）需要跨时间步积累梯度；
# ② 梯度累积（Gradient Accumulation）用 k 步小 batch 模拟大 batch。
# 但普通单步训练中忘记清零是最常见的 bug 之一。

print("\n陷阱演示：不清零时梯度在多次 backward 间持续累加")

wt = torch.tensor(0.0, requires_grad=True)
bt = torch.tensor(0.0, requires_grad=True)

for i in range(3):
    yt = wt * x + bt
    lt = ((yt - y_true) ** 2).mean()
    lt.backward()
    print(f"  第{i+1}次 backward: w.grad={wt.grad.item():.4f}  ← 每轮均在前一次基础上叠加")

# ---- .data / .detach() / .item() 的语义区别 ----
print("\n三种访问 tensor 值的方式（语义上有本质差异）:")
t = torch.tensor(3.14, requires_grad=True)

# .item()：将 0 维 tensor 转为 Python 标量，完全脱离计算图，最安全
print(f"  .item()   = {t.item():<8}  → Python float，与计算图无关")

# .data：直接访问底层存储，不追踪操作；但修改 .data 会破坏版本计数器
# （version counter），可能导致 backward 时出现 RuntimeError 或静默错误。
# 在需要跳过梯度追踪的原地修改（如参数更新）中仍有使用，但风险高于 detach()。
print(f"  .data     = {t.data}  → tensor，绕过 autograd 版本检查（慎用）")

# .detach()：创建与原 tensor 共享存储但不参与计算图的新 tensor，
# 且受版本计数器保护——若原 tensor 被修改，访问 detach 结果时会正确报错。
# 推荐在需要"从图中取出数值"时使用 detach()。
print(f"  .detach() = {t.detach()}  → tensor，安全脱离计算图（推荐）")
