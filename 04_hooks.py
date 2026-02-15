# 04_hooks.py
# 用途：Tensor Hook 和 Module Hook 机制
#
# Hook（钩子）在论文复现中非常重要。
# 比如 Grad-CAM（类激活图可视化）就需要 hook 来提取中间层的梯度。
# 很多人跳过这部分，结果看到别人代码里的 register_hook 就懵了。

import torch
import torch.nn as nn

# ========== 问题：中间变量的梯度拿不到 ==========
#
# 03 里说过，非叶子节点的梯度默认不保留。
# 但有时候我们就是需要它，怎么办？

print("=== 问题：中间变量的梯度默认不保存 ===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1       # 中间变量
loss = (y ** 2).sum()
loss.backward()

print(f"x.grad = {x.grad}")   # 有值
print(f"y.grad = {y.grad}")   # None！拿不到

print()

# ========== 方案1：retain_grad() ==========
# 最简单粗暴的方法：让 PyTorch 保留这个中间变量的梯度

print("=== 方案1：retain_grad() ===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1

y.retain_grad()  # 告诉 PyTorch：y 的梯度我也要

loss = (y ** 2).sum()
loss.backward()

print(f"x.grad = {x.grad}")
print(f"y.grad = {y.grad}")  # 这次有了！
# y.grad = ∂loss/∂y = 2y = 2*(2x+1) = [6, 10, 14]

print()

# ========== 方案2：register_hook() —— 更灵活 ==========
# hook 的意思是：当某个 tensor 的梯度被计算出来时，自动调用你指定的函数
# 你可以在这个函数里做任何事：打印、保存、修改梯度...

print("=== 方案2：register_hook() ===")

# 用一个列表来收集梯度
saved_grads = {}

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1

# 注册一个 hook：当 y 的梯度算出来时，把它存起来
def save_grad(name):
    """返回一个 hook 函数，把梯度存到字典里"""
    def hook_fn(grad):
        saved_grads[name] = grad.clone()  # clone 防止被覆盖
    return hook_fn

h = y.register_hook(save_grad('y'))  # h 是 hook 的句柄，后面可以用来移除

loss = (y ** 2).sum()
loss.backward()

print(f"通过 hook 拿到的 y 的梯度: {saved_grads['y']}")

# 用完了记得移除 hook，否则每次 backward 都会触发
h.remove()

print()

# ========== register_hook 的实战用法：修改梯度 ==========
# 有些论文会在反向传播时修改梯度，比如梯度裁剪、梯度反转（Domain Adaptation）

print("=== 用 hook 修改梯度 ===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1

# 梯度反转：把梯度乘以 -1（用在对抗训练中）
# hook 函数如果有返回值，就会替换掉原来的梯度
h = y.register_hook(lambda grad: -grad)

loss = (y ** 2).sum()
loss.backward()

print(f"x.grad（梯度被反转了）= {x.grad}")
# 正常的 x.grad 应该是 [12, 20, 28]
# 反转后变成 [-12, -20, -28]

h.remove()

print()

# ========== Module Hook：针对神经网络层 ==========
# 上面的 register_hook 是针对 tensor 的。
# 如果你想 hook 整个网络层（nn.Module），用 register_forward_hook / register_backward_hook

print("=== Module Hook：查看中间层的输入输出 ===")

# 搞个简单的两层网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 4)  # 3维输入 -> 4维输出
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)  # 4维 -> 1维
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = SimpleNet()

# 存中间层信息的地方
activations = {}
gradients = {}

# forward hook：前向传播经过某层时触发
def forward_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().clone()
    return hook

# full backward hook：反向传播经过某层时触发
def backward_hook(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0].detach().clone()
    return hook

# 给 relu 层挂上 hook
h1 = model.relu.register_forward_hook(forward_hook('relu'))
h2 = model.relu.register_full_backward_hook(backward_hook('relu'))

# 跑一遍前向+反向
x = torch.randn(2, 3)  # batch=2, 特征=3
output = model(x)
loss = output.sum()
loss.backward()

print(f"ReLU 层的输出（激活值）:\n{activations['relu']}")
print(f"ReLU 层反向传播时的梯度:\n{gradients['relu']}")

# 清理
h1.remove()
h2.remove()

print()

# ========== 实际应用场景 ==========
print("=== 实际场景：简化版 Grad-CAM 思路 ===")
print("""
Grad-CAM 的核心思路很简单：
1. 用 forward_hook 拿到某个卷积层的输出（特征图 A）
2. 用 backward_hook 拿到该层的梯度（∂loss/∂A）
3. 对梯度做 GAP（全局平均池化）得到权重 α
4. 加权求和：热力图 = ReLU(Σ αk * Ak)

这就是为什么 hook 在论文复现中这么重要。
如果你不会 hook，Grad-CAM、特征可视化这些都写不了。
""")

# ========== 用 register_hook 查看线性回归中间变量的梯度 ==========

print("=== 线性回归中 y_pred 的梯度 ===")

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([3.0, 5.0, 7.0])

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = w * x + b

# 给 y_pred 挂个 hook，看看它的梯度长什么样
y_pred.register_hook(lambda grad: print(f"  y_pred 的梯度（∂Loss/∂y_pred）: {grad}"))

loss = ((y_pred - y_true) ** 2).mean()
loss.backward()

# y_pred 的梯度 = ∂Loss/∂y_pred = (2/N) * (y_pred - y_true)
# y_pred = [0, 0, 0], y_true = [3, 5, 7]
# 所以梯度 = (2/3) * [-3, -5, -7] = [-2, -3.3333, -4.6667]
print(f"w.grad = {w.grad}")
print(f"b.grad = {b.grad}")
