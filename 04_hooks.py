# 04_hooks.py
# Tensor Hook 与 Module Hook 机制
#
# Hook 是 PyTorch 提供的"拦截器"接口，允许在前向/反向传播的特定位置注册回调函数，以观测或修改中间激活值和梯度。
#
# 工程应用（非完整列举）：
#   ① Grad-CAM 可视化（Selvaraju et al., ICCV 2017）
#      —— 提取最后一个卷积层的 feature map 及其梯度，加权生成热力图
#   ② 梯度反转（Gradient Reversal Layer）（Ganin et al., JMLR 2016）
#      —— 在域适应中对判别器的梯度取反，使特征提取器学习域不变特征
#   ③ 特征提取 / 中间层监控 —— 无需修改 forward 代码即可插拔式获取任意层输出
#   ④ 调试工具 —— 检测梯度消失/爆炸、NaN/Inf

import torch
import torch.nn as nn

# ========== 问题：非叶子节点的梯度默认不保留 ==========
# 原因：PyTorch 在 backward 执行后立即释放中间激活值的梯度，以将显存复杂度维持在 O(参数量) 而非 O(所有中间节点)。

print("=== 问题：非叶子节点梯度默认丢弃 ===")

x    = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y    = x * 2 + 1      # y 是中间变量，is_leaf=False
loss = (y ** 2).sum()
loss.backward()

print(f"x.grad = {x.grad}")   # ✓ 叶子节点有梯度
print(f"y.grad = {y.grad}")   # None：中间节点梯度已被释放
print()

# ========== 方案 1：retain_grad() ==========
# 在 forward 前调用，令 PyTorch 保留该非叶子节点的梯度缓冲区。
# 缺点：每个保留节点都持续占用显存，仅适用于调试场景。

print("=== 方案 1：retain_grad() ===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1
y.retain_grad()   # 告知 PyTorch 保留 y 的梯度

loss = (y ** 2).sum()
loss.backward()

# ∂loss/∂y = 2y = 2*(2x+1)：在 x=[1,2,3] 处 = [6, 10, 14]
print(f"x.grad = {x.grad}")
print(f"y.grad = {y.grad}  （期望 [6, 10, 14]）")
print()

# ========== 方案 2：register_hook() —— 更通用的梯度拦截器 ==========
# 当 tensor 的梯度被计算出来时，PyTorch 自动调用所有注册在该 tensor 上的 hook。
# hook 签名：hook(grad) -> Tensor or None
#   返回 None    —— 不修改梯度，仅作观测
#   返回 Tensor  —— 用返回值替换原梯度（可实现梯度裁剪、缩放、反转等）
# hook 句柄（handle）的 .remove() 方法可在不再需要时注销，防止重复触发。

print("=== 方案 2：register_hook()（观测梯度）===")

saved_grads = {}

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1

def save_grad(name):
    """闭包工厂，捕获 name，返回将梯度存入字典的 hook 函数"""
    def hook_fn(grad):
        saved_grads[name] = grad.clone()  # clone 避免后续 in-place 操作覆盖缓冲区
    return hook_fn

h = y.register_hook(save_grad('y'))

loss = (y ** 2).sum()
loss.backward()

print(f"hook 捕获的 y.grad = {saved_grads['y']}")
h.remove()   # 观测结束后移除，避免每次 backward 都触发
print()

# ========== 用 hook 修改梯度：梯度反转层（GRL）==========
# Gradient Reversal Layer（梯度反转层）是域适应中的经典技巧：
# 前向传播为恒等映射，反向传播将梯度乘以 -λ。
# 通常用自定义 Function 实现（见 07），此处用 hook 演示等效效果。
# 参考：Ganin & Lempitsky (ICML 2015 / JMLR 2016)。

print("=== register_hook 修改梯度：梯度反转（域适应 GRL）===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1

# 反转梯度：hook 返回非 None 即替换原梯度
h = y.register_hook(lambda grad: -grad)

loss = (y ** 2).sum()
loss.backward()

# 正常梯度（无反转）x.grad 应为 [12, 20, 28]
print(f"x.grad（梯度已反转）= {x.grad}  （正常值的相反数）")
h.remove()
print()

# ========== Module Hook：针对 nn.Module 层 ==========
# Tensor Hook 仅作用于单个 tensor；Module Hook 可拦截整个层的输入输出。
#
# register_forward_hook(hook(module, input, output))
#   —— 前向传播执行完该层后触发，input/output 均为 tuple
#
# register_full_backward_hook(hook(module, grad_input, grad_output))
#   —— 反向传播流经该层时触发
#   —— 注意：register_backward_hook 已废弃，对含有多输入/输出的模块行为不一致
#            应改用 register_full_backward_hook

print("=== Module Hook：拦截网络层的激活值与梯度 ===")

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 4)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model      = SimpleNet()
activations = {}
gradients   = {}

def forward_hook(name):
    def hook(module, input, output):
        # detach() 防止 hook 内的操作意外加入主计算图
        activations[name] = output.detach().clone()
    return hook

def backward_hook(name):
    def hook(module, grad_input, grad_output):
        # grad_output[0]：流入该层的梯度（即 ∂loss/∂output）
        gradients[name] = grad_output[0].detach().clone()
    return hook

h1 = model.relu.register_forward_hook(forward_hook('relu'))
h2 = model.relu.register_full_backward_hook(backward_hook('relu'))

x_in = torch.randn(2, 3)
out  = model(x_in)
loss = out.sum()
loss.backward()

print(f"ReLU 输出（激活值）:\n{activations['relu']}")
print(f"ReLU 反传梯度（∂loss/∂ReLU_output）:\n{gradients['relu']}")
# 观察：ReLU 反传梯度中，原输出为 0 的位置（被截断的负值）梯度也为 0，
# 这正是 ReLU 不可导点（x=0）处梯度为 0 的体现。

h1.remove()
h2.remove()
print()

# ========== 实际场景：Grad-CAM 原理速览 ==========
print("=== Grad-CAM 原理（Selvaraju et al., ICCV 2017）===")
print("""
Grad-CAM 的完整流程：
  1. register_forward_hook  → 保存目标卷积层的 feature map A^k  （形状：[C, H, W]）
  2. register_full_backward_hook → 保存 ∂y^c/∂A^k              （类别 c 对 feature map 的梯度）
  3. 全局平均池化（GAP）计算权重：α^c_k = (1/Z) Σᵢ Σⱼ (∂y^c/∂A^k_ij)
  4. 加权求和后接 ReLU：
       L^c_Grad-CAM = ReLU(Σ_k α^c_k · A^k)
     —— ReLU 确保只保留对类别 c 有正贡献的特征

直觉解释：梯度大的 channel 对分类更重要；
         ReLU 过滤掉"对 c 类有害"的特征响应区域。
不会 hook 就无法复现 Grad-CAM，这是 hook 在研究中必须掌握的核心原因。
""")

# ========== 补充：线性回归中用 hook 观测 y_pred 的梯度 ==========
print("=== 补充：线性回归 y_pred 梯度验证 ===")

x_data = torch.tensor([1.0, 2.0, 3.0])
y_data = torch.tensor([3.0, 5.0, 7.0])
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = w * x_data + b

# ∂MSE/∂ŷᵢ = (2/N)(ŷᵢ - yᵢ)；在 w=b=0 时 ŷ=0，
# 故 ∂L/∂ŷ = (2/3)*[-3, -5, -7] = [-2, -3.333, -4.667]
y_pred.register_hook(
    lambda grad: print(f"  y_pred 的梯度 ∂L/∂ŷ: {grad}")
)

loss = ((y_pred - y_data) ** 2).mean()
loss.backward()
print(f"w.grad = {w.grad}  b.grad = {b.grad}")
