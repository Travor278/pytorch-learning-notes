# 06_gradient_accumulation_and_tricks.py
# 用途：梯度累加、梯度裁剪、梯度检查 等实战中常碰到的技巧
#
# 这些不是理论概念，是你写训练代码时一定会遇到的问题。
# 比如 GPU 显存不够大，batch_size 只能设 4，但论文里用的是 32。
# 梯度累加就是干这个的。

import torch
import torch.nn as nn

# ========== 梯度累加（Gradient Accumulation）==========
# 场景：你想用 batch_size=32 训练，但显存只够 batch_size=8
# 解决办法：跑 4 次 batch_size=8，梯度累加起来，再更新一次参数
# 效果等价于 batch_size=32（近似）

print("=== 梯度累加 ===")

# 搞个小网络
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 假数据
data = torch.randn(32, 10)    # 32 个样本
target = torch.randn(32, 1)

# --- 对比1：正常的 batch_size=32 ---
optimizer.zero_grad()
output = model(data)
loss = nn.MSELoss()(output, target)
loss.backward()
# 看看梯度的值
grad_normal = model.weight.grad.clone()
print(f"正常 batch=32 的梯度（前5个）: {grad_normal[0, :5]}")

# --- 对比2：梯度累加，4次 batch=8 ---
optimizer.zero_grad()  # 先清零
accumulation_steps = 4
batch_size = 8

for i in range(accumulation_steps):
    # 每次取 8 个样本
    start = i * batch_size
    end = start + batch_size
    mini_batch = data[start:end]
    mini_target = target[start:end]
    
    output = model(mini_batch)
    # 注意：loss 要除以累加步数！
    # 因为 MSELoss 默认算的是这个 mini-batch 的均值，
    # 但我们要的是整个 batch 的均值
    loss = nn.MSELoss()(output, mini_target) / accumulation_steps
    loss.backward()  # 梯度会自动累加到 .grad 上

grad_accumulated = model.weight.grad.clone()
print(f"累加 4x8 的梯度（前5个）: {grad_accumulated[0, :5]}")
print(f"两者差距: {(grad_normal - grad_accumulated).abs().max().item():.8f}")
# 差距非常小（浮点误差），说明效果等价

# 累加完毕后，一次性更新
optimizer.step()

print()

# ========== 梯度裁剪（Gradient Clipping）==========
# 训练 RNN/LSTM/Transformer 时经常碰到梯度爆炸（gradient exploding）。
# 梯度裁剪就是：如果梯度的范数超过阈值，就把它缩放回来。

print("=== 梯度裁剪 ===")

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 制造一个很大的梯度
x = torch.randn(1, 10) * 100  # 输入值很大
y = torch.tensor([[1.0]])

optimizer.zero_grad()
output = model(x)
loss = nn.MSELoss()(output, y)
loss.backward()

# 看看梯度有多大
grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
print(f"裁剪前梯度范数: {grad_norm_before:.2f}")

# 假装重来一次
optimizer.zero_grad()
output = model(x)
loss = nn.MSELoss()(output, y)
loss.backward()

# 裁剪梯度到范数 <= 1.0
grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"裁剪后梯度范数: {grad_norm_after:.2f}")

# 看看实际的梯度值变小了
actual_norm = 0
for p in model.parameters():
    if p.grad is not None:
        actual_norm += p.grad.norm().item() ** 2
actual_norm = actual_norm ** 0.5
print(f"裁剪后实际梯度范数: {actual_norm:.4f}")

print()

# ========== 梯度检查（Gradient Check）==========
# 用数值微分验证 autograd 算的对不对
# 调试自定义 autograd Function 时很有用

print("=== 梯度检查（数值验证）===")

# 自己写个简单版本，不用 torch.autograd.gradcheck
def numerical_gradient(f, x, eps=1e-5):
    """用有限差分法算数值梯度"""
    grad = torch.zeros_like(x)
    for i in range(x.shape[0]):
        x_plus = x.clone()
        x_plus[i] += eps
        x_minus = x.clone()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def my_func(x):
    """f(x) = x[0]² * x[1] + sin(x[2])"""
    return x[0]**2 * x[1] + torch.sin(x[2])

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# autograd 的结果
y = my_func(x)
y.backward()
grad_auto = x.grad.clone()

# 数值微分的结果
grad_num = numerical_gradient(my_func, x.detach())

print(f"Autograd:  {grad_auto}")
print(f"数值微分:  {grad_num}")
print(f"最大误差: {(grad_auto - grad_num).abs().max().item():.2e}")

# 也可以用 PyTorch 自带的（更严格）
from torch.autograd import gradcheck
x_check = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64)
# gradcheck 需要 float64 精度
test = gradcheck(lambda x: x[0]**2 * x[1] + torch.sin(x[2]), (x_check,))
print(f"PyTorch gradcheck 结果: {test}")

print()

# ========== torch.no_grad() vs torch.inference_mode() ==========
# 推理时不需要梯度，关掉可以省内存和加速

print("=== no_grad() vs inference_mode() ===")

x = torch.randn(100, 100, requires_grad=True)

# no_grad：不记录计算图，但 tensor 还是"正常"的
with torch.no_grad():
    y = x * 2
    print(f"no_grad 下: y.requires_grad = {y.requires_grad}")

# inference_mode：更激进的优化，PyTorch 1.9+ 推荐用这个做推理
with torch.inference_mode():
    y = x * 2
    print(f"inference_mode 下: y.requires_grad = {y.requires_grad}")

print("""
inference_mode 比 no_grad 快，因为它内部做了更多优化。
但有限制：inference_mode 下创建的 tensor 不能在外面用于需要梯度的计算。
推理用 inference_mode，训练中临时不算梯度用 no_grad。
""")

# ========== 冻结部分参数（Freeze Layers）==========
# 迁移学习中，你经常需要冻结 backbone，只训练最后几层

print("=== 冻结参数 ===")

# 假装这是个预训练模型
class PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
        )
        self.head = nn.Linear(20, 5)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

model = PretrainedModel()

# 冻结 backbone：把 requires_grad 设为 False
for param in model.backbone.parameters():
    param.requires_grad = False

# 只有 head 的参数需要更新
trainable = [name for name, p in model.named_parameters() if p.requires_grad]
frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
print(f"可训练参数: {trainable}")
print(f"冻结参数: {frozen}")

# 另一个好处：冻结的层不算梯度，省显存
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

x = torch.randn(4, 10)
output = model(x)
loss = output.sum()
loss.backward()

print(f"backbone 第一层权重梯度: {model.backbone[0].weight.grad}")  # None！
print(f"head 权重梯度不为空: {model.head.weight.grad is not None}")

print()
print("=== 总结 ===")
print("""
1. 梯度累加：显存不够就累加，loss 记得除步数
2. 梯度裁剪：clip_grad_norm_ 防止梯度爆炸
3. 梯度检查：调试自定义 backward 的利器
4. inference_mode > no_grad（推理时）
5. 冻结参数：requires_grad=False + 只把需要训练的参数给 optimizer
""")
