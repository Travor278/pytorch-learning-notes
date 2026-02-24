# 06_gradient_accumulation_and_tricks.py
# 训练工程中的核心梯度技巧
#
# 主题：
#   1. 梯度累加（Gradient Accumulation）
#   2. 梯度裁剪（Gradient Clipping）
#   3. 梯度数值验证（Gradient Check）
#   4. torch.no_grad() vs torch.inference_mode()
#   5. 参数冻结（Freeze Layers）

import torch
import torch.nn as nn

# ========== 1. 梯度累加（Gradient Accumulation）==========
#
# 问题背景：
#   批量大小（batch size）在理论上影响优化轨迹——大 batch 梯度估计方差低，
#   但受限于 GPU 显存，往往无法直接设大。
#
# 解决方案：
#   将 batch_size=B 拆成 k 个 mini-batch（每个大小 B/k），
#   累加 k 步梯度后再调用一次 optimizer.step()。
#   效果近似等价于 batch_size=B 的单步更新。
#
# 严格等价条件：
#   ① 损失函数对样本线性可分解（MSE、CE 均满足）；
#   ② 累加期间无 Batch Normalization（BN 的统计量基于 mini-batch，
#      k 步累加并不等价于在 B 个样本上计算 BN，会引入统计偏差）。
#
# 延伸参考：Smith et al., "Don't Decay the Learning Rate, Increase the Batch Size"
#           ICLR 2018——等效大 batch 训练时学习率亦应相应缩放。

print("=== 梯度累加 ===")

model     = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

data   = torch.randn(32, 10)
target = torch.randn(32, 1)

# 对照组：完整 batch=32 的单步梯度
optimizer.zero_grad()
loss_full = nn.MSELoss()(model(data), target)
loss_full.backward()
grad_full = model.weight.grad.clone()
print(f"完整 batch=32 的梯度（前5个）: {grad_full[0, :5]}")

# 实验组：4 步累加，每步 batch=8
optimizer.zero_grad()
k          = 4
mini_size  = 8

for i in range(k):
    s, e  = i * mini_size, (i + 1) * mini_size
    # 关键：损失须除以 k，使累加后的梯度等效于在 B 个样本上的均值梯度。
    # 若不除 k，等同于把学习率放大了 k 倍。
    loss_mini = nn.MSELoss()(model(data[s:e]), target[s:e]) / k
    loss_mini.backward()   # .grad 自动累加，不清零

grad_accum = model.weight.grad.clone()
print(f"累加 4×8 的梯度（前5个）: {grad_accum[0, :5]}")
print(f"最大绝对误差（浮点精度）: {(grad_full - grad_accum).abs().max().item():.2e}")

optimizer.step()   # 累加完成后统一更新
print()

# ========== 2. 梯度裁剪（Gradient Clipping）==========
#
# 梯度爆炸（gradient explosion）在 RNN/LSTM/Transformer 中尤为常见。
# 成因：在时间/深度方向上反复与权重矩阵相乘，若谱范数 > 1 则梯度范数指数增长。
# 参考：Pascanu et al., "On the difficulty of training recurrent neural networks"
#       ICML 2013——正式引入梯度裁剪作为训练稳定化策略。
#
# clip_grad_norm_(params, max_norm, norm_type=2.0) 的数学操作：
#   ① 计算全局梯度范数：g_norm = (Σ_p ‖∇p‖²)^(1/2)
#   ② 若 g_norm > max_norm，则对所有参数梯度缩放：
#      ∇p ← ∇p · (max_norm / g_norm)
#   这保持了各参数梯度的相对比例，仅缩放全局幅度。

print("=== 梯度裁剪 ===")

model     = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_large = torch.randn(1, 10) * 100   # 极大输入值制造大梯度
y_target = torch.tensor([[1.0]])

optimizer.zero_grad()
loss = nn.MSELoss()(model(x_large), y_target)
loss.backward()

# 裁剪前全局梯度范数
norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
print(f"裁剪前全局梯度 L2 范数: {norm_before:.2f}")

# 重新计算梯度后裁剪到 max_norm=1.0
optimizer.zero_grad()
loss = nn.MSELoss()(model(x_large), y_target)
loss.backward()

norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"裁剪后全局梯度 L2 范数: {norm_after:.2f}（上报的是裁剪前的实际范数）")

# 验证裁剪是否生效
actual = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None) ** 0.5
print(f"裁剪后实际梯度 L2 范数: {actual:.4f}  （应 ≤ 1.0）")
print()

# ========== 3. 梯度数值验证（Gradient Check）==========
#
# 梯度验证的理论依据：中心差商（Central Difference）近似一阶导数：
#
#   ∂f/∂xᵢ ≈ [f(x + εeᵢ) - f(x - εeᵢ)] / (2ε)
#
# 该近似的截断误差为 O(ε²)，远优于前向差商的 O(ε)。
# 相对误差标准：
#   |autograd - numerical| / max(|autograd|, |numerical|, ε) < threshold
#
# 注意事项：
#   ① 必须使用 float64（double），float32 的数值精度不足以验证高精度梯度；
#   ② ε 通常取 1e-5~1e-6，过小则数值精度不稳定，过大则截断误差增大；
#   ③ 对含有随机性（Dropout）或不连续点（max、ReLU 在零点）的函数需特别注意。

print("=== 梯度数值验证 ===")

def numerical_gradient(f, x, eps=1e-5):
    """中心差商数值微分，返回与 x 同形状的梯度估计"""
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_p, x_m  = x.clone(), x.clone()
        x_p.view(-1)[i] += eps
        x_m.view(-1)[i] -= eps
        grad.view(-1)[i] = (f(x_p) - f(x_m)) / (2 * eps)
    return grad

def my_func(x):
    """f(x) = x₀² · x₁ + sin(x₂)"""
    return x[0]**2 * x[1] + torch.sin(x[2])

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = my_func(x)
y.backward()
grad_auto = x.grad.clone()

grad_num = numerical_gradient(my_func, x.detach())

print(f"Autograd:    {grad_auto}")
print(f"数值微分:    {grad_num}")
print(f"最大相对误差: {(grad_auto - grad_num).abs().max().item():.2e}")

# torch.autograd.gradcheck：PyTorch 官方实现，更严格的双精度验证
from torch.autograd import gradcheck
x_f64 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64)
result = gradcheck(lambda x: x[0]**2 * x[1] + torch.sin(x[2]), (x_f64,))
print(f"PyTorch gradcheck 通过: {result}")
print()

# ========== 4. no_grad() vs inference_mode() ==========
#
# torch.no_grad()：
#   关闭 requires_grad 追踪，tensor 不进入计算图。
#   但 version counter 仍更新，某些 autograd 检查仍有效。
#   适用于：参数更新步骤、混合 autograd/非 autograd 代码中的局部禁用。
#
# torch.inference_mode()（PyTorch 1.9+）：
#   在 no_grad 基础上进一步禁用 version counter 更新，
#   允许更多内核优化（如避免记录 view 操作的历史）。
#   限制：inference_mode 下创建的 tensor 不能被任何 autograd 操作消费，
#         否则抛出 RuntimeError（有别于 no_grad 的静默容许）。
#   适用于：纯推理路径（eval loop、部署服务），性能更优。
#
# 规则：推理用 inference_mode，训练循环中临时关闭梯度用 no_grad。

print("=== no_grad() vs inference_mode() ===")

x = torch.randn(4, 4, requires_grad=True)

with torch.no_grad():
    y_no = x * 2
    print(f"no_grad        下 y.requires_grad = {y_no.requires_grad}")

with torch.inference_mode():
    y_inf = x * 2
    print(f"inference_mode 下 y.requires_grad = {y_inf.requires_grad}")

print("""
inference_mode 的额外限制：
  y_inf 不能被 backward/grad 使用（即使离开上下文管理器）。
  若不小心在 inference_mode 输出上调用了需要梯度的操作，
  PyTorch 会在运行时报错，而非静默产生错误结果。
""")

# ========== 5. 参数冻结（Freeze Layers）==========
#
# 迁移学习（Transfer Learning）的标准范式：
#   ① 加载在大规模数据集（如 ImageNet）上预训练的 backbone；
#   ② 冻结 backbone 权重（requires_grad=False），仅更新 head 层；
#   ③ Fine-tune 阶段可逐步解冻更多层（Gradual Unfreezing，Howard & Ruder 2018）。
#
# 显存收益：冻结层不仅不更新，其激活值的梯度也不会被计算和存储，
# 可显著降低反向传播的显存峰值（对于深度 backbone 尤为显著）。

print("=== 参数冻结（迁移学习场景）===")

class PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 20),
        )
        self.head = nn.Linear(20, 5)

    def forward(self, x):
        return self.head(self.backbone(x))

model = PretrainedModel()

# 冻结 backbone：requires_grad=False 阻止梯度传播至此处及更早的节点
for param in model.backbone.parameters():
    param.requires_grad = False

trainable = [name for name, p in model.named_parameters() if p.requires_grad]
frozen    = [name for name, p in model.named_parameters() if not p.requires_grad]
print(f"可训练参数: {trainable}")
print(f"冻结参数:   {frozen}")

# 优化器只接受可训练参数，避免为冻结参数分配动量缓冲区（节省显存）
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

x_in  = torch.randn(4, 10)
loss  = model(x_in).sum()
loss.backward()

print(f"backbone[0].weight.grad = {model.backbone[0].weight.grad}  （冻结，无梯度）")
print(f"head.weight.grad 非空: {model.head.weight.grad is not None}  （可训练，有梯度）")

print()
print("=== 总结 ===")
print("""
1. 梯度累加：k 步 mini-batch 模拟 B 大 batch；loss 必须除 k；BN 层下不严格等价
2. 梯度裁剪：clip_grad_norm_ 按比例缩放全局梯度，保持方向不变；防止 RNN 梯度爆炸
3. 梯度验证：中心差商 O(ε²) 精度；必须用 float64；是调试自定义 backward 的标准工具
4. inference_mode > no_grad（推理性能）；no_grad 更灵活（允许 autograd 混用）
5. 冻结层：requires_grad=False + 仅将可训练参数传给 optimizer，兼顾正确性与显存效率
""")
