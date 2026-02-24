# 07_custom_autograd_function.py
# 自定义 Autograd Function：手写 forward 与 backward
#
# 使用场景：
#   ① PyTorch 未实现的运算（非标准激活函数、特殊层）；
#   ② 需要用 C++/CUDA 实现高效 forward，但 autograd 无法自动推导 backward；
#   ③ 需要在反向传播中注入特殊逻辑（量化训练的 STE、对抗训练的梯度反转）；
#   ④ 出于效率将多个运算融合（fused kernel），绕过逐算子的中间 tensor。
#
# 实现约定（继承 torch.autograd.Function）：
#   - forward / backward 均为 @staticmethod
#   - ctx（Context）对象用于在 forward 与 backward 间传递状态：
#       ctx.save_for_backward(*tensors)  —— 存 tensor（必须用此接口，否则内存管理异常）
#       ctx.arbitrary_attr = value       —— 存非 tensor 的标量/超参数等
#   - backward 的返回值数量必须与 forward 的输入参数数量严格一致；
#     不需要梯度的输入对应返回 None。

import torch
from torch.autograd import Function, gradcheck

# ========== 例 1：自定义 ReLU ==========
# ReLU(x) = max(0, x)
# 前向：y = x · 1(x > 0)
# 反向：∂L/∂x = ∂L/∂y · 1(x > 0)   ← 局部梯度为指示函数
# 在 x=0 处，梯度严格来说不存在（左导 0，右导 1），
# PyTorch 官方约定 ReLU'(0) = 0（次梯度选择），此处保持一致。

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)   # 保存原始输入，供 backward 生成掩码
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output：从后续节点传来的梯度 ∂L/∂y（链式法则上游部分）
        局部梯度（Jacobian 对角元）：dy/dx = 1 if x > 0 else 0
        返回：grad_input = grad_output ⊙ mask（逐元素乘，非矩阵乘）
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0       # x=0 处取次梯度 0
        return grad_input

print("=== 自定义 ReLU ===")
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

y_custom = MyReLU.apply(x)     # 必须用 .apply()，不能直接实例化调用
y_custom.sum().backward()
print(f"输入:            {x.data}")
print(f"MyReLU 输出:     {y_custom.data}")
print(f"MyReLU 梯度:     {x.grad}  （期望 [0,0,0,1,1]）")

# 与官方 ReLU 对比
x2 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
torch.relu(x2).sum().backward()
print(f"官方 ReLU 梯度:  {x2.grad}")
print()

# ========== 例 2：Straight-Through Estimator（STE）==========
#
# 问题：二值化（sign 函数）在几乎所有点梯度为 0（阶跃函数），
#       若直接反传，梯度归零，网络无法学习。
#
# STE（Straight-Through Estimator）的思路（Bengio et al., 2013）：
#   前向：使用真实的不可微运算（sign、round、argmax 等）；
#   反向：用恒等映射（或其他平滑近似）代替真实导数，直通梯度。
#
# 理论支撑：将量化算子视为"带可控噪声的随机化操作"，
# STE 是该随机化模型的无偏估计量在高温极限下的退化形式。
# 参考：Bengio et al., "Estimating or Propagating Gradients Through
#       Stochastic Neurons for Conditional Computation", arXiv 2013.
# 应用：BinaryConnect, XNOR-Net, DoReFa-Net 等量化神经网络均以 STE 为基础。

class BinarizeSTE(Function):
    @staticmethod
    def forward(ctx, input):
        # 前向：真实量化操作
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向：STE —— 梯度直通，视 sign 为恒等映射
        # 更保守的变体：只在 |x| ≤ 1 时直通（Hinton 2012 讲义中的版本），
        # 即 return grad_output * (input.abs() <= 1).float()
        # 此处使用最基础版本（无截断）
        return grad_output

print("=== Straight-Through Estimator（量化训练）===")
x = torch.tensor([-0.5, 0.3, -0.8, 0.1, 0.9], requires_grad=True)
y = BinarizeSTE.apply(x)
print(f"输入:           {x.data}")
print(f"二值化输出:     {y.data}   （前向：sign 操作）")

loss = (y * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])).sum()
loss.backward()
print(f"梯度（STE）:    {x.grad}   （反向：直通，非零）")
print()

# ========== 例 3：多输入自定义 Function + gradcheck 验证 ==========
#
# z = x²y + y³
# ∂z/∂x = 2xy
# ∂z/∂y = x² + 3y²
#
# gradcheck 的验证流程：
#   对每个输入的每个元素，用中心差商计算数值梯度，与 autograd 梯度比较。
#   相对误差阈值默认 1e-5。必须使用 float64，因为 float32 的数值精度
#   与 eps≈1e-6 的差商不匹配，会导致误判。

class MyOp(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x ** 2 * y + y ** 3

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        # 链式法则：grad_input = grad_output · (∂z/∂input)
        grad_x = grad_output * (2 * x * y)
        grad_y = grad_output * (x ** 2 + 3 * y ** 2)
        # 返回值数量 == forward 输入参数数量（此处为 2）
        return grad_x, grad_y

print("=== 多输入自定义 Function ===")
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = MyOp.apply(x, y)
z.backward()

print(f"z = x²y + y³ = {z.item():.1f}  （期望 {2**2*3 + 3**3}）")
print(f"∂z/∂x = 2xy  = {x.grad.item():.1f}  （期望 {2*2*3}）")
print(f"∂z/∂y = x²+3y² = {y.grad.item():.1f}  （期望 {2**2 + 3*3**2}）")

# gradcheck：float64 + 双精度中心差商，严格验证 backward 实现正确性
x64 = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
y64 = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
print(f"gradcheck 验证: {gradcheck(MyOp.apply, (x64, y64), eps=1e-6)}")
print()

# ========== 例 4：ctx 的使用规范 ==========
#
# save_for_backward 的设计原因：
#   PyTorch 在内部对 saved tensor 进行版本追踪——若 forward 保存的 tensor
#   在 backward 调用前被 in-place 修改，PyTorch 会检测到版本号变化并抛出错误，
#   而非静默地使用被污染的数据。直接用 ctx.attr = tensor 绕过此保护。
#
# 非 tensor 的超参数（int, float, bool）直接赋给 ctx 属性，不需要 save_for_backward。
# backward 中不需要梯度的对应输入（如 scale）返回 None，
# PyTorch 据此决定是否继续向更早的节点传播梯度。

class ScaleFunction(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)        # tensor：受版本保护
        ctx.scale = scale               # 非 tensor：直接存为属性
        return x * scale

    @staticmethod
    def backward(ctx, grad_output):
        x,    = ctx.saved_tensors
        scale = ctx.scale
        # scale 是常量超参数，无需梯度，对应返回 None
        return grad_output * scale, None

print("=== ctx 使用规范 ===")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = ScaleFunction.apply(x, 5.0)
y.sum().backward()

print(f"输入:    {x.data}")
print(f"输出:    {y.data}  （x × 5）")
print(f"梯度:    {x.grad}  （期望 [5, 5, 5]）")

print()
print("=== 总结 ===")
print("""
1. 继承 Function，实现 @staticmethod forward / backward
2. ctx.save_for_backward 仅存 tensor（含版本保护）；非 tensor 用 ctx.attr 直接赋值
3. backward 返回数量 = forward 输入参数数量，不需要梯度的返回 None
4. 调用方式必须是 MyFunc.apply(args)，不能直接调用 forward
5. STE 是量化网络训练的核心：前向量化，反向直通（Bengio et al., 2013）
6. gradcheck 是验证自定义 backward 正确性的标准工具，必须使用 float64
""")
