# 07_custom_autograd_function.py
# 用途：自定义 autograd Function —— 自己写前向和反向传播
#
# 什么时候需要自定义？
# 1. PyTorch 没有实现的数学运算（比如某些论文提出的奇怪激活函数）
# 2. 你想用 C/C++ 实现高效的 forward，但 autograd 推不出 backward
# 3. 想在反向传播中做一些特殊操作（比如 Straight-Through Estimator）
#
# 这是进阶内容，但理解它能让你真正明白 PyTorch autograd 在底层做了什么。

import torch
from torch.autograd import Function

# ========== 例1：自己实现 ReLU ==========
# 先拿最简单的 ReLU 练手，你知道 ReLU(x) = max(0, x)
# forward: y = x if x > 0 else 0
# backward: dy/dx = 1 if x > 0 else 0

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        """
        ctx 是一个上下文对象，用来在 forward 和 backward 之间传递信息。
        你可以把 forward 中需要用到的东西存进去，backward 再取出来。
        """
        # 存住 input，backward 要用它判断哪些位置 > 0
        ctx.save_for_backward(input)
        # forward 逻辑就是把负数变成 0
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output 是从后面传过来的梯度（链式法则的上一环）。
        我们要算的是：grad_input = grad_output * (local gradient)
        """
        # 取出 forward 存的 input
        input, = ctx.saved_tensors
        # ReLU 的局部梯度：x > 0 时为 1，否则为 0
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 测一下
print("=== 自定义 ReLU ===")

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

# 用我们自己的 ReLU
my_relu = MyReLU.apply  # .apply 是调用自定义 Function 的方式
y = my_relu(x)
loss = y.sum()
loss.backward()

print(f"输入: {x.data}")
print(f"MyReLU 输出: {y.data}")
print(f"梯度: {x.grad}")
# 梯度应该是 [0, 0, 0, 1, 1]（负数位置梯度为0）

# 对比 PyTorch 自带的
x2 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y2 = torch.relu(x2)
y2.sum().backward()
print(f"官方 ReLU 梯度: {x2.grad}")

print()

# ========== 例2：Straight-Through Estimator (STE) ==========
# 这是量化（Quantization）训练中的核心技巧。
#
# 问题：二值化（binarize）操作不可导（阶跃函数梯度为0）
# 解决：forward 做二值化，backward 直接传梯度（假装没有二值化）
# 这就是 STE（Straight-Through Estimator）

class Binarize(Function):
    @staticmethod
    def forward(ctx, input):
        # 前向：大于0输出+1，否则-1
        return input.sign()
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向：直接把梯度原样传回去（straight-through）
        # 理论上 sign() 的梯度是 0（几乎处处为0），
        # 但如果真传0回去，网络就学不动了
        return grad_output  # 骗 autograd：假装 forward 是恒等函数

print("=== Straight-Through Estimator ===")

x = torch.tensor([-0.5, 0.3, -0.8, 0.1, 0.9], requires_grad=True)
binarize = Binarize.apply

y = binarize(x)
print(f"输入: {x.data}")
print(f"二值化输出: {y.data}")  # [-1, 1, -1, 1, 1]

# 假设后面有个 loss
loss = (y * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])).sum()
loss.backward()

print(f"梯度（直通估计）: {x.grad}")
# 梯度不是0！因为 STE 绕过了 sign 的零梯度

print()

# ========== 例3：带两个输入的自定义 Function ==========
# 自定义一个 z = x² * y + y³ 的运算

class MyOp(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x ** 2 * y + y ** 3
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        # ∂z/∂x = 2xy
        grad_x = grad_output * (2 * x * y)
        # ∂z/∂y = x² + 3y²
        grad_y = grad_output * (x ** 2 + 3 * y ** 2)
        # 返回的数量必须跟 forward 的输入参数一一对应
        return grad_x, grad_y

print("=== 多输入自定义 Function ===")

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = MyOp.apply(x, y)
z.backward()

print(f"z = x²y + y³ = {z.item()}")
print(f"∂z/∂x = 2xy = {x.grad.item()} (期望: {2*2*3})")
print(f"∂z/∂y = x² + 3y² = {y.grad.item()} (期望: {4+27})")

# 用 gradcheck 验证我们写的 backward 对不对
from torch.autograd import gradcheck
x_test = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
y_test = torch.tensor(3.0, requires_grad=True, dtype=torch.float64)
test = gradcheck(MyOp.apply, (x_test, y_test), eps=1e-6)
print(f"gradcheck 验证: {test}")

print()

# ========== 例4：save_for_backward 的注意事项 ==========
print("=== ctx 的用法细节 ===")

class DemoFunction(Function):
    @staticmethod
    def forward(ctx, x, scale):
        # save_for_backward 只能存 tensor
        ctx.save_for_backward(x)
        # 非 tensor 的东西用 ctx 的属性存
        ctx.scale = scale
        return x * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale = ctx.scale  # 取出非 tensor 的值
        return grad_output * scale, None  # scale 不需要梯度，返回 None

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = DemoFunction.apply(x, 5.0)  # scale=5
y.sum().backward()
print(f"输入: {x.data}")
print(f"输出 (x * 5): {y.data}")
print(f"梯度: {x.grad}")  # [5, 5, 5]

print()
print("=== 总结 ===")
print("""
1. 自定义 Function 需要实现 forward 和 backward 两个 staticmethod
2. ctx.save_for_backward() 存 tensor，ctx.xxx 存其他东西
3. backward 返回值的数量 = forward 输入参数的数量（不需要梯度的返回 None）
4. 用 gradcheck 可以验证你的 backward 写对了没有
5. STE（Straight-Through Estimator）是量化训练的基础
6. 调用方式是 MyFunction.apply(args)，不是 MyFunction(args)
""")
