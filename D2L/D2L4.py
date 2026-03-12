#%%
import torch
x = torch.arange(4.0)
x
# %%
x.requires_grad_(True) # 标记 x 需要梯度
x.grad # x 的梯度，初始为 None
# %%
y = 2 * torch.dot(x, x)
y
#%%
y.backward() # 反向传播，计算 x 的梯度
x.grad # x 的梯度
# %%
x.grad == 4 * x.data # 验证梯度是否正确
# %%
# 在默认情况下，PyTorch 会累积梯度，因此在每次反向传播前需要清零
x.grad.zero_() # 清零 x 的梯度
y = x.sum()
y.backward() # 反向传播，计算 x 的梯度
x.grad # x 的梯度
# %%
# 对非标量调用‘backward()’需要传入一个 gradient 参数，该参数指定了反向传播的初始梯度
x.grad.zero_() # 清零 x 的梯度
y = x * x
# 等价于y.backward(torch.ones(len(x)))，即每个元素的初始梯度为1
y.sum().backward() # 反向传播，计算 x 的梯度
x.grad # x 的梯度
# %%
x.grad.zero_()
y = x * x
u = y.detach() # detach() 从计算图中分离 y，使其不再需要梯度
z = u * x
z.sum().backward() # 反向传播，计算 x 的梯度
x.grad == u
# %%
x.grad.zero_()
y.sum().backward() # 反向传播，计算 x 的梯度
x.grad == 2 * x.data
# %%
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

a.grad == d / a # 链式法则：d/dx f(g(x)) = f'(g(x)) * g'(x)