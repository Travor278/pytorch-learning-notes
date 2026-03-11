#%%
import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x - y, x * y, x / y, x**y
# %%
x = torch.arange(4.0)
x
# %%
x[3]
# %%
len(x), x.shape
# %%
A = torch.arange(20).reshape(5, 4)
A
# %%
A.T, A.T.shape
# %%
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B, B.sum(), B.sum(axis=0), B.sum(axis=1), B == B.T
# %%
X = torch.arange(24).reshape(2, 3, 4)
X, X.sum(), X.sum(axis=0), X.sum(axis=1), X.sum(axis=2)
# %%
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
A, B, A == B, A + B, A - B, A * B, A / (B + 1e-5)
# %%
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
# %%
x = torch.arange(4, dtype=torch.float32)
x, x.sum(), x.sum().item() # item()将单元素张量转换为Python数值
# %%
A.mean(), A.sum() / A.numel() # numel()返回张量中的元素数量
# %%
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
# %%
sum_A = A.sum(axis=1, keepdim=True) # keepdim=True保持原有维度
sum_A, sum_A.shape
# %%
A / sum_A
# %%
A.cumsum(axis=0) # 沿着轴0累积求和
# %%
A.cumsum(axis=1) # 沿着轴1累积求和
# %%
y = torch.ones(4, dtype=torch.float32)
x, y, torch.dot(x, y), x.sum() # dot()计算两个张量的点积
# %%
torch.norm(x) # norm()计算张量的范数
# %%
A.shape, x.shape, torch.mv(A, x) # mv()计算矩阵和向量的乘积
# %%
B = torch.ones(4, 3)
A.shape, B.shape, torch.mm(A, B) # mm()计算矩阵和矩阵的乘积
# %%
u = torch.tensor([-3.0, 4.0])
torch.norm(u), torch.sqrt((u**2).sum()) # 计算向量的范数
# %%
torch.abs(u), torch.abs(u).sum() # 计算向量的L1范数