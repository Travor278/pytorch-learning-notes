#%%
import torch
x = torch.arange(12)
#%%
print(x)
# %%
print(x.shape)
# %%
print(x.numel())
# %%
X = x.reshape(3, 4)
print(X)
# %%
torch.zeros((2, 3, 4))
# %%
torch.ones((2, 3, 4))
# %%
torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]])
# %%
torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]]).shape
# %%
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]).shape
# %%
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x**y
# %%
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1), torch.cat((X, Y), dim=-1), torch.cat((X, Y), dim=-1).shape
# %%
x == y
# %%
X.sum(), X.sum(axis=0), X.sum(axis=1)
# %%
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
# %%
a + b
# %%
X[-1], X[1:3], X[0:3:2]
# %%
X[1, 2] = 9
X
# %%
X[0:2, :] = 12
X
# %%
before = id(Y)
Y = Y + X
id(Y) == before
# %%
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
# %%
before = id(X)
X += Y
id(X) == before
# %%
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
# %%
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)