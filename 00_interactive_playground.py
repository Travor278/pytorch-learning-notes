#%%
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version in build: {torch.version.cuda}")

#%%
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * 2 + 1).pow(2).sum()
y.backward()

print("x:", x)
print("y:", y)
print("x.grad:", x.grad)

#%%
def f(v):
    return torch.stack([
        v[0] ** 2 + v[1],
        v[1] ** 2 + v[2],
        v[2] * v[0],
    ])

from torch.autograd.functional import jacobian

v = torch.tensor([1.0, 2.0, 3.0])
j = jacobian(f, v)
print("Jacobian:\n", j)

#%%
print("Done. Add more #%% cells and run with Shift+Enter in VS Code.")
