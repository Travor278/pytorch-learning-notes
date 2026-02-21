import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.0, 2.0, 5.0])

# 将输入和目标调整为 (batch_size, num_features) 的形状，以适应 L1Loss 的要求
inputs = torch.reshape(inputs, (1, 1, 1, 3)) # (N, C, H, W)
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')  # reduction='mean' 是默认值，表示返回所有元素的平均损失
                                # reduction='sum' 则返回所有元素的损失之和
result = loss(inputs, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))  # (N, C)
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)