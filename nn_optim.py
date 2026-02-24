import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Flatten
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10(root="./dataset2", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Moli(nn.Module):
    def __init__(self):
        super(Moli, self).__init__() # 继承父类的 __init__ 方法
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),  # [B, 3, 32, 32] -> [B, 32, 32, 32]
            MaxPool2d(2),                 # [B, 32, 32, 32] -> [B, 32, 16, 16]
            Conv2d(32, 32, 5, padding=2), # [B, 32, 16, 16] -> [B, 32, 16, 16]
            MaxPool2d(2),                 # [B, 32, 16, 16] -> [B, 32, 8, 8]
            Conv2d(32, 64, 5, padding=2), # [B, 32, 8, 8] -> [B, 64, 8, 8]
            MaxPool2d(2),                 # [B, 64, 8, 8] -> [B, 64, 4, 4]
            Flatten(),                    # [B, 64, 4, 4] -> [B, 1024]
            Linear(1024, 64),             # [B, 1024] -> [B, 64]
            Linear(64, 10)                # [B, 64] -> [B, 10]
        )

    def forward(self, x):
        x = self.model1(x)
        return x
    
loss = nn.CrossEntropyLoss()
moli = Moli()
optim = torch.optim.SGD(moli.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = moli(imgs)
        result_loss = nn.CrossEntropyLoss()(outputs, targets)
        optim.zero_grad()  # 梯度清零
        result_loss.backward()  # 反向传播计算梯度
        optim.step()  # 更新参数
        running_loss = running_loss + result_loss
    print(running_loss)