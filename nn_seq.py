import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter

class Moli(nn.Module):
    # def __init__(self):
    #     super(Moli, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 32, 5, padding=2)  # [B, 3, 32, 32] -> [B, 32, 32, 32]
    #     self.maxpool1 = nn.MaxPool2d(2)              # [B, 32, 32, 32] -> [B, 32, 16, 16]
    #     self.conv2 = nn.Conv2d(32, 32, 5, padding=2) # [B, 32, 16, 16] -> [B, 32, 16, 16]
    #     self.maxpool2 = nn.MaxPool2d(2)              # [B, 32, 16, 16] -> [B, 32, 8, 8]
    #     self.conv3 = nn.Conv2d(32, 64, 5, padding=2) # [B, 32, 8, 8] -> [B, 64, 8, 8]
    #     self.maxpool3 = nn.MaxPool2d(2)              # [B, 64, 8, 8] -> [B, 64, 4, 4]
    #     self.flatten = nn.Flatten()                  # [B, 64, 4, 4] -> [B, 1024]
    #     self.linear1 = Linear(1024, 64)              # [B, 1024] -> [B, 64]
    #     self.linear2 = Linear(64, 10)                # [B, 64] -> [B, 10]
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
    
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.maxpool1(x)
    #     x = self.conv2(x)
    #     x = self.maxpool2(x)
    #     x = self.conv3(x)
    #     x = self.maxpool3(x)
    #     x = self.flatten(x)
    #     x = self.linear1(x)
    #     x = self.linear2(x)
    #     return x
    def forward(self, x):
        x = self.model1(x)
        return x

moli = Moli()
print(moli)

# 测试前向传播
input = torch.ones((64, 3, 32, 32))  # 模拟一个 batch 的输入
output = moli(input)
print(output.shape)  # [64, 10]

writer = SummaryWriter("logs_seq")
writer.add_graph(moli, input)       # 可视化整个模型结构
writer.close()
# tensorboard --logdir=logs_seq
# torch.onnx.export(moli, input, "moli_model.onnx") # https://netron.app/