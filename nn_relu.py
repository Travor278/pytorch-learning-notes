import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


input = torch.tensor([[-1, -0.5],
                      [-1, 3]])
# ReLU函数的作用是：将输入中的负数部分置为0，正数部分保持不变。
# Sigmoid函数的作用是：将输入映射到0和1之间，适用于二分类问题的输出层。
# 这个操作可以帮助我们引入非线性，使得神经网络能够学习更复杂的函数。
output = torch.reshape(input, (-1, 1, 2, 2)) # 将输入张量的形状调整为 (batch_size, channels, height, width)，这里 batch_size=1，channels=1，height=2，width=2
print(output.shape)

dataset = torchvision.datasets.CIFAR10(root="./dataset2", train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64) # 创建一个数据加载器，批量加载 CIFAR10 数据集

class Moli(nn.Module): # 定义一个名为 Moli 的神经网络类，继承自 nn.Module
    def __init__(self): # 初始化函数，定义网络的层
        super(Moli, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input): # 前向传播函数，定义网络的计算过程
        output = self.sigmoid1(input) # 对输入进行 Sigmoid 激活函数的计算，得到输出
        return output

moli = Moli()

writer = SummaryWriter("logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = moli(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
# tensorboard --logdir=logs_maxpool --samples_per_plugin images=9999
# 在浏览器中打开 http://localhost:6006/ 查看结果