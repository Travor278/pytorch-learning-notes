import torch
# 现在也可以直接使用 torch.load 加载模型了，之前的 torch.load 只能加载模型参数
# from model_save import Moli # 导入 Moli 类定义，以便加载模型时能够找到该类
import torchvision
# from torch import nn

# 方式1 -> 保存方式1：加载模型
# PyTorch 2.6 把 torch.load 的 weights_only 默认值从 False 改成了 True
model = torch.load("vgg16_method1.pth", weights_only=False)
print(model)

# 方式2 -> 保存方式2：加载模型参数
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(model)

# 陷阱1
# class Moli(nn.Module):
#     def __init__(self):
#         super(Moli, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

#     def forward(self, x):
#         return self.conv1(x)

# moli = Moli()
model = torch.load("moli_mothod1.pth", weights_only=False)
print(model)