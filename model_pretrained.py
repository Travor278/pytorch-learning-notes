import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet(root="./data_image_net", split="train", download=True, 
#                                            transform=torchvision.transforms.ToTensor())
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(root="./dataset2", train=False, download=True, 
                                          transform=torchvision.transforms.ToTensor())

# 在classifier中添加一个线性层，输入特征数为1000，输出特征数为10
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
# 替换掉原来的线性层，输入特征数为4096，输出特征数为10
vgg16_false.classifier[6] = nn.Linear(4096, 10) 
print(vgg16_false)