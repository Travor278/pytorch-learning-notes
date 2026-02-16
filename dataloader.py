import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10(root='./dataset2', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
# batch_size=64: 每个批次加载64个样本
# shuffle=True: 打乱数据集
# num_workers=0: 使用主线程加载数据
# drop_last=True: 如果数据集大小不能被批次大小整除，丢弃最后一个不完整的批次

# 测试数据集中第一个样本的图像和标签
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(img.shape)
        # print(target)
        # break
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step += 1

writer.close()