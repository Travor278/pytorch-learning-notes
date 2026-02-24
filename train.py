import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Moli

train_data = torchvision.datasets.CIFAR10(root='./dataset2', train=True, download=True, 
                                         transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./dataset2', train=False, download=True, 
                                        transform=torchvision.transforms.ToTensor())

# length of dataset
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
moli = Moli()

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()

# 优化器(随机梯度下降)
learning_rate = 0.01
optimizer = torch.optim.SGD(moli.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))
    # 训练步骤开始
    moli.train()  # 将模型设置为训练模式
    for data in train_dataloader:
        imgs, targets = data
        outputs = moli(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    moli.eval()  # 将模型设置为评估模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = moli(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(moli, "moli_{}.pth".format(i))
    # torch.save(moli.state_dict(), "moli_{}.pth".format(i)) # 官方推荐的保存模型方式，保存模型参数
    print("模型已保存")

writer.close()