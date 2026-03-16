#%% # 图片分类数据集
import torch
import torchvision
import time
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt
#%%
# 通过ToTensor实例将图像数据从PIL类型转换为32位浮点数，并除以255使其归一化到[0,1]范围内
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, 
                                                transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, 
                                               transform=trans, download=True)
len(mnist_train), len(mnist_test)
# %%
mnist_train[0][0].shape
# %%
def get_fashion_mnist_labels(labels):
    """return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            "img = img.numpy()"
            ax.imshow(img.numpy())
        else:
            "img = img.asnumpy()"
            ax.imshow(img)
# %%
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
# %%
batch_size = 256

def get_dataloader_workers():
    """Use 4 processes to read the data (Windows 下改为 0 避免多进程报错)."""
    return 0

train_iter = data.DataLoader(mnist_train,batch_size, shuffle=True, 
                             num_workers=get_dataloader_workers())

start_time = time.time()
for X, y in train_iter:
    continue
print(f'{time.time() - start_time:.2f} sec')
# %%
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True,
                                                    transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False,
                                                   transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))