#%% # softmax回归从零开始实现
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
#%% 1. 数据加载
def get_fashion_mnist_labels(labels):
    """返回 Fashion-MNIST 数据集的文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def load_data_fashion_mnist(batch_size):
    """下载 Fashion-MNIST 数据集并加载到内存。"""
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True,
                                                    transform=trans, download=True)
    mnist_test  = torchvision.datasets.FashionMNIST(root="./data", train=False,
                                                    transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,  num_workers=0),
            torch.utils.data.DataLoader(mnist_test,  batch_size, shuffle=False, num_workers=0))

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
#%% 2. 初始化模型参数
num_inputs  = 784
num_outputs = 10

# 权重 W: (784, 10)，偏置 b: (10,)
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
#%% 验证求和操作
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)
#%% 3. Softmax 操作
def softmax(X):
    X_exp = torch.exp(X)
    # 对每一行求和，keepdim=True 保持二维以触发广播
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 每行概率和为 1

# 验证
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)
#%% 4. 定义模型
def net(X):
    # X: (batch_size, 1, 28, 28) → reshape 成 (batch_size, 784)
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
#%% 5. 定义交叉熵损失函数
def cross_entropy(y_hat, y):
    # 取每个样本预测为真实类别的概率，再取对数的负值
    return -torch.log(y_hat[range(len(y_hat)), y])

# 验证
y     = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
cross_entropy(y_hat, y)
#%% 6. 分类精度
def accuracy(y_hat, y):
    """计算预测正确的样本数。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)
#%% 7. 累加器
class Accumulator:
    """对 n 个变量累加求和。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
#%% 8. 在测试集上评估精度
def evaluate_accuracy(net, data_iter):
    """计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

evaluate_accuracy(net, test_iter)
#%% 9. 训练一个 epoch
def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期。"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            metric.add(float(l.mean()) * len(y), accuracy(y_hat, y), y.numel())
        else:
            # 自定义优化器（如上面的 sgd）
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
#%% 10. 动画绘图器（简化版，用静态图替代动画）
class Animator:
    """训练过程可视化（每 epoch 刷新一次折线图）。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None,
                 xlim=None, ylim=None, figsize=(6, 4)):
        self.legend  = legend or []
        self.xlabel  = xlabel
        self.ylabel  = ylabel
        self.xlim    = xlim
        self.ylim    = ylim
        self.figsize = figsize
        self.X = []
        self.Y = []   # list of lists，每条曲线一个列表

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        self.X.append(x)
        self.Y.append(list(y))

    def show(self):
        _, ax = plt.subplots(figsize=self.figsize)
        if self.Y:
            n_lines = len(self.Y[0])
            ys = [[row[i] for row in self.Y] for i in range(n_lines)]
            fmts = ['-', 'm--', 'g-.', 'r:']
            for i, curve in enumerate(ys):
                label = self.legend[i] if i < len(self.legend) else f'line {i}'
                ax.plot(self.X, curve, fmts[i % len(fmts)], label=label)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)
        ax.legend()
        plt.tight_layout()
        plt.show()
#%% 11. 完整训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（第3章）。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 1.0],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
        print(f'epoch {epoch+1:2d}  loss {train_loss:.4f}  '
              f'train acc {train_acc:.3f}  test acc {test_acc:.3f}')
    animator.show()
#%% 12. 定义优化器并开始训练
lr = 0.1

def updater(batch_size):
    return sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
#%% 13. 预测
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表。"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy(), cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i], fontsize=8)
    plt.tight_layout()
    plt.show()
    return axes

def predict_ch3(net, test_iter, n=6):
    """预测标签（第3章）。"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)