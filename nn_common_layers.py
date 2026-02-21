"""
nn 常用层速览
已学过：Linear、ReLU/Sigmoid、MaxPool2d、Conv2d
本文件补充：
  1. BatchNorm2d   —— 批归一化
  2. Dropout       —— 随机丢弃，防过拟合
  3. AvgPool2d / AdaptiveAvgPool2d —— 平均池化
  4. Flatten       —— 展平层（替代手动 reshape）
  5. nn.Sequential —— 顺序容器，简化模型写法
  6. Embedding     —— 词嵌入（NLP 入口）
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 公用数据
dataset = torchvision.datasets.CIFAR10(root="./dataset2", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor()
)
dataloader = DataLoader(dataset, batch_size=64)
imgs, _ = next(iter(dataloader))   # imgs: [64, 3, 32, 32]

# 1. BatchNorm2d（批归一化）
# 作用：对每个 channel 做归一化（均值≈0，方差≈1），加速训练收敛，减少对初始化和学习率的敏感性。
# 参数：num_features = 输入的 channel 数
# 注意：train() 模式用 batch 统计量；eval() 模式用训练期间的滑动均值/方差。

# %%
bn = nn.BatchNorm2d(num_features=3)
out_bn = bn(imgs)
print("BatchNorm2d 输入:", imgs.shape)   # [64, 3, 32, 32]
print("BatchNorm2d 输出:", out_bn.shape) # [64, 3, 32, 32]  形状不变，数值被归一化
print("输出均值（应接近 0）:", out_bn.mean().item())
print("输出方差（应接近 1）:", out_bn.var().item())


# 2. Dropout
# 作用：训练时随机将部分神经元输出置 0，强迫网络学习冗余特征，防止过拟合。
# 参数：p = 每个元素被置 0 的概率（常用 0.5）
# 注意：eval() 模式下 Dropout 自动关闭，不会丢弃任何值。

# %%
dropout = nn.Dropout(p=0.5)
x = torch.ones(1, 10)
print("\nDropout 输入:", x)
print("Dropout 输出（训练模式，约一半变 0）:", dropout(x))

dropout.eval()
print("Dropout 输出（eval 模式，全部保留）:", dropout(x))
dropout.train()

# Dropout2d：对整个 channel 随机置 0（适合卷积后的 feature map）
dropout2d = nn.Dropout2d(p=0.3)
out_d2 = dropout2d(imgs)
print("\nDropout2d 输出形状:", out_d2.shape)   # [64, 3, 32, 32]


# 3. AvgPool2d 和 AdaptiveAvgPool2d（平均池化）
# AvgPool2d：固定 kernel_size，取窗口内平均值。
# AdaptiveAvgPool2d：指定输出尺寸，自动推断 stride/kernel_size。
#   —— 分类网络末尾常见用法：AdaptiveAvgPool2d((1,1)) 全局平均池化

# %%
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
out_avg = avg_pool(imgs)
print("\nAvgPool2d 输入:", imgs.shape)    # [64, 3, 32, 32]
print("AvgPool2d 输出:", out_avg.shape)  # [64, 3, 16, 16]

adaptive = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 全局平均池化
out_ada = adaptive(imgs)
print("AdaptiveAvgPool2d 输出:", out_ada.shape)       # [64, 3,  1,  1]


# 4. Flatten（展平层）
# 作用：将多维张量从指定维度开始展平成一维，通常放在卷积层和全连接层之间。
# 比手写 view(-1, ...) 更直观，且在 Sequential 中可直接作为一层使用。

# %%
flatten = nn.Flatten(start_dim=1)  # 保留 batch 维度，其余全部展平
out_flat = flatten(out_ada)
print("\nFlatten 输入:", out_ada.shape)  # [64, 3, 1, 1]
print("Flatten 输出:", out_flat.shape)  # [64, 3]


# 5. nn.Sequential（顺序容器）
# 作用：将多个层按顺序打包成一个模块，前向传播自动依次调用每一层，不需要手写 forward。

# %%
# 用 Sequential 搭一个小型分类头
classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((4, 4)),   # [B, 3, 4, 4]
    nn.Flatten(start_dim=1),        # [B, 48]
    nn.Linear(48, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 10),              # CIFAR-10 共 10 类
)
print("\nSequential 结构:\n", classifier)
out_cls = classifier(imgs)
print("Sequential 输出:", out_cls.shape)  # [64, 10]

# 也可以用 OrderedDict 给每层命名（方便按名字访问）
from collections import OrderedDict
named_model = nn.Sequential(OrderedDict([
    ('pool',    nn.AdaptiveAvgPool2d((4, 4))),
    ('flatten', nn.Flatten(start_dim=1)),
    ('fc',      nn.Linear(48, 10)),
]))
print("按名字访问层:", named_model.fc)


# 6. Embedding（词嵌入）
# 作用：将离散 token（整数 ID）映射为连续的稠密向量，
#       是 NLP 任务（文本分类、翻译、语言模型）的标准入口层。
# 参数：num_embeddings = 词表大小（token 种类数）
#      embedding_dim  = 每个 token 映射到的向量维度

# %%
vocab_size  = 1000   # 词表 1000 个词
embed_dim   = 64     # 每个词用 64 维向量表示
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

# 模拟一个 batch：4 个句子，每句 10 个 token ID
token_ids = torch.randint(0, vocab_size, (4, 10))  # [4, 10]
word_vectors = embedding(token_ids)
print("\nEmbedding 输入（token IDs）:", token_ids.shape)   # [4, 10]
print("Embedding 输出（向量）      :", word_vectors.shape) # [4, 10, 64]

# padding_idx：指定某个 ID 对应的向量始终为 0（常用于填充符 <PAD>）
embedding_with_pad = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
print("padding_idx=0 时，ID=0 的向量（全零）:", embedding_with_pad(torch.tensor([0])))


# 综合示例：用上述层搭一个完整的小型 CNN 分类器
# %%
class SmallCNN(nn.Module):
    """
    输入: [B, 3, 32, 32]  (CIFAR-10)
    输出: [B, 10]          (10 类 logits)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 32, 32, 32]
            nn.BatchNorm2d(32),                           # 归一化
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 32, 16, 16]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                 # [B, 64,  4,  4]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # [B, 1024]
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SmallCNN()
print("\nSmallCNN 结构:\n", model)
out = model(imgs)
print("SmallCNN 输出:", out.shape)  # [64, 10]

# TensorBoard 可视化各层输出（以 features 输出为例）
writer = SummaryWriter("logs_layers")
features_out = model.features(imgs)   # [64, 64, 4, 4]

# 取前 3 个 channel 写入 TensorBoard（每个 channel 当成灰度图）
writer.add_images("conv_features_ch0", features_out[:, 0:1], 0)
writer.add_images("conv_features_ch1", features_out[:, 1:2], 0)
writer.add_images("conv_features_ch2", features_out[:, 2:3], 0)
writer.close()
print("\n已写入 TensorBoard，运行：tensorboard --logdir=logs_layers")
