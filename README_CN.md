[English](README.md) | 中文

# PyTorch 学习笔记

这是一个从零开始的 PyTorch 学习仓库，当前覆盖五条线：
- Autograd 核心原理（`00` 到 `08`）
- Dataset / DataLoader / TensorBoard 数据流实践
- `nn.Module` 各类层、损失函数与完整网络搭建
- 模型保存、加载与预训练模型使用
- CPU / GPU 完整训练流程

## 目录结构

### 1）Autograd 系列（核心）

| 文件 | 主题 | 内容 |
|------|------|------|
| `00_interactive_playground.py` | 交互式试验场 | PyTorch 环境检查、简单 autograd 示例、Jacobian 计算 |
| `01_manual_gradient.py` | 手动梯度计算 | 手写链式法则实现梯度下降，拟合 `y = 2x + 1` |
| `02_autograd_basics.py` | Autograd 入门 | `requires_grad`、`backward()`、`no_grad()`、梯度累加陷阱 |
| `03_computational_graph.py` | 计算图机制 | `grad_fn`、叶子张量、`retain_graph`、`detach()` |
| `04_hooks.py` | Hook 机制 | Tensor/Module Hook、梯度查看与修改 |
| `05_jacobian_and_advanced_autograd.py` | Jacobian 与高阶导数 | VJP、Jacobian、Hessian 基础 |
| `06_gradient_accumulation_and_tricks.py` | 训练技巧 | 梯度累加、梯度裁剪、梯度检查、参数冻结 |
| `07_custom_autograd_function.py` | 自定义 Function | 自定义 ReLU/STE/多输入函数 + `gradcheck` |
| `08_tensor_operations_and_gpu.py` | 张量与 GPU | 创建、变形、索引、广播、GPU、NumPy 互转 |

### 2）数据流与可视化

| 文件 | 说明 |
|------|------|
| `generate_labels.py` | 为蚂蚁/蜜蜂图像生成标签 txt 文件 |
| `read_data.py` | 构建自定义 `Dataset`（hymenoptera） |
| `test_Tf.py` | `transforms.ToTensor()` 基础用法 |
| `Useful_TF.py` | 常用 transforms：`Normalize`、`Resize`、`Compose`、`RandomCrop` |
| `test_tb.py` | TensorBoard 基础：记录图像和标量 |
| `dataset_transform.py` | CIFAR-10 下载 + transforms 组合 + TensorBoard 预览 |
| `dataloader.py` | 使用 `DataLoader` 按批加载 CIFAR-10 并记录多轮图像 |

### 3）nn.Module —— 层、损失函数与完整网络搭建

| 文件 | 说明 |
|------|------|
| `nn_module.py` | 最小自定义 `nn.Module`（`forward` 逻辑示例） |
| `nn_relu.py` | `ReLU` / `Sigmoid` 激活函数演示，并对 CIFAR-10 批数据做可视化 |
| `nn_maxpool.py` | `MaxPool2d` 下采样演示（`ceil_mode=True`）并写入 TensorBoard |
| `nn_conv.py` | `F.conv2d` 的 stride / padding 对比实验 |
| `nn_conv2d.py` | `nn.Conv2d` 处理 CIFAR-10，并在 TensorBoard 可视化输入输出 |
| `nn_linear.py` | `nn.Linear` 基础用法：将 CIFAR-10 图像展平后接全连接层 |
| `nn_seq.py` | 用 `nn.Sequential` 搭建 Conv→Pool→Flatten→Linear 网络，TensorBoard 可视化计算图 |
| `nn_loss.py` | 常用损失函数：`L1Loss`、`MSELoss`、`CrossEntropyLoss` |
| `nn_loss_network.py` | 完整 CNN（Conv2d + MaxPool2d + Flatten + Linear）+ 损失计算 + `backward()` |
| `nn_optim.py` | 加入 `torch.optim.SGD`：`zero_grad` → `backward` → `step` 完整训练循环，跑 20 轮 |
| `nn_common_layers.py` | 常用层汇总：`BatchNorm2d`、`Dropout`、`AvgPool2d`、`AdaptiveAvgPool2d`、`Flatten`、`Embedding` |

### 4）模型保存、加载与预训练模型

| 文件 | 说明 |
|------|------|
| `model.py` | 独立定义 `Moli` CNN 网络结构（供 `train.py` 导入） |
| `model_save.py` | 两种保存方式：整个模型 vs `state_dict`（官方推荐）；自定义类保存陷阱演示 |
| `model_load.py` | 两种加载方式：方式1需 `weights_only=False`；方式2先建模型再 `load_state_dict`；PyTorch 2.6 注意事项 |
| `model_pretrained.py` | 加载有/无预训练权重的 VGG16；用 `add_module` 追加层或直接替换层修改分类头 |

### 5）完整训练流程

| 文件 | 说明 |
|------|------|
| `train.py` | CPU 完整训练：数据加载 → 模型 → 损失 → 优化器 → train/eval 循环 → TensorBoard → 每轮保存模型 |
| `train_gpu_1.py` | GPU 训练（`.cuda()` 写法）：硬编码 CUDA 设备，加入 `time` 计时 |
| `train_gpu_2.py` | GPU 训练（`.to(device)` 写法，推荐）：设备变量统一管理，CPU/GPU 一行切换 |

### 6）控制台速查脚本

| 文件 | 说明 |
|------|------|
| `Console.py` | 快速检查 CUDA 可用性与 `torch` API 的交互式小脚本 |

## 数据集与本地目录

- `data/`：MNIST
- `dataset/`：hymenoptera（ants vs bees）
- `dataset2/`：CIFAR-10

数据集、TensorBoard 日志和训练生成的 `.pth` 模型文件体积较大，已通过 `.gitignore` 排除，不会提交到 GitHub。

## 环境

- Python 3.x
- PyTorch
- torchvision
- TensorBoard

```bash
pip install torch torchvision tensorboard
```

## 推荐学习顺序

1. 先按 `00` → `08` 学完 Autograd 主线。
2. 再看 transforms / dataset 相关脚本（`test_Tf.py`、`Useful_TF.py`、`read_data.py`、`dataset_transform.py`）。
3. 接着学习批处理和可视化（`dataloader.py`、`test_tb.py`）。
4. 学习 nn 模块，按顺序：激活与池化 → 卷积 → 全连接 → Sequential 组网 → 损失函数 → 优化器 → 常用层。
5. 学习模型保存与加载（`model_save.py`、`model_load.py`、`model_pretrained.py`）。
6. 最后跑完整训练流程：`train.py`（CPU）→ `train_gpu_1.py` → `train_gpu_2.py`（推荐 GPU 写法）。
