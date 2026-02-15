[English](README.md) | 中文

# PyTorch Autograd 学习笔记

从零开始系统学习 PyTorch 自动求导机制（Autograd），包含从手动梯度推导到自定义 Function 的完整学习路径，以及数据加载和 TensorBoard 可视化的实践。

## 目录结构

### Autograd 系列（核心）

| 文件 | 主题 | 内容 |
|------|------|------|
| `00_interactive_playground.py` | 交互式试验场 | PyTorch 环境检查、简单 autograd 示例、Jacobian 计算 |
| `01_manual_gradient.py` | 手动梯度计算 | 纯手写链式法则实现梯度下降，拟合 `y = 2x + 1` |
| `02_autograd_basics.py` | Autograd 入门 | `requires_grad`、`backward()`、`no_grad()`、梯度累加陷阱 |
| `03_computational_graph.py` | 动态计算图 | `grad_fn`、叶子节点、`retain_graph`、`detach()` |
| `04_hooks.py` | Hook 机制 | Tensor Hook、Module Hook、梯度修改、Grad-CAM 思路 |
| `05_jacobian_and_advanced_autograd.py` | Jacobian 与高阶导数 | VJP、Jacobian 矩阵、高阶导数、Hessian 矩阵 |
| `06_gradient_accumulation_and_tricks.py` | 梯度实战技巧 | 梯度累加、梯度裁剪、梯度检查、参数冻结 |
| `07_custom_autograd_function.py` | 自定义 Autograd Function | 自定义 ReLU、STE、多输入函数、`gradcheck` 验证 |
| `08_tensor_operations_and_gpu.py` | 张量操作与 GPU | 张量创建/变形/索引、矩阵运算、广播、GPU 加速、NumPy 互转 |

### 数据与可视化

| 文件 | 说明 |
|------|------|
| `read_data.py` | 自定义 Dataset，继承 `torch.utils.data.Dataset` |
| `generate_labels.py` | 为蚂蚁/蜜蜂图像数据集生成标签文件 |
| `test_tb.py` | TensorBoard 基础：记录图像和标量曲线 |
| `test_Tf.py` | `transforms.ToTensor()` 基础用法 |
| `Useful_TF.py` | 常用 transforms 汇总：Normalize、Resize、Compose、RandomCrop |

## 使用的数据集

- **MNIST** — 手写数字识别
- **CIFAR-10** — 10 类彩色图像分类
- **Hymenoptera** — 蚂蚁 vs 蜜蜂二分类（迁移学习经典数据集）

> 数据集文件较大，未包含在仓库中。运行相关脚本时会自动下载，或手动放置到 `data/` 和 `dataset/` 目录下。

## 环境

- Python 3.x
- PyTorch
- torchvision
- TensorBoard

```bash
pip install torch torchvision tensorboard
```

## 推荐学习顺序

按文件编号 `00` → `08` 顺序阅读 Autograd 系列，每个文件都包含详细的中文注释和代码示例，建议在 IDE 中分块运行（支持 `# %%` 交互式执行）。
