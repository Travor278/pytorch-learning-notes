[中文](README_CN.md) | English

# PyTorch Autograd Learning Notes

A systematic, from-scratch study of PyTorch's automatic differentiation (Autograd) — covering everything from manual gradient derivation to custom Autograd Functions, plus hands-on practice with data loading and TensorBoard visualization.

## Project Structure

### Autograd Series (Core)

| File | Topic | Description |
|------|-------|-------------|
| `00_interactive_playground.py` | Interactive Playground | PyTorch environment check, simple autograd demo, Jacobian computation |
| `01_manual_gradient.py` | Manual Gradient | Implement gradient descent with hand-written chain rule, fitting `y = 2x + 1` |
| `02_autograd_basics.py` | Autograd Basics | `requires_grad`, `backward()`, `no_grad()`, gradient accumulation pitfall |
| `03_computational_graph.py` | Dynamic Computational Graph | `grad_fn`, leaf nodes, `retain_graph`, `detach()` |
| `04_hooks.py` | Hook Mechanism | Tensor hooks, module hooks, gradient modification, Grad-CAM idea |
| `05_jacobian_and_advanced_autograd.py` | Jacobian & Higher-Order Derivatives | VJP, Jacobian matrix, higher-order derivatives, Hessian matrix |
| `06_gradient_accumulation_and_tricks.py` | Gradient Tricks | Gradient accumulation, gradient clipping, gradient checking, parameter freezing |
| `07_custom_autograd_function.py` | Custom Autograd Function | Custom ReLU, STE, multi-input functions, `gradcheck` verification |
| `08_tensor_operations_and_gpu.py` | Tensor Operations & GPU | Tensor creation/reshape/indexing, matrix operations, broadcasting, GPU acceleration, NumPy interop |

### Data & Visualization

| File | Description |
|------|-------------|
| `read_data.py` | Custom Dataset inheriting `torch.utils.data.Dataset` |
| `generate_labels.py` | Generate label files for the ants/bees image dataset |
| `test_tb.py` | TensorBoard basics: logging images and scalar curves |
| `test_Tf.py` | `transforms.ToTensor()` basic usage |
| `Useful_TF.py` | Common transforms overview: Normalize, Resize, Compose, RandomCrop |

## Datasets Used

- **MNIST** — Handwritten digit recognition
- **CIFAR-10** — 10-class color image classification
- **Hymenoptera** — Ants vs. Bees binary classification (classic transfer learning dataset)

> Datasets are too large to include in the repository. They will be downloaded automatically when running the relevant scripts, or you can manually place them under `data/` and `dataset/`.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- TensorBoard

```bash
pip install torch torchvision tensorboard
```

## Recommended Learning Order

Follow the file numbering `00` → `08` through the Autograd series. Each file contains detailed comments (in Chinese) and code examples. It is recommended to run them interactively in your IDE using `# %%` cell blocks.
