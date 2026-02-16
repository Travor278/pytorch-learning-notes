[中文](README_CN.md) | English

# PyTorch Learning Notes

From-scratch PyTorch study notes focused on three tracks:
- Autograd fundamentals (`00` to `08`)
- Dataset / DataLoader / TensorBoard practice
- `nn.Module` and convolution demos (`Conv2d`)

## Project Structure

### 1) Autograd Series (Core)

| File | Topic | Description |
|------|-------|-------------|
| `00_interactive_playground.py` | Interactive Playground | PyTorch check, simple autograd demo, Jacobian calculation |
| `01_manual_gradient.py` | Manual Gradient | Hand-written chain rule + gradient descent for `y = 2x + 1` |
| `02_autograd_basics.py` | Autograd Basics | `requires_grad`, `backward()`, `no_grad()`, accumulation pitfall |
| `03_computational_graph.py` | Computational Graph | `grad_fn`, leaf tensors, `retain_graph`, `detach()` |
| `04_hooks.py` | Hook Mechanism | Tensor/Module hooks, gradient inspection & modification |
| `05_jacobian_and_advanced_autograd.py` | Jacobian & Higher-Order Derivatives | VJP, Jacobian, Hessian basics |
| `06_gradient_accumulation_and_tricks.py` | Training Tricks | Accumulation, clipping, grad check, parameter freezing |
| `07_custom_autograd_function.py` | Custom Function | Custom ReLU / STE / multi-input with `gradcheck` |
| `08_tensor_operations_and_gpu.py` | Tensor Ops & GPU | Creation, reshape, indexing, broadcasting, GPU, NumPy interop |

### 2) Data Pipeline & TensorBoard

| File | Description |
|------|-------------|
| `generate_labels.py` | Generate label txt files for ants/bees images |
| `read_data.py` | Build a custom `Dataset` for hymenoptera images |
| `test_Tf.py` | Basic `transforms.ToTensor()` usage |
| `Useful_TF.py` | Common transforms: `Normalize`, `Resize`, `Compose`, `RandomCrop` |
| `test_tb.py` | TensorBoard basics for image/scalar logging |
| `dataset_transform.py` | CIFAR-10 download + transform pipeline + TensorBoard preview |
| `dataloader.py` | Load CIFAR-10 with `DataLoader` and log batches by epoch |

### 3) nn.Module & Convolution Practice

| File | Description |
|------|-------------|
| `nn_module.py` | Minimal custom `nn.Module` (`forward` demo) |
| `nn_conv.py` | `torch.nn.functional.conv2d` with stride/padding comparison |
| `nn_conv2d.py` | `nn.Conv2d` on CIFAR-10 + input/output visualization in TensorBoard |

## Datasets and Local Paths

- `data/`: MNIST
- `dataset/`: hymenoptera (ants vs bees)
- `dataset2/`: CIFAR-10

Large datasets and TensorBoard logs are intentionally excluded from git via `.gitignore`.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- TensorBoard

```bash
pip install torch torchvision tensorboard
```

## Suggested Learning Path

1. Start with `00` -> `08` (Autograd core).
2. Move to transforms/dataset scripts (`test_Tf.py`, `Useful_TF.py`, `read_data.py`, `dataset_transform.py`).
3. Continue with batching and visualization (`dataloader.py`, `test_tb.py`).
4. Finish with model building basics (`nn_module.py`, `nn_conv.py`, `nn_conv2d.py`).
