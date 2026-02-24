[中文](README_CN.md) | English

# PyTorch Learning Notes

From-scratch PyTorch study notes covering seven tracks:
- Autograd fundamentals (`00` to `08`)
- Dataset / DataLoader / TensorBoard practice
- `nn.Module` layers, loss functions, and network building
- Model saving, loading, and pretrained models
- Full training loop on CPU and GPU
- Advanced topic modules: training engineering, loss functions, model architecture, PEFT, generative models, evaluation & inference, multimodal

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

### 3) nn.Module — Layers, Loss & Building Networks

| File | Description |
|------|-------------|
| `nn_module.py` | Minimal custom `nn.Module` (`forward` demo) |
| `nn_relu.py` | `ReLU` / `Sigmoid` activation demo on CIFAR-10 batches with TensorBoard |
| `nn_maxpool.py` | `MaxPool2d` feature downsampling demo (`ceil_mode=True`) with TensorBoard |
| `nn_conv.py` | `torch.nn.functional.conv2d` with stride/padding comparison |
| `nn_conv2d.py` | `nn.Conv2d` on CIFAR-10 + input/output visualization in TensorBoard |
| `nn_linear.py` | `nn.Linear` basics: flatten CIFAR-10 images and pass through a fully-connected layer |
| `nn_seq.py` | Build a Conv→Pool→Flatten→Linear network with `nn.Sequential`; visualize compute graph in TensorBoard |
| `nn_loss.py` | Common loss functions: `L1Loss`, `MSELoss`, `CrossEntropyLoss` |
| `nn_loss_network.py` | Full CNN (Conv2d + MaxPool2d + Flatten + Linear) + loss computation + `backward()` |
| `nn_optim.py` | Add `torch.optim.SGD` to the full network: `zero_grad` → `backward` → `step` complete training loop over 20 epochs |
| `nn_common_layers.py` | Overview of other common layers: `BatchNorm2d`, `Dropout`, `AvgPool2d`, `AdaptiveAvgPool2d`, `Flatten`, `Embedding` |

### 4) Model Saving, Loading & Pretrained Models

| File | Description |
|------|-------------|
| `model.py` | Standalone `Moli` CNN definition (imported by `train.py`) |
| `model_save.py` | Two save strategies: full model (`torch.save(model)`) vs state dict (`model.state_dict()`); custom class save trap demo |
| `model_load.py` | Load method 1 with `weights_only=False`; load method 2 with `load_state_dict`; PyTorch 2.6 pitfall notes |
| `model_pretrained.py` | Load VGG16 with/without pretrained weights; modify classifier by `add_module` or layer replacement |

### 5) Full Training Loop

| File | Description |
|------|-------------|
| `train.py` | Complete CPU training: data → model → loss → optimizer → train/eval loop → TensorBoard → save checkpoint each epoch |
| `train_gpu_1.py` | GPU training using `.cuda()` — hardcoded CUDA device, add timing with `time` module |
| `train_gpu_2.py` | GPU training using `.to(device)` — recommended pattern, device-agnostic (swap `"cuda"` to `"cpu"` freely) |

### 6) Quick Console Check

| File | Description |
|------|-------------|
| `Console.py` | Simple CUDA availability and `torch` API quick inspection snippets |

## Datasets and Local Paths

- `data/`: MNIST
- `dataset/`: hymenoptera (ants vs bees)
- `dataset2/`: CIFAR-10

Datasets, TensorBoard logs, and trained `.pth` model files are intentionally excluded from git via `.gitignore`.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- TensorBoard

```bash
pip install torch torchvision tensorboard
```

### 7) Advanced Topic Modules

#### `training_engineering/`

| File | Description |
|------|-------------|
| `optimizer_scheduler.py` | AdamW vs Adam (decoupled weight decay), parameter groups, StepLR / CosineAnnealingLR / OneCycleLR, linear warmup + cosine decay |
| `amp_training.py` | FP16 / BF16 / FP32 formats, `autocast` + `GradScaler`, loss scaling theory, CPU vs GPU speed comparison |
| `gradient_checkpoint.py` | Activation recomputation, `torch.utils.checkpoint`, memory–compute trade-off, segment checkpoint strategy |
| `distributed_training.py` | DataParallel vs DDP, Ring-AllReduce principle, single-machine multi-GPU `mp.spawn` setup |

#### `loss_functions/`

| File | Description |
|------|-------------|
| `kl_divergence.py` | KL divergence definition, forward vs reverse KL (mean-seeking vs mode-seeking), temperature scaling, knowledge distillation (Hinton 2015), connection to SEED |
| `contrastive_loss.py` | InfoNCE, NT-Xent (SimCLR), Triplet loss, in-batch negatives and temperature |
| `focal_loss.py` | Focal loss for class imbalance (Lin et al. 2017), modulating factor `(1−p_t)^γ`, multi-class extension |
| `label_smoothing.py` | Label smoothing theory, KL-penalty interpretation, `F.cross_entropy(label_smoothing=)`, calibration vs distillation trade-off |

#### `model_architecture/`

| File | Description |
|------|-------------|
| `attention_mechanism.py` | Scaled dot-product attention, causal mask, multi-head attention from scratch, `nn.MultiheadAttention` |
| `positional_encoding.py` | Sinusoidal PE (Vaswani 2017), learnable PE (BERT), RoPE (LLaMA / LLaVA) |
| `residual_connection.py` | Residual gradient flow analysis, BasicBlock with projection shortcut, Post-LN vs Pre-LN, RMSNorm |
| `normalization.py` | BatchNorm / LayerNorm / GroupNorm / InstanceNorm — statistics dimensions, selection guide |

#### `peft/`

| File | Description |
|------|-------------|
| `lora_principle.py` | LoRA math `W = W₀ + (α/r)BA`, zero-init of B, `LoRALinear` wrapper, applying LoRA to model layers |
| `adapter_layers.py` | Bottleneck adapter (Houlsby 2019), gated vs serial insertion, PEFT method comparison table |

#### `generative_models/`

| File | Description |
|------|-------------|
| `vae_basics.py` | ELBO derivation, reparameterization trick `z = μ + σε`, KL analytical solution, β-VAE |
| `diffusion_basics.py` | DDPM forward process `q(x_t|x_0)`, noise prediction loss `L_simple`, sinusoidal time embedding, ancestral sampling |
| `gan_training.py` | GAN minimax objective, non-saturating loss, training loop, mode collapse analysis, WGAN-GP gradient penalty |

#### `evaluation_inference/`

| File | Description |
|------|-------------|
| `sampling_strategies.py` | Greedy, temperature sampling, Top-k, Top-p (nucleus, Holtzman 2020), repetition penalty |
| `beam_search.py` | Beam search algorithm, length penalty (Wu et al. 2016), diverse beam search concept, strategy selection guide |

#### `multimodal/`

| File | Description |
|------|-------------|
| `clip_contrastive.py` | CLIP symmetric InfoNCE, learnable temperature `logit_scale`, zero-shot classification, role in LLaVA / SEED |
| `cross_attention.py` | Cross-attention Q/K/V from different modalities, gated cross-attention (Flamingo 2022), Q-Former (BLIP-2 2023) |

## Suggested Learning Path

1. Start with `00` -> `08` (Autograd core).
2. Move to transforms/dataset scripts (`test_Tf.py`, `Useful_TF.py`, `read_data.py`, `dataset_transform.py`).
3. Continue with batching and visualization (`dataloader.py`, `test_tb.py`).
4. Work through `nn` modules: activations & pooling → convolution → linear → Sequential → loss functions → optimizer → common layers.
5. Study model persistence: `model_save.py` → `model_load.py` → `model_pretrained.py`.
6. Run a full training loop: `train.py` (CPU) → `train_gpu_1.py` → `train_gpu_2.py` (recommended GPU pattern).
