[中文](README_CN.md) | English

# PyTorch Learning Notes

From-scratch PyTorch study notes covering ten tracks:
- Autograd fundamentals (`autograd/00` to `autograd/08`)
- Dataset / DataLoader / TensorBoard practice
- `nn.Module` layers, loss functions, and network building
- Model saving, loading, and pretrained models
- Full training loop on CPU and GPU
- Advanced topic modules: training engineering, loss functions, model architecture, PEFT, generative models, evaluation & inference, multimodal
- Transformer (Encoder-Decoder, paper reproduction from scratch)
- GPT (Decoder-only LLM with RoPE, KV Cache, BPE tokenizer)
- `mini_llm/`: MiniMind-style small LLM engineering track for pretraining, SFT, LoRA, DPO/RL, and serving

## Project Structure

### 1) Autograd Series (Core)

| File | Topic | Description |
|------|-------|-------------|
| `autograd/00_interactive_playground.py` | Interactive Playground | PyTorch check, simple autograd demo, Jacobian calculation |
| `autograd/01_manual_gradient.py` | Manual Gradient | Hand-written chain rule + gradient descent for `y = 2x + 1` |
| `autograd/02_autograd_basics.py` | Autograd Basics | `requires_grad`, `backward()`, `no_grad()`, accumulation pitfall |
| `autograd/03_computational_graph.py` | Computational Graph | `grad_fn`, leaf tensors, `retain_graph`, `detach()` |
| `autograd/04_hooks.py` | Hook Mechanism | Tensor/Module hooks, gradient inspection & modification |
| `autograd/05_jacobian_and_advanced_autograd.py` | Jacobian & Higher-Order Derivatives | VJP, Jacobian, Hessian basics |
| `autograd/06_gradient_accumulation_and_tricks.py` | Training Tricks | Accumulation, clipping, grad check, parameter freezing |
| `autograd/07_custom_autograd_function.py` | Custom Function | Custom ReLU / STE / multi-input with `gradcheck` |
| `autograd/08_tensor_operations_and_gpu.py` | Tensor Ops & GPU | Creation, reshape, indexing, broadcasting, GPU, NumPy interop |

### 2) Data Pipeline & TensorBoard

| File | Description |
|------|-------------|
| `xiaotudui/generate_labels.py` | Generate label txt files for ants/bees images |
| `xiaotudui/read_data.py` | Build a custom `Dataset` for hymenoptera images |
| `xiaotudui/test_Tf.py` | Basic `transforms.ToTensor()` usage |
| `xiaotudui/Useful_TF.py` | Common transforms: `Normalize`, `Resize`, `Compose`, `RandomCrop` |
| `xiaotudui/test_tb.py` | TensorBoard basics for image/scalar logging |
| `xiaotudui/dataset_transform.py` | CIFAR-10 download + transform pipeline + TensorBoard preview |
| `xiaotudui/dataloader.py` | Load CIFAR-10 with `DataLoader` and log batches by epoch |

### 3) nn.Module — Layers, Loss & Building Networks

| File | Description |
|------|-------------|
| `xiaotudui/nn_module.py` | Minimal custom `nn.Module` (`forward` demo) |
| `xiaotudui/nn_relu.py` | `ReLU` / `Sigmoid` activation demo on CIFAR-10 batches with TensorBoard |
| `xiaotudui/nn_maxpool.py` | `MaxPool2d` feature downsampling demo (`ceil_mode=True`) with TensorBoard |
| `xiaotudui/nn_conv.py` | `torch.nn.functional.conv2d` with stride/padding comparison |
| `xiaotudui/nn_conv2d.py` | `nn.Conv2d` on CIFAR-10 + input/output visualization in TensorBoard |
| `xiaotudui/nn_linear.py` | `nn.Linear` basics: flatten CIFAR-10 images and pass through a fully-connected layer |
| `xiaotudui/nn_seq.py` | Build a Conv→Pool→Flatten→Linear network with `nn.Sequential`; visualize compute graph in TensorBoard |
| `xiaotudui/nn_loss.py` | Common loss functions: `L1Loss`, `MSELoss`, `CrossEntropyLoss` |
| `xiaotudui/nn_loss_network.py` | Full CNN (Conv2d + MaxPool2d + Flatten + Linear) + loss computation + `backward()` |
| `xiaotudui/nn_optim.py` | Add `torch.optim.SGD` to the full network: `zero_grad` → `backward` → `step` complete training loop over 20 epochs |
| `xiaotudui/nn_common_layers.py` | Overview of other common layers: `BatchNorm2d`, `Dropout`, `AvgPool2d`, `AdaptiveAvgPool2d`, `Flatten`, `Embedding` |

### 4) Model Saving, Loading & Pretrained Models

| File | Description |
|------|-------------|
| `xiaotudui/model.py` | Standalone `Moli` CNN definition (imported by `xiaotudui/train.py`) |
| `xiaotudui/model_save.py` | Two save strategies: full model (`torch.save(model)`) vs state dict (`model.state_dict()`); custom class save trap demo |
| `xiaotudui/model_load.py` | Load method 1 with `weights_only=False`; load method 2 with `load_state_dict`; PyTorch 2.6 pitfall notes |
| `xiaotudui/model_pretrained.py` | Load VGG16 with/without pretrained weights; modify classifier by `add_module` or layer replacement |

### 5) Full Training Loop

| File | Description |
|------|-------------|
| `xiaotudui/train.py` | Complete CPU training: data → model → loss → optimizer → train/eval loop → TensorBoard → save checkpoint each epoch |
| `xiaotudui/train_gpu_1.py` | GPU training using `.cuda()` — hardcoded CUDA device, add timing with `time` module |
| `xiaotudui/train_gpu_2.py` | GPU training using `.to(device)` — recommended pattern, device-agnostic (swap `"cuda"` to `"cpu"` freely) |

### 6) Quick Console Check

| File | Description |
|------|-------------|
| `xiaotudui/Console.py` | Simple CUDA availability and `torch` API quick inspection snippets |

---

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
| `diffusion_basics.py` | DDPM forward process `q(x_t\|x_0)`, noise prediction loss `L_simple`, sinusoidal time embedding, ancestral sampling |
| `gan_training.py` | GAN minimax objective, non-saturating loss, training loop, mode collapse analysis, WGAN-GP gradient penalty |

#### `evaluation_inference/`

| File | Description |
|------|-------------|
| `sampling_strategies.py` | Greedy, temperature sampling, Top-k, Top-p (nucleus, Holtzman 2020), repetition penalty |
| `beam_search.py` | Beam search algorithm, length penalty (Wu et al. 2016), diverse beam search concept, strategy selection guide |
| `systematic_evaluation.py` | Systematic LLM evaluation overview: offline objective metrics, judge-based open-ended evaluation, pairwise battle, win rate, Elo, bootstrap intuition |
| `mmlu_eval.py` | Teaching-style MMLU evaluation: few-shot multiple-choice prompt building, answer parsing, per-subject accuracy, macro average |
| `mt_bench_eval.py` | Teaching-style MT-Bench pipeline: multi-turn question set, answer generation, judge scoring, turn/category aggregation |
| `lmsys_arena_eval.py` | Teaching-style LMSYS/Chatbot Arena pipeline: anonymized A/B battle, vote logging, win rate and Elo update |

#### `multimodal/`

| File | Description |
|------|-------------|
| `clip_contrastive.py` | CLIP symmetric InfoNCE, learnable temperature `logit_scale`, zero-shot classification, role in LLaVA / SEED |
| `cross_attention.py` | Cross-attention Q/K/V from different modalities, gated cross-attention (Flamingo 2022), Q-Former (BLIP-2 2023) |

---

### 8) Transformer/

Complete Encoder-Decoder Transformer reproduced from *Attention Is All You Need* (Vaswani et al. 2017), implemented as annotated notebooks.

| File | Description |
|------|-------------|
| `MHA.py` | Multi-head attention from scratch: Q/K/V projection, scaled dot-product, head split/merge, attention weight output |
| `FFN.py` | Position-wise Feed Forward Network: two linear layers with ReLU, dimension expansion from `d_model` to `d_ff` |
| `PostionalEncoding.py` | Sinusoidal positional encoding: sine/cosine at alternating dimensions, added to token embeddings |
| `create_mask.py` | Three mask types explained and implemented: source padding mask, target causal mask, memory mask |
| `Encoder.py` | Encoder layer (bidirectional self-attention + FFN + Add&Norm); stacked into full Encoder |
| `Decoder.py` | Decoder layer (masked self-attention + cross-attention + FFN + Add&Norm); stacked into full Decoder |
| `Transformer.py` | Full Encoder-Decoder Transformer assembled from the above modules; includes a Generator (linear + log-softmax) output head |
| `train.py` | Train on a small parallel text corpus (`data/parallel_toy.tsv`): build vocabulary, teacher-forcing training loop, mask construction |
| `test.py` | Shape and forward-pass tests for MHA, mask utilities, and the full Transformer |

---

### 9) GPT/

Decoder-only GPT language model trained on classical Chinese text (~400k characters). All components implemented from scratch.

| File | Description |
|------|-------------|
| `MHA.py` | Causal multi-head self-attention with optional RoPE on Q/K; returns attention weights; supports KV cache (`past_kv`) |
| `RoPE.py` | Rotary Position Embedding: `rotate_half` using LLaMA-style pairing, precomputed `cos`/`sin` tables, applied to Q and K |
| `FFN.py` | Feed-forward block with GELU activation and configurable `d_ff` hidden dimension |
| `GPTBlock.py` | Pre-LN decoder block: LayerNorm → self-attention → residual; LayerNorm → FFN → residual; passes through KV cache |
| `GPT.py` | Full GPT model: token embedding + optional learned positional embedding (fallback when `use_rope=False`), stacked `GPTBlock`s, final LayerNorm, weight-tied `lm_head` |
| `create_mask.py` | Causal mask utility (`make_causal_mask`): upper-triangular boolean mask for autoregressive decoding |
| `tokenizer.py` | `CharTokenizer`: character-level tokenizer; special tokens pad/bos/eos/unk, `min_freq` filtering, save/load to JSON |
| `bpe_tokenizer.py` | `BPETokenizer`: Byte Pair Encoding tokenizer trained on corpus; incremental pair-count updates for fast training; greedy priority merge for encoding; save/load to JSON |
| `train_gpt.py` | Training script: cosine LR warmup (`get_lr`), tqdm progress bar, char or BPE tokenizer selection (`--tokenizer`), checkpoint save/resume, gradient clipping |
| `sample.py` | Generation script: KV-cache autoregressive decoding, temperature scaling, top-k filtering, top-p (nucleus) sampling |
| `test_gpt.py` | Unit tests: GELU activation, weight tying, Pre-LN block behavior, forward shapes, RoPE correctness, KV cache vs full-forward equivalence |

#### Training

```bash
# Character-level tokenizer
python GPT/train_gpt.py --text-path GPT/_train_text_large.txt --device cpu --max-steps 2000

# BPE tokenizer (recommended)
python GPT/train_gpt.py --text-path GPT/_train_text_large.txt \
  --tokenizer bpe --bpe-vocab-size 8500 \
  --d-model 256 --num-heads 8 --num-layers 6 --d-ff 1024 \
  --device cpu --max-steps 2000
```

#### Sampling

The repository does not include pretrained weights. Train the model first or
place a compatible checkpoint at the path passed to `--ckpt`.

```bash
python GPT/sample.py --ckpt GPT/checkpoints/gpt_char_best.pt \
  --prompt "春风" --temperature 0.8 --top-p 0.9 --max-new-tokens 120
```

---

### 10) mini_llm/

MiniMind-style small LLM engineering track. It does not reuse the teaching
`GPT/` implementation; it is organized as a project that can grow into a
trainable, tunable, evaluable, and servable stack.

| Path | Description |
|------|-------------|
| `mini_llm/model/` | Model structure: presets, Dense/MoE decoder-only model, LoRA, RoPE and GQA |
| `mini_llm/trainer/` | Training entrypoints: tokenizer, pretrain, full_sft, lora, dpo, ppo, grpo, agent, distillation |
| `mini_llm/scripts/` | Engineering scripts: CLI chat, conversion, tool-call eval, OpenAI API serving, WebUI |
| `mini_llm/dataset/` | Local JSONL data directory; real data is ignored, format notes are tracked |
| `mini_llm/eval_llm.py` | Inference and evaluation entrypoint |

Current status: dense model training, JSONL pretraining, full SFT, native LoRA,
checkpoint evaluation, and DPO are runnable. MoE and RL-stage infrastructure
are in progress; KV cache and production-grade evaluation remain planned. See
[`mini_llm/README.md`](mini_llm/README.md) for presets and commands.

---

## Datasets and Local Paths

- `data/`: MNIST
- `dataset/`: hymenoptera (ants vs bees)
- `dataset2/`: CIFAR-10
- `GPT/_train_text_large.txt`: ~400k characters of classical Chinese (四书五经, 宋词, 全唐诗)

Datasets, TensorBoard logs, and trained model files are excluded from git via `.gitignore`.

## Repository Hygiene

- Model weights and checkpoint directories are intentionally not versioned.
- Keep reproducible configs, tokenizer code, evaluation results, and small
  metadata files in Git; keep generated `.pt`, `.pth`, `.ckpt`,
  `.safetensors`, and similar artifacts local or in external model storage.
- Training scripts create their output directories as needed. A fresh clone
  therefore contains source code and documentation, but no pretrained model.

## Requirements

```bash
pip install torch torchvision tensorboard tqdm
```

## Suggested Learning Path

1. `autograd/00` → `autograd/08`: Autograd core
2. `xiaotudui/` transforms / Dataset scripts → DataLoader / TensorBoard
3. `xiaotudui/nn` layers: activations → convolution → linear → Sequential → loss → optimizer → common layers
4. Model persistence: `xiaotudui/model_save.py` → `xiaotudui/model_load.py` → `xiaotudui/model_pretrained.py`
5. Full training loop: `xiaotudui/train.py` (CPU) → `xiaotudui/train_gpu_1.py` → `xiaotudui/train_gpu_2.py`
6. Advanced modules: `training_engineering/`, `model_architecture/`, `loss_functions/`, `peft/`, `generative_models/`, `evaluation_inference/`, `multimodal/`
7. `Transformer/`: build an Encoder-Decoder Transformer from the paper
8. `GPT/`: build a Decoder-only GPT, train on real text, generate samples
9. `mini_llm/`: MiniMind-style engineering path, from model core to pretraining, SFT, alignment, and serving
