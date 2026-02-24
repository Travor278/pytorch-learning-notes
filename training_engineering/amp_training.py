# amp_training.py
# 混合精度训练（Automatic Mixed Precision, AMP）
#
# 核心思想：用 FP16/BF16 做前向和反向传播，加速计算并节省显存，
# 但在 FP32 下维护权重更新，防止精度损失。
# 来源：Micikevicius et al., "Mixed Precision Training", ICLR 2018.

import torch
import torch.nn as nn
import time

# ========== 1. 浮点格式回顾 ==========
print("=== 浮点格式对比 ===")
#
# FP32（single）：1 符号 + 8 指数 + 23 尾数，动态范围约 1.2e-38 ~ 3.4e38
# FP16（half）  ：1 符号 + 5 指数 + 10 尾数，动态范围约 6e-5 ~ 6.5e4
#                 梯度如果落在 FP16 表示范围之外就会变成 0（下溢）或 Inf（上溢）
# BF16          ：1 符号 + 8 指数 + 7 尾数，动态范围与 FP32 相同但精度更低
#                 对梯度下溢不敏感，是 TPU 和 Ampere+ GPU 的首选格式
#
# Tensor Core（Volta 架构起）：
#   专门的硬件单元，以 FP16/BF16 为输入做矩阵乘，结果累加到 FP32。
#   吞吐量是纯 FP32 的 8~16 倍（理论峰值）。

print(f"FP32 max: {torch.finfo(torch.float32).max:.2e}")
print(f"FP16 max: {torch.finfo(torch.float16).max:.2e}")
print(f"BF16 max: {torch.finfo(torch.bfloat16).max:.2e}")
print(f"FP16 min（正规化）: {torch.finfo(torch.float16).tiny:.2e}")

# 演示 FP16 下溢问题
x_fp32 = torch.tensor(1e-5, dtype=torch.float32)
x_fp16 = x_fp32.half()
print(f"\nFP32 1e-5 → FP16: {x_fp16.item():.2e}  （下溢为 0）" if x_fp16 == 0 else
      f"\nFP32 1e-5 → FP16: {x_fp16.item():.2e}")
print()

# ========== 2. Loss Scaling（梯度缩放）==========
print("=== Loss Scaling 原理 ===")
#
# 问题：反向传播中的梯度值往往很小（如 1e-6），落在 FP16 表示下限以下，
#       直接变为 0，梯度消失。
#
# 解决方案（Micikevicius et al. 2018）：
#   前向计算后，将 loss 乘以一个大的缩放因子 S（如 2^16 = 65536），
#   使梯度人为放大 S 倍，从而落在 FP16 的有效范围内。
#   在 optimizer.step() 前，再将梯度除以 S 恢复原始幅度，然后更新 FP32 权重。
#
# GradScaler 的动态策略：
#   - 如果梯度中出现 Inf/NaN，本步跳过 optimizer.step()，并将 S 减半
#   - 连续 growth_interval 步无 Inf/NaN，则将 S 乘以 growth_factor 增大
#   这样 S 自动收敛到既不下溢又不上溢的合适值。

# GradScaler 示例（无 CUDA 时 CPU 上也可演示基本用法）
scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu',
                               init_scale=2**16,
                               growth_factor=2.0,
                               backoff_factor=0.5,
                               growth_interval=2000)
print(f"初始 scale factor: {scaler.get_scale():.0f}")
print()

# ========== 3. autocast 的使用规范 ==========
print("=== autocast 使用规范 ===")
#
# torch.amp.autocast 自动决定每个算子使用 FP16 还是 FP32：
#   - 矩阵乘（mm / bmm / linear）、卷积：降至 FP16/BF16，Tensor Core 加速
#   - Softmax、LayerNorm、损失函数：保留 FP32，防止数值不稳定
#   - 不需要手动标注每个算子，框架内置了白名单/黑名单机制
#
# BF16 vs FP16：
#   - A100/H100/RTX 4000 系列支持 BF16，动态范围同 FP32，不需要 loss scaling
#   - 旧卡（V100/P100）只支持 FP16，需要 GradScaler

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype_amp  = torch.bfloat16 if (torch.cuda.is_available() and
                                 torch.cuda.is_bf16_supported()) else torch.float16

print(f"运行设备: {device_str}")
print(f"AMP dtype: {dtype_amp}")
print()

# ========== 4. 完整 AMP 训练循环 ==========
print("=== 完整 AMP 训练循环 ===")

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

device   = torch.device(device_str)
model    = ConvNet().to(device)
criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler   = torch.amp.GradScaler(device_str, enabled=(dtype_amp == torch.float16))

# 模拟 3 个 batch 的训练
for step in range(3):
    x = torch.randn(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    optimizer.zero_grad()

    # autocast 内部：矩阵乘/卷积自动降精度，损失保持 FP32
    with torch.amp.autocast(device_str, dtype=dtype_amp):
        logits = model(x)
        loss   = criterion(logits, y)

    # scaler 放大 loss，防止梯度下溢
    scaler.scale(loss).backward()

    # unscale 梯度后做裁剪（必须在 scaler.step 之前）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 若梯度无 Inf/NaN，执行更新；否则跳过本步并调小 scale
    scaler.step(optimizer)
    scaler.update()

    print(f"step={step+1}, loss={loss.item():.4f}, scale={scaler.get_scale():.0f}")

print()

# ========== 5. CPU 上对比 FP32 vs AMP 速度（仅有 CUDA 时有意义）==========
print("=== 性能对比（需 CUDA）===")

if torch.cuda.is_available():
    SIZE  = 2048
    model_large = nn.Linear(SIZE, SIZE).cuda()
    x_in  = torch.randn(256, SIZE, device='cuda')

    # FP32 基准
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = model_large(x_in)
    torch.cuda.synchronize()
    t_fp32 = time.perf_counter() - t0

    # AMP
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.amp.autocast('cuda', dtype=dtype_amp):
            _ = model_large(x_in)
    torch.cuda.synchronize()
    t_amp = time.perf_counter() - t0

    print(f"FP32 ×50: {t_fp32:.3f}s")
    print(f"AMP  ×50: {t_amp:.3f}s  （加速 {t_fp32/t_amp:.1f}×）")
else:
    print("无 CUDA，跳过性能对比。")

print()
print("=== 总结 ===")
print("""
1. AMP 核心：前/反向用 FP16/BF16，权重更新保持 FP32，兼顾速度与精度
2. loss scaling 解决 FP16 梯度下溢，GradScaler 动态维护合适的缩放因子
3. BF16 动态范围与 FP32 相同，A100+ 无需 GradScaler（enabled=False 即可）
4. scaler.unscale_(opt) 必须在 clip_grad_norm_ 之前，否则裁剪阈值被放大
5. BatchNorm / LayerNorm / softmax 自动保留 FP32（autocast 白名单机制）
""")
