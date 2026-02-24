# gradient_checkpoint.py
# 梯度检查点（Gradient Checkpointing / Activation Recomputation）
#
# 核心动机：训练深度模型时，反向传播需要保存所有层的激活值（前向输出），
# 以便计算梯度。对于 Transformer，激活值显存占用 O(L·B·S·d)，远超模型参数本身。
# Gradient Checkpointing 以额外计算换显存：只保存部分"检查点"激活，
# 其余在反向传播时重新计算。
#
# 来源：Chen et al., "Training Deep Nets with Sublinear Memory Cost", arXiv 2016.
# 显存从 O(L) 降至 O(√L)（均匀间隔检查点时），代价是额外约 33% 前向计算。

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# ========== 1. 显存占用的根源 ==========
print("=== 激活值显存分析 ===")
#
# 标准反向传播需要在内存中同时保存：
#   ① 模型参数（weights）：固定，约 4 bytes/param（FP32）
#   ② 梯度（gradients）：与参数等大
#   ③ 优化器状态（Adam 需要 m, v）：参数量的 2 倍
#   ④ 激活值（activations）：前向传播的中间结果
#
# 对 Transformer：每层激活约 12·B·S·d bytes（FP32），
# 12 层 BERT-base（d=768, S=512, B=16）激活值 ≈ 12×12×16×512×768 ×4 ≈ 4.5 GB
# 这往往是训练显存瓶颈，而不是参数本身（参数只有 ~440 MB）。
#
# Gradient Checkpointing 的策略：
#   将网络分为若干段（segment），每段的输入作为检查点（checkpoint）保存，
#   段内的中间激活在反向传播时按需重算。
#   均匀分段 √L 个检查点时，总显存 O(√L)，重算代价约一次额外前向。

class TransformerBlock(nn.Module):
    """一个简化的 Transformer encoder block（省略 attention，用于演示开销）"""
    def __init__(self, d_model=256):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_model * 4)
        self.fc2  = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.norm(x + self.fc2(self.act(self.fc1(x))))

# ========== 2. 普通前向 vs checkpoint 前向 ==========
print("=== 普通前向 vs checkpoint 前向 ===")

d_model = 256
N_LAYERS = 16
blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(N_LAYERS)])

x = torch.randn(4, 64, d_model, requires_grad=True)  # (batch, seq, d_model)

# ---- 普通前向：所有激活保留在内存 ----
def forward_normal(x, blocks):
    for block in blocks:
        x = block(x)
    return x

# ---- checkpoint 前向：段内激活不保留，反向时重算 ----
def forward_with_checkpoint(x, blocks):
    for block in blocks:
        # checkpoint 包装后：不保存 block 内部激活，反向时重新执行 block.forward
        # use_reentrant=False：新版推荐写法，支持 torch.compile，避免潜在的 in-place 问题
        x = checkpoint.checkpoint(block, x, use_reentrant=False)
    return x

# 两者输出应相同
out_normal = forward_normal(x.clone().detach().requires_grad_(True), blocks)
out_ckpt   = forward_with_checkpoint(x.clone().detach().requires_grad_(True), blocks)

print(f"普通前向输出 shape: {out_normal.shape}")
print(f"checkpoint 输出 shape: {out_ckpt.shape}")
print(f"输出一致（数值相同）: {torch.allclose(out_normal, out_ckpt, atol=1e-5)}")
print()

# ========== 3. 显存对比（仅有 CUDA 时精确统计）==========
print("=== 显存对比 ===")

if torch.cuda.is_available():
    d = 512
    blks = nn.ModuleList([TransformerBlock(d) for _ in range(24)]).cuda()

    def measure_memory(use_ckpt):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        xi = torch.randn(8, 128, d, device='cuda', requires_grad=True)
        if use_ckpt:
            for blk in blks:
                xi = checkpoint.checkpoint(blk, xi, use_reentrant=False)
        else:
            for blk in blks:
                xi = blk(xi)
        xi.sum().backward()
        return torch.cuda.max_memory_allocated() / 1024**2  # MB

    mem_normal = measure_memory(False)
    mem_ckpt   = measure_memory(True)
    print(f"普通训练峰值显存: {mem_normal:.1f} MB")
    print(f"gradient ckpt 显存: {mem_ckpt:.1f} MB  （节省 {(1 - mem_ckpt/mem_normal)*100:.0f}%）")
else:
    print("无 CUDA，跳过显存统计。CPU 上可验证正确性，但无法准确测量显存。")
print()

# ========== 4. 分段策略（Segment Checkpoint）==========
print("=== 分段策略 ===")
#
# 不必每层都 checkpoint，可以每隔 k 层设一个检查点。
# k=1（每层）：最省显存，重算开销最大（+100% 前向）
# k=√L（均匀）：理论最优的显存-计算权衡（O(√L) 显存，O(1) 额外前向）
# k=L（不用）：标准训练，无额外计算
#
# 实践中 k=2 或 k=4 是常用的折中点：
#   - 显存节省 50%~75%
#   - 训练速度只慢 15%~25%

def forward_segment_checkpoint(x, blocks, ckpt_every=2):
    """每隔 ckpt_every 层做一次 checkpoint"""
    for i, block in enumerate(blocks):
        if i % ckpt_every == 0:
            x = checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x

x2 = torch.randn(4, 64, d_model, requires_grad=True)
out2 = forward_segment_checkpoint(x2, blocks, ckpt_every=2)
loss2 = out2.sum()
loss2.backward()
print(f"分段 checkpoint（每2层）梯度计算正常: {x2.grad is not None}")
print()

# ========== 5. 实际使用场景与注意事项 ==========
print("=== 注意事项 ===")
print("""
1. checkpoint 内不能有 in-place 操作（+=，ReLU inplace 等），会破坏重算一致性
2. 随机性操作（Dropout）在重算时需固定随机种子，checkpoint 内部自动处理（preserve_rng_state=True）
3. use_reentrant=False 是 PyTorch 2.x 推荐写法，兼容 torch.compile 和 DDP
4. HuggingFace Transformers：model.gradient_checkpointing_enable() 一键开启
5. 典型场景：长序列（S>2048）、大 batch size、显存受限的大模型微调
""")
