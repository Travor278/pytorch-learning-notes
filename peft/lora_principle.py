# lora_principle.py
# LoRA：低秩分解的参数高效微调
#
# 全量微调（Full Fine-tuning）的问题：
#   对每个下游任务都复制一份完整的大模型参数（LLaMA-7B 约 28GB FP32），
#   存储和通信成本极高，多任务部署几乎不可行。
#
# LoRA 的核心假设（Hu et al. 2021）：
#   预训练模型的权重更新 ΔW 本征维度（intrinsic rank）很低——
#   模型"需要学"的增量信息可以用远小于全参数的低秩矩阵表示。
#
# 参考：Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. LoRA 数学原理 ==========
print("=== LoRA 原理 ===")
#
# 原始权重矩阵 W₀ ∈ R^{d×k}，全量微调学习 ΔW ∈ R^{d×k}。
# LoRA 约束 ΔW = B·A，其中：
#   A ∈ R^{r×k}  （down-projection）
#   B ∈ R^{d×r}  （up-projection）
#   r << min(d, k)（低秩约束）
#
# 前向计算：
#   h = W₀ x + (α/r) · B·A·x
#       ↑            ↑
#   冻结原始权重  可训练的低秩增量（α 是缩放超参数）
#
# 参数量对比：
#   全量微调: d × k 参数（如 4096×4096 = 16.7M）
#   LoRA r=8: r×(d+k) = 8×(4096+4096) = 65K  ← 缩减 256 倍
#
# α/r 的设计：
#   α 固定（默认等于 r），等比于 r 缩放增量幅度。
#   这样调整 r 时不需要重新调 α——
#   r=8 和 r=32 用相同 α，增量幅度等效（均乘以 α/r）。
#
# 初始化：
#   A 用高斯初始化（类似原始线性层），B 初始化为 0。
#   → 训练开始时 ΔW = B·A = 0，模型行为与预训练相同，确保稳定的起点。

d_out, d_in, r = 4096, 4096, 8
params_full = d_out * d_in
params_lora = r * (d_in + d_out)
print(f"全量微调参数: {params_full:,}")
print(f"LoRA r={r} 参数: {params_lora:,}  （减少 {params_full // params_lora}×）")
print()

# ========== 2. LoRALinear 实现 ==========
print("=== LoRALinear 实现 ===")

class LoRALinear(nn.Module):
    """
    将预训练的 nn.Linear 包装为 LoRA 版本。
    原始权重 W₀ 冻结，只训练低秩分解参数 A 和 B。
    """
    def __init__(self, original_linear: nn.Linear, r: int, alpha: float = None):
        super().__init__()
        self.r     = r
        self.alpha = alpha if alpha is not None else r   # 默认 α = r
        self.scale = self.alpha / r                      # 最终缩放因子

        d_out, d_in = original_linear.weight.shape

        # 冻结原始权重（不参与梯度计算）
        self.weight = original_linear.weight
        self.bias   = original_linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # 可训练的低秩参数
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))  # B 初始化为 0

        # A 的初始化：高斯，使得 A·x 的方差合理
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        # 原始前向：W₀ x + b（W₀ 已冻结）
        out = F.linear(x, self.weight, self.bias)
        # LoRA 增量：(α/r) · B · A · x
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T   # (... , d_out)
        return out + self.scale * lora_out

    def merge_weights(self):
        """
        推理时可将 LoRA 合并回原始权重：W = W₀ + (α/r)·B·A
        合并后等价于普通 Linear，零额外计算开销
        """
        with torch.no_grad():
            merged_weight = self.weight + self.scale * (self.lora_B @ self.lora_A)
        return merged_weight

# 测试
original = nn.Linear(512, 256)
lora_layer = LoRALinear(original, r=8, alpha=16)

# 参数统计
total_params    = sum(p.numel() for p in lora_layer.parameters())
trainable_params= sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
print(f"总参数量:  {total_params:,}")
print(f"可训练参数: {trainable_params:,}  （只有 lora_A 和 lora_B）")
print(f"训练比例:  {trainable_params / total_params * 100:.2f}%")

x_test = torch.randn(4, 64, 512)
out    = lora_layer(x_test)
print(f"\n输入: {x_test.shape} → 输出: {out.shape}")

# 验证初始化：LoRA 增量应为 0（B=0）
with torch.no_grad():
    out_orig  = F.linear(x_test, original.weight, original.bias)
    out_lora  = lora_layer(x_test)
    delta     = (out_orig - out_lora).abs().max().item()
print(f"初始 LoRA 增量（期望≈0）: {delta:.6f}")
print()

# ========== 3. 将模型的 Linear 层替换为 LoRA ==========
print("=== 替换模型中的 Linear 层 ===")
#
# 实际应用中，只对注意力层的 Q、V 投影矩阵加 LoRA（Hu et al. 建议）。
# 对 K 和 FFN 层加 LoRA 效果提升有限但参数增加明显。
# 实验显示 Q+V LoRA ≈ 全量微调效果（在推荐量 r=4~16 时）。

class TinyTransformerLayer(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.o_proj  = nn.Linear(d_model, d_model)
        self.ffn_fc1 = nn.Linear(d_model, d_model * 4)
        self.ffn_fc2 = nn.Linear(d_model * 4, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 简化：不做完整 attention，直接用 v 作为 attention 输出
        out = self.norm(x + self.o_proj(v))
        out = self.norm(out + self.ffn_fc2(F.gelu(self.ffn_fc1(out))))
        return out

def apply_lora(model, r=8, alpha=16, target_modules=('q_proj', 'v_proj')):
    """
    将指定模块名替换为 LoRA 版本，其余层冻结
    """
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad_(False)

    # 替换目标层
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # 获取模块名的最后一级
        short_name = name.split('.')[-1]
        if short_name not in target_modules:
            continue

        # 用 LoRALinear 替换
        parent_name, _, child_name = name.rpartition('.')
        parent = model if not parent_name else dict(model.named_modules())[parent_name]
        setattr(parent, child_name if child_name else name,
                LoRALinear(module, r=r, alpha=alpha))

model = TinyTransformerLayer(d_model=256)
total_before = sum(p.numel() for p in model.parameters())

apply_lora(model, r=8, target_modules=('q_proj', 'v_proj'))

total_after     = sum(p.numel() for p in model.parameters())
trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数量:  {total_after:,}")
print(f"可训练参数: {trainable_after:,}")
print(f"训练比例:  {trainable_after / total_after * 100:.2f}%  （对比全量100%）")

x = torch.randn(2, 16, 256)
out = model(x)
print(f"前向传播正常: {out.shape}")
print()

# ========== 4. r 的选择 ==========
print("=== rank r 的选择 ===")
print("""
r 的影响：
  r=1~4  : 极少参数，适合简单任务（文本分类）或测试
  r=8~16 : 大多数论文的推荐值，在参数效率和性能间折中
  r=32+  : 接近全量微调，适合复杂任务（代码生成、数学推理）
  r=d    : 退化为全量微调（LoRA 的上界）

不同层的适合 r：
  注意力 Q/V：r=8 通常足够
  FFN 层    ：提升效果有限，通常不加
  Embedding ：不推荐加 LoRA（词汇分布变化需要全量更新）

实践经验（LLaMA-7B 上）：
  r=8, alpha=16, 只加 q_proj/v_proj → MMLU 接近全量微调 95% 性能
  参数量：约 4M / 7B ≈ 0.06%
""")

print("=== 总结 ===")
print("""
1. LoRA: W = W₀ + (α/r)·B·A，W₀ 冻结，只训练 A（高斯初始化）和 B（零初始化）
2. B 零初始化确保训练起点 ΔW=0，保持预训练模型的稳定性
3. 推理时可合并 ΔW 到 W₀，不增加推理延迟
4. 通常只对 Q/V 投影加 LoRA，FFN 和 K 收益有限
5. r=8~16, α=r 是大多数场景的合理起点
""")
