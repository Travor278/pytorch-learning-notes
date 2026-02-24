# clip_contrastive.py
# CLIP：图文对比学习与零样本视觉-语言对齐
#
# CLIP（Contrastive Language-Image Pre-training）通过在 4 亿图文对上做
# 对比学习，学到统一的图文语义空间，实现强大的零样本迁移能力。
# 是 LLaVA / SEED 等多模态大模型的视觉编码器基础。
#
# 参考：Radford et al., "Learning Transferable Visual Models from Natural Language Supervision",
#       ICML 2021（CLIP 原文）

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. CLIP 的训练目标 ==========
print("=== CLIP 训练目标 ===")
print("""
训练数据：N 个图文对 {(I_1, T_1), ..., (I_N, T_N)}
编码器：
  图像编码器 f（ViT 或 ResNet）：I → z_I ∈ R^d
  文本编码器 g（Transformer）：T → z_T ∈ R^d

目标：让配对的 (z_I, z_T) 余弦相似度高，非配对的低。
这是双向 InfoNCE（NT-Xent 的变体）：
  L = (1/2) [L_{I→T} + L_{T→I}]

  L_{I→T} 第 i 行：
    -log[ exp(cos(z_I_i, z_T_i)/τ) / Σ_j exp(cos(z_I_i, z_T_j)/τ) ]
  （给定图像 i，在 N 个文本中找出配对的文本 i）

对称地，L_{T→I} 从文本找图像。
τ 是可学习的温度参数（初始化为 log(1/0.07)=e^{2.66}，约 0.07）。

关键设计：
  ① 大 batch size（CLIP 用 32768！）：提供充足的负样本
  ② L2 归一化嵌入：余弦相似度 = 内积，无需除以范数
  ③ 可学习 τ：自动找到最合适的分布温度
""")

# ========== 2. CLIP 损失实现 ==========
print("=== CLIP 损失实现 ===")

class CLIPLoss(nn.Module):
    def __init__(self, init_temp_logit=2.659):  # log(1/0.07) ≈ 2.659
        super().__init__()
        # 可学习的温度（logit_scale = log(1/τ)，τ = exp(-logit_scale)）
        self.logit_scale = nn.Parameter(torch.tensor(init_temp_logit))

    def forward(self, image_embeds, text_embeds):
        """
        image_embeds: (N, d) 图像嵌入（L2 归一化后）
        text_embeds : (N, d) 文本嵌入（L2 归一化后）
        """
        # L2 归一化（确保余弦相似度 = 内积）
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds  = F.normalize(text_embeds,  dim=-1)

        # 限制 logit_scale 防止数值爆炸（原文 clip 到 log(100)=4.605）
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # 相似度矩阵 (N, N)：sim[i][j] = 图像 i 与文本 j 的余弦相似度
        sim_matrix = logit_scale * torch.mm(image_embeds, text_embeds.T)

        # 对角线是配对样本（正样本），其余是负样本
        N      = image_embeds.size(0)
        labels = torch.arange(N, device=image_embeds.device)

        # 双向 InfoNCE 对称平均
        loss_i2t = F.cross_entropy(sim_matrix, labels)    # 图像→文本
        loss_t2i = F.cross_entropy(sim_matrix.T, labels)  # 文本→图像
        loss     = (loss_i2t + loss_t2i) / 2

        return loss, sim_matrix

clip_loss = CLIPLoss()
print(f"初始温度 τ: {clip_loss.logit_scale.exp().item():.4f}  "
      f"（1/τ={1/clip_loss.logit_scale.exp().item():.2f}）")

# 模拟一个 batch
N, d = 16, 512
torch.manual_seed(42)
image_embeds = torch.randn(N, d)  # 实际为 ViT 输出
text_embeds  = torch.randn(N, d)  # 实际为 Transformer 输出

# 让配对的图文嵌入略微相似（加小扰动）
text_embeds = text_embeds + 0.2 * image_embeds

loss, sim = clip_loss(image_embeds, text_embeds)
print(f"\n批量 N={N}, d={d}")
print(f"CLIP 损失: {loss.item():.4f}")
print(f"相似度矩阵对角线（配对）均值: {sim.diagonal().mean().item():.3f}")
print(f"相似度矩阵非对角线（非配对）均值: {(sim.sum() - sim.trace()) / (N*(N-1)):.3f}")
print()

# ========== 3. 零样本分类 ==========
print("=== 零样本图像分类 ===")
#
# CLIP 的零样本分类流程：
#   ① 对每个类别，将其名称放入模板："a photo of a {class_name}"
#   ② 用文本编码器将所有类别模板编码为 z_T（class_embeds）
#   ③ 将测试图像编码为 z_I
#   ④ 分类 = argmax_{class} cos(z_I, z_T_class)
#
# 这相当于把每个类别文本作为"分类器权重"，
# 用自然语言描述替代了 ImageNet 式的线性分类头。

def zero_shot_classify(image_embed, class_embeds, class_names):
    """
    image_embed  : (d,) 单张图像嵌入（L2 归一化）
    class_embeds : (C, d) 各类别文本嵌入（L2 归一化）
    """
    image_embed  = F.normalize(image_embed, dim=-1)
    class_embeds = F.normalize(class_embeds, dim=-1)

    # 计算与每个类别的余弦相似度
    sims   = torch.mv(class_embeds, image_embed)  # (C,)
    probs  = F.softmax(sims * 100, dim=-1)         # 温度=100（接近 argmax）
    pred   = probs.argmax().item()

    print(f"预测类别: {class_names[pred]}  （置信度 {probs[pred].item():.3f}）")
    for i, (name, p) in enumerate(zip(class_names, probs)):
        marker = " ← 预测" if i == pred else ""
        print(f"  {name:15s}: {p.item():.3f}{marker}")

# 模拟 3 个类别的文本嵌入（猫/狗/鸟）
class_names = ["cat", "dog", "bird"]
C = len(class_names)
d = 512

torch.manual_seed(0)
# 真实 CLIP 中，这里是文本编码器的输出；这里用随机向量模拟
class_embeds = torch.randn(C, d)

# 模拟一张"猫"图像（嵌入接近 class_embeds[0]）
image_embed  = class_embeds[0] + 0.3 * torch.randn(d)

print("零样本分类示例（图像接近 'cat' 类别的嵌入）:")
zero_shot_classify(image_embed, class_embeds, class_names)
print()

# ========== 4. CLIP 在多模态大模型中的角色 ==========
print("=== CLIP 在多模态大模型中的角色 ===")
print("""
LLaVA / SEED-LLaVA 的架构：
  ┌──────────────┐   ┌───────────────────┐   ┌──────────────┐
  │  图像        │→  │ CLIP ViT（冻结）  │→  │ Projection   │→  LLM (LLaMA)
  │              │   │ 视觉编码器        │   │ MLP/Linear   │
  └──────────────┘   └───────────────────┘   └──────────────┘

CLIP ViT 作为冻结的视觉编码器：
  ① 将图像切成 14×14 patches，编码为 token 序列
  ② 已在 4 亿图文对上预训练，具备强大的语义理解能力
  ③ 冻结 CLIP 权重，只训练 Projection 层和 LLM 部分

SEED（Li et al. 2024）的改进：
  不直接用 CLIP token 作为视觉输入，
  而是将图像 tokenize 为离散 token（通过 VQVAE），
  然后通过知识蒸馏让 LLM 学会从 token 序列还原图像语义。
  核心损失：KL(q_student ‖ p_teacher)，student = LLM，teacher = CLIP

为什么 CLIP 是基础：
  CLIP 的嵌入空间已经与自然语言对齐，
  LLM 可以通过相对少量的 alignment 训练与 CLIP 空间对接，
  而不需要从头学习视觉语义。
""")

print("=== 总结 ===")
print("""
1. CLIP 用双向 InfoNCE 在图文对上对比学习，使图像和文本嵌入到同一语义空间
2. 可学习温度 τ（logit_scale 参数），限制上限防止数值溢出
3. 零样本分类：文本类别描述 → 文本嵌入，与图像嵌入做最近邻
4. CLIP ViT 是现代多模态大模型（LLaVA/SEED）的标准视觉编码器
5. 大 batch（CLIP 用 32768）对对比学习效果至关重要
""")
