# beam_search.py
# Beam Search：序列搜索与长度惩罚
#
# Beam Search 是机器翻译、摘要等需要高质量输出场景的标准解码算法。
# 在每步保留得分最高的 B 个候选序列（beam），兼顾搜索质量和计算开销。
#
# 参考：
#   - Sutskever et al., "Sequence to Sequence Learning with NNs", NeurIPS 2014
#   - Wu et al., "Google's NMT System", arXiv 2016（长度惩罚）

import torch
import torch.nn.functional as F
import math

# ========== 1. Beam Search 核心算法 ==========
print("=== Beam Search 原理 ===")
print("""
Greedy 解码（beam=1）：
  每步只保留最优 token，时间复杂度 O(T·V)

Beam Search（beam=B）：
  每步保留 B 条路径（beam），对每条路径扩展 V 个 token，
  从 B×V 个候选中选出得分最高的 B 个继续搜索。
  时间复杂度 O(T·B·V)，空间 O(T·B)

得分：log 概率的累加：
  score(x_{1:t}) = Σ_{i=1}^{t} log p(x_i | x_{<i})

问题：直接用 log 概率之和会偏向短序列（每一步 log p ≤ 0）
解决：长度惩罚（Length Penalty，Wu et al. 2016）：
  score_norm(x) = score(x) / ((5 + |x|)^α / (5 + 1)^α)
  α ∈ [0, 1]，α=0 不惩罚，α=1 完全归一化到平均 log prob
  GNMT 使用 α=0.6~0.7
""")

# ========== 2. 简化 Beam Search 实现 ==========
print("=== Beam Search 实现 ===")

class SimpleLanguageModel:
    """
    用于演示的微型"语言模型"：
    词表大小 8，每步产生下一个 token 的分布（固定，不依赖历史）
    """
    def __init__(self, vocab_size=8, seed=42):
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        # 预定义每步的条件分布（模拟不同 token 的续写概率）
        self.transitions = torch.randn(vocab_size, vocab_size)

    def next_logits(self, token_id, step=None):
        """给定当前 token，返回下一步 logit（简化：只依赖上一个 token）"""
        return self.transitions[token_id % self.vocab_size]

def beam_search(lm, init_token, beam_size, max_steps, alpha=0.6, eos_id=7):
    """
    lm        : 语言模型（提供 next_logits 接口）
    init_token: 起始 token id
    beam_size : beam 宽度
    max_steps : 最大生成长度
    alpha     : 长度惩罚系数
    eos_id    : 结束标记（到达 eos 停止扩展该 beam）

    返回：(最优序列, 归一化得分)
    """
    # 初始化：B 条路径各为 [init_token]，得分均为 0
    beams  = [[init_token]]   # 每个 beam 是一条路径
    scores = [0.0]            # 对应的累积 log 概率

    completed = []  # 已结束（到达 EOS）的 beam

    for step in range(max_steps):
        all_candidates = []

        for beam_idx, (beam, score) in enumerate(zip(beams, scores)):
            if beam[-1] == eos_id:
                # 已结束的 beam 直接保留（不再扩展）
                all_candidates.append((beam, score))
                continue

            # 获取下一步 logit
            last_token = beam[-1]
            logits     = lm.next_logits(last_token, step)
            log_probs  = F.log_softmax(logits, dim=-1)

            # 扩展所有可能的下一个 token
            for token_id in range(lm.vocab_size):
                new_beam  = beam + [token_id]
                new_score = score + log_probs[token_id].item()
                all_candidates.append((new_beam, new_score))

        # 按归一化得分排序，保留 top-B
        def length_penalty(length, alpha):
            return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

        def normalized_score(beam, score):
            return score / length_penalty(len(beam), alpha)

        all_candidates.sort(key=lambda x: normalized_score(x[0], x[1]), reverse=True)
        top_candidates = all_candidates[:beam_size]

        beams  = [c[0] for c in top_candidates]
        scores = [c[1] for c in top_candidates]

        # 检查是否所有 beam 都结束
        if all(b[-1] == eos_id for b in beams):
            break

    # 返回归一化得分最高的 beam
    best_beam, best_score = max(
        zip(beams, scores),
        key=lambda x: normalized_score(x[0], x[1])
    )
    return best_beam, normalized_score(best_beam, best_score)

lm = SimpleLanguageModel(vocab_size=8)

print("不同 beam size 的搜索结果（init_token=0）:")
for beam_size in [1, 3, 5]:
    seq, score = beam_search(lm, init_token=0, beam_size=beam_size, max_steps=6, alpha=0.6)
    print(f"  beam_size={beam_size}: 序列={seq}, 归一化得分={score:.4f}")
print()

# ========== 3. 长度惩罚的影响 ==========
print("=== 长度惩罚（alpha 参数）===")
print("""
没有长度惩罚（α=0）：beam search 倾向于输出短序列
  因为每个 token 都贡献负的 log 概率，越长总分越低

过度长度惩罚（α→1）：倾向于很长的输出
  强制按序列长度平均，模型会堆砌词语

实践推荐（翻译、摘要）：
  α=0.6~0.7（GNMT 论文建议）
  机器翻译：α=0.6，beam=4~6
  文本摘要：α=0.8~1.0（摘要较短，需要较强的长度惩罚）
  对话生成：通常不用 beam search（多样性更重要，用 top-p）
""")

print("不同 alpha 对长度的影响:")
for alpha in [0.0, 0.3, 0.6, 1.0]:
    seqs_lengths = []
    for init in range(8):  # 从不同起始 token 搜索
        seq, _ = beam_search(lm, init_token=init, beam_size=3, max_steps=8, alpha=alpha)
        seqs_lengths.append(len(seq))
    avg_len = sum(seqs_lengths) / len(seqs_lengths)
    print(f"  α={alpha:.1f}: 平均生成长度 {avg_len:.1f}")
print()

# ========== 4. Diverse Beam Search（简介）==========
print("=== Diverse Beam Search（简介）===")
print("""
标准 Beam Search 的问题：B 条 beam 往往高度相似（只有末尾 1~2 个词不同）
本质原因：都从同一前缀扩展，早期歧异小

Diverse Beam Search（Vijayakumar et al. 2016）：
  将 B 条 beam 分成 G 组，每组 beam_size / G 条。
  同组内用标准 beam search；不同组之间引入 diversity penalty：
    对于已被其他组选择的 token，降低当前组选中的可能性。
  使得 B 条 beam 覆盖更多样的候选答案。

实际应用：
  - 多样化摘要/翻译生成（给用户提供多个选项）
  - 数据增强
  - 评分模型重排（beam 生成候选 + reranker 选最优）
""")

# ========== 5. Beam Search vs 采样的对比 ==========
print("=== 解码策略选型 ===")
print("""
任务                    推荐解码策略
────────────────────────────────────────────
机器翻译（准确性优先）  Beam Search（beam=4, α=0.6）
文本摘要（准确性优先）  Beam Search（beam=4, α=0.8）
开放域对话（多样性）    Top-p（p=0.9）+ 温度（T=0.8）
创意写作               Top-p + 高温度（T=1.0~1.2）
代码生成               Greedy 或低温度（T=0.2）Top-p
数学/推理（思维链）     贪心 + 重复多次取最优（self-consistency）
""")
