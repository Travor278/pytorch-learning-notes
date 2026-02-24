# sampling_strategies.py
# 文本生成中的解码策略：Greedy、Top-k、Top-p（Nucleus）与温度采样
#
# 自回归语言模型每步产生一个词表分布，如何从这个分布中"选词"
# 直接影响生成文本的质量、多样性和连贯性。
#
# 参考：Holtzman et al., "The Curious Case of Neural Text Degeneration", ICLR 2020（Top-p）

import torch
import torch.nn.functional as F

# ========== 1. 贪心解码（Greedy Decoding）==========
print("=== 贪心解码 ===")
#
# 每步选择概率最大的 token：
#   token = argmax p(x_t | x_{<t})
#
# 优点：确定性，可复现，速度快
# 缺点：局部最优不等于全局最优——
#       有时选一个当前概率略低的词，后续可以接更通顺的句子
#       导致"重复性退化"（repetition degeneration）：
#       "The cat sat on the mat. The cat sat on the mat. The cat..."

def greedy_decode(logits):
    """logits: (vocab_size,)"""
    return logits.argmax().item()

# 模拟一个小词表的 logit 分布
vocab_size = 8
torch.manual_seed(42)
logits = torch.tensor([0.1, 2.5, 0.3, 1.8, 0.5, 0.2, 1.1, 0.4])

probs  = F.softmax(logits, dim=-1)
token  = greedy_decode(logits)
print(f"概率分布: {probs.numpy().round(3).tolist()}")
print(f"贪心选取 token: {token}  （概率 {probs[token]:.3f}）")
print()

# ========== 2. 温度采样（Temperature Sampling）==========
print("=== 温度采样 ===")
#
# 与 KL 蒸馏中的温度含义完全相同：
#   p_T(x) = softmax(logits / T)
#
#   T < 1：分布更"尖锐"，倾向于选高概率词，输出更保守、重复
#   T = 1：原始分布，按模型真实概率随机采样
#   T > 1：分布更"平坦"，低概率词也有机会被选，输出更多样但可能不连贯
#   T → 0：等价于贪心；T → ∞：等价于均匀随机

def temperature_sample(logits, T=1.0, n_samples=1000):
    probs = F.softmax(logits / T, dim=-1)
    tokens = torch.multinomial(probs, n_samples, replacement=True)
    return tokens, probs

print(f"{'T':>6}  {'max_prob':>10}  {'entropy':>10}  {'token 0~7 采样频率'}:")
for T in [0.5, 1.0, 1.5, 2.0]:
    tokens, probs = temperature_sample(logits, T, n_samples=5000)
    counts = torch.bincount(tokens, minlength=vocab_size).float() / 5000
    entropy = -(probs * probs.log()).sum().item()
    print(f"  T={T:.1f}: max_p={probs.max().item():.3f}  H={entropy:.3f}  "
          f"{counts.numpy().round(3).tolist()}")
print()

# ========== 3. Top-k 采样 ==========
print("=== Top-k 采样 ===")
#
# 每步只在概率最高的 k 个 token 中采样，其余 token 概率置为 0。
# 这避免了低概率的"奇怪"词被采到，同时保留一定多样性。
#
# 缺点：k 是固定超参数，但最优的 k 随上下文而变——
#   有时模型非常确信下一个词（分布很尖，k=10 已经足够）；
#   有时多个词都合理（分布很平，k=10 截断了太多合理选择）。

def top_k_sample(logits, k=5, T=1.0):
    """保留 top-k 个 token，其余置为 -inf"""
    if k == 0:
        return temperature_sample(logits, T, 1)[0][0].item()

    top_k_vals, top_k_idx = torch.topk(logits, k)
    filtered = torch.full_like(logits, float('-inf'))
    filtered.scatter_(0, top_k_idx, top_k_vals)   # 恢复 top-k 的 logit
    probs = F.softmax(filtered / T, dim=-1)
    token = torch.multinomial(probs, 1).item()
    return token

print(f"Top-k 采样结果（100 次，k=3, T=1.0）:")
k_counts = torch.zeros(vocab_size)
for _ in range(1000):
    t = top_k_sample(logits, k=3, T=1.0)
    k_counts[t] += 1
print(f"  采样频率: {(k_counts/1000).numpy().round(3).tolist()}")
print(f"  非 top-3 词被采到: {(k_counts[[0,2,4,5,7]] > 0).any().item()}  （应为 False）")
print()

# ========== 4. Top-p（Nucleus）采样 ==========
print("=== Top-p（Nucleus）采样 ===")
#
# Holtzman et al. 2020 提出：
#   每步动态选取累积概率恰好超过 p 的最小 token 集合（nucleus），
#   只在这个集合内采样。
#
# 相比 Top-k 的优势：
#   当分布集中时（模型很确信），nucleus 很小（仅 1~2 个词）
#   当分布均匀时（多个词合理），nucleus 很大（允许更多多样性）
#   k 随上下文动态变化，更合理
#
# 参数 p=0.9 通常是最优的（GPT-2/3 生成时的默认值）

def top_p_sample(logits, p=0.9, T=1.0):
    """Nucleus 采样（Top-p）"""
    # 按概率降序排列
    probs_sorted, indices_sorted = torch.sort(
        F.softmax(logits / T, dim=-1), descending=True
    )
    # 计算累积概率
    cumsum = torch.cumsum(probs_sorted, dim=-1)
    # 找到累积概率超过 p 的位置，之后的 token 置为 0
    # 注意 shift 一位：确保选取累积恰好超过 p 的那个 token
    cumsum_shifted = torch.roll(cumsum, 1, dims=-1)
    cumsum_shifted[0] = 0.0
    remove_mask = cumsum_shifted > p   # 这些位置概率太低，移除
    probs_sorted[remove_mask] = 0.0
    probs_sorted /= probs_sorted.sum()  # 重新归一化

    # 采样后还原到原始 index
    token_rank = torch.multinomial(probs_sorted, 1).item()
    return indices_sorted[token_rank].item()

print(f"Top-p 采样结果（1000 次，p=0.9）:")
p_counts = torch.zeros(vocab_size)
for _ in range(1000):
    t = top_p_sample(logits, p=0.9, T=1.0)
    p_counts[t] += 1
print(f"  采样频率: {(p_counts/1000).numpy().round(3).tolist()}")

# 展示不同 p 对 nucleus 大小的影响
print(f"\n不同 p 对应 nucleus 大小（当前分布）:")
probs_sorted, _ = torch.sort(F.softmax(logits, dim=-1), descending=True)
cumsum = torch.cumsum(probs_sorted, dim=0)
for p_val in [0.5, 0.7, 0.9, 0.95]:
    nucleus_size = (cumsum <= p_val).sum().item() + 1
    print(f"  p={p_val}: nucleus_size={nucleus_size} 个 token")
print()

# ========== 5. Repetition Penalty ==========
print("=== Repetition Penalty（重复惩罚）===")
#
# 贪心解码容易重复（"the cat the cat the cat..."）。
# 重复惩罚：将已生成过的 token 的 logit 除以惩罚因子 θ > 1，
#           降低它们被再次采到的概率。
#
# Keskar et al. 2019 & 业界常用：θ ≈ 1.1~1.3

def apply_repetition_penalty(logits, past_tokens, penalty=1.3):
    """
    logits     : (vocab_size,) 当前步的 logit
    past_tokens: 已生成的 token id 列表
    penalty    : 惩罚因子（> 1），越大重复越少
    """
    for token_id in set(past_tokens):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty  # 负值时乘以 penalty 使其更负
    return logits

logits_rep = logits.clone()
past = [1, 1, 3]   # token 1 已出现 2 次，token 3 出现 1 次

print(f"惩罚前 token 1 logit: {logits_rep[1]:.3f}, token 3 logit: {logits_rep[3]:.3f}")
logits_penalized = apply_repetition_penalty(logits_rep, past, penalty=1.3)
print(f"惩罚后 token 1 logit: {logits_penalized[1]:.3f}, token 3 logit: {logits_penalized[3]:.3f}")
print()

print("=== 总结 ===")
print("""
1. 贪心解码：确定性快速，但容易重复，缺乏多样性
2. 温度采样：T<1 保守，T>1 多样；T=0.7~1.0 是文本生成常用范围
3. Top-k：只在 k 个最高概率词中采样；k=40~50 常见，但 k 是固定值
4. Top-p（推荐）：动态 nucleus，分布集中时自动缩小候选集；p=0.9 是默认值
5. 实践：T + Top-p 组合（如 T=0.8, p=0.9）是当前生产系统的主流设置
""")
