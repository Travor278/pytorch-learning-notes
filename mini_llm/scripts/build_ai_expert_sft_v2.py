from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "mini_llm" / "dataset"
EVAL_DIR = PROJECT_ROOT / "mini_llm" / "evals"


REFERENCE_DATASETS = [
    "HuggingFaceTB/smol-smoltalk",
    "mlabonne/FineTome-100k",
    "Open-Orca/OpenOrca",
    "glaiveai/glaive-function-calling-v2",
]


CARDS: list[dict[str, Any]] = [
    {
        "topic": "llm_architecture",
        "subtopic": "mha_mqa_gqa",
        "difficulty": "basic",
        "questions": [
            "MHA、MQA 和 GQA 的核心区别是什么？",
            "请从 KV cache 的角度对比 MHA、MQA、GQA。",
            "为什么很多新 LLM 会用 GQA 而不是标准 MHA？",
        ],
        "core": "MHA 为每个查询头保留独立的 K、V 头，表达能力强但 KV cache 最大。MQA 让所有查询头共享一组 K、V，显著减少推理显存和带宽。GQA 把查询头分成若干组，每组共享一组 K、V，是 MHA 效果和 MQA 推理效率之间的折中。",
        "details": "如果有 32 个 query heads，MHA 也有 32 组 KV，MQA 只有 1 组 KV，GQA 可能有 4 或 8 组 KV。推理时每生成一个 token 都要保存历史 K、V，所以减少 KV 头数能直接降低长上下文显存。",
        "practice": "工程上，小模型可先实现 MHA 保证正确性，再加入 n_kv_heads 参数支持 MQA/GQA。训练和推理必须保证 query heads 能整除 kv heads，否则 repeat_kv 或分组映射会出错。",
        "pitfall": "不要把 MQA/GQA 理解成减少 Q 头；它们主要共享 K、V，Q 头数量通常仍然保持较多。",
        "confusion": "有人说 MQA 是把所有 Q、K、V 都合成一个头，所以等价于单头注意力。",
        "required": ["MHA", "MQA", "GQA", "K", "V", "KV cache", "共享"],
        "forbidden": ["前额叶", "神经元", "BM25"],
        "eval_prompt": "用不超过 120 字解释 MHA、MQA、GQA 在 KV 头数量上的区别。",
    },
    {
        "topic": "llm_architecture",
        "subtopic": "rope",
        "difficulty": "basic",
        "questions": [
            "RoPE 位置编码的核心思想是什么？",
            "为什么 decoder-only LLM 常用 RoPE？",
            "RoPE 和绝对位置 embedding 有什么区别？",
        ],
        "core": "RoPE 通过按位置旋转 Q、K 向量，把位置信息注入注意力分数。旋转后的 Q 和 K 点积自然包含相对位置信息，因此很适合自回归注意力。",
        "details": "RoPE 不需要给 token embedding 直接相加一个绝对位置向量，而是在注意力内部作用于 Q、K。它通常与长上下文外推方法一起使用，例如 NTK scaling、YaRN 或位置插值。",
        "practice": "实现时要缓存 cos/sin，并保证不同 head 维度按偶数维成对旋转。扩展上下文长度时要谨慎测试困惑度和实际生成，因为简单拉长 max_position 不等于可靠外推。",
        "pitfall": "RoPE 不是一种训练损失，也不是检索算法；它解决的是序列位置信息如何进入 attention。",
        "confusion": "有人说 RoPE 是把 BM25 的词频分数加入注意力矩阵。",
        "required": ["RoPE", "Q", "K", "旋转", "相对位置"],
        "forbidden": ["BM25", "词频", "倒排索引"],
        "eval_prompt": "RoPE 为什么能表达相对位置信息？",
    },
    {
        "topic": "llm_architecture",
        "subtopic": "rmsnorm",
        "difficulty": "basic",
        "questions": [
            "RMSNorm 和 LayerNorm 有什么区别？",
            "为什么很多 LLM 使用 RMSNorm？",
            "RMSNorm 在 Transformer block 中起什么作用？",
        ],
        "core": "LayerNorm 会减去均值并按方差归一化，RMSNorm 只使用均方根做缩放，不显式中心化。RMSNorm 计算更简单，通常足以稳定 LLM 训练。",
        "details": "现代 decoder-only LLM 常用 pre-norm 结构，在 attention 和 FFN 之前做 RMSNorm。这样可以改善梯度流动，减少深层网络训练不稳定的问题。",
        "practice": "实现 RMSNorm 时通常保留可学习权重 weight，不需要 bias。数值上要加 eps，避免均方根接近 0 时出现不稳定。",
        "pitfall": "不要把 RMSNorm 当成正则化方法；它是归一化层，不直接替代 weight decay 或 dropout。",
        "confusion": "有人说 RMSNorm 的主要作用是随机丢弃神经元，类似 dropout。",
        "required": ["RMSNorm", "LayerNorm", "均方根", "归一化", "eps"],
        "forbidden": ["dropout", "随机丢弃"],
        "eval_prompt": "请说明 RMSNorm 与 LayerNorm 的主要区别。",
    },
    {
        "topic": "llm_architecture",
        "subtopic": "swiglu",
        "difficulty": "basic",
        "questions": [
            "SwiGLU 前馈网络为什么常见？",
            "LLM 里的 SwiGLU 起什么作用？",
            "SwiGLU 和普通 FFN 有什么不同？",
        ],
        "core": "SwiGLU 是一种带门控的前馈网络，通常用一支 SiLU 激活后的投影去门控另一支投影，再映射回隐藏维度。它比普通 ReLU/GELU FFN 有更强的表达能力。",
        "details": "许多 LLM 用 SwiGLU 替代两层 MLP。常见结构是 gate_proj、up_proj 和 down_proj，其中 gate 与 up 相乘后再 down 投影。",
        "practice": "实现时要注意中间维度通常不是简单的 4 倍 hidden size，而会按模型配置缩放并对齐到某个 multiple。权重形状错了会影响参数量和 checkpoint 兼容。",
        "pitfall": "SwiGLU 不是注意力机制；它位于 FFN/MLP 部分，和 QKV 投影不同。",
        "confusion": "有人说 SwiGLU 是一种新的 KV cache 管理算法。",
        "required": ["SwiGLU", "门控", "SiLU", "FFN", "up_proj", "down_proj"],
        "forbidden": ["KV cache", "PagedAttention"],
        "eval_prompt": "SwiGLU 相比普通两层 FFN 多了什么结构？",
    },
    {
        "topic": "llm_architecture",
        "subtopic": "dense_vs_moe",
        "difficulty": "basic",
        "questions": [
            "Dense 模型和 MoE 模型的本质区别是什么？",
            "为什么 MoE 参数多但单 token 计算量不一定很大？",
            "Dense FFN 和 MoE FFN 在激活参数上有什么不同？",
        ],
        "core": "Dense 模型中每个 token 都经过同一组完整 FFN 参数。MoE 把 FFN 拆成多个专家，每个 token 只路由到少数专家，因此总参数可以很大，但单 token 激活参数较少。",
        "details": "MoE 的关键是稀疏激活：路由器为每个 token 选择 top-k 专家。这样模型容量来自大量专家参数，计算量则主要由被激活的专家决定。",
        "practice": "工程上 MoE 需要处理专家并行、容量限制、token dispatch、combine、负载均衡损失和溢出 token。它不是把 Dense 模型简单复制几份就结束。",
        "pitfall": "不要把 MoE 的总参数量等同于每个 token 的实际计算量；MoE 报告里常区分 total parameters 和 active parameters。",
        "confusion": "有人说 MoE 推理时每个 token 都会经过全部专家，所以一定比 Dense 慢很多。",
        "required": ["Dense", "MoE", "专家", "路由", "top-k", "稀疏激活"],
        "forbidden": ["全部专家", "每个 token 都经过所有专家"],
        "eval_prompt": "Dense 模型和 MoE 模型在每个 token 激活参数上有什么区别？",
    },
    {
        "topic": "moe",
        "subtopic": "routing",
        "difficulty": "intermediate",
        "questions": [
            "MoE 中 routing 是怎么工作的？",
            "MoE 路由器如何选择专家？",
            "请说明 top-k routing 的基本流程。",
        ],
        "core": "MoE 路由器通常是一个线性层，为每个 token 计算所有专家的 logits。经过 softmax 得到专家权重后，选择 top-k 专家处理该 token，并把专家输出按路由权重加权合并。",
        "details": "训练时路由既要让 token 找到合适专家，也要避免所有 token 挤到少数专家。常见实现还会设置 expert capacity，超过容量的 token 可能被丢弃、延迟或回退到残差路径。",
        "practice": "调试 routing 要看每个专家收到的 token 数、路由熵、溢出率和 auxiliary loss。只看总 loss 可能看不出专家塌缩。",
        "pitfall": "路由器不是检索器，也不是把 token 送到向量数据库；它是在模型内部选择 FFN 专家。",
        "confusion": "有人说 MoE routing 是用 BM25 给专家建倒排索引。",
        "required": ["路由器", "logits", "softmax", "top-k", "专家", "加权"],
        "forbidden": ["BM25", "倒排索引", "向量数据库"],
        "eval_prompt": "MoE 的 top-k routing 通常包含哪几个步骤？",
    },
    {
        "topic": "moe",
        "subtopic": "load_balance",
        "difficulty": "intermediate",
        "questions": [
            "MoE 专家负载不均衡如何解决？",
            "什么是 MoE 的 load balancing loss？",
            "能否通过修改损失函数缓解 MoE 专家塌缩？",
        ],
        "core": "MoE 负载不均衡指路由器总把 token 分给少数专家，导致部分专家过载、部分专家得不到训练。常见缓解方式是加入 load balancing auxiliary loss，鼓励专家接收的 token 占比和路由概率更均匀。",
        "details": "除了辅助损失，还可以使用 capacity factor、路由噪声、top-k routing、专家 dropout、分组路由和更细的监控。损失函数能缓解问题，但不能替代正确的并行调度和容量设计。",
        "practice": "训练日志应记录每个专家 token 计数、最大/最小负载比、溢出 token 比例和 aux loss。若少数专家长期占比过高，应降低路由过度自信或增大均衡约束。",
        "pitfall": "不要把负载均衡 loss 设得无限大；过强均衡会迫使 token 去不合适的专家，损害模型质量。",
        "confusion": "有人说只要把 cross entropy 降低，MoE 负载自然一定均衡。",
        "required": ["负载均衡", "auxiliary loss", "专家", "token 占比", "capacity"],
        "forbidden": ["一定均衡", "无需监控"],
        "eval_prompt": "为什么 MoE 需要 load balancing auxiliary loss？",
    },
    {
        "topic": "peft",
        "subtopic": "lora_core",
        "difficulty": "basic",
        "questions": [
            "LoRA 的核心思想是什么？",
            "请简洁解释 LoRA 如何减少微调成本。",
            "LoRA 为什么属于参数高效微调？",
        ],
        "core": "LoRA 冻结原模型权重，只训练低秩增量矩阵，用 BA 近似权重更新 Delta W。前向时等价于原线性层输出加上一个低秩旁路输出。",
        "details": "如果原权重 W 是 d_out x d_in，LoRA 通常设置 A 为 r x d_in，B 为 d_out x r，r 远小于 d_in 和 d_out。这样训练参数远少于全参微调，训练后还可以把 BA 合并进 W。",
        "practice": "LLM 中常把 LoRA 插到 q_proj、k_proj、v_proj、o_proj 或 FFN 投影层。小数据集上要用较低学习率、合适 rank 和验证集，避免把底座风格训坏。",
        "pitfall": "LoRA 与神经科学没有直接关系；不要把它解释成模拟大脑神经元连接。",
        "confusion": "有人说 LoRA 是给模型添加新的神经元记忆模块，主要模拟前额叶学习。",
        "required": ["LoRA", "冻结", "低秩", "A", "B", "Delta W"],
        "forbidden": ["前额叶", "神经元", "大脑"],
        "eval_prompt": "LoRA 如何用低秩矩阵近似权重更新？",
    },
    {
        "topic": "peft",
        "subtopic": "lora_init_rank",
        "difficulty": "intermediate",
        "questions": [
            "LoRA 中 A、B 矩阵通常如何初始化？",
            "为什么 LoRA 常把 B 初始化为 0？",
            "LoRA 的 rank 应该怎么选？",
        ],
        "core": "常见 LoRA 初始化是 A 随机初始化，B 初始化为 0，使训练开始时低秩分支输出为 0，不会突然改变基座模型行为。rank r 控制低秩更新容量。",
        "details": "r 越大，表达能力越强，但训练参数、显存和过拟合风险也越高。小任务常用 r=4 或 8，复杂领域适配可尝试 16、32，再用验证集和实际生成质量选择。",
        "practice": "如果一开始 loss 或生成突然崩坏，要检查是否把 A、B 都随机初始化且缩放过大。alpha 通常用于缩放 LoRA 分支，实际缩放常写成 alpha / r。",
        "pitfall": "rank 不是越大越好；在小数据或弱底座上，过高 rank 会更容易记忆噪声。",
        "confusion": "有人说 LoRA 的 B 必须随机初始化，否则模型学不到任何东西。",
        "required": ["A", "B", "初始化", "B 为 0", "rank", "alpha"],
        "forbidden": ["必须随机初始化", "rank 越大越好"],
        "eval_prompt": "LoRA 为什么常用 A 随机、B 为 0 的初始化？",
    },
    {
        "topic": "peft",
        "subtopic": "qlora",
        "difficulty": "intermediate",
        "questions": [
            "QLoRA 和 LoRA 有什么区别？",
            "QLoRA 为什么能显著省显存？",
            "QLoRA 中量化的是哪些参数？",
        ],
        "core": "QLoRA 通常把冻结的基座模型权重量化到 4bit，同时只训练 LoRA 低秩参数。它减少了基座权重占用的显存，使单卡微调更大的模型成为可能。",
        "details": "QLoRA 的训练计算仍需要反量化参与前向和反向，但优化器状态主要对应 LoRA 参数。典型技术包括 NF4、double quantization 和 paged optimizer。",
        "practice": "使用 QLoRA 时要关注量化精度、目标模块、rank、学习率和显存峰值。过度量化或错误 target modules 会导致收敛差。",
        "pitfall": "QLoRA 不是把 LoRA 矩阵本身压成 4bit 后训练；通常训练的 LoRA 参数仍保持较高精度。",
        "confusion": "有人说 QLoRA 的核心是把所有梯度都量化成 1bit。",
        "required": ["QLoRA", "4bit", "基座模型", "冻结", "LoRA", "显存"],
        "forbidden": ["1bit", "所有梯度"],
        "eval_prompt": "QLoRA 相比普通 LoRA 主要省在哪里？",
    },
    {
        "topic": "alignment",
        "subtopic": "sft",
        "difficulty": "basic",
        "questions": [
            "预训练和 SFT 的区别是什么？",
            "为什么底座没训好时 SFT 只能学到回答腔调？",
            "SFT 数据构建最重要的是什么？",
        ],
        "core": "预训练用 next-token prediction 学语言、知识和模式，目标是让模型会续写。SFT 用指令或对话数据训练模型按用户意图回答，目标是让模型学会遵循指令和输出格式。",
        "details": "如果底座语言建模能力很弱，SFT 往往只能教会模板和语气，不能凭空补足大量知识。高质量 SFT 需要准确答案、稳定格式、主题覆盖、去重和合适配比。",
        "practice": "专业 SFT 应先做小型固定 eval，确认关键概念能答对，再扩大训练。不要只看训练 loss，因为 loss 下降可能来自记住模板。",
        "pitfall": "SFT 不是万能知识注入器；错误答案会被模型稳定模仿。",
        "confusion": "有人说只要 SFT 样本足够多，就不需要预训练。",
        "required": ["预训练", "SFT", "next-token", "指令", "格式", "质量"],
        "forbidden": ["不需要预训练", "万能"],
        "eval_prompt": "为什么 SFT 不能完全替代预训练？",
    },
    {
        "topic": "alignment",
        "subtopic": "dpo",
        "difficulty": "intermediate",
        "questions": [
            "DPO 的原理是什么？",
            "请写出 DPO 的 loss 公式并解释。",
            "DPO 和 PPO 在训练数据上有什么不同？",
        ],
        "core": "DPO 是离线偏好优化方法，直接使用 chosen/rejected 成对回答训练策略模型。它把奖励建模和策略优化合并成一个分类式目标，不需要在线 rollout。",
        "details": "常见损失为 -log sigmoid(beta * [(log pi_theta(y+|x)-log pi_theta(y-|x)) - (log pi_ref(y+|x)-log pi_ref(y-|x))])。它鼓励策略模型相对参考模型更偏向 chosen 而不是 rejected。",
        "practice": "DPO 数据要保证 chosen 和 rejected 的差异真实且可学习。beta 控制偏离参考模型的强度，太大可能过拟合偏好，太小则更新弱。",
        "pitfall": "DPO 不需要单独训练 reward model，但仍然需要参考模型和高质量偏好对。",
        "confusion": "有人说 DPO 必须先在线采样，再用环境奖励回传 advantage。",
        "required": ["DPO", "chosen", "rejected", "参考模型", "log sigmoid", "beta"],
        "forbidden": ["必须在线采样", "advantage"],
        "eval_prompt": "写出 DPO loss 的核心形式，并说明 chosen/rejected 的作用。",
    },
    {
        "topic": "alignment",
        "subtopic": "ppo_grpo",
        "difficulty": "intermediate",
        "questions": [
            "PPO、DPO、GRPO 的区别是什么？",
            "GRPO 为什么可以减少对 value model 的依赖？",
            "LLM 对齐里 PPO 的基本流程是什么？",
        ],
        "core": "PPO 是在线强化学习方法，通常用策略模型生成回答，奖励模型打分，再用优势函数和 KL 约束更新。DPO 是离线偏好优化，不需要 rollout。GRPO 用同一 prompt 的一组回答做相对奖励归一化，减少对独立 value model 的依赖。",
        "details": "PPO 更灵活，但工程复杂、显存和采样成本高。DPO 简单稳定，依赖偏好数据质量。GRPO 适合可验证奖励或多样本比较场景，但仍要控制 KL 和奖励噪声。",
        "practice": "小模型项目里，通常先做 SFT，再用 DPO 学偏好；PPO/GRPO 等到 rollout、奖励和评估闭环稳定后再上。",
        "pitfall": "GRPO 不是 DPO 的同义词；它仍然是基于采样回答和奖励的强化学习式优化。",
        "confusion": "有人说 GRPO 就是不带参考模型的 DPO。",
        "required": ["PPO", "DPO", "GRPO", "rollout", "奖励", "KL"],
        "forbidden": ["同义词", "不带参考模型的 DPO"],
        "eval_prompt": "简要对比 DPO、PPO 和 GRPO。",
    },
    {
        "topic": "alignment",
        "subtopic": "reward_hacking",
        "difficulty": "intermediate",
        "questions": [
            "什么是 reward hacking？",
            "RLHF 中为什么需要 KL 约束？",
            "奖励模型可能带来哪些风险？",
        ],
        "core": "Reward hacking 指模型找到奖励函数漏洞，获得高奖励但输出质量或真实目标变差。RLHF 中常用 KL 约束限制策略偏离参考模型，减少语言退化和极端投机行为。",
        "details": "奖励模型只是真实偏好的近似，可能偏爱长答案、固定格式或看似自信的错误回答。KL、人工评估、规则验证和多样化偏好数据能降低风险。",
        "practice": "训练时要同时监控奖励、KL、长度、重复率和固定 eval。奖励升高但人工质量下降时，应降低学习率、增大 KL 或回退数据。",
        "pitfall": "奖励分数高不等于模型真的更好；必须结合固定评测和人工抽查。",
        "confusion": "有人说 RLHF 只要 reward 越高越好，不需要看 KL 或人工质量。",
        "required": ["reward hacking", "奖励模型", "KL", "参考模型", "人工评估"],
        "forbidden": ["越高越好", "不需要评估"],
        "eval_prompt": "为什么 RLHF/PPO 训练 LLM 时常加入 KL 惩罚？",
    },
    {
        "topic": "inference",
        "subtopic": "kv_cache",
        "difficulty": "basic",
        "questions": [
            "KV Cache 在自回归推理中有什么作用？",
            "为什么生成时要缓存 K 和 V？",
            "KV cache 为什么会占用大量显存？",
        ],
        "core": "自回归生成每一步只新增一个 token，历史 token 的 K、V 不会改变。KV Cache 保存历史 K、V，下一步只计算新 token，并用新 Q 去 attend 历史 K、V。",
        "details": "KV cache 显存随 batch size、层数、KV 头数、head_dim 和上下文长度增长。长上下文和多并发请求会让 KV cache 成为推理显存瓶颈。",
        "practice": "实现 generate 时要区分 prefill 和 decode 阶段。prefill 处理整段 prompt，decode 每次追加一个 token 并更新 cache。",
        "pitfall": "KV cache 不是训练数据缓存，也不是 tokenizer 词表缓存。",
        "confusion": "有人说 KV cache 是把训练集样本缓存起来防止过拟合。",
        "required": ["KV Cache", "自回归", "K", "V", "prefill", "decode"],
        "forbidden": ["训练集", "词表缓存"],
        "eval_prompt": "KV Cache 为什么能减少自回归生成中的重复计算？",
    },
    {
        "topic": "inference",
        "subtopic": "paged_attention",
        "difficulty": "intermediate",
        "questions": [
            "vLLM 的 PagedAttention 解决了什么问题？",
            "PagedAttention 为什么能提升推理吞吐？",
            "PagedAttention 和普通 KV cache 管理有什么区别？",
        ],
        "core": "PagedAttention 把 KV cache 按块分页管理，类似操作系统的虚拟内存。它避免为每个请求预留连续大块显存，减少碎片并提高并发请求下的显存利用率。",
        "details": "在动态 batch 和不同长度请求混合时，普通连续 KV 分配容易浪费显存。PagedAttention 用 block table 维护逻辑序列到物理 KV block 的映射，因此更适合服务端高吞吐推理。",
        "practice": "PagedAttention 主要优化推理内存管理和调度，不改变模型训练目标。评估它要看吞吐、首 token 延迟、显存利用率和并发能力。",
        "pitfall": "PagedAttention 不是一种新的注意力数学公式，也不是 BM25 检索算法。",
        "confusion": "有人说 PagedAttention 是把 BM25 打分加入 attention，所以能检索更准。",
        "required": ["PagedAttention", "KV cache", "分页", "显存碎片", "并发", "vLLM"],
        "forbidden": ["BM25", "检索更准", "训练损失"],
        "eval_prompt": "PagedAttention 如何改善 KV cache 的显存管理？",
    },
    {
        "topic": "inference",
        "subtopic": "flash_attention",
        "difficulty": "intermediate",
        "questions": [
            "FlashAttention 的核心思想是什么？",
            "FlashAttention 为什么更省显存？",
            "FlashAttention 和 PagedAttention 分别解决什么问题？",
        ],
        "core": "FlashAttention 通过分块计算和优化内存访问，避免显式保存完整 attention 矩阵。它在保持精确 attention 结果的同时减少 HBM 读写和显存占用。",
        "details": "FlashAttention 主要优化 attention kernel 的计算和内存效率；PagedAttention 主要优化推理服务中的 KV cache 分配和调度。两者可以互补，但不是同一层问题。",
        "practice": "训练或 prefill 阶段常受 attention 矩阵显存影响，FlashAttention 收益明显。decode 阶段常受 KV cache 和小矩阵开销影响，需要结合 batching 和 cache 管理。",
        "pitfall": "FlashAttention 不是近似 attention；经典 FlashAttention 计算的是精确 softmax attention。",
        "confusion": "有人说 FlashAttention 是通过随机丢弃一半 token 来近似加速。",
        "required": ["FlashAttention", "分块", "HBM", "显存", "精确 attention"],
        "forbidden": ["随机丢弃", "近似"],
        "eval_prompt": "FlashAttention 和 PagedAttention 的优化对象有什么不同？",
    },
    {
        "topic": "inference",
        "subtopic": "quantization",
        "difficulty": "intermediate",
        "questions": [
            "INT8/INT4 量化为什么能加速推理？",
            "GPTQ 和 AWQ 大致有什么区别？",
            "量化会带来哪些风险？",
        ],
        "core": "量化把权重或激活从 FP16/BF16 压到 INT8、INT4 等低位宽，减少显存占用和内存带宽压力。有硬件 kernel 支持时，低位矩阵乘法还能提高吞吐。",
        "details": "GPTQ 常强调基于近似二阶信息的逐层后训练量化，AWQ 强调保护重要激活通道以降低权重量化误差。二者都常用于无需重新训练或少量校准的推理量化。",
        "practice": "量化前要准备校准集，并在目标任务 eval 上比较困惑度、准确率和生成质量。小模型或专业知识模型过度量化可能让概念边界更不稳。",
        "pitfall": "量化不是无损压缩；位宽越低，速度和质量的 trade-off 越明显。",
        "confusion": "有人说 INT4 量化一定不损失任何精度，而且所有 GPU 都会自动加速。",
        "required": ["量化", "INT8", "INT4", "GPTQ", "AWQ", "校准"],
        "forbidden": ["无损", "一定不损失", "所有 GPU"],
        "eval_prompt": "推理量化为什么需要校准集和任务评估？",
    },
    {
        "topic": "inference",
        "subtopic": "speculative_decoding",
        "difficulty": "intermediate",
        "questions": [
            "Speculative Decoding 是什么？",
            "投机解码如何加速大模型生成？",
            "draft model 和 target model 在投机解码中分别做什么？",
        ],
        "core": "投机解码用小的 draft model 先生成多个候选 token，再由大的 target model 并行验证。被接受的 token 可以一次前进多步，被拒绝时按规则修正。",
        "details": "它的目标是在保持目标模型分布正确的前提下减少大模型逐 token 调用次数。收益取决于 draft model 速度、候选接受率和验证开销。",
        "practice": "部署时要监控接受率、端到端 latency 和吞吐。draft model 太弱会导致频繁拒绝，反而增加复杂度。",
        "pitfall": "投机解码不是简单用小模型替代大模型；最终分布仍由 target model 校验约束。",
        "confusion": "有人说投机解码就是直接相信小模型输出，不再调用大模型。",
        "required": ["投机解码", "draft model", "target model", "验证", "接受率"],
        "forbidden": ["不再调用大模型", "直接替代"],
        "eval_prompt": "投机解码为什么需要 target model 验证 draft token？",
    },
    {
        "topic": "rag",
        "subtopic": "bm25",
        "difficulty": "basic",
        "questions": [
            "BM25 的计算原理是什么？",
            "BM25 为什么适合关键词检索？",
            "BM25 中 k1 和 b 大致控制什么？",
        ],
        "core": "BM25 是经典稀疏检索打分方法，根据查询词在文档中的词频、逆文档频率和文档长度归一化来计算相关性。IDF 提高稀有查询词的权重，词频项有饱和效果。",
        "details": "k1 控制词频增长的饱和速度，b 控制文档长度归一化强度。BM25 不依赖神经网络，适合精确关键词、术语和编号检索。",
        "practice": "RAG 中 BM25 常作为强基线，也常与向量检索混合召回。对专业术语、函数名、错误码和缩写，BM25 往往比纯向量检索更稳。",
        "pitfall": "BM25 不是 attention 机制，也不使用 Q、K、V。",
        "confusion": "有人说 BM25 是 Transformer 里计算注意力权重的公式。",
        "required": ["BM25", "词频", "IDF", "文档长度", "k1", "b"],
        "forbidden": ["Q", "K", "V", "注意力权重"],
        "eval_prompt": "BM25 打分中 IDF、k1、b 分别大致起什么作用？",
    },
    {
        "topic": "rag",
        "subtopic": "dense_retrieval",
        "difficulty": "basic",
        "questions": [
            "embedding 检索和 BM25 有什么区别？",
            "稀疏检索和向量检索如何互补？",
            "为什么 RAG 常用 hybrid search？",
        ],
        "core": "BM25 依赖词面匹配，擅长关键词、术语和精确字符串。embedding 检索把文本编码成向量，按语义相似度召回，擅长同义改写和概念相近的问题。",
        "details": "Hybrid search 结合 BM25 和向量检索，能同时覆盖精确匹配和语义召回。随后用 reranker 对候选文档重新排序，提高上下文质量。",
        "practice": "专业知识库里建议混合召回：BM25 保证术语不丢，embedding 找语义相近材料，reranker 控制最终上下文数量。",
        "pitfall": "向量检索不是 BM25 的完全替代；纯向量模型可能漏掉短编号、变量名和罕见术语。",
        "confusion": "有人说有了 embedding 后，BM25 在所有场景都没有价值。",
        "required": ["BM25", "embedding", "语义", "关键词", "hybrid", "reranker"],
        "forbidden": ["完全替代", "没有价值"],
        "eval_prompt": "为什么专业 RAG 常把 BM25 和向量检索混合使用？",
    },
    {
        "topic": "rag",
        "subtopic": "chunking",
        "difficulty": "intermediate",
        "questions": [
            "RAG 文档切分应该注意什么？",
            "chunk size 和 overlap 如何影响检索？",
            "为什么 RAG 需要 reranker？",
        ],
        "core": "RAG 文档切分要在语义完整性和召回精度之间折中。chunk 太大容易引入噪声，太小可能丢失上下文；overlap 能缓解边界切断问题。",
        "details": "Reranker 接收 query 和候选 chunk，做更精细的相关性判断。它通常比第一阶段召回更慢，但能显著提高送入生成模型的上下文质量。",
        "practice": "技术文档适合按标题、段落、代码块和表格边界切分，而不是固定字数硬切。评估时要看 answer recall、context precision 和人工可追溯性。",
        "pitfall": "把整本文档塞进上下文不等于 RAG；这会浪费窗口并增加干扰。",
        "confusion": "有人说 chunk 越大越好，因为模型能看到更多文字。",
        "required": ["chunk", "overlap", "召回", "reranker", "上下文"],
        "forbidden": ["越大越好", "整本文档"],
        "eval_prompt": "RAG 中 chunk 太大或太小分别有什么问题？",
    },
    {
        "topic": "rag",
        "subtopic": "rag_evaluation",
        "difficulty": "intermediate",
        "questions": [
            "如何评估 RAG 系统？",
            "RAG 的检索质量和生成质量应分别看什么？",
            "RAG 为什么需要可追溯引用？",
        ],
        "core": "RAG 评估要拆成检索和生成两部分。检索看相关文档是否被召回、排序是否靠前；生成看答案是否正确、是否基于上下文、是否给出可追溯引用。",
        "details": "常见指标包括 recall@k、MRR、nDCG、context precision、faithfulness 和 answer correctness。专业系统还要看无法回答时是否能承认资料不足。",
        "practice": "固定 eval set 应包含有答案、无答案、相似干扰文档和需要多跳组合的样例。只看最终回答会掩盖检索失败还是生成失败。",
        "pitfall": "RAG 不是把幻觉责任完全推给检索；生成模型仍可能无视证据或编造引用。",
        "confusion": "有人说只要向量库召回了内容，生成答案一定可信。",
        "required": ["RAG", "recall@k", "排序", "faithfulness", "引用"],
        "forbidden": ["一定可信", "无需评估"],
        "eval_prompt": "为什么 RAG 评估要拆分检索质量和生成质量？",
    },
    {
        "topic": "vlm",
        "subtopic": "clip",
        "difficulty": "basic",
        "questions": [
            "CLIP 的图文对齐原理是什么？",
            "CLIP 为什么能做零样本分类？",
            "CLIP 的 contrastive loss 在学什么？",
        ],
        "core": "CLIP 使用图像编码器和文本编码器，把匹配的图文对拉近，把不匹配图文拉远。训练目标是对比学习，让图像和文本进入同一个语义向量空间。",
        "details": "零样本分类时，把类别写成文本 prompt，计算图像向量与各类别文本向量的相似度，选择最相似的类别。CLIP 学到的是图文语义对齐，不是直接生成长文本。",
        "practice": "工程上要注意 prompt 模板、图像预处理和温度参数。中文场景可使用中文 CLIP 或先做多语言文本对齐。",
        "pitfall": "CLIP 不是 LLaVA；CLIP 通常不直接进行多轮视觉问答。",
        "confusion": "有人说 CLIP 是一个带 decoder 的聊天模型，可以直接回答复杂图像问题。",
        "required": ["CLIP", "图像编码器", "文本编码器", "对比学习", "零样本"],
        "forbidden": ["聊天模型", "decoder"],
        "eval_prompt": "CLIP 如何用图文对比学习支持零样本分类？",
    },
    {
        "topic": "vlm",
        "subtopic": "vit_patch",
        "difficulty": "basic",
        "questions": [
            "ViT 如何把图像输入 Transformer？",
            "Vision Transformer 的 patch embedding 是什么？",
            "为什么 ViT 需要位置编码？",
        ],
        "core": "ViT 把图像切成固定大小的 patch，将每个 patch 展平并线性投影成 token embedding，再加位置编码送入 Transformer。这样图像被转成类似文本 token 的序列。",
        "details": "patch size 会影响序列长度和细节保留。patch 越小，token 越多，计算更贵但细节更丰富；patch 越大，计算更省但可能损失局部信息。",
        "practice": "实现时要保证图像尺寸能被 patch size 整除，或采用 resize/padding。位置编码用于告诉模型 patch 的空间顺序。",
        "pitfall": "ViT 不是逐像素把每个 RGB 值当成一个 token；通常是 patch 级 token。",
        "confusion": "有人说 ViT 会把每个像素都当成独立 token，所以 224x224 图像有 50176 个 token。",
        "required": ["ViT", "patch", "embedding", "位置编码", "Transformer"],
        "forbidden": ["每个像素", "50176 个 token"],
        "eval_prompt": "ViT 的 patch embedding 把图像变成了什么形式？",
    },
    {
        "topic": "vlm",
        "subtopic": "llava",
        "difficulty": "basic",
        "questions": [
            "LLaVA 的基本结构是什么？",
            "LLaVA 如何把视觉特征接入大语言模型？",
            "LLaVA 中 projector 的作用是什么？",
        ],
        "core": "LLaVA 通常由视觉编码器、视觉投影层和语言模型组成。视觉编码器提取图像特征，projector 把视觉特征映射到 LLM hidden space，再作为视觉 token 与文本 token 一起输入 LLM。",
        "details": "训练通常分两步：先做图文对齐预训练，让 projector 学会连接视觉和语言空间；再做多模态指令 SFT，让模型学会图像问答、描述和推理。",
        "practice": "工程接入时要明确图像 token 插入位置、projector 维度、视觉特征层选择和 tokenizer 中的 image placeholder。文本 LLM 本身不认识原始像素。",
        "pitfall": "LLaVA 不是把图片文件路径直接喂给 LLM；必须先经过视觉编码器和投影层。",
        "confusion": "有人说 LLaVA 只需要把图片转成 base64 字符串拼进 prompt。",
        "required": ["LLaVA", "视觉编码器", "projector", "LLM", "视觉 token"],
        "forbidden": ["base64", "文件路径直接"],
        "eval_prompt": "LLaVA 中视觉 encoder、projector、LLM 分别承担什么作用？",
    },
    {
        "topic": "vlm",
        "subtopic": "vlm_training",
        "difficulty": "intermediate",
        "questions": [
            "VLM 训练通常分哪些阶段？",
            "图文大模型如何从对齐训练过渡到指令微调？",
            "多模态 SFT 和普通文本 SFT 有什么不同？",
        ],
        "core": "VLM 常先做图文对齐训练，让视觉特征能进入语言模型空间；再用多模态指令数据做 SFT，让模型学会根据图像和文本指令回答。后续还可做偏好优化或任务专项训练。",
        "details": "对齐阶段常训练 projector 或少量模块，指令阶段可解冻更多 LLM/vision 参数。数据包括描述、VQA、OCR、定位、图表理解和推理。",
        "practice": "小项目应先固定 vision encoder 和 LLM，只训练 projector 验证链路，再逐步解冻。eval 要覆盖有图、无图、OCR、数量和空间关系。",
        "pitfall": "文本 SFT 数据不能直接让模型学会看图；必须有图像特征参与训练。",
        "confusion": "有人说只要给 LLM 读足够多图片描述，它就能直接处理图像像素。",
        "required": ["VLM", "图文对齐", "projector", "多模态指令", "图像特征"],
        "forbidden": ["直接处理像素", "只靠文本"],
        "eval_prompt": "为什么 VLM 通常先做图文对齐再做多模态 SFT？",
    },
    {
        "topic": "vlm",
        "subtopic": "vlm_hallucination",
        "difficulty": "intermediate",
        "questions": [
            "VLM 容易出现哪些幻觉？",
            "图文模型为什么会看错图？",
            "如何缓解 VLM 的视觉幻觉？",
        ],
        "core": "VLM 幻觉包括不存在的物体、错误属性、数量错误、空间关系错误和 OCR 误读。原因可能是视觉分辨率不足、图像特征对齐弱、训练数据偏差或语言先验过强。",
        "details": "缓解方法包括更高质量图文数据、困难负样本、OCR/定位专项训练、拒答样本、视觉 grounding 评估和基于证据的回答约束。",
        "practice": "评估时不要只看描述是否流畅，要检查是否忠于图像。对机器人/VLA 场景，数量和空间关系错误会直接影响动作。",
        "pitfall": "VLM 回答自信不代表它真的看到了图像证据。",
        "confusion": "有人说 VLM 只要语言模型足够大，就不会产生视觉幻觉。",
        "required": ["VLM", "幻觉", "物体", "空间关系", "OCR", "grounding"],
        "forbidden": ["不会产生", "只要语言模型足够大"],
        "eval_prompt": "列举三类常见 VLM 幻觉，并说明一个缓解思路。",
    },
    {
        "topic": "vla",
        "subtopic": "vla_definition",
        "difficulty": "basic",
        "questions": [
            "VLA 模型是什么？",
            "Vision-Language-Action 模型解决什么问题？",
            "VLA 和 VLM 的区别是什么？",
        ],
        "core": "VLA 模型把视觉感知、语言指令理解和动作生成连接起来，用于机器人或具身智能。它不仅回答问题，还要输出动作序列、控制信号或可执行计划。",
        "details": "VLM 主要生成语言回答，VLA 还必须考虑环境状态、机器人动作空间、时序决策和闭环反馈。因此 VLA 的错误会影响实际执行，而不只是回答文本质量。",
        "practice": "工程上要明确 observation、language instruction、action representation、控制频率和安全边界。VLA 数据通常来自遥操作示范、机器人轨迹或仿真环境。",
        "pitfall": "不要把 VLA 等同于普通看图聊天；动作输出和环境反馈是它的核心差异。",
        "confusion": "有人说 VLA 就是把 LLaVA 的回答后面加一句机器人去执行。",
        "required": ["VLA", "视觉", "语言", "动作", "机器人", "闭环"],
        "forbidden": ["普通看图聊天", "加一句"],
        "eval_prompt": "VLA 相比 VLM 多了哪些关键约束？",
    },
    {
        "topic": "vla",
        "subtopic": "imitation_learning",
        "difficulty": "basic",
        "questions": [
            "机器人 imitation learning 是什么？",
            "行为克隆在 VLA 中有什么作用？",
            "为什么行为克隆会遇到分布偏移？",
        ],
        "core": "Imitation learning 从专家示范中学习状态到动作的映射。行为克隆是最直接的方法，把专家动作当监督标签，用监督学习训练策略。",
        "details": "行为克隆简单稳定，适合建立初始策略，但模型一旦偏离专家轨迹，遇到训练中没见过的状态，错误可能逐步累积，这就是分布偏移问题。",
        "practice": "缓解方法包括收集更多纠错示范、DAgger、加入噪声扰动、闭环评估和结合强化学习微调。机器人数据要记录时间同步、相机、关节、末端执行器和动作频率。",
        "pitfall": "行为克隆不是强化学习；它主要依赖示范数据，而不是通过奖励在线探索。",
        "confusion": "有人说行为克隆必须在真实机器人上随机探索才能学习。",
        "required": ["imitation learning", "行为克隆", "专家示范", "分布偏移", "DAgger"],
        "forbidden": ["必须随机探索", "强化学习"],
        "eval_prompt": "行为克隆为什么容易出现误差累积？",
    },
    {
        "topic": "vla",
        "subtopic": "diffusion_policy",
        "difficulty": "intermediate",
        "questions": [
            "Diffusion Policy 的核心思想是什么？",
            "扩散策略为什么适合机器人动作生成？",
            "Diffusion Policy 和直接回归动作有什么不同？",
        ],
        "core": "Diffusion Policy 把动作序列生成看成条件去噪过程，从噪声逐步还原未来一段动作。它能表达多峰动作分布，适合抓取和操作中存在多种可行动作轨迹的场景。",
        "details": "相比直接回归均值动作，扩散策略不容易把多个可行模式平均成一个不可执行动作。条件可以来自图像、低维状态和语言指令。",
        "practice": "训练时要定义 action horizon、observation horizon、噪声调度和控制频率。部署时常用 receding horizon，只执行预测序列的前几步再重新规划。",
        "pitfall": "Diffusion Policy 生成的是动作序列，不是图像生成模型直接拿来画机器人。",
        "confusion": "有人说扩散策略是先生成一张目标图片，再让机器人照着图片移动。",
        "required": ["Diffusion Policy", "动作序列", "去噪", "多峰", "receding horizon"],
        "forbidden": ["目标图片", "画机器人"],
        "eval_prompt": "Diffusion Policy 为什么比直接回归动作更适合多峰动作分布？",
    },
    {
        "topic": "vla",
        "subtopic": "action_tokenization",
        "difficulty": "intermediate",
        "questions": [
            "VLA 中动作可以如何表示？",
            "动作 tokenization 有什么利弊？",
            "连续动作和离散动作建模有什么区别？",
        ],
        "core": "VLA 动作可以表示为连续控制向量、离散动作类别、离散化后的 action tokens，或高层计划加低层控制器。动作 tokenization 把连续动作分桶或编码成 token，使语言模型可以像生成文本一样生成动作。",
        "details": "离散化便于复用 LLM 架构和交叉熵训练，但会带来量化误差。连续动作回归更自然，但需要合适的损失、尺度归一化和控制稳定性处理。",
        "practice": "选择动作表示要看机器人硬件、控制频率和任务粒度。机械臂末端位姿、夹爪开合和移动底盘速度不应盲目用同一个 token 方案。",
        "pitfall": "动作 token 不是普通文字 token；它必须能映射回具体控制量。",
        "confusion": "有人说把动作写成中文词语就完成了动作 tokenization。",
        "required": ["动作", "tokenization", "连续", "离散", "控制量", "量化误差"],
        "forbidden": ["中文词语", "普通文字"],
        "eval_prompt": "VLA 中把连续动作离散成 action tokens 有什么好处和风险？",
    },
    {
        "topic": "agent",
        "subtopic": "react",
        "difficulty": "basic",
        "questions": [
            "ReAct 方法的核心思想是什么？",
            "Agent 里的 Thought-Action-Observation 是什么？",
            "为什么 Agent 要把推理和行动交替进行？",
        ],
        "core": "ReAct 把推理和行动交替组织：模型先思考下一步，再调用工具或执行动作，然后观察结果并继续推理。它让语言推理能被外部环境反馈修正。",
        "details": "Thought 用于规划，Action 表示工具调用或环境操作，Observation 是工具或环境返回。多轮循环能减少盲目生成，但也需要停止条件和错误恢复。",
        "practice": "工程实现应把用户可见回答和内部推理/工具调用分离，并记录每一步的输入输出。工具失败时要让 Agent 能重试、换工具或向用户澄清。",
        "pitfall": "ReAct 不是让模型无限输出思考文本；重点是行动与观察形成闭环。",
        "confusion": "有人说 ReAct 就是让模型把思考过程写得越长越好。",
        "required": ["ReAct", "Thought", "Action", "Observation", "工具", "反馈"],
        "forbidden": ["越长越好", "无限"],
        "eval_prompt": "ReAct 中 Thought、Action、Observation 分别代表什么？",
    },
    {
        "topic": "agent",
        "subtopic": "tool_calling_schema",
        "difficulty": "intermediate",
        "questions": [
            "Tool Calling 和普通文本生成有什么区别？",
            "函数调用模型需要学会什么？",
            "为什么工具调用需要固定 schema？",
        ],
        "core": "Tool Calling 要求模型在合适时机输出结构化工具名和参数，而不是只生成自然语言。模型要学会判断是否需要工具、选择工具、填写参数，并根据返回结果继续回答。",
        "details": "固定 schema 能约束参数类型、必填字段和工具描述，降低解析失败率。常见格式是 JSON 函数调用，例如 name 加 arguments。",
        "practice": "训练数据应包含需要调用工具、不需要调用工具、工具不可用、参数缺失和工具返回错误的样例。解析器要校验 JSON，不应直接执行未校验参数。",
        "pitfall": "工具调用不是让模型随便编 API；可用工具必须由系统提供。",
        "confusion": "有人说模型想用什么工具就可以自己发明工具名。",
        "required": ["Tool Calling", "工具", "schema", "JSON", "参数", "返回结果"],
        "forbidden": ["自己发明工具", "随便编 API"],
        "eval_prompt": "为什么函数调用训练需要固定工具 schema 和参数校验？",
    },
    {
        "topic": "agent",
        "subtopic": "agent_memory",
        "difficulty": "basic",
        "questions": [
            "Agent 记忆可以分为哪些类型？",
            "LLM Agent 为什么需要 memory？",
            "长期记忆和短期上下文有什么区别？",
        ],
        "core": "Agent 记忆常分为短期上下文、长期向量记忆、任务状态记忆和用户偏好记忆。短期上下文用于当前对话，长期记忆用于跨会话检索经验或用户信息。",
        "details": "记忆能帮助 Agent 保持目标、复用历史信息和个性化响应。但记忆也带来隐私、过期、冲突和错误写入风险。",
        "practice": "工程上要给记忆设置写入条件、检索条件、权限、过期时间和用户删除机制。不能把所有上下文都永久保存。",
        "pitfall": "记忆不是越多越好；错误记忆会持续污染后续决策。",
        "confusion": "有人说 Agent 应该把每一句对话永久写入向量库。",
        "required": ["Agent", "短期上下文", "长期记忆", "向量", "隐私", "过期"],
        "forbidden": ["永久写入", "越多越好"],
        "eval_prompt": "Agent 的长期记忆为什么需要写入规则和过期机制？",
    },
    {
        "topic": "agent",
        "subtopic": "agentic_rl",
        "difficulty": "intermediate",
        "questions": [
            "Agentic RL 和普通 RLHF 有什么区别？",
            "为什么 Agent 训练需要环境反馈？",
            "如何给工具调用 Agent 设计奖励？",
        ],
        "core": "普通 RLHF 多优化单轮回答偏好；Agentic RL 更关注多步任务中规划、工具调用和最终任务完成。它需要环境、工具或验证器反馈来判断行动是否有效。",
        "details": "奖励可以来自任务成功率、工具参数正确性、步骤成本、最终答案验证和安全约束。多步任务的 credit assignment 更难，因为最终失败可能由早期某一步造成。",
        "practice": "先用 SFT 学会工具格式，再用可验证任务做 RL。日志要保留 trajectory，便于定位是规划错、工具选错、参数错还是观察理解错。",
        "pitfall": "Agentic RL 不只是让回答更讨喜；它优化的是可执行任务链。",
        "confusion": "有人说 Agentic RL 只需要给最终文本打好评差评。",
        "required": ["Agentic RL", "多步", "工具调用", "环境反馈", "trajectory", "奖励"],
        "forbidden": ["只需要最终文本", "单轮"],
        "eval_prompt": "Agentic RL 为什么比单轮 RLHF 更需要 trajectory 日志？",
    },
    {
        "topic": "training_engineering",
        "subtopic": "mixed_precision",
        "difficulty": "basic",
        "questions": [
            "BF16、FP16、FP32 训练有什么区别？",
            "为什么新 GPU 上常优先用 BF16？",
            "小模型训练有必要强行 FP32 吗？",
        ],
        "core": "FP32 数值范围和精度高，但显存和带宽开销大。FP16 更省显存但数值范围小，容易溢出，需要 loss scaling。BF16 保留接近 FP32 的指数范围，同时节省显存，训练更稳。",
        "details": "支持 BF16 的新 GPU 上，LLM 训练常优先用 BF16。FP32 可用于调试数值问题，但在 8GB 显存上通常会显著降低可训练模型大小和 batch。",
        "practice": "选择精度要看硬件支持、显存、稳定性和速度。训练日志若出现 NaN/Inf，应检查学习率、梯度裁剪、loss scaling 和数据异常。",
        "pitfall": "FP32 不会自动让模型更聪明；它只是数值精度更高。",
        "confusion": "有人说只要换成 FP32，模型知识能力就会大幅提升。",
        "required": ["FP32", "FP16", "BF16", "显存", "数值范围", "loss scaling"],
        "forbidden": ["更聪明", "大幅提升知识"],
        "eval_prompt": "为什么 8GB 显存训练 LLM 通常不优先选 FP32？",
    },
    {
        "topic": "training_engineering",
        "subtopic": "gradient_accumulation",
        "difficulty": "basic",
        "questions": [
            "梯度累积有什么作用？",
            "micro batch 和 effective batch 有什么区别？",
            "为什么单卡小显存训练常用 gradient accumulation？",
        ],
        "core": "梯度累积把多个 micro batch 的梯度累加后再更新一次参数，用较小显存模拟更大的 effective batch。effective batch size 等于 micro batch size 乘以累积步数再乘以数据并行卡数。",
        "details": "它不减少单个样本前向反向的计算量，但降低了每次需要同时放入显存的 batch。累积步数增大后，每个 optimizer step 之间间隔更久，学习率和日志解释要相应调整。",
        "practice": "训练日志应区分 micro step 和 optimizer step。保存 checkpoint 和学习率调度通常按 optimizer step 计数。",
        "pitfall": "梯度累积不能增加模型容量；它只是 batch 维度的显存折中。",
        "confusion": "有人说梯度累积能让 8GB 显卡直接装下任意大模型。",
        "required": ["梯度累积", "micro batch", "effective batch", "optimizer step", "显存"],
        "forbidden": ["任意大模型", "增加模型容量"],
        "eval_prompt": "梯度累积如何用小显存模拟更大的 batch？",
    },
    {
        "topic": "training_engineering",
        "subtopic": "checkpointing",
        "difficulty": "basic",
        "questions": [
            "训练中间 checkpoint 有什么用？",
            "为什么不应该无限保留所有 step 权重？",
            "checkpoint 和 eval 结果应该如何管理？",
        ],
        "core": "中间 checkpoint 用于断点恢复、回退到更好阶段和分析训练退化。它保存模型权重、优化器状态、step 和训练配置。",
        "details": "每个完整 checkpoint 可能很大，长期训练要设置保留策略，例如保留 last、best、若干关键里程碑和日志，删除密集中间快照。",
        "practice": "最好把 checkpoint、日志和固定 eval 输出绑定。这样清理文件时仍能知道哪个权重效果最好，避免只凭最后一步判断。",
        "pitfall": "中间权重不是越多越安全；没有 eval 记录的大量权重只会占空间。",
        "confusion": "有人说每 100 步保存的所有 checkpoint 都必须永久保留。",
        "required": ["checkpoint", "断点恢复", "优化器状态", "last", "best", "eval"],
        "forbidden": ["永久保留", "越多越安全"],
        "eval_prompt": "训练时为什么通常只保留 last、best 和少数里程碑 checkpoint？",
    },
    {
        "topic": "training_engineering",
        "subtopic": "data_mixture",
        "difficulty": "intermediate",
        "questions": [
            "专业 SFT 数据应该如何和通用数据混合？",
            "为什么 AI 专业知识 SFT 不能只堆专业问答？",
            "数据配比对小模型有什么影响？",
        ],
        "core": "专业 SFT 要提高目标领域能力，但不能完全挤掉通用对话和基础语言能力。小模型容量有限，专业数据比例过高会导致偏科、模板化和灾难性遗忘。",
        "details": "合理做法是保留一部分通用 SFT，加入高密度专业样本，并用固定 eval 监控通用能力和专业能力。专业样本应去重、准确、短而密。",
        "practice": "可以先用 5% 到 15% 专业数据做混合 SFT，观察 eval，再逐步提高。若模型开始对普通问题也强行回答 AI 术语，说明配比或步数过头。",
        "pitfall": "训练 loss 下降不代表配比正确；模型可能只是记住专业模板。",
        "confusion": "有人说专业模型只需要专业数据，通用 SFT 越少越好。",
        "required": ["专业 SFT", "通用数据", "配比", "灾难性遗忘", "固定 eval"],
        "forbidden": ["只需要专业数据", "越少越好"],
        "eval_prompt": "为什么给小模型做 AI 专业 SFT 时仍要混入通用数据？",
    },
    {
        "topic": "training_engineering",
        "subtopic": "validation_loss",
        "difficulty": "basic",
        "questions": [
            "为什么训练时要看 validation loss？",
            "训练 loss 下降但效果变差可能是什么原因？",
            "如何判断是否继续增加训练步数？",
        ],
        "core": "训练 loss 只说明模型更适应训练数据，validation loss 和固定 eval 才能反映泛化。训练 loss 下降但评测变差，可能是过拟合、数据污染、学习率过高或训练分布偏移。",
        "details": "是否继续训练应看验证集趋势、固定 prompt 回归、重复率和人工样例。若 valid loss 持续下降且生成质量提升，可以加步数；若停滞或退化，应早停或调整数据。",
        "practice": "固定 eval set 要版本化，不要每次训练后临时换题。否则不同 checkpoint 的表现无法比较。",
        "pitfall": "不要只根据最后一个训练 loss 判断模型好坏。",
        "confusion": "有人说 loss 越低，模型回答一定越正确。",
        "required": ["validation loss", "训练 loss", "过拟合", "固定 eval", "早停"],
        "forbidden": ["一定越正确", "只看训练 loss"],
        "eval_prompt": "训练 loss 降低但固定 eval 变差，通常说明什么？",
    },
    {
        "topic": "tokenizer",
        "subtopic": "bpe_vocab",
        "difficulty": "basic",
        "questions": [
            "BPE tokenizer 的基本原理是什么？",
            "专业 AI 语料为什么可能需要更大的 vocab？",
            "小模型 vocab size 应该如何权衡？",
        ],
        "core": "BPE 从字符或字节级 token 开始，反复合并语料中高频相邻 token 对，形成子词词表。合适的 tokenizer 能减少序列长度，并让常见术语被更稳定地表示。",
        "details": "AI 专业语料包含 PagedAttention、torch.nn、GRPO、LLaVA 等英文缩写和代码符号。词表过小会切得太碎，浪费上下文；词表过大又会增加 embedding 参数并稀释低频 token 学习。",
        "practice": "对 3 亿级小模型，8k 能跑通，16k 往往更适合中文、英文术语和代码混合场景。换 tokenizer 通常意味着需要重新训练或至少重建 embedding。",
        "pitfall": "更大词表不是免费提升；它会改变参数量、数据效率和 checkpoint 兼容性。",
        "confusion": "有人说 tokenizer 只影响输入显示，不影响训练和参数。",
        "required": ["BPE", "vocab", "子词", "序列长度", "embedding", "checkpoint"],
        "forbidden": ["不影响训练", "免费提升"],
        "eval_prompt": "为什么把 8k tokenizer 换成 16k 往往需要重新训练？",
    },
    {
        "topic": "tokenizer",
        "subtopic": "special_tokens",
        "difficulty": "basic",
        "questions": [
            "聊天模型为什么需要 special tokens？",
            "SFT 中 user/assistant 标记有什么作用？",
            "为什么 tokenizer 要包含工具调用相关标记？",
        ],
        "core": "Special tokens 用于明确对话边界、角色、开始结束和工具调用结构。SFT 中 user/assistant 标记能让模型区分用户输入和需要学习生成的助手输出。",
        "details": "常见标记包括 BOS、EOS、PAD、user、assistant、system、tool_call、tool_response。没有稳定标记时，模型容易把用户内容和助手内容混在一起。",
        "practice": "训练和推理必须使用同一 chat template。若 SFT 数据用一种格式，eval prompt 用另一种格式，小模型很容易失配。",
        "pitfall": "special tokens 不是普通文本装饰；它们影响 tokenizer、loss mask 和生成停止条件。",
        "confusion": "有人说 role 标记只是给人看的，模型训练时可以随便改。",
        "required": ["special tokens", "user", "assistant", "chat template", "loss mask", "EOS"],
        "forbidden": ["随便改", "给人看的"],
        "eval_prompt": "为什么 SFT 和推理必须保持同一套 chat template？",
    },
    {
        "topic": "scaling_law",
        "subtopic": "data_compute",
        "difficulty": "intermediate",
        "questions": [
            "Scaling Law 对小模型训练有什么启发？",
            "SFT 数据集是不是越大越好？",
            "为什么 3 亿参数模型也需要大量 token？",
        ],
        "core": "Scaling Law 的启发是参数量、数据量和算力要匹配。3 亿参数模型如果只看很少 token，通常仍然欠训练；但 SFT 阶段不是简单越大越好，质量和分布更关键。",
        "details": "预训练更依赖大量高质量 token 来学习语言和知识，SFT 更依赖准确、干净、覆盖目标行为的数据。低质量 SFT 扩大规模会放大错误风格和幻觉。",
        "practice": "判断是否加数据或加步数，要看验证 loss、固定 eval 和人工检查。高质量小数据常比脏的大数据更适合专业 SFT。",
        "pitfall": "不要把预训练 scaling law 直接套成 SFT 数据越大越好。",
        "confusion": "有人说 SFT 样本越多越好，即使包含重复和错误答案也能平均掉。",
        "required": ["Scaling Law", "参数量", "数据量", "预训练", "SFT", "质量"],
        "forbidden": ["越多越好", "错误答案能平均掉"],
        "eval_prompt": "为什么 SFT 数据规模增加不一定带来更好的专业能力？",
    },
    {
        "topic": "distributed_training",
        "subtopic": "deepspeed_zero",
        "difficulty": "intermediate",
        "questions": [
            "DeepSpeed ZeRO 的核心思想是什么？",
            "ZeRO Stage 1/2/3 大致有什么区别？",
            "DeepSpeed 为什么能降低大模型训练显存？",
        ],
        "core": "DeepSpeed ZeRO 通过把优化器状态、梯度和参数在数据并行卡之间分片，减少每张卡上的冗余显存。Stage 越高，分片对象越多，显存越省但通信更复杂。",
        "details": "Stage 1 分片优化器状态，Stage 2 进一步分片梯度，Stage 3 连参数也分片。ZeRO-Offload 还可以把部分状态放到 CPU 或 NVMe，但会引入带宽瓶颈。",
        "practice": "单卡 8GB 训练 3 亿模型时 DeepSpeed 帮助有限，因为没有多卡可分片；更直接的手段是 BF16、梯度累积、checkpointing、LoRA 或更小模型。",
        "pitfall": "DeepSpeed 不是自动让所有训练都更快；省显存常以通信和工程复杂度为代价。",
        "confusion": "有人说单卡上开 DeepSpeed ZeRO-3 就能像多卡一样分片参数。",
        "required": ["DeepSpeed", "ZeRO", "优化器状态", "梯度", "参数分片", "Stage 3"],
        "forbidden": ["自动更快", "单卡像多卡"],
        "eval_prompt": "DeepSpeed ZeRO 为什么主要在多卡训练中节省显存？",
    },
    {
        "topic": "evaluation",
        "subtopic": "fixed_eval",
        "difficulty": "basic",
        "questions": [
            "为什么小模型训练要有固定 eval set？",
            "专业知识模型应该如何做回归测试？",
            "自动评估和人工评估如何配合？",
        ],
        "core": "固定 eval set 能让不同 checkpoint 在同一批问题上比较，避免每次凭感觉换题。专业知识模型需要覆盖概念解释、对比、公式、误区纠正和工程边界。",
        "details": "自动评估可以先用关键词召回、禁忌词、长度和重复率做粗筛；人工评估负责判断准确性、边界和表达。小模型输出波动大，单个 prompt 不能代表整体能力。",
        "practice": "eval 样本应包含标准答案、required keywords、forbidden keywords、难度和主题。训练集和 eval prompt 应尽量去重，避免直接背题。",
        "pitfall": "只看一个 LoRA 问题或一个 BM25 问题，很容易误判模型整体进展。",
        "confusion": "有人说只要训练 loss 降了，就不需要固定测试题。",
        "required": ["固定 eval", "checkpoint", "关键词", "禁忌词", "人工评估"],
        "forbidden": ["不需要测试", "只看 loss"],
        "eval_prompt": "固定 eval set 对比较不同 checkpoint 有什么价值？",
    },
    {
        "topic": "data_quality",
        "subtopic": "sft_quality",
        "difficulty": "basic",
        "questions": [
            "构建高质量 SFT 数据要注意什么？",
            "为什么专业知识 SFT 要避免同词不同义混淆？",
            "AI 专业数据清洗有哪些关键步骤？",
        ],
        "core": "高质量 SFT 数据应单样本单知识点、答案准确、术语一致、格式稳定、长度适中，并覆盖多种问法。专业知识尤其要避免把相邻概念混在一起。",
        "details": "清洗步骤包括去重、长度过滤、语言检测、敏感噪声删除、术语表统一、答案事实检查和 train/eval 去污染。高质量比单纯数量更重要。",
        "practice": "对小模型，回答最好短而密，少写大段泛泛价值判断。每个主题要有正向解释、对比题、误区纠正和工程场景。",
        "pitfall": "不要把网上抓来的相关文本直接当 SFT；它们可能没有指令格式，也可能有错误或过时内容。",
        "confusion": "有人说只要样本里出现 LoRA、MoE、BM25 这些词，模型就会学会概念。",
        "required": ["SFT", "去重", "术语一致", "事实检查", "train/eval", "去污染"],
        "forbidden": ["直接当 SFT", "只要出现词"],
        "eval_prompt": "为什么 AI 专业 SFT 数据要做术语统一和误区样本？",
    },
]


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def assistant_answer(card: dict[str, Any], mode: str) -> str:
    if mode == "concept":
        return f"{card['core']}\n\n关键细节：{card['details']}"
    if mode == "engineering":
        return f"{card['core']}\n\n落地时要注意：{card['practice']}"
    if mode == "interview":
        return f"简答：{card['core']} {card['pitfall']}"
    if mode == "misconception":
        return f"这个说法不准确。{card['pitfall']} 正确理解是：{card['core']}"
    if mode == "checklist":
        return "可以按这几项检查：\n" + "\n".join(
            [
                f"- 核心概念：{card['core']}",
                f"- 关键边界：{card['pitfall']}",
                f"- 工程落地：{card['practice']}",
            ]
        )
    if mode == "teacher":
        return f"先抓住一句话：{card['core']}\n\n再看边界：{card['details']} {card['pitfall']}"
    raise ValueError(f"Unknown mode: {mode}")


def user_prompt(card: dict[str, Any], mode: str, index: int) -> str:
    questions = list(card["questions"])
    question = questions[index % len(questions)]
    if mode == "concept":
        return question
    if mode == "engineering":
        return f"从工程实现角度说明：{question}"
    if mode == "interview":
        return f"面试中如果被问到“{question}”，请给一个准确但不啰嗦的回答。"
    if mode == "misconception":
        confusion = str(card["confusion"])
        confusion = re.sub(r"^有人说[：:\s]*", "", confusion).strip()
        return f"有人说：“{confusion}” 请指出问题并给出正确解释。"
    if mode == "checklist":
        return f"给我一个判断清单，帮助确认自己是否真的理解：{question}"
    if mode == "teacher":
        return f"用适合小模型学习的方式讲清楚：{question}"
    raise ValueError(f"Unknown mode: {mode}")


def iter_train_records(seed: int) -> list[dict[str, Any]]:
    modes = ["concept", "engineering", "interview", "misconception", "checklist", "teacher"]
    records: list[dict[str, Any]] = []
    for card in CARDS:
        for index, mode in enumerate(modes):
            records.append(
                {
                    "source": "curated_ai_expert_v2",
                    "reference_datasets": REFERENCE_DATASETS,
                    "topic": card["topic"],
                    "subtopic": card["subtopic"],
                    "task_type": mode,
                    "difficulty": card["difficulty"],
                    "quality_tags": [
                        "single_concept",
                        "short_dense_answer",
                        "terminology_controlled",
                        "misconception_guarded",
                    ],
                    "conversations": [
                        {"role": "user", "content": user_prompt(card, mode, index)},
                        {"role": "assistant", "content": assistant_answer(card, mode)},
                    ],
                }
            )
    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def iter_eval_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, card in enumerate(CARDS, start=1):
        records.append(
            {
                "id": f"ai_expert_eval_v2_{index:03d}",
                "source": "curated_ai_expert_eval_v2",
                "topic": card["topic"],
                "subtopic": card["subtopic"],
                "task_type": "concept_regression",
                "difficulty": card["difficulty"],
                "prompt": card["eval_prompt"],
                "reference": card["core"],
                "required_keywords": card["required"],
                "forbidden_keywords": card["forbidden"],
                "min_required_keyword_recall": 0.55,
                "max_new_tokens": 160,
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_train(records: list[dict[str, Any]]) -> None:
    seen_prompts: set[str] = set()
    for index, record in enumerate(records, start=1):
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError(f"record {index}: expected two-turn conversation")
        user = conversations[0]
        assistant = conversations[1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError(f"record {index}: expected user then assistant")
        prompt = compact(str(user.get("content", "")))
        answer = compact(str(assistant.get("content", "")))
        if len(prompt) < 8 or len(answer) < 40:
            raise ValueError(f"record {index}: too short")
        key = prompt.lower()
        if key in seen_prompts:
            raise ValueError(f"duplicate prompt: {prompt}")
        seen_prompts.add(key)


def validate_eval(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> None:
    train_prompts = {
        compact(str(record["conversations"][0]["content"])).lower()
        for record in train_records
    }
    ids: set[str] = set()
    for record in eval_records:
        record_id = str(record["id"])
        if record_id in ids:
            raise ValueError(f"duplicate eval id: {record_id}")
        ids.add(record_id)
        prompt = compact(str(record["prompt"]))
        if prompt.lower() in train_prompts:
            raise ValueError(f"eval prompt leaks exact train prompt: {prompt}")
        if not record.get("required_keywords"):
            raise ValueError(f"eval record missing required keywords: {record_id}")


def summarize(records: list[dict[str, Any]], label: str) -> None:
    topic_counts = Counter(str(record.get("topic", "unknown")) for record in records)
    type_counts = Counter(str(record.get("task_type", "unknown")) for record in records)
    print(f"{label}: {len(records):,}")
    print("topics: " + ", ".join(f"{k}={v}" for k, v in topic_counts.most_common()))
    print("types: " + ", ".join(f"{k}={v}" for k, v in type_counts.most_common()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build curated AI expert SFT v2 and fixed eval set.")
    parser.add_argument("--out-path", type=Path, default=DATASET_DIR / "ai_expert_sft_v2.jsonl")
    parser.add_argument("--eval-out-path", type=Path, default=EVAL_DIR / "ai_expert_eval_v2.jsonl")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_records = iter_train_records(args.seed)
    eval_records = iter_eval_records()
    validate_train(train_records)
    validate_eval(train_records, eval_records)
    summarize(train_records, "train")
    summarize(eval_records, "eval")
    print("reference datasets: " + ", ".join(REFERENCE_DATASETS))
    if args.dry_run:
        print("dry_run: skipped writing files")
        return
    write_jsonl(args.out_path, train_records)
    write_jsonl(args.eval_out_path, eval_records)
    print(f"wrote: {args.out_path}")
    print(f"wrote: {args.eval_out_path}")


if __name__ == "__main__":
    main()
