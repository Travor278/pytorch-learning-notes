from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "mini_llm" / "dataset"


TOPICS: list[dict[str, object]] = [
    {
        "category": "llm_architecture",
        "questions": [
            "MHA、MQA 和 GQA 有什么区别？",
            "请对比多头注意力、MQA 和 GQA。",
        ],
        "answer": "MHA 为每个注意力头保留独立的 Q、K、V；MQA 让多个查询头共享同一组 K、V，主要减少 KV cache 和推理显存；GQA 把查询头分组，每组共享一组 K、V，是效果和推理效率之间的折中。",
    },
    {
        "category": "llm_architecture",
        "questions": [
            "Transformer 解码器块通常由哪些部分组成？",
            "一个 decoder-only LLM block 里通常有什么模块？",
        ],
        "answer": "典型 decoder-only LLM block 包含归一化层、自注意力层、残差连接、前馈网络以及另一个残差连接。现代 LLM 常用 RMSNorm、RoPE、SwiGLU/GLU 类 FFN，并使用 causal mask 保证只能看见当前位置之前的 token。",
    },
    {
        "category": "llm_architecture",
        "questions": [
            "RoPE 位置编码的核心思想是什么？",
            "为什么很多 LLM 使用 RoPE？",
        ],
        "answer": "RoPE 通过对 Q、K 向量做位置相关的旋转，把相对位置信息注入注意力分数。它天然适合自回归注意力，能较好表达相对距离，也便于通过 NTK scaling、YaRN 等方法做长上下文外推。",
    },
    {
        "category": "llm_architecture",
        "questions": [
            "RMSNorm 和 LayerNorm 有什么区别？",
            "为什么 LLM 常用 RMSNorm？",
        ],
        "answer": "LayerNorm 会减去均值再按方差归一化，RMSNorm 只按均方根缩放，不显式中心化。RMSNorm 计算更简单，训练稳定性通常足够好，所以很多 LLM 用它替代 LayerNorm。",
    },
    {
        "category": "llm_architecture",
        "questions": [
            "SwiGLU 前馈网络为什么常见？",
            "LLM 里的 SwiGLU 起什么作用？",
        ],
        "answer": "SwiGLU 是带门控的前馈结构，通常把一支激活后的投影和另一支投影相乘，再映射回隐藏维度。它比普通 ReLU/GELU FFN 表达能力更强，在许多大模型中能带来更好的训练效果。",
    },
    {
        "category": "tokenizer",
        "questions": [
            "BPE tokenizer 的基本原理是什么？",
            "为什么小模型也需要训练 tokenizer？",
        ],
        "answer": "BPE 从字符或字节级 token 开始，反复合并语料中最常见的相邻 token 对，形成子词词表。合适的 tokenizer 能降低序列长度、覆盖常见词形，并让模型更高效地学习语言统计规律。",
    },
    {
        "category": "training",
        "questions": [
            "预训练和 SFT 的区别是什么？",
            "为什么先 pretrain 再 SFT？",
        ],
        "answer": "预训练用 next-token prediction 学语言、知识和模式，目标是让模型会续写；SFT 用指令对话数据教模型按用户意图回答，目标是让模型会聊天和遵循格式。底座没训稳时，SFT 往往只能学到回答腔调。",
    },
    {
        "category": "training",
        "questions": [
            "SFT 数据越大越好吗？",
            "SFT 数据集是不是越大模型越强？",
        ],
        "answer": "SFT 不是简单越大越好。高质量、格式一致、覆盖目标任务的数据更重要。低质量或分布混乱的数据会引入噪声、幻觉和错误风格，通常需要做清洗、去重、配比和小规模验证。",
    },
    {
        "category": "training",
        "questions": [
            "预训练数据和 SFT 数据分布差异很大怎么办？",
            "领域数据和通用数据差异大时怎么微调？",
        ],
        "answer": "可以先做领域继续预训练，让模型熟悉领域语言和概念，再做 SFT 学交互格式。SFT 时混入一部分通用数据、降低学习率、控制步数，可以减少灾难性遗忘和风格过拟合。",
    },
    {
        "category": "scaling_law",
        "questions": [
            "Scaling Law 对小模型训练有什么启发？",
            "为什么 3 亿参数模型也需要很多 token？",
        ],
        "answer": "Scaling Law 的核心启发是参数量、数据量和算力要匹配。3 亿参数模型如果只看几千万 token，通常还处于严重欠训练状态；更多高质量 token 会继续改善语言建模能力，只是收益会逐渐递减。",
    },
    {
        "category": "peft",
        "questions": [
            "LoRA 的核心思想是什么？",
            "请简单解释 LoRA。",
        ],
        "answer": "LoRA 冻结原模型权重，只在部分线性层旁边训练低秩矩阵 A 和 B，用低秩增量近似权重更新。这样训练参数和显存开销更小，任务完成后还可以把增量合并回原权重。",
    },
    {
        "category": "peft",
        "questions": [
            "LoRA 中 A、B 矩阵通常怎么初始化？",
            "LoRA 为什么常把 B 初始化为 0？",
        ],
        "answer": "常见做法是 A 随机初始化，B 初始化为 0。这样训练开始时 LoRA 分支输出近似为 0，不会突然破坏基座模型，随后再通过训练逐步学习任务相关增量。",
    },
    {
        "category": "peft",
        "questions": [
            "LoRA 的 rank 应该怎么选？",
            "LoRA rank 越大越好吗？",
        ],
        "answer": "rank 控制低秩增量的容量。rank 越大表达能力越强，但参数、显存、训练时间和过拟合风险也会增加。小任务常用 4 或 8，复杂任务可用 16、32 或更高，需要用验证集和生成效果来选。",
    },
    {
        "category": "peft",
        "questions": [
            "QLoRA 和 LoRA 有什么区别？",
            "QLoRA 为什么能省显存？",
        ],
        "answer": "LoRA 通常在普通精度基座上训练低秩增量；QLoRA 会把基座模型量化到 4bit 等低精度，只训练 LoRA 参数。它把大部分权重存储显存压低，因此能在单卡上微调更大的模型。",
    },
    {
        "category": "moe",
        "questions": [
            "Dense 模型和 MoE 模型的本质区别是什么？",
            "为什么 MoE 能有很多参数但计算量不一定很大？",
        ],
        "answer": "Dense 模型每个 token 都经过所有前馈参数；MoE 模型把前馈层拆成多个专家，路由器只为每个 token 选择少数专家。MoE 的总参数量很大，但单 token 激活参数较少，所以计算是稀疏的。",
    },
    {
        "category": "moe",
        "questions": [
            "MoE 中 routing 是怎么工作的？",
            "MoE 路由器如何选择专家？",
        ],
        "answer": "路由器通常用线性层为每个 token 计算所有专家的分数，经过 softmax 后选择 top-k 专家。token 被送入选中的专家计算，专家输出再按路由权重加权求和。",
    },
    {
        "category": "moe",
        "questions": [
            "MoE 为什么会有专家负载不均衡问题？",
            "MoE 专家塌缩是什么意思？",
        ],
        "answer": "如果路由器总把 token 分给少数专家，这些专家会过载，其他专家几乎得不到训练信号，称为负载不均衡或专家塌缩。它会降低并行效率，也会浪费模型容量。",
    },
    {
        "category": "moe",
        "questions": [
            "如何缓解 MoE 的负载不均衡？",
            "MoE 负载均衡通常怎么做？",
        ],
        "answer": "常见方法包括加入 load balancing auxiliary loss、限制专家 capacity、使用 top-k routing、对路由分数加噪声、expert parallel 调度，以及监控每个专家的 token 占比和溢出率。",
    },
    {
        "category": "alignment",
        "questions": [
            "DPO、PPO 和 GRPO 的区别是什么？",
            "请对比 DPO、PPO、GRPO。",
        ],
        "answer": "DPO 是离线偏好优化，直接用 chosen/rejected 样本训练策略，不需要在线 rollout。PPO 是强化学习方法，通常需要奖励模型或环境反馈，并用优势函数和 KL 约束更新策略。GRPO 用一组回答的相对奖励估计优势，常用于减少对 value model 的依赖。",
    },
    {
        "category": "alignment",
        "questions": [
            "DPO 的损失函数是什么？",
            "DPO loss 怎么写？",
        ],
        "answer": "DPO 常用损失为 -log sigmoid(beta * [(log pi(y+|x)-log pi(y-|x)) - (log ref(y+|x)-log ref(y-|x))])。它鼓励策略模型相对参考模型更偏向 chosen 回答，而不是 rejected 回答。",
    },
    {
        "category": "alignment",
        "questions": [
            "RLHF 中 KL 约束有什么作用？",
            "为什么 PPO 训练 LLM 时要加 KL 惩罚？",
        ],
        "answer": "KL 约束用于限制策略模型偏离参考模型太远，避免奖励模型漏洞、语言质量下降和模式崩坏。它相当于给强化学习更新加安全边界，让模型在优化奖励的同时保持原有语言能力。",
    },
    {
        "category": "inference",
        "questions": [
            "KV Cache 在自回归推理中有什么作用？",
            "为什么推理时要缓存 K 和 V？",
        ],
        "answer": "自回归生成每次只新增一个 token，过去 token 的 K、V 不会改变。KV Cache 把它们缓存起来，下一步只计算新 token 的 Q/K/V，并复用历史 K/V，从而显著减少重复计算。",
    },
    {
        "category": "inference",
        "questions": [
            "vLLM 的 PagedAttention 解决了什么问题？",
            "PagedAttention 为什么能提升吞吐？",
        ],
        "answer": "PagedAttention 把 KV cache 像操作系统分页一样管理，避免为每个请求预留连续大块显存。它减少显存碎片，提高 KV cache 利用率，从而让更多请求并发推理。",
    },
    {
        "category": "inference",
        "questions": [
            "FlashAttention 的核心思想是什么？",
            "FlashAttention 为什么更快更省显存？",
        ],
        "answer": "FlashAttention 通过分块计算和重排内存访问，避免显式保存完整注意力矩阵，并减少 HBM 读写。它保持精确 attention 结果，同时显著降低显存占用和提升速度。",
    },
    {
        "category": "inference",
        "questions": [
            "Speculative Decoding 是什么？",
            "投机解码如何加速大模型生成？",
        ],
        "answer": "Speculative Decoding 用一个小模型先草拟多个 token，再由大模型并行验证这些 token。若草稿被接受，就一次前进多步；若被拒绝，则回退修正。它能在保持分布正确的前提下减少大模型调用次数。",
    },
    {
        "category": "inference",
        "questions": [
            "量化为什么能加速推理？",
            "INT8/INT4 量化主要带来什么收益？",
        ],
        "answer": "量化把权重或激活从 FP16/BF16 压到 INT8、INT4 等低位宽，减少显存占用和内存带宽压力。有硬件支持时还可提高矩阵乘法吞吐，但过度量化可能损失精度。",
    },
    {
        "category": "distributed_training",
        "questions": [
            "DeepSpeed ZeRO 的核心思想是什么？",
            "ZeRO 为什么能省显存？",
        ],
        "answer": "ZeRO 把优化器状态、梯度和参数分片到多张卡上，而不是每张卡都完整保存一份。Stage 越高，分片越彻底，显存占用越低，但通信和工程复杂度也更高。",
    },
    {
        "category": "rag",
        "questions": [
            "BM25 的计算原理是什么？",
            "BM25 为什么适合关键词检索？",
        ],
        "answer": "BM25 根据查询词在文档中的词频、逆文档频率和文档长度归一化打分。k1 控制词频饱和，b 控制长度惩罚。它不依赖神经网络，适合关键词匹配和稀疏检索基线。",
    },
    {
        "category": "rag",
        "questions": [
            "RAG 系统一般由哪些步骤组成？",
            "检索增强生成的基本流程是什么？",
        ],
        "answer": "RAG 通常包括文档切分、向量化或稀疏索引、召回、重排序、上下文拼接和生成。它把外部知识注入 prompt，降低模型记忆负担，并让答案可以追溯到资料来源。",
    },
    {
        "category": "rag",
        "questions": [
            "embedding 检索和 BM25 有什么区别？",
            "稀疏检索和向量检索如何互补？",
        ],
        "answer": "BM25 依赖词面匹配，擅长精确关键词；embedding 检索依赖语义相似度，擅长同义改写和模糊语义。实际 RAG 常把两者混合召回，再用 reranker 做精排。",
    },
    {
        "category": "vlm",
        "questions": [
            "CLIP 的图文对齐原理是什么？",
            "CLIP 为什么能做零样本分类？",
        ],
        "answer": "CLIP 用图像编码器和文本编码器分别得到向量，通过对比学习拉近匹配图文、拉远不匹配图文。推理时把类别写成文本提示，与图像向量比较相似度，就能做零样本分类。",
    },
    {
        "category": "vlm",
        "questions": [
            "ViT 如何把图像输入 Transformer？",
            "Vision Transformer 的 patch embedding 是什么？",
        ],
        "answer": "ViT 把图像切成固定大小的 patch，把每个 patch 展平后线性映射成 token embedding，再加位置编码送入 Transformer。这样图像就被表示成类似文本 token 的序列。",
    },
    {
        "category": "vlm",
        "questions": [
            "LLaVA 的基本结构是什么？",
            "LLaVA 如何把视觉特征接入大语言模型？",
        ],
        "answer": "LLaVA 通常由视觉编码器、投影层和语言模型组成。视觉编码器提取图像特征，投影层把视觉特征映射到 LLM 隐空间，再作为视觉 token 与文本 token 一起输入语言模型。",
    },
    {
        "category": "vlm",
        "questions": [
            "VLM 训练通常分哪些阶段？",
            "训练图文大模型一般怎么做？",
        ],
        "answer": "常见流程是先做图文对齐预训练，让投影层或部分模型学会把视觉特征接到语言空间；再用多模态指令数据做 SFT，让模型学会问答、描述、OCR、定位和推理等交互能力。",
    },
    {
        "category": "vlm",
        "questions": [
            "多模态模型中的 projector 有什么作用？",
            "为什么 VLM 需要视觉投影层？",
        ],
        "answer": "视觉编码器输出的特征维度和分布通常与 LLM embedding 空间不同。projector 负责把视觉特征映射到 LLM 可理解的隐藏空间，使图像 patch 或区域特征能作为视觉 token 被语言模型处理。",
    },
    {
        "category": "vlm",
        "questions": [
            "VLM 容易出现哪些幻觉？",
            "图文模型为什么会看错图？",
        ],
        "answer": "VLM 幻觉常见于不存在的物体、错误属性、数量错误、空间关系错误和 OCR 误读。原因可能是视觉分辨率不足、训练数据偏差、语言先验过强，或图像特征与文本生成没有充分对齐。",
    },
    {
        "category": "vla",
        "questions": [
            "VLA 模型是什么？",
            "Vision-Language-Action 模型解决什么问题？",
        ],
        "answer": "VLA 模型把视觉感知、语言指令理解和动作生成连在一起，用于机器人或具身智能。它不仅回答问题，还要根据图像和指令输出动作序列、控制信号或可执行计划。",
    },
    {
        "category": "vla",
        "questions": [
            "VLA 和 VLM 的区别是什么？",
            "为什么说 VLA 比 VLM 更接近机器人控制？",
        ],
        "answer": "VLM 主要把图像和文本映射到语言回答；VLA 还要输出动作，并受到物理环境、机器人动力学和任务反馈约束。VLA 更强调时序决策、可执行性和闭环控制。",
    },
    {
        "category": "vla",
        "questions": [
            "机器人 imitation learning 是什么？",
            "VLA 里行为克隆有什么作用？",
        ],
        "answer": "Imitation learning 让模型从专家示范中学习状态到动作的映射。行为克隆是最直接的方法，用监督学习拟合专家动作，适合建立初始策略，但可能在分布偏移时累积错误。",
    },
    {
        "category": "vla",
        "questions": [
            "Diffusion Policy 的核心思想是什么？",
            "扩散策略为什么适合机器人动作生成？",
        ],
        "answer": "Diffusion Policy 把动作序列生成看成去噪过程，从噪声逐步还原出未来一段动作。它能表达多峰动作分布，适合抓取、操作等存在多种可行轨迹的机器人任务。",
    },
    {
        "category": "agent",
        "questions": [
            "AI Agent 通常由哪些组件组成？",
            "一个 LLM Agent 的基本架构是什么？",
        ],
        "answer": "LLM Agent 通常包含模型大脑、工具接口、任务规划、记忆、执行循环和反馈评估。模型负责理解和决策，工具负责调用外部能力，记忆保存上下文，反馈用于修正计划。",
    },
    {
        "category": "agent",
        "questions": [
            "ReAct 方法的核心思想是什么？",
            "Agent 里的 Thought-Action-Observation 是什么？",
        ],
        "answer": "ReAct 把推理和行动交替组织：模型先思考下一步，再调用工具或执行动作，然后观察结果并继续推理。这样能把语言推理与外部环境反馈结合起来，减少盲目生成。",
    },
    {
        "category": "agent",
        "questions": [
            "Tool Calling 和普通文本生成有什么区别？",
            "函数调用模型需要学什么？",
        ],
        "answer": "Tool Calling 要求模型在合适时机输出结构化工具名和参数，而不是只生成自然语言。它需要学会判断是否需要工具、选择哪个工具、填对参数，并根据工具返回继续回答。",
    },
    {
        "category": "agent",
        "questions": [
            "Agent 记忆可以分为哪些类型？",
            "LLM Agent 为什么需要 memory？",
        ],
        "answer": "Agent 记忆常分为短期上下文记忆、长期向量记忆、任务状态记忆和用户画像记忆。记忆能帮助模型跨多轮保持目标、复用历史经验，但也需要做权限、过期和隐私控制。",
    },
    {
        "category": "agent",
        "questions": [
            "Agentic RL 和普通 RLHF 有什么区别？",
            "为什么 Agent 训练需要环境反馈？",
        ],
        "answer": "普通 RLHF 多优化单轮回答偏好；Agentic RL 更关注多步任务中的工具调用、规划和执行结果。它通常需要环境或验证器反馈，比如任务是否完成、工具调用是否正确、最终答案是否可验证。",
    },
    {
        "category": "eval",
        "questions": [
            "如何评估一个小语言模型是否训好了？",
            "小模型训练后应该看哪些指标？",
        ],
        "answer": "先看训练和验证 loss 是否稳定下降，再做固定 prompt 回归测试，覆盖闲聊、摘要、代码、计算、知识问答和安全拒答。还要检查重复、胡编、格式错误和对 prompt 模板的依赖。",
    },
    {
        "category": "eval",
        "questions": [
            "为什么模型会出现重复生成？",
            "LLM 输出一直重复同一句话通常是什么原因？",
        ],
        "answer": "重复生成可能来自底座欠训练、数据中重复模板过多、SFT 过拟合、解码参数不合适或缺少重复惩罚。若贪心和采样都重复，通常说明模型分布本身还没学稳。",
    },
    {
        "category": "data",
        "questions": [
            "构建 AI 专业知识 SFT 数据时要注意什么？",
            "专业知识库怎么避免把模型训偏？",
        ],
        "answer": "专业 SFT 数据要覆盖多种问法、保持答案准确简洁、避免单一模板，并与通用 SFT 混合训练。专业数据比例不宜过高，否则小模型容易偏科，丢失闲聊、摘要和基础语言能力。",
    },
]


def iter_examples() -> Iterable[dict[str, object]]:
    for item in TOPICS:
        category = str(item["category"])
        answer = str(item["answer"])
        questions = item["questions"]
        if not isinstance(questions, list):
            raise TypeError(f"questions must be a list for category={category}")
        for question in questions:
            yield {
                "source": "curated_ai_knowledge_v1",
                "category": category,
                "conversations": [
                    {"role": "user", "content": str(question)},
                    {"role": "assistant", "content": answer},
                ],
            }


def write_jsonl(path: Path, examples: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build curated AI knowledge SFT data.")
    parser.add_argument("--out-path", type=Path, default=DATASET_DIR / "ai_knowledge_sft.jsonl")
    parser.add_argument(
        "--legacy-path",
        type=Path,
        default=DATASET_DIR / "llm_knowledge_sft.jsonl",
        help="Also overwrite the old LLM-only knowledge file unless --no-legacy is set.",
    )
    parser.add_argument("--no-legacy", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = list(iter_examples())
    write_jsonl(args.out_path, examples)
    if not args.no_legacy:
        write_jsonl(args.legacy_path, examples)
    categories = sorted({str(example["category"]) for example in examples})
    print(f"wrote {len(examples)} examples to {args.out_path}")
    if not args.no_legacy:
        print(f"wrote {len(examples)} examples to {args.legacy_path}")
    print("categories: " + ", ".join(categories))


if __name__ == "__main__":
    main()
