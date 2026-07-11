# mini_llm

`mini_llm/` is the engineering track for a MiniMind-style small LLM stack.
It is intentionally separate from `GPT/`: `GPT/` stays as the teaching
implementation, while this folder grows toward a trainable and deployable
project layout.

The layout follows the upstream MiniMind shape:

```text
mini_llm/
  dataset/              # local JSONL datasets, ignored except README
  model/                # config, Dense/MoE model, LoRA
  trainer/              # tokenizer, pretrain, SFT, LoRA, DPO, PPO/GRPO, agent
  scripts/              # chat, conversion, OpenAI-compatible serving, WebUI
  eval_llm.py           # inference/eval entrypoint
```

Target path:

1. `model/model_minimind.py`: Qwen-style decoder-only core with RMSNorm,
   RoPE, GQA, SwiGLU, optional MoE, KV cache.
2. `trainer/train_tokenizer.py`: BPE/ByteLevel tokenizer and chat special
   tokens.
3. `trainer/train_pretrain.py`: `{"text": ...}` causal LM pretraining.
4. `trainer/train_full_sft.py`: `{"conversations": [...]}` supervised tuning.
5. `trainer/train_lora.py`: native LoRA training and merge/export.
6. `trainer/train_dpo.py`: preference optimization.
7. `trainer/train_ppo.py`, `trainer/train_grpo.py`, `trainer/train_agent.py`:
   rollout-based alignment and tool-use training.
8. `scripts/serve_openai_api.py`: OpenAI-compatible local serving.

Current status:

- M1 complete: Dense `MiniMindForCausalLM` forward/loss/backward works.
- Default config matches `minimind-3` scale: about 63.9M parameters.
- M2 complete enough for local experiments: JSONL pretraining data flow,
  byte tokenizer, lightweight BPE tokenizer, BF16/FP16 autocast, gradient
  accumulation, and checkpoint resume.
- Target training preset: `mini310m`, about 310.2M parameters with
  `vocab_size=6400`. The config is tracked at `configs/mini310m.json`.
- Larger vocab presets are available: `mini310m_8k` is about 312.0M params,
  and `mini320m_16k` is exactly about 320.0M params.
- M3 complete: `eval_llm.py` can load checkpoints and run naive autoregressive
  generation. KV cache is still planned.
- M4 complete: SFT conversation JSONL data flow and full-parameter SFT.
- M5 complete: native LoRA injection, LoRA SFT, and merged checkpoint export.
- M6 started: dense MoE block with top-k routing and load-balancing aux loss.
- M7 started: DPO preference training works; PPO/GRPO have explicit runnable
  boundaries pending rollout/reward infrastructure.
- KV cache and production-grade tokenizer/eval remain planned.

Tokenizer:

```bash
python mini_llm/trainer/train_tokenizer.py \
  --tokenizer bpe \
  --data-path mini_llm/dataset/pretrain_t2t_mini.jsonl \
  --tokenizer-path mini_llm/tokenizer.json \
  --vocab-size 8192 \
  --min-freq 2 \
  --max-chars 1000000
```

Smoke command:

```bash
python mini_llm/trainer/train_pretrain.py \
  --data-path mini_llm/dataset/pretrain.jsonl \
  --model-size tiny \
  --block-size 128 \
  --batch-size 8 \
  --max-steps 100
```

Pretraining streams JSONL by default, which is the recommended mode for the
MiniMind mini/full files. Add `--in-memory` only for tiny tests.

310M target command:

```bash
python mini_llm/trainer/train_pretrain.py \
  --data-path mini_llm/dataset/pretrain_t2t_mini.jsonl \
  --model-size mini310m_8k \
  --tokenizer-path mini_llm/tokenizer.json \
  --block-size 512 \
  --batch-size 1 \
  --dry-run

python mini_llm/trainer/train_pretrain.py \
  --data-path mini_llm/dataset/pretrain_t2t_mini.jsonl \
  --model-size mini310m_8k \
  --tokenizer-path mini_llm/tokenizer.json \
  --block-size 512 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --precision auto \
  --max-steps 100
```

Resume:

```bash
python mini_llm/trainer/train_pretrain.py \
  --data-path mini_llm/dataset/pretrain_t2t_mini.jsonl \
  --model-size mini310m_8k \
  --tokenizer-path mini_llm/tokenizer.json \
  --block-size 512 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --precision auto \
  --max-steps 200 \
  --resume-from latest
```

Eval/generate:

```bash
python mini_llm/eval_llm.py \
  --checkpoint latest \
  --tokenizer-path mini_llm/tokenizer.json \
  --prompt "你好，介绍一下你自己" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 50
```

SFT:

```bash
python mini_llm/trainer/train_full_sft.py \
  --data-path mini_llm/dataset/sft_t2t_mini.jsonl \
  --tokenizer-path mini_llm/tokenizer.json \
  --out-dir mini_llm/checkpoints \
  --init-from latest \
  --block-size 512 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-5 \
  --precision auto
```

LoRA SFT:

```bash
python mini_llm/trainer/train_lora.py \
  --data-path mini_llm/dataset/sft.jsonl \
  --tokenizer-path mini_llm/tokenizer.json \
  --out-dir mini_llm/checkpoints \
  --init-from latest \
  --rank 8 \
  --alpha 16 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --precision auto
```

DPO:

```bash
python mini_llm/trainer/train_dpo.py \
  --data-path mini_llm/dataset/dpo.jsonl \
  --tokenizer-path mini_llm/tokenizer.json \
  --out-dir mini_llm/checkpoints \
  --init-from latest \
  --batch-size 1 \
  --precision auto
```

Next priorities are KV cache/generation speed, LoRA-DPO memory reduction, and
then LLaVA engineering integration.

VLM should grow separately from `LLaVA/` as an engineering stack around the
existing ViT, projector, LLM bridge, data pipeline, training, and evaluation.
