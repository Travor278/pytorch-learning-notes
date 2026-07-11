# mini_llm dataset

Keep local training files here. Actual data is ignored by git.

Downloaded MiniMind mini files:

- `pretrain_t2t_mini.jsonl`: pretraining, SHA256
  `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`
- `sft_t2t_mini.jsonl`: SFT, SHA256
  `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518`
- `dpo.jsonl`: preference training, SHA256
  `ee934a8a455ccc99d1334d63e1254dd1d64f497fd067cfcbb71e3043f5b46768`

If direct access to Hugging Face is unavailable without a proxy, `hf-mirror.com`
works as a no-proxy mirror. Always verify hashes after downloading.

Expected formats:

Pretrain:

```json
{"text": "Transformer uses self-attention to model context."}
```

Run a tiny pretrain smoke test:

```bash
python mini_llm/trainer/train_pretrain.py --data-path mini_llm/dataset/pretrain.jsonl --model-size tiny
```

Train a lightweight BPE tokenizer before serious runs:

```bash
python mini_llm/trainer/train_tokenizer.py --tokenizer bpe --data-path mini_llm/dataset/pretrain.jsonl --vocab-size 8192
```

SFT:

```json
{"conversations": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}
```

DPO:

```json
{"chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "good"}], "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "bad"}]}
```

Curated AI knowledge SFT:

```bash
python mini_llm/scripts/build_ai_knowledge_sft.py
```

This generates `ai_knowledge_sft.jsonl` and refreshes the legacy
`llm_knowledge_sft.jsonl` with LLM, VLM, VLA, Agent, RAG, MoE, alignment,
inference, and training/evaluation examples. Use it as a small domain mix,
not as the only SFT dataset.

Curated AI expert SFT v2:

```bash
python mini_llm/scripts/build_ai_expert_sft_v2.py
```

This generates `ai_expert_sft_v2.jsonl` and the fixed evaluation set
`mini_llm/evals/ai_expert_eval_v2.jsonl`. The v2 set is a compact,
high-density expert anchor set for LLM, VLM, VLA, Agent, RAG, PEFT, MoE,
alignment, inference, tokenizer, and training-engineering concepts. Its shape
borrows from community SFT datasets such as SmolTalk, FineTome, OpenOrca, and
Glaive function-calling: clear conversation format, short dense answers,
single-concept samples, misconception guards, and structured eval metadata.

Run the fixed expert eval without loading a checkpoint:

```bash
python mini_llm/scripts/eval_expert_suite.py --dry-run
```

Run it on a checkpoint:

```bash
python mini_llm/scripts/eval_expert_suite.py \
  --checkpoint mini_llm/checkpoints/minimind_mini_8k_sft50k_ai_hf_mix/sft_last.pt \
  --tokenizer-path mini_llm/tokenizer.json \
  --device cuda \
  --precision auto
```

Mixed SFT:

```bash
python mini_llm/scripts/build_mixed_sft.py
```

This builds `sft_ai_mix.jsonl` from a shuffled sample of MiniMind SFT plus
repeated AI knowledge examples. The default ratio is about 9% domain examples.

Public HF AI SFT collection:

```bash
python mini_llm/scripts/collect_ai_sft_from_hf.py
```

This streams public Hugging Face datasets, filters for AI/ML/LLM/VLM/Agent/code
keywords, normalizes several formats into `conversations`, and writes
`ai_hf_sft.jsonl`. It uses a repo-local HF cache at `mini_llm/.hf_cache`.
Treat this as a candidate pool: spot-check before mixing into training.

Stable mixed AI SFT:

```bash
python mini_llm/scripts/build_ai_mixed_sft.py
```

This builds `sft_ai_hf_mix.jsonl` from MiniMind SFT, the stricter HF AI
candidate pool, and repeated curated anchor examples. The default mix is roughly
88% MiniMind SFT, 9% HF AI examples, and 3% curated anchors.

High-quality expert v2 mix without HF candidate noise:

```bash
python mini_llm/scripts/build_ai_mixed_sft.py \
  --base-path mini_llm/dataset/sft_t2t_mini.jsonl \
  --curated-path mini_llm/dataset/ai_expert_sft_v2.jsonl \
  --out-path mini_llm/dataset/sft_ai_expert_v2_mix.jsonl \
  --base-samples 50000 \
  --hf-ai-samples 0 \
  --curated-repeat 20
```

This creates a training mix of about 90% MiniMind SFT and 10% curated expert v2
anchors. Use this before trying broader HF AI candidate pools.

Short smoke files in this directory are for pipeline checks only. They are not
large enough for meaningful model quality.
