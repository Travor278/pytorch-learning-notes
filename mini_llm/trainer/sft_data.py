from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

try:
    from .tokenizer import ByteTokenizer
    from .trainer_utils import iter_jsonl
except ImportError:
    from tokenizer import ByteTokenizer
    from trainer_utils import iter_jsonl


def normalize_conversations(record: dict[str, Any]) -> list[dict[str, str]]:
    conversations = record.get("conversations") or record.get("messages")
    if isinstance(conversations, list):
        normalized = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).strip().lower()
            content = turn.get("content")
            if role and isinstance(content, str) and content.strip():
                normalized.append({"role": role, "content": content.strip()})
        if normalized:
            return normalized

    instruction = record.get("instruction") or record.get("prompt") or record.get("question")
    output = record.get("output") or record.get("answer") or record.get("response")
    if isinstance(instruction, str) and isinstance(output, str):
        return [
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": output.strip()},
        ]
    return []


def encode_conversation(
    conversations: list[dict[str, str]],
    tokenizer: ByteTokenizer,
    block_size: int,
) -> tuple[list[int], list[int]]:
    input_ids = [tokenizer.bos_token_id]
    labels = [-100]

    for turn in conversations:
        role = turn["role"]
        content = turn["content"]
        header = f"<|{role}|>\n"
        header_ids = tokenizer.encode(header)
        content_ids = tokenizer.encode(content + "\n")
        input_ids.extend(header_ids)
        labels.extend([-100] * len(header_ids))
        input_ids.extend(content_ids)
        if role == "assistant":
            labels.extend(content_ids)
        else:
            labels.extend([-100] * len(content_ids))

    input_ids.append(tokenizer.eos_token_id)
    if conversations and conversations[-1]["role"] == "assistant":
        labels.append(tokenizer.eos_token_id)
    else:
        labels.append(-100)

    input_ids = input_ids[:block_size]
    labels = labels[:block_size]
    return input_ids, labels


class SFTJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer,
        block_size: int,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = self._build_examples()
        if not self.examples:
            raise ValueError(f"No usable SFT examples found in {self.path}")

    def _build_examples(self) -> list[dict[str, torch.Tensor]]:
        examples: list[dict[str, torch.Tensor]] = []
        for record in iter_jsonl(self.path):
            conversations = normalize_conversations(record)
            if not conversations:
                continue
            input_ids, labels = encode_conversation(conversations, self.tokenizer, self.block_size)
            if len(input_ids) < 2 or all(label == -100 for label in labels):
                continue
            examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class StreamingSFTJsonlDataset(IterableDataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer,
        block_size: int,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        for line_index, record in enumerate(iter_jsonl(self.path)):
            if line_index % num_workers != worker_id:
                continue
            conversations = normalize_conversations(record)
            if not conversations:
                continue
            input_ids, labels = encode_conversation(
                conversations,
                self.tokenizer,
                self.block_size,
            )
            if len(input_ids) < 2 or all(label == -100 for label in labels):
                continue
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].numel() for item in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for row, item in enumerate(batch):
            seq_len = item["input_ids"].numel()
            input_ids[row, :seq_len] = item["input_ids"]
            labels[row, :seq_len] = item["labels"]
            attention_mask[row, :seq_len] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
