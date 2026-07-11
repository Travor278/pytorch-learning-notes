from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

try:
    from .tokenizer import ByteTokenizer
    from .trainer_utils import iter_jsonl
except ImportError:
    from tokenizer import ByteTokenizer
    from trainer_utils import iter_jsonl


class PretrainJsonlDataset(Dataset):
    """In-memory JSONL pretraining dataset.

    The expected input format is one JSON object per line:
      {"text": "..."}
    Texts are encoded, concatenated, and sliced into block-size chunks.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer,
        block_size: int,
        text_key: str = "text",
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = self._build_examples(text_key=text_key, add_bos=add_bos, add_eos=add_eos)

        if not self.examples:
            raise ValueError(f"No usable pretrain examples found in {self.path}")

    def _build_examples(self, text_key: str, add_bos: bool, add_eos: bool) -> list[list[int]]:
        token_ids: list[int] = []
        for record in iter_jsonl(self.path):
            text = record.get(text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            token_ids.extend(self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos))

        examples: list[list[int]] = []
        for start in range(0, len(token_ids), self.block_size):
            chunk = token_ids[start : start + self.block_size]
            if len(chunk) >= 2:
                examples.append(chunk)
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.examples[index], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].numel() for item in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for row, item in enumerate(batch):
            ids = item["input_ids"]
            seq_len = ids.numel()
            input_ids[row, :seq_len] = ids
            labels[row, :seq_len] = item["labels"]
            attention_mask[row, :seq_len] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class StreamingPretrainJsonlDataset(IterableDataset):
    """Streaming JSONL dataset for large pretraining files."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer,
        block_size: int,
        text_key: str = "text",
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_key = text_key
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        buffer: list[int] = []

        for line_index, record in enumerate(iter_jsonl(self.path)):
            if line_index % num_workers != worker_id:
                continue
            text = record.get(self.text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            buffer.extend(
                self.tokenizer.encode(text, add_bos=self.add_bos, add_eos=self.add_eos)
            )
            while len(buffer) >= self.block_size:
                chunk = buffer[: self.block_size]
                del buffer[: self.block_size]
                ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": ids, "labels": ids.clone()}

        if len(buffer) >= 2:
            ids = torch.tensor(buffer[: self.block_size], dtype=torch.long)
            yield {"input_ids": ids, "labels": ids.clone()}
