from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

try:
    from .sft_data import encode_conversation, normalize_conversations
    from .tokenizer import ByteTokenizer
    from .trainer_utils import iter_jsonl
except ImportError:
    from sft_data import encode_conversation, normalize_conversations
    from tokenizer import ByteTokenizer
    from trainer_utils import iter_jsonl


def _messages_from_value(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return normalize_conversations({"conversations": value})
    if isinstance(value, str) and value.strip():
        return [{"role": "assistant", "content": value.strip()}]
    return []


def normalize_preference_pair(record: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    chosen = _messages_from_value(record.get("chosen"))
    rejected = _messages_from_value(record.get("rejected"))
    if not chosen or not rejected:
        return [], []

    prompt = record.get("prompt") or record.get("question") or record.get("instruction")
    if isinstance(prompt, list):
        prompt_messages = _messages_from_value(prompt)
    elif isinstance(prompt, str) and prompt.strip():
        prompt_messages = [{"role": "user", "content": prompt.strip()}]
    else:
        prompt_messages = []
    if prompt_messages:
        chosen = prompt_messages + chosen
        rejected = prompt_messages + rejected
    return chosen, rejected


class PreferenceJsonlDataset(Dataset):
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
            raise ValueError(f"No usable preference examples found in {self.path}")

    def _build_examples(self) -> list[dict[str, torch.Tensor]]:
        examples: list[dict[str, torch.Tensor]] = []
        for record in iter_jsonl(self.path):
            chosen_messages, rejected_messages = normalize_preference_pair(record)
            if not chosen_messages or not rejected_messages:
                continue
            chosen_ids, chosen_labels = encode_conversation(
                chosen_messages,
                self.tokenizer,
                self.block_size,
            )
            rejected_ids, rejected_labels = encode_conversation(
                rejected_messages,
                self.tokenizer,
                self.block_size,
            )
            if all(label == -100 for label in chosen_labels + rejected_labels):
                continue
            examples.append(
                {
                    "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
                    "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
                    "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
                    "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
                }
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


@dataclass
class PreferenceCollator:
    pad_token_id: int

    def _pad(self, items: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(item.numel() for item in items)
        output = torch.full((len(items), max_len), pad_value, dtype=torch.long)
        for row, item in enumerate(items):
            output[row, : item.numel()] = item
        return output

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        chosen_input_ids = self._pad([item["chosen_input_ids"] for item in batch], self.pad_token_id)
        chosen_labels = self._pad([item["chosen_labels"] for item in batch], -100)
        rejected_input_ids = self._pad(
            [item["rejected_input_ids"] for item in batch],
            self.pad_token_id,
        )
        rejected_labels = self._pad([item["rejected_labels"] for item in batch], -100)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_input_ids.ne(self.pad_token_id).long(),
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_input_ids.ne(self.pad_token_id).long(),
        }
