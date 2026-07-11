from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"


class ByteTokenizer:
    """A deterministic UTF-8 byte tokenizer.

    This is the bootstrap tokenizer for the M2 training loop. It is deliberately
    dependency-free and lossless for multilingual text. A real BPE tokenizer can
    replace it later without changing the pretraining data flow.
    """

    def __init__(self, special_tokens: SpecialTokens | None = None):
        self.special_tokens = special_tokens or SpecialTokens()
        self.special_token_to_id = {
            self.special_tokens.pad: 0,
            self.special_tokens.bos: 1,
            self.special_tokens.eos: 2,
            self.special_tokens.unk: 3,
        }
        self.id_to_special_token = {v: k for k, v in self.special_token_to_id.items()}
        self.byte_offset = len(self.special_token_to_id)
        self.vocab_size = self.byte_offset + 256

    @property
    def pad_token_id(self) -> int:
        return self.special_token_to_id[self.special_tokens.pad]

    @property
    def bos_token_id(self) -> int:
        return self.special_token_to_id[self.special_tokens.bos]

    @property
    def eos_token_id(self) -> int:
        return self.special_token_to_id[self.special_tokens.eos]

    @property
    def unk_token_id(self) -> int:
        return self.special_token_to_id[self.special_tokens.unk]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        ids.extend(byte + self.byte_offset for byte in text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], skip_special_tokens: bool = True) -> str:
        byte_values = bytearray()
        pieces: list[str] = []
        for token_id in ids:
            if token_id in self.id_to_special_token:
                if byte_values:
                    pieces.append(byte_values.decode("utf-8", errors="replace"))
                    byte_values.clear()
                if not skip_special_tokens:
                    pieces.append(self.id_to_special_token[token_id])
                continue
            byte_id = token_id - self.byte_offset
            if 0 <= byte_id <= 255:
                byte_values.append(byte_id)
            elif not skip_special_tokens:
                pieces.append(self.special_tokens.unk)
        if byte_values:
            pieces.append(byte_values.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def to_dict(self) -> dict:
        return {
            "type": "byte",
            "special_tokens": {
                "pad": self.special_tokens.pad,
                "bos": self.special_tokens.bos,
                "eos": self.special_tokens.eos,
                "unk": self.special_tokens.unk,
            },
            "byte_offset": self.byte_offset,
            "vocab_size": self.vocab_size,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "byte":
            raise ValueError(f"Unsupported tokenizer type: {payload.get('type')}")
        tokens = payload.get("special_tokens", {})
        return cls(
            SpecialTokens(
                pad=tokens.get("pad", "<pad>"),
                bos=tokens.get("bos", "<bos>"),
                eos=tokens.get("eos", "<eos>"),
                unk=tokens.get("unk", "<unk>"),
            )
        )


class BPETokenizer:
    """A small character-level BPE tokenizer for local mini_llm experiments."""

    def __init__(
        self,
        stoi: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: SpecialTokens | None = None,
    ):
        self.special_tokens = special_tokens or SpecialTokens()
        self.stoi = stoi
        self.itos = {idx: token for token, idx in stoi.items()}
        self.merges = merges
        self._merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        min_freq: int = 2,
        special_tokens: SpecialTokens | None = None,
    ) -> "BPETokenizer":
        tokens = special_tokens or SpecialTokens()
        initial_tokens = [tokens.pad, tokens.bos, tokens.eos, tokens.unk]
        initial_tokens.extend(char for char in sorted(set(text)) if char not in initial_tokens)
        stoi = {token: idx for idx, token in enumerate(initial_tokens)}

        word_freqs: dict[tuple[str, ...], int] = {}
        for line, freq in Counter(line for line in text.splitlines() if line.strip()).items():
            word_freqs[tuple(line)] = freq

        pair_counts: Counter[tuple[str, str]] = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += freq

        merges: list[tuple[str, str]] = []
        while len(stoi) < vocab_size:
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_freq:
                break

            new_token = "".join(best_pair)
            if new_token in stoi:
                pair_counts.pop(best_pair, None)
                break
            stoi[new_token] = len(stoi)
            merges.append(best_pair)

            affected = [
                (word, freq)
                for word, freq in list(word_freqs.items())
                if cls._contains_pair(word, best_pair)
            ]
            updates: Counter[tuple[str, ...]] = Counter()
            for word, freq in affected:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)

                new_word = cls._merge_word(word, best_pair)
                updates[new_word] += freq
                word_freqs.pop(word, None)

                for i in range(len(new_word) - 1):
                    pair_counts[(new_word[i], new_word[i + 1])] += freq

            for word, freq in updates.items():
                word_freqs[word] = word_freqs.get(word, 0) + freq

            pair_counts.pop(best_pair, None)
        return cls(stoi=stoi, merges=merges, special_tokens=tokens)

    @staticmethod
    def _contains_pair(word: tuple[str, ...], pair: tuple[str, str]) -> bool:
        a, b = pair
        return any(word[i] == a and word[i + 1] == b for i in range(len(word) - 1))

    @staticmethod
    def _merge_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        a, b = pair
        merged: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return tuple(merged)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_token_id(self) -> int:
        return self.stoi[self.special_tokens.pad]

    @property
    def bos_token_id(self) -> int:
        return self.stoi[self.special_tokens.bos]

    @property
    def eos_token_id(self) -> int:
        return self.stoi[self.special_tokens.eos]

    @property
    def unk_token_id(self) -> int:
        return self.stoi[self.special_tokens.unk]

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        while len(tokens) >= 2:
            best_rank = len(self.merges)
            best_index = -1
            for i in range(len(tokens) - 1):
                rank = self._merge_rank.get((tokens[i], tokens[i + 1]), len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_index = i
            if best_index < 0:
                break
            tokens = (
                tokens[:best_index]
                + [tokens[best_index] + tokens[best_index + 1]]
                + tokens[best_index + 2 :]
            )
        return tokens

    def _tokenize(self, text: str) -> list[str]:
        pieces: list[str] = []
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if line:
                pieces.extend(self._apply_merges(list(line)))
            if index < len(lines) - 1:
                pieces.append("\n")
        return pieces

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        ids.extend(self.stoi.get(token, self.unk_token_id) for token in self._tokenize(text))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], skip_special_tokens: bool = True) -> str:
        pieces: list[str] = []
        special_values = {
            self.special_tokens.pad,
            self.special_tokens.bos,
            self.special_tokens.eos,
            self.special_tokens.unk,
        }
        for token_id in ids:
            token = self.itos.get(int(token_id), self.special_tokens.unk)
            if skip_special_tokens and token in special_values:
                continue
            pieces.append(token)
        return "".join(pieces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "bpe",
            "stoi": self.stoi,
            "merges": self.merges,
            "special_tokens": {
                "pad": self.special_tokens.pad,
                "bos": self.special_tokens.bos,
                "eos": self.special_tokens.eos,
                "unk": self.special_tokens.unk,
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokens = payload.get("special_tokens", {})
        special_tokens = SpecialTokens(
            pad=tokens.get("pad", "<pad>"),
            bos=tokens.get("bos", "<bos>"),
            eos=tokens.get("eos", "<eos>"),
            unk=tokens.get("unk", "<unk>"),
        )
        stoi = {str(key): int(value) for key, value in payload["stoi"].items()}
        merges = [tuple(item) for item in payload.get("merges", [])]
        return cls(stoi=stoi, merges=merges, special_tokens=special_tokens)


Tokenizer = ByteTokenizer | BPETokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tokenizer_type = payload.get("type")
    if tokenizer_type == "byte":
        return ByteTokenizer.load(path)
    if tokenizer_type == "bpe":
        return BPETokenizer.load(path)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def load_or_create_tokenizer(path: str | Path | None = None) -> Tokenizer:
    if path is None:
        return ByteTokenizer()
    path = Path(path)
    if path.exists():
        return load_tokenizer(path)
    tokenizer = ByteTokenizer()
    tokenizer.save(path)
    return tokenizer
