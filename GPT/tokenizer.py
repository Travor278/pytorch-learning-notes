from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class CharTokenizer:
    def __init__(
        self,
        stoi: dict[str, int],
        itos: dict[int, str],
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
    ) -> None:
        self.stoi = stoi
        self.itos = itos
        self.special_tokens = special_tokens

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

    @classmethod
    def from_text(
        cls,
        text: str,
        min_freq: int = 1,
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
    ) -> "CharTokenizer":
        counter = Counter(text)

        vocab_chars = [
            ch
            for ch, freq in sorted(counter.items(), key=lambda item: item[0])
            if freq >= min_freq and ch not in special_tokens
        ]

        tokens = [*special_tokens, *vocab_chars]
        stoi = {token: idx for idx, token in enumerate(tokens)}
        itos = {idx: token for token, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos, special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        chars: list[str] = []
        for idx in ids:
            token = self.itos.get(idx, self.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "stoi": self.stoi,
            "special_tokens": list(self.special_tokens),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        stoi = {str(token): int(idx) for token, idx in payload["stoi"].items()}
        itos = {idx: token for token, idx in stoi.items()}
        special_tokens = tuple(payload.get("special_tokens", list(SPECIAL_TOKENS)))
        return cls(stoi=stoi, itos=itos, special_tokens=special_tokens)
