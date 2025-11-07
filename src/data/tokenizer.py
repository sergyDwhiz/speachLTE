"""
Simple character-level tokenizer with configurable vocabulary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(slots=True)
class TokenizerConfig:
    """Configuration for initializing tokenizers."""

    vocab_path: Path
    blank_token: str = "<blank>"
    unk_token: str = "<unk>"


class TextTokenizer:
    """Loads vocabulary from disk and encodes/decodes text sequences."""

    def __init__(self, token_to_id: Dict[str, int], blank_token: str = "<blank>", unk_token: str = "<unk>") -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.blank_token = blank_token
        self.unk_token = unk_token

    @classmethod
    def from_file(cls, vocab_path: Path, blank_token: str = "<blank>", unk_token: str = "<unk>") -> "TextTokenizer":
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        token_to_id = data.get("token_to_id", data)
        return cls(token_to_id=token_to_id, blank_token=blank_token, unk_token=unk_token)

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> "TextTokenizer":
        return cls.from_file(config.vocab_path, blank_token=config.blank_token, unk_token=config.unk_token)

    def encode(self, text: str) -> List[int]:
        """Convert string to token ids."""
        if not text:
            return [self.blank_id]
        ids = []
        for char in text:
            ids.append(self.token_to_id.get(char, self.unk_id))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token ids back to string."""
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, self.unk_token)
            if token in {self.blank_token, self.unk_token}:
                continue
            tokens.append(token)
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def blank_id(self) -> int:
        return self.token_to_id[self.blank_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]
