"""Tokenizer for LLM Forge.

Two backends behind one small interface:

  * "bpe"  — a real Byte-Pair Encoding tokenizer (HuggingFace `tokenizers`), the same
             algorithm GPT-2 uses. Trained on your corpus, saved as tokenizer.json.
  * "char" — a trivial character-level tokenizer (no training, no dependencies beyond
             stdlib) that's perfect for tiny demos and understanding the concept.

Read alongside lessons/02-tokenization.md.
"""

from __future__ import annotations

import json
import os
from typing import List

SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>", "<|user|>", "<|assistant|>"]


class CharTokenizer:
    """Maps each unique character to an integer. Zero training required."""

    def __init__(self, vocab: List[str]):
        self.itos = list(vocab)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        vocab = SPECIAL_TOKENS + sorted(set(text))
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else ""
            out.append("" if tok in SPECIAL_TOKENS else tok)
        return "".join(out)

    def token_id(self, special: str) -> int:
        return self.stoi[special]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "char", "vocab": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["vocab"])


class BPETokenizer:
    """Thin wrapper around HuggingFace `tokenizers` (Byte-Pair Encoding)."""

    def __init__(self, tk):
        self._tk = tk

    @classmethod
    def train(cls, text: str, vocab_size: int = 2048) -> "BPETokenizer":
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        tk = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|unk|>"] + SPECIAL_TOKENS,
            show_progress=False,
        )
        tk.train_from_iterator([text], trainer=trainer)
        return cls(tk)

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self._tk.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tk.decode(ids)

    def token_id(self, special: str) -> int:
        return self._tk.token_to_id(special)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._tk.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(path))


def load_tokenizer(path: str):
    """Load whichever tokenizer kind was saved at `path`."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(64)
    if '"kind": "char"' in head or '"kind":"char"' in head:
        return CharTokenizer.load(path)
    return BPETokenizer.load(path)


def train_tokenizer(text: str, kind: str = "bpe", vocab_size: int = 2048):
    if kind == "char":
        return CharTokenizer.train(text)
    return BPETokenizer.train(text, vocab_size=vocab_size)
