"""Tokenizer for LLM Forge.

Two backends behind one small interface:

  * "bpe"  — a real Byte-Pair Encoding tokenizer (HuggingFace `tokenizers`), the same
             algorithm GPT-2 uses. Trained on your corpus, saved as tokenizer.json.
  * "char" — a trivial character-level tokenizer (no training, no dependencies beyond
             stdlib) that's perfect for tiny demos and understanding the concept.

Both treat the reserved control strings in ``SPECIAL_TOKENS`` (e.g. ``<|user|>``) as single
atomic tokens via longest-match encoding, so the chat template round-trips correctly.

Read alongside lessons/02-tokenization.md.
"""

from __future__ import annotations

import hashlib
import json
import os

SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>", "<|user|>", "<|assistant|>"]


def text_fingerprint(text: str) -> str:
    """Stable short fingerprint of a corpus, used to detect tokenizer/corpus drift."""
    h = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
    return f"sha1:{h[:16]}:{len(text)}"


def _meta_path(tokenizer_path: str) -> str:
    return tokenizer_path + ".meta.json"


def save_tokenizer_meta(
    tokenizer_path: str, kind: str, vocab_size: int, corpus_fingerprint: str | None = None
) -> None:
    """Write a small sidecar describing how a tokenizer was trained."""
    meta = {"kind": kind, "vocab_size": vocab_size, "corpus_fingerprint": corpus_fingerprint}
    with open(_meta_path(tokenizer_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_tokenizer_meta(tokenizer_path: str) -> dict | None:
    """Read the sidecar metadata for a tokenizer, or None if it does not exist."""
    p = _meta_path(tokenizer_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _encode_with_specials(text: str, encode_plain, special_to_id: dict) -> list[int]:
    """Longest-match tokenization: reserved specials are atomic, everything else plain.

    `encode_plain(str) -> List[int]` handles ordinary (non-special) substrings.
    """
    specials = sorted(special_to_id, key=len, reverse=True)
    ids: list[int] = []
    i, n = 0, len(text)
    buf_start = 0
    while i < n:
        matched = None
        if text[i] == "<":  # cheap gate: all specials start with '<'
            for sp in specials:
                if text.startswith(sp, i):
                    matched = sp
                    break
        if matched is not None:
            if buf_start < i:
                ids.extend(encode_plain(text[buf_start:i]))
            ids.append(special_to_id[matched])
            i += len(matched)
            buf_start = i
        else:
            i += 1
    if buf_start < n:
        ids.extend(encode_plain(text[buf_start:]))
    return ids


class CharTokenizer:
    """Maps each unique character to an integer. Zero training required."""

    kind = "char"

    def __init__(self, vocab: list[str]):
        self.itos = list(vocab)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self._special_to_id = {s: self.stoi[s] for s in SPECIAL_TOKENS if s in self.stoi}

    @classmethod
    def train(cls, text: str) -> CharTokenizer:
        vocab = SPECIAL_TOKENS + sorted(set(text))
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        def plain(s: str) -> list[int]:
            return [self.stoi[c] for c in s if c in self.stoi]

        return _encode_with_specials(text, plain, self._special_to_id)

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else ""
            out.append("" if tok in SPECIAL_TOKENS else tok)
        return "".join(out)

    def pieces(self, text: str) -> list[dict]:
        """Return [{'id', 'piece'}] for each token, for the tokenizer playground."""
        out = []
        for i in self.encode(text):
            piece = self.itos[i] if 0 <= i < len(self.itos) else ""
            out.append({"id": i, "piece": piece})
        return out

    def token_id(self, special: str) -> int:
        return self.stoi[special]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "char", "vocab": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> CharTokenizer:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["vocab"])


class BPETokenizer:
    """Thin wrapper around HuggingFace `tokenizers` (Byte-Pair Encoding)."""

    kind = "bpe"

    def __init__(self, tk):
        self._tk = tk

    @classmethod
    def train(cls, text: str, vocab_size: int = 2048) -> BPETokenizer:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tk = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|unk|>"] + SPECIAL_TOKENS,
            show_progress=False,
        )
        tk.train_from_iterator([text], trainer=trainer)
        # Ensure specials are registered as atomic added tokens so encode() never splits them.
        tk.add_special_tokens(SPECIAL_TOKENS)
        return cls(tk)

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tk.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tk.decode(ids)

    def pieces(self, text: str) -> list[dict]:
        """Return [{'id', 'piece'}] for each token, for the tokenizer playground.

        ByteLevel BPE marks a leading space as 'Ġ' and a newline as 'Ċ'; we translate those
        back to a real space/newline so the chips read naturally.
        """
        enc = self._tk.encode(text)
        out = []
        for tid, tokstr in zip(enc.ids, enc.tokens, strict=True):
            piece = tokstr.replace("\u0120", " ").replace("\u010a", "\n")
            out.append({"id": tid, "piece": piece})
        return out

    def token_id(self, special: str) -> int:
        tid = self._tk.token_to_id(special)
        if tid is None:
            raise KeyError(special)
        return tid

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._tk.save(path)

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))


def load_tokenizer(path: str):
    """Load whichever tokenizer kind was saved at `path`."""
    with open(path, encoding="utf-8") as f:
        head = f.read(64)
    if '"kind": "char"' in head or '"kind":"char"' in head:
        return CharTokenizer.load(path)
    return BPETokenizer.load(path)


def train_tokenizer(text: str, kind: str = "bpe", vocab_size: int = 2048):
    if kind == "char":
        return CharTokenizer.train(text)
    if kind == "bpe":
        return BPETokenizer.train(text, vocab_size=vocab_size)
    raise ValueError(f"Unknown tokenizer kind '{kind}' (expected 'bpe' or 'char')")
