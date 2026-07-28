"""Shared pytest fixtures and helpers for the LLM Forge test suite.

Everything here is intentionally tiny (a micro char model, a handful of steps) so the full
suite runs in seconds on CPU.
"""

import json
import os

import pytest

from llmforge.config import ModelConfig, TrainConfig
from llmforge.data import PretrainData
from llmforge.model import GPT
from llmforge.tokenizer import save_tokenizer_meta, text_fingerprint, train_tokenizer
from llmforge.train import train

TINY_CORPUS = (
    "the little cat sat on the warm mat. the little dog ran in the green park. "
    "the sun was warm and the cat was happy. a little bird sang in the tall tree. "
    "the dog and the cat played all day in the park near the big house. "
) * 30

CHAT_ROWS = [
    {"prompt": "say hello", "response": "hello there, nice to meet you."},
    {"prompt": "talk about a cat", "response": "the cat sat on the mat in the sun."},
    {"prompt": "talk about a dog", "response": "the dog ran in the park all day."},
] * 6


@pytest.fixture
def char_tokenizer():
    return train_tokenizer(TINY_CORPUS, kind="char")


def make_char_tokenizer(text=TINY_CORPUS):
    return train_tokenizer(text, kind="char")


def tiny_model(tok, block_size=32):
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=32
    )
    return GPT(cfg)


@pytest.fixture
def base_checkpoint(tmp_path):
    """Train a micro base model + tokenizer into tmp_path/base and return its path."""
    tok = make_char_tokenizer()
    tok_path = os.path.join(str(tmp_path), "tokenizer.json")
    tok.save(tok_path)
    save_tokenizer_meta(tok_path, "char", tok.vocab_size, text_fingerprint(TINY_CORPUS))

    model = tiny_model(tok)
    ids = tok.encode(TINY_CORPUS)
    ds = PretrainData(ids, block_size=32, device="cpu")
    out = os.path.join(str(tmp_path), "base")
    cfg = TrainConfig(
        batch_size=8,
        block_size=32,
        steps=10,
        eval_interval=5,
        eval_iters=2,
        sample_every=0,
        device="cpu",
        out_dir=out,
    )
    train(model, ds, cfg, tokenizer=tok, tokenizer_path=tok_path)
    return out


@pytest.fixture
def chat_jsonl(tmp_path):
    path = os.path.join(str(tmp_path), "chat.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in CHAT_ROWS:
            f.write(json.dumps(row) + "\n")
    return path
