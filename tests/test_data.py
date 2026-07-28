"""Corpus validation, SFT truncation, masking, and dropped-example accounting."""

import json
import os

import pytest

from llmforge.data import PretrainData, SFTData, _format_example
from llmforge.tokenizer import train_tokenizer
from tests.conftest import TINY_CORPUS


def _tok():
    return train_tokenizer(TINY_CORPUS + " abcdefghijklmnopqrstuvwxyz 0123456789", kind="char")


def test_pretrain_rejects_tiny_corpus():
    tok = _tok()
    ids = tok.encode("the cat")  # far fewer than 2*(block_size+1)
    with pytest.raises(ValueError) as e:
        PretrainData(ids, block_size=64, device="cpu")
    assert "too small" in str(e.value).lower()


def test_pretrain_window_bounds_sampleable():
    tok = _tok()
    ids = tok.encode(TINY_CORPUS)
    ds = PretrainData(ids, block_size=16, device="cpu")
    x, y = ds.get_batch("train", 4)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


def test_sft_long_prompt_never_overflows():
    tok = _tok()
    block_size = 24
    long_prompt = "the cat " * 100  # deliberately much longer than the context
    out = _format_example(tok, long_prompt, "the dog ran in the park.", block_size)
    assert out is not None
    x, y = out
    assert len(x) <= block_size
    assert len(y) <= block_size


def test_sft_masks_prompt_tokens():
    tok = _tok()
    out = _format_example(tok, "say hi", "hello there.", 48)
    assert out is not None
    x, y = out
    # at least one masked (-1) label at the start, and at least one real target
    assert y[0] == -1
    assert any(t != -1 for t in y)


def test_sft_drops_unusable_and_counts(tmp_path):
    tok = _tok()
    path = os.path.join(str(tmp_path), "chat.jsonl")
    rows = [
        {"prompt": "hi", "response": "hello there friend."},
        {"prompt": "x" * 500, "response": ""},  # empty response -> unusable
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ds = SFTData(path, tok, block_size=32, device="cpu")
    assert ds.dropped >= 1
    assert len(ds.examples) >= 1


def test_sft_all_unusable_raises(tmp_path):
    tok = _tok()
    path = os.path.join(str(tmp_path), "chat.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "x" * 200, "response": ""}) + "\n")
    with pytest.raises(ValueError):
        SFTData(path, tok, block_size=16, device="cpu")
