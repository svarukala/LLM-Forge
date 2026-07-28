"""Resume support, accurate stop metadata, and tokenizer-reuse compatibility."""

import os

from llmforge.checkpoint import checkpoint_info
from llmforge.config import TrainConfig
from llmforge.data import PretrainData
from llmforge.tokenizer import (
    load_tokenizer_meta,
    save_tokenizer_meta,
    text_fingerprint,
    train_tokenizer,
)
from llmforge.train import train
from tests.conftest import TINY_CORPUS, make_char_tokenizer, tiny_model


def _dataset(tok, block_size=32):
    ids = tok.encode(TINY_CORPUS)
    return PretrainData(ids, block_size=block_size, device="cpu")


def test_stop_metadata_is_accurate(tmp_path):
    tok = make_char_tokenizer()
    tok_path = os.path.join(str(tmp_path), "tok.json")
    tok.save(tok_path)
    model = tiny_model(tok)
    ds = _dataset(tok)
    out = os.path.join(str(tmp_path), "base")

    # stop after 3 completed steps
    calls = {"n": 0}

    def stop_flag():
        calls["n"] += 1
        return calls["n"] > 3  # allow 3 iterations, then request stop

    cfg = TrainConfig(
        batch_size=8,
        block_size=32,
        steps=50,
        eval_interval=100,
        eval_iters=2,
        sample_every=0,
        device="cpu",
        out_dir=out,
    )
    result = train(model, ds, cfg, tokenizer=tok, tokenizer_path=tok_path, stop_flag=stop_flag)
    assert result["status"] == "stopped"
    assert result["completed_steps"] == 3
    info = checkpoint_info(out)
    assert info["meta"]["status"] == "stopped"
    assert info["meta"]["completed_steps"] == 3
    assert info["resumable"] is True


def test_resume_continues_from_saved_step(tmp_path):
    tok = make_char_tokenizer()
    tok_path = os.path.join(str(tmp_path), "tok.json")
    tok.save(tok_path)
    model = tiny_model(tok)
    ds = _dataset(tok)
    out = os.path.join(str(tmp_path), "base")

    cfg1 = TrainConfig(
        batch_size=8,
        block_size=32,
        steps=5,
        eval_interval=100,
        eval_iters=2,
        sample_every=0,
        device="cpu",
        out_dir=out,
    )
    r1 = train(model, ds, cfg1, tokenizer=tok, tokenizer_path=tok_path)
    assert r1["completed_steps"] == 5

    # resume to 10 steps total using a fresh model object loaded from the checkpoint
    model2 = tiny_model(tok)
    from llmforge.checkpoint import load_checkpoint

    model2, _, _ = load_checkpoint(out, device="cpu")
    cfg2 = TrainConfig(
        batch_size=8,
        block_size=32,
        steps=10,
        eval_interval=100,
        eval_iters=2,
        sample_every=0,
        device="cpu",
        out_dir=out,
    )
    r2 = train(model2, ds, cfg2, tokenizer=tok, tokenizer_path=tok_path, resume_dir=out)
    assert r2["completed_steps"] == 10


def test_tokenizer_meta_detects_incompatibility(tmp_path):
    tok = train_tokenizer(TINY_CORPUS, kind="char")
    tok_path = os.path.join(str(tmp_path), "tok.json")
    tok.save(tok_path)
    fp = text_fingerprint(TINY_CORPUS)
    save_tokenizer_meta(tok_path, "char", tok.vocab_size, fp)

    meta = load_tokenizer_meta(tok_path)
    assert meta["kind"] == "char"
    assert meta["corpus_fingerprint"] == fp

    # requesting BPE on the same path must be detected as incompatible
    requested_kind = "bpe"
    compatible = meta["kind"] == requested_kind
    assert compatible is False

    # a different corpus changes the fingerprint (so reuse would be unsafe)
    assert text_fingerprint(TINY_CORPUS + " extra") != fp
