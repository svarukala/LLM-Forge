"""Checkpoint save/load, schema validation, path-containment, and resume state."""

import json
import os

import pytest

from llmforge.checkpoint import checkpoint_info, load_checkpoint, load_train_state, save_checkpoint
from tests.conftest import make_char_tokenizer, tiny_model


def _save_tiny(tmp_path, name="ckpt", train_state=None, meta=None):
    tok = make_char_tokenizer()
    tok_path = os.path.join(str(tmp_path), "tokenizer.json")
    tok.save(tok_path)
    model = tiny_model(tok)
    out = os.path.join(str(tmp_path), name)
    save_checkpoint(
        out, model, tok_path, meta=meta or {"final_loss": 1.23}, train_state=train_state
    )
    return out, model


def test_save_and_load_roundtrip(tmp_path):
    out, model = _save_tiny(tmp_path)
    loaded, tok_path, cfg = load_checkpoint(out, device="cpu")
    assert loaded.cfg.n_layer == model.cfg.n_layer
    assert loaded.cfg.vocab_size == model.cfg.vocab_size
    # tokenizer is copied INSIDE the checkpoint (self-contained, no traversal)
    assert os.path.dirname(os.path.abspath(tok_path)) == os.path.abspath(out)
    assert os.path.exists(tok_path)


def test_tokenizer_is_contained(tmp_path):
    out, _ = _save_tiny(tmp_path)
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert not os.path.isabs(cfg["tokenizer"])
    assert ".." not in cfg["tokenizer"]


def test_reject_traversal_tokenizer_path(tmp_path):
    out, _ = _save_tiny(tmp_path)
    cfg_path = os.path.join(out, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["tokenizer"] = "..\\escape.json"
    cfg["schema_version"] = 2  # already migrated, so no auto-migration
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    with pytest.raises(ValueError):
        load_checkpoint(out, device="cpu")


def test_reject_missing_model_fields(tmp_path):
    out, _ = _save_tiny(tmp_path)
    cfg_path = os.path.join(out, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    del cfg["model"]["n_layer"]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    with pytest.raises(ValueError):
        load_checkpoint(out, device="cpu")


def test_checkpoint_info_reports_resumable(tmp_path):
    out, _ = _save_tiny(tmp_path, train_state={"step": 5})
    info = checkpoint_info(out)
    assert info["resumable"] is True
    assert info["schema_version"] >= 2
    state = load_train_state(out)
    assert state["step"] == 5


def test_checkpoint_info_non_resumable(tmp_path):
    out, _ = _save_tiny(tmp_path)
    info = checkpoint_info(out)
    assert info["resumable"] is False
    assert load_train_state(out) is None


def test_legacy_checkpoint_migrates(tmp_path):
    # simulate a pre-v2 checkpoint whose tokenizer lives inside the checkpoint under an old name
    out, _ = _save_tiny(tmp_path)
    inside = os.path.join(out, "tokenizer.json")
    legacy = os.path.join(out, "legacy_tok.json")
    os.replace(inside, legacy)
    cfg_path = os.path.join(out, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("schema_version", None)
    cfg["tokenizer"] = "legacy_tok.json"  # contained (no traversal)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    # loading should auto-migrate: pull tokenizer to the standard name + bump schema
    model, tok_path, cfg2 = load_checkpoint(out, device="cpu")
    assert cfg2["schema_version"] >= 2
    assert os.path.dirname(os.path.abspath(tok_path)) == os.path.abspath(out)


def test_legacy_external_tokenizer_rejected_by_default(tmp_path):
    # a legacy config that points its tokenizer OUTSIDE the checkpoint must be refused
    out, _ = _save_tiny(tmp_path)
    external = os.path.join(str(tmp_path), "tokenizer_legacy.json")
    os.replace(os.path.join(out, "tokenizer.json"), external)
    cfg_path = os.path.join(out, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("schema_version", None)
    cfg["tokenizer"] = os.path.join("..", "tokenizer_legacy.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    with pytest.raises(ValueError):
        load_checkpoint(out, device="cpu")


def test_legacy_external_tokenizer_migrates_when_trusted(tmp_path):
    # the same escaping legacy layout migrates only when trust_migration is explicit
    out, _ = _save_tiny(tmp_path)
    external = os.path.join(str(tmp_path), "tokenizer_legacy.json")
    os.replace(os.path.join(out, "tokenizer.json"), external)
    cfg_path = os.path.join(out, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("schema_version", None)
    cfg["tokenizer"] = os.path.join("..", "tokenizer_legacy.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    model, tok_path, cfg2 = load_checkpoint(out, device="cpu", trust_migration=True)
    assert cfg2["schema_version"] >= 2
    assert os.path.dirname(os.path.abspath(tok_path)) == os.path.abspath(out)
