"""Saving and loading models in the portable safetensors format.

A checkpoint is a self-contained directory under ``runs/`` containing:
  * model.safetensors   — the weights (safe, portable tensor format)
  * tokenizer.json      — the tokenizer used to train it (kept *inside* the checkpoint so
                          the checkpoint is portable and the tokenizer path can be validated)
  * config.json         — schema version, model architecture, metadata, fingerprints
  * train_state.pt      — (optional) optimizer + RNG + step state, so training can resume

The tokenizer is stored inside the checkpoint so loading never has to trust an arbitrary
external path (guards against path traversal), and so a shared checkpoint always carries the
exact tokenizer it was trained with.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_model, save_model

from .config import ModelConfig
from .model import GPT

SCHEMA_VERSION = 2
TOKENIZER_FILENAME = "tokenizer.json"
TRAIN_STATE_FILENAME = "train_state.pt"


def _package_version() -> str:
    try:
        from . import __version__  # type: ignore

        return str(__version__)
    except Exception:
        return "0.0.0"


def _copy_tokenizer_into(out_dir: str, tokenizer_path: str) -> str | None:
    """Copy the tokenizer (and its .meta.json sidecar) into the checkpoint directory."""
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        return None
    dest = os.path.join(out_dir, TOKENIZER_FILENAME)
    src_abs = os.path.abspath(tokenizer_path)
    dest_abs = os.path.abspath(dest)
    if src_abs != dest_abs:
        with open(tokenizer_path, "rb") as s, open(dest, "wb") as d:
            d.write(s.read())
        meta_src = tokenizer_path + ".meta.json"
        if os.path.exists(meta_src):
            with open(meta_src, "rb") as s, open(dest + ".meta.json", "wb") as d:
                d.write(s.read())
    return TOKENIZER_FILENAME


def save_checkpoint(
    out_dir: str,
    model: GPT,
    tokenizer_path: str,
    meta: dict | None = None,
    train_state: dict | None = None,
) -> None:
    """Persist a checkpoint. If ``train_state`` is given, training can later resume from it."""
    os.makedirs(out_dir, exist_ok=True)
    # save_model handles tied/shared weights (wte.weight is tied to lm_head.weight)
    save_model(model, os.path.join(out_dir, "model.safetensors"))

    tok_rel = _copy_tokenizer_into(out_dir, tokenizer_path)
    config = {
        "schema_version": SCHEMA_VERSION,
        "llmforge_version": _package_version(),
        "model": model.cfg.as_dict(),
        "tokenizer": tok_rel,
        "meta": meta or {},
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if train_state is not None:
        torch.save(train_state, os.path.join(out_dir, TRAIN_STATE_FILENAME))


def _maybe_migrate_legacy(out_dir: str, config: dict, trust_migration: bool = False) -> dict:
    """Upgrade a pre-v2 checkpoint in place: pull an externally-referenced tokenizer inside
    the checkpoint directory and stamp the current schema version. Safe to call repeatedly.

    Security: the legacy tokenizer path is *untrusted* metadata. Before reading/copying it we
    verify it resolves **inside** the checkpoint directory. A legacy path that is absolute or
    escapes via ``..`` is refused unless ``trust_migration=True`` is explicitly passed, so a
    malicious config can never make us read an arbitrary server file.
    """
    if config.get("schema_version", 1) >= SCHEMA_VERSION:
        return config
    tok_rel = config.get("tokenizer")
    new_tok_rel = tok_rel
    if tok_rel:
        legacy_src = os.path.abspath(os.path.join(out_dir, tok_rel))
        root = os.path.abspath(out_dir)
        contained = os.path.isabs(tok_rel) is False and (
            legacy_src == root or os.path.commonpath([legacy_src, root]) == root
        )
        if not contained and not trust_migration:
            raise ValueError(
                f"Refusing to migrate legacy checkpoint {out_dir}: its tokenizer path "
                f"{tok_rel!r} points outside the checkpoint directory. If you trust this "
                f"checkpoint's source, re-run with trusted migration enabled."
            )
        local = os.path.join(out_dir, TOKENIZER_FILENAME)
        if not os.path.exists(local) and os.path.exists(legacy_src):
            with open(legacy_src, "rb") as s, open(local, "wb") as d:
                d.write(s.read())
            meta_src = legacy_src + ".meta.json"
            if os.path.exists(meta_src):
                with open(meta_src, "rb") as s, open(local + ".meta.json", "wb") as d:
                    d.write(s.read())
        new_tok_rel = TOKENIZER_FILENAME if os.path.exists(local) else None
    config = dict(config)
    config["schema_version"] = SCHEMA_VERSION
    config["tokenizer"] = new_tok_rel
    config.setdefault("llmforge_version", _package_version())
    print(f"[llm-forge] migrated legacy checkpoint {out_dir} to schema v{SCHEMA_VERSION}")
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


def _validate_config(out_dir: str, config: dict) -> None:
    if "model" not in config or not isinstance(config["model"], dict):
        raise ValueError(f"Checkpoint {out_dir} has no valid 'model' section in config.json")
    schema = config.get("schema_version", 1)
    if schema > SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint {out_dir} was written with schema v{schema}, but this LLM Forge "
            f"understands up to v{SCHEMA_VERSION}. Please upgrade llm-forge."
        )
    required = {"vocab_size", "block_size", "n_layer", "n_head", "n_embd"}
    missing = required - set(config["model"])
    if missing:
        raise ValueError(f"Checkpoint {out_dir} model config missing fields: {sorted(missing)}")

    tok_rel = config.get("tokenizer")
    if tok_rel is not None:
        # tokenizer must be contained within the checkpoint directory (no traversal/absolute)
        if os.path.isabs(tok_rel):
            raise ValueError(
                f"Checkpoint {out_dir} tokenizer path must be relative, got {tok_rel!r}"
            )
        resolved = os.path.abspath(os.path.join(out_dir, tok_rel))
        root = os.path.abspath(out_dir)
        if os.path.commonpath([resolved, root]) != root:
            raise ValueError(
                f"Checkpoint {out_dir} tokenizer path {tok_rel!r} escapes the checkpoint directory"
            )


def load_checkpoint(out_dir: str, device: str = "cpu", trust_migration: bool = False):
    """Return (model, tokenizer_path, config_dict) after validating the checkpoint."""
    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No checkpoint at {out_dir} (missing config.json)")
    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)
    config = _maybe_migrate_legacy(out_dir, config, trust_migration=trust_migration)
    _validate_config(out_dir, config)

    cfg = ModelConfig(**config["model"])
    model = GPT(cfg)
    load_model(model, os.path.join(out_dir, "model.safetensors"))
    model.to(device)
    model.eval()

    tok_rel = config.get("tokenizer")
    tok_path = os.path.normpath(os.path.join(out_dir, tok_rel)) if tok_rel else None
    return model, tok_path, config


def load_train_state(out_dir: str) -> dict | None:
    """Load optimizer/step/RNG state for resuming, or None if this checkpoint has none.

    Loaded with ``weights_only=True`` so a checkpoint can never execute arbitrary code on
    resume: everything we save (optimizer tensors, ints, RNG tensors/lists) is a plain
    tensor or primitive container, so no pickled classes are needed. See SECURITY.md.
    """
    p = os.path.join(out_dir, TRAIN_STATE_FILENAME)
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=True)


def checkpoint_info(out_dir: str, trust_migration: bool = False) -> dict:
    """Return a human-friendly summary of a checkpoint (used by the checkpoint-info CLI)."""
    with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    config = _maybe_migrate_legacy(out_dir, config, trust_migration=trust_migration)
    _validate_config(out_dir, config)
    info = {
        "path": out_dir,
        "schema_version": config.get("schema_version", 1),
        "llmforge_version": config.get("llmforge_version", "unknown"),
        "model": config["model"],
        "tokenizer": config.get("tokenizer"),
        "meta": config.get("meta", {}),
        "resumable": os.path.exists(os.path.join(out_dir, TRAIN_STATE_FILENAME)),
    }
    return info
