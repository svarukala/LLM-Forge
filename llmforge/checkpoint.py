"""Saving and loading models in the portable safetensors format.

A checkpoint is a directory under runs/ containing:
  * model.safetensors   — the weights
  * config.json         — model architecture + tokenizer path + training metadata

safetensors is a simple, safe, cross-platform tensor format — the same one used across
the open-model ecosystem — so models you make here are easy to share with teammates.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from safetensors.torch import save_model, load_model

from .config import ModelConfig
from .model import GPT


def save_checkpoint(out_dir: str, model: GPT, tokenizer_path: str, meta: Optional[dict] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # save_model handles tied/shared weights (wte.weight is tied to lm_head.weight)
    save_model(model, os.path.join(out_dir, "model.safetensors"))
    config = {
        "model": model.cfg.as_dict(),
        "tokenizer": os.path.relpath(tokenizer_path, out_dir) if tokenizer_path else None,
        "meta": meta or {},
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_checkpoint(out_dir: str, device: str = "cpu"):
    """Return (model, tokenizer_path, config_dict)."""
    with open(os.path.join(out_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    cfg = ModelConfig(**config["model"])
    model = GPT(cfg)
    load_model(model, os.path.join(out_dir, "model.safetensors"))
    model.to(device)
    model.eval()

    tok_rel = config.get("tokenizer")
    tok_path = os.path.normpath(os.path.join(out_dir, tok_rel)) if tok_rel else None
    return model, tok_path, config
