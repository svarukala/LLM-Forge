"""Generate text from a trained checkpoint (the 'inference' side of the model)."""

from __future__ import annotations

import torch

from .checkpoint import load_checkpoint
from .tokenizer import load_tokenizer
from .config import pick_device


def generate_text(out_dir: str, prompt: str, max_new_tokens: int = 200,
                  temperature: float = 0.8, top_k: int = 40,
                  device: str | None = None) -> str:
    device = pick_device(device)
    model, tok_path, _ = load_checkpoint(out_dir, device=device)
    tokenizer = load_tokenizer(tok_path)

    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())
