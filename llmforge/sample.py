"""Generate text from a trained checkpoint (the 'inference' side of the model)."""

from __future__ import annotations

import torch

from .checkpoint import load_checkpoint
from .config import pick_device
from .data import EOT
from .tokenizer import load_tokenizer


def generate_text(
    out_dir: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    stop_at_eot: bool = True,
    device: str | None = None,
) -> str:
    device = pick_device(device)
    model, tok_path, _ = load_checkpoint(out_dir, device=device)
    if not tok_path:
        raise ValueError(f"Checkpoint {out_dir} has no tokenizer to load.")
    tokenizer = load_tokenizer(tok_path)

    eos_id = None
    if stop_at_eot:
        try:
            eos_id = tokenizer.token_id(EOT)
        except Exception:
            eos_id = None

    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_id,
    )
    # decode() strips special tokens, so a trailing EOT won't appear in the text
    return tokenizer.decode(out[0].tolist())
