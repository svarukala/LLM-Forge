"""Data loading for pre-training and fine-tuning.

Pre-training: one long stream of token ids; we grab random contiguous windows.
Fine-tuning (SFT): prompt/response pairs formatted into a chat template, where the loss
is only computed on the assistant's response tokens (the model learns to *answer*, not to
parrot the question).

Read alongside lessons/04-pretraining.md and lessons/05-finetuning.md.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import torch

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOT = "<|endoftext|>"


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class PretrainData:
    """A single long token stream; yields random (x, y) windows for next-token prediction."""

    def __init__(self, ids: List[int], block_size: int, device: str, val_frac: float = 0.1):
        data = torch.tensor(ids, dtype=torch.long)
        n_val = max(block_size + 1, int(len(data) * val_frac))
        self.train = data[:-n_val]
        self.val = data[-n_val:]
        self.block_size = block_size
        self.device = device

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.train if split == "train" else self.val
        max_start = len(source) - self.block_size - 1
        ix = torch.randint(0, max(1, max_start), (batch_size,))
        x = torch.stack([source[i:i + self.block_size] for i in ix])
        y = torch.stack([source[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)


def _format_example(tokenizer, prompt: str, response: str, block_size: int):
    """Build (input_ids, target_ids) for one chat example, masking the prompt in the loss.

    Crucially, if the example is longer than the context window we truncate the *response
    body* but always keep the trailing <|endoftext|> token, so the model learns where an
    answer ends and stops cleanly at inference (instead of running to the length cap).
    """
    prompt_ids = tokenizer.encode(f"{USER} {prompt}\n{ASSISTANT} ")
    try:
        eot = tokenizer.token_id(EOT)
    except Exception:
        eot = None
    body_ids = tokenizer.encode(response)

    max_total = block_size + 1
    reserve = 1 if eot is not None else 0
    avail = max_total - len(prompt_ids) - reserve
    body_ids = body_ids[:max(0, avail)]
    ids = prompt_ids + body_ids + ([eot] if eot is not None else [])

    x = ids[:-1]
    y = ids[1:]
    # mask (-1) the prompt portion so loss is computed only on the response (incl. the EOT)
    mask_len = min(len(prompt_ids) - 1, len(y))
    y = [-1] * mask_len + y[mask_len:]
    return x, y


class SFTData:
    """Supervised fine-tuning data from a JSONL file of {"prompt", "response"} rows."""

    def __init__(self, path: str, tokenizer, block_size: int, device: str):
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                x, y = _format_example(tokenizer, row["prompt"], row["response"], block_size)
                if len(x) < 2:
                    continue
                self.examples.append((x, y))
        if not self.examples:
            raise ValueError(f"No usable examples found in {path}")
        self.block_size = block_size
        self.device = device
        self.pad_id = _safe_pad_id(tokenizer)

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import random
        picks = [random.choice(self.examples) for _ in range(batch_size)]
        maxlen = max(len(x) for x, _ in picks)
        xs, ys = [], []
        for x, y in picks:
            pad = maxlen - len(x)
            xs.append(x + [self.pad_id] * pad)
            ys.append(y + [-1] * pad)   # padded targets are ignored in the loss
        x = torch.tensor(xs, dtype=torch.long, device=self.device)
        y = torch.tensor(ys, dtype=torch.long, device=self.device)
        return x, y


def _safe_pad_id(tokenizer) -> int:
    try:
        return tokenizer.token_id("<|pad|>")
    except Exception:
        return 0
