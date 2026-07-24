"""The training loop, shared by pre-training and fine-tuning.

A single `train()` function drives both:
  * pre-training reads a long text stream (PretrainData)
  * fine-tuning reads chat pairs with a masked prompt (SFTData)

Both are just "predict the next token" — fine-tuning simply starts from pre-trained
weights and only scores the assistant's tokens. That symmetry is the whole point.

Read alongside lessons/04-pretraining.md.
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Optional

import torch

from .config import ModelConfig, TrainConfig, pick_device
from .model import GPT
from .checkpoint import save_checkpoint


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to 10% of the peak learning rate."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    return cfg.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def _estimate_loss(model: GPT, dataset, cfg: TrainConfig) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    model: GPT,
    dataset,
    cfg: TrainConfig,
    tokenizer=None,
    tokenizer_path: str = "",
    on_event: Optional[Callable[[dict], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> GPT:
    """Train `model` on `dataset`. Emits progress dicts via `on_event` (for the dashboard).

    `on_event` receives dicts like:
      {"type": "step",  "step": 50, "loss": 3.1, "lr": 3e-4, "tok_per_sec": 12000}
      {"type": "eval",  "step": 50, "train": 3.0, "val": 3.2}
      {"type": "sample","step": 100, "text": "..."}
      {"type": "done",  "out_dir": "runs/base"}
    """
    device = pick_device(cfg.device)
    model.to(device)
    model.train()
    torch.manual_seed(cfg.seed)

    optimizer = model.configure_optimizer(cfg.weight_decay, cfg.learning_rate)
    os.makedirs(cfg.out_dir, exist_ok=True)

    def emit(evt: dict) -> None:
        if on_event:
            on_event(evt)

    emit({"type": "start", "params": model.num_params(), "device": device,
          "steps": cfg.steps})

    t0 = time.time()
    tokens_since = 0
    for step in range(1, cfg.steps + 1):
        if stop_flag and stop_flag():
            emit({"type": "stopped", "step": step})
            break

        lr = _lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = dataset.get_batch("train", cfg.batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        tokens_since += x.numel()
        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            tps = tokens_since / dt if dt > 0 else 0.0
            emit({"type": "step", "step": step, "loss": float(loss.item()),
                  "lr": lr, "tok_per_sec": tps})
            t0, tokens_since = time.time(), 0

        if step % cfg.eval_interval == 0 or step == cfg.steps:
            metrics = _estimate_loss(model, dataset, cfg)
            emit({"type": "eval", "step": step, **metrics})

        if tokenizer is not None and (step % cfg.sample_every == 0):
            text = _quick_sample(model, tokenizer, device)
            emit({"type": "sample", "step": step, "text": text})

    save_checkpoint(cfg.out_dir, model, tokenizer_path,
                    meta={"steps": cfg.steps, "final_loss": float(loss.item())})
    emit({"type": "done", "out_dir": cfg.out_dir})
    return model


@torch.no_grad()
def _quick_sample(model: GPT, tokenizer, device: str, n: int = 80) -> str:
    model.eval()
    start = tokenizer.encode("The") or [0]
    idx = torch.tensor([start], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n, temperature=0.8, top_k=40)
    model.train()
    return tokenizer.decode(out[0].tolist())
