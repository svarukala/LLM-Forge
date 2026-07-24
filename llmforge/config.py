"""Central configuration for LLM Forge's PyTorch pipeline.

Small by default so everything runs on a laptop CPU in minutes. Bump these up
(and use a CUDA GPU) once you want a model that writes something coherent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Optional


RUNS_DIR = os.environ.get("LLMFORGE_RUNS", "runs")


@dataclass
class ModelConfig:
    """GPT architecture hyper-parameters."""

    vocab_size: int = 2048
    block_size: int = 128        # context length (how many tokens the model sees)
    n_layer: int = 4             # number of transformer blocks
    n_head: int = 4              # attention heads per block
    n_embd: int = 128            # embedding / hidden width
    dropout: float = 0.1
    bias: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Optimization hyper-parameters shared by pre-training and fine-tuning."""

    batch_size: int = 16
    block_size: int = 128
    steps: int = 500
    eval_interval: int = 50
    eval_iters: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 50
    seed: int = 1337
    device: Optional[str] = None   # auto-detected if None
    compile: bool = False          # torch.compile (off by default for portability)
    sample_every: int = 100        # print a sample this often during training
    out_dir: str = field(default_factory=lambda: os.path.join(RUNS_DIR, "base"))

    def as_dict(self) -> dict:
        return asdict(self)


def pick_device(requested: Optional[str] = None) -> str:
    """Choose the best available device unless one is explicitly requested."""
    if requested:
        return requested
    try:
        import torch
    except ImportError:  # playground path doesn't need torch
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon, if a teammate is on a Mac
    return "cpu"
