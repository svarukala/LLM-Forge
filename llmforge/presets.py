"""Hardware detection and named training presets for LLM Forge.

The whole point of the workshop is to *feel* why scale matters. A 4M-param model on
a CPU writes grammatical English but can't reliably follow a prompt ("write about a
dog" -> a story about a bird). A ~25M+ param model on a GPU starts to bind the prompt
to the content and stops cleanly. These presets make that trade-off one flag away, and
we auto-recommend the right one for the machine you're on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Preset:
    """A complete, verified set of knobs for one hardware tier."""

    name: str
    blurb: str
    tier: str            # "cpu" or "gpu"
    # tokenizer
    tok_kind: str
    vocab_size: int
    # model
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    # training
    batch_size: int
    pretrain_steps: int
    finetune_steps: int
    lr: float
    finetune_lr: float
    # data sizing guidance (MB of pre-train corpus)
    data_mb: float
    approx_params_m: float
    est_time: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "blurb": self.blurb,
            "tier": self.tier,
            "tok_kind": self.tok_kind,
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "batch_size": self.batch_size,
            "pretrain_steps": self.pretrain_steps,
            "finetune_steps": self.finetune_steps,
            "lr": self.lr,
            "finetune_lr": self.finetune_lr,
            "data_mb": self.data_mb,
            "approx_params_m": self.approx_params_m,
            "est_time": self.est_time,
        }


PRESETS = {
    # Runs on any laptop/devbox CPU in ~15-20 min. Coherent English, loose topic-following.
    "cpu": Preset(
        name="cpu",
        blurb="CPU-friendly. Coherent sentences in ~15-20 min. Topic-following is loose "
              "(this is the teaching moment about scale).",
        tier="cpu",
        tok_kind="bpe", vocab_size=4096,
        block_size=128, n_layer=4, n_head=4, n_embd=256,
        batch_size=32, pretrain_steps=1200, finetune_steps=500,
        lr=3e-4, finetune_lr=1e-4,
        data_mb=6.0, approx_params_m=4.2, est_time="~15-20 min on CPU",
    ),
    # Needs a CUDA GPU. ~25M params, binds prompt->content, stops cleanly.
    "gpu": Preset(
        name="gpu",
        blurb="GPU-friendly (~25M params). Follows the prompt topic and stops cleanly. "
              "Needs CUDA; ~1-2 hrs on a consumer GPU.",
        tier="gpu",
        tok_kind="bpe", vocab_size=8192,
        block_size=256, n_layer=8, n_head=8, n_embd=512,
        batch_size=48, pretrain_steps=5000, finetune_steps=1500,
        lr=6e-4, finetune_lr=1e-4,
        data_mb=30.0, approx_params_m=25.0, est_time="~1-2 hrs on a consumer CUDA GPU",
    ),
    # For the brave with a beefy GPU: GPT-2-small-ish. Noticeably better prose.
    "gpu-large": Preset(
        name="gpu-large",
        blurb="Bigger GPU preset (~85M params, GPT-2-small class). Best prose; needs a "
              "strong GPU and more data/time.",
        tier="gpu",
        tok_kind="bpe", vocab_size=8192,
        block_size=512, n_layer=12, n_head=12, n_embd=768,
        batch_size=32, pretrain_steps=12000, finetune_steps=3000,
        lr=6e-4, finetune_lr=8e-5,
        data_mb=100.0, approx_params_m=85.0, est_time="several hrs on a strong CUDA GPU",
    ),
}


@dataclass
class Hardware:
    device: str            # cpu | cuda | mps
    gpu_name: Optional[str]
    gpu_mem_gb: Optional[float]
    cuda: bool

    def as_dict(self) -> dict:
        return {
            "device": self.device,
            "gpu_name": self.gpu_name,
            "gpu_mem_gb": self.gpu_mem_gb,
            "cuda": self.cuda,
        }


def detect_hardware() -> Hardware:
    """Inspect the machine and report what accelerator (if any) is available."""
    try:
        import torch
    except ImportError:
        return Hardware(device="cpu", gpu_name=None, gpu_mem_gb=None, cuda=False)

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            mem = round(props.total_memory / (1024 ** 3), 1)
            return Hardware(device="cuda", gpu_name=props.name, gpu_mem_gb=mem, cuda=True)
        except Exception:
            return Hardware(device="cuda", gpu_name="CUDA GPU", gpu_mem_gb=None, cuda=True)

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return Hardware(device="mps", gpu_name="Apple Silicon (MPS)", gpu_mem_gb=None, cuda=False)

    return Hardware(device="cpu", gpu_name=None, gpu_mem_gb=None, cuda=False)


def recommend_preset(hw: Optional[Hardware] = None) -> str:
    """Pick the best preset name for the detected hardware."""
    hw = hw or detect_hardware()
    if hw.device == "cuda":
        if hw.gpu_mem_gb is not None and hw.gpu_mem_gb >= 16:
            return "gpu-large"
        return "gpu"
    # MPS is real but PyTorch coverage/perf varies; keep newcomers on the safe path.
    return "cpu"


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Choose from: {', '.join(PRESETS)}")
    return PRESETS[name]


def describe() -> str:
    """Human-readable hardware + recommendation summary (used by the CLI 'doctor')."""
    hw = detect_hardware()
    rec = recommend_preset(hw)
    lines = ["LLM Forge :: hardware check", "-" * 32]
    if hw.device == "cuda":
        mem = f"{hw.gpu_mem_gb} GB" if hw.gpu_mem_gb else "unknown mem"
        lines.append(f"Accelerator : CUDA GPU detected -> {hw.gpu_name} ({mem})")
    elif hw.device == "mps":
        lines.append(f"Accelerator : Apple Silicon (MPS) detected -> {hw.gpu_name}")
    else:
        lines.append("Accelerator : none (CPU only)")
    lines.append(f"Recommended : --preset {rec}")
    lines.append("")
    p = get_preset(rec)
    lines.append(f"  {p.name}: {p.blurb}")
    lines.append(f"  ~{p.approx_params_m:.0f}M params, {p.est_time}")
    lines.append("")
    lines.append("All presets:")
    for name, pp in PRESETS.items():
        star = " (recommended)" if name == rec else ""
        lines.append(f"  - {name}{star}: ~{pp.approx_params_m:.0f}M params, {pp.est_time}")
    return "\n".join(lines)
