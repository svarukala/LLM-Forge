"""The training loop, shared by pre-training and fine-tuning.

A single `train()` function drives both:
  * pre-training reads a long text stream (PretrainData)
  * fine-tuning reads chat pairs with a masked prompt (SFTData)

Both are just "predict the next token" — fine-tuning simply starts from pre-trained
weights and only scores the assistant's tokens. That symmetry is the whole point.

The loop is reproducible (all RNGs seeded), reports rich metrics (loss, perplexity,
throughput, elapsed/ETA), writes accurate completed-step metadata, and can be resumed
from a saved optimizer/step state.

Read alongside lessons/04-pretraining.md.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable

import torch

from .checkpoint import load_train_state, save_checkpoint
from .config import TrainConfig, pick_device, set_seed
from .model import GPT


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to 10% of the peak learning rate."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    return cfg.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 20 else float("inf")


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


def _rng_state() -> dict:
    # Keep every value here restorable WITHOUT pickle (see load_train_state's weights_only
    # load): torch RNG state is a ByteTensor, Python's is a tuple of ints, and NumPy's array
    # is converted to a plain list so nothing requires arbitrary class reconstruction.
    state: dict = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        np_state: tuple = tuple(np.random.get_state(legacy=True))
        arr = np.asarray(np_state[1], dtype=np.uint32).tolist()
        state["numpy"] = {
            "keys": arr,
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached": float(np_state[4]),
        }
    except ImportError:
        pass
    return state


def _restore_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        # torch.load may return the inner tuple as a list; random.setstate needs tuples.
        py = state["python"]
        if isinstance(py, list):
            py = (py[0], tuple(py[1]), py[2])
        random.setstate(py)
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        try:
            import numpy as np

            np_state = state["numpy"]
            keys = np.array(np_state["keys"], dtype=np.uint32)
            np.random.set_state(
                ("MT19937", keys, np_state["pos"], np_state["has_gauss"], np_state["cached"])
            )
        except ImportError:
            pass


def _dataset_rng_state(dataset) -> dict | None:
    """Snapshot a dataset's own RNG so a resumed run draws the same future batches."""
    fn = getattr(dataset, "rng_state", None)
    if callable(fn):
        result = fn()
        return result if isinstance(result, dict) else None
    return None


def _restore_dataset_rng(dataset, state) -> None:
    fn = getattr(dataset, "set_rng_state", None)
    if callable(fn) and state:
        fn(state)


def train(
    model: GPT,
    dataset,
    cfg: TrainConfig,
    tokenizer=None,
    tokenizer_path: str = "",
    on_event: Callable[[dict], None] | None = None,
    stop_flag: Callable[[], bool] | None = None,
    resume_dir: str | None = None,
    extra_meta: dict | None = None,
) -> dict:
    """Train `model` on `dataset`. Emits progress dicts via `on_event` (for the dashboard).

    Returns a result dict: {"status", "completed_steps", "final_loss", "out_dir"}.
    `status` is "completed" if all steps ran, or "stopped" if interrupted via `stop_flag`.
    """
    if cfg.steps <= 0:
        raise ValueError(f"steps must be >= 1, got {cfg.steps}")
    if cfg.grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {cfg.grad_accum}")

    device = pick_device(cfg.device)
    set_seed(cfg.seed, cfg.deterministic)
    model.to(device)
    model.train()

    optimizer = model.configure_optimizer(cfg.weight_decay, cfg.learning_rate)

    start_step = 0
    if resume_dir:
        state = load_train_state(resume_dir)
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
            start_step = int(state.get("step", 0))
            _restore_rng_state(state.get("rng", {}))
            _restore_dataset_rng(dataset, state.get("dataset_rng"))

    use_amp = bool(cfg.amp and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    def emit(evt: dict) -> None:
        if on_event:
            on_event(evt)

    emit(
        {
            "type": "start",
            "params": model.num_params(),
            "device": device,
            "steps": cfg.steps,
            "start_step": start_step,
        }
    )

    def _save(status: str, completed_steps: int, final_loss: float) -> None:
        meta = {
            "steps_requested": cfg.steps,
            "completed_steps": completed_steps,
            "final_loss": final_loss,
            "final_perplexity": _perplexity(final_loss),
            "status": status,
            "seed": cfg.seed,
            "train_config": cfg.as_dict(),
        }
        if extra_meta:
            meta.update(extra_meta)
        train_state = {
            "optimizer": optimizer.state_dict(),
            "step": completed_steps,
            "rng": _rng_state(),
            "dataset_rng": _dataset_rng_state(dataset),
        }
        save_checkpoint(cfg.out_dir, model, tokenizer_path, meta=meta, train_state=train_state)

    t0 = time.time()
    run_start = time.time()
    tokens_since = 0
    last_loss = float("nan")
    completed = start_step
    status = "completed"

    for step in range(start_step + 1, cfg.steps + 1):
        if stop_flag and stop_flag():
            status = "stopped"
            emit({"type": "stopped", "step": step - 1})
            break

        lr = _lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_tokens = 0
        for _ in range(cfg.grad_accum):
            x, y = dataset.get_batch("train", cfg.batch_size)
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                scaler.scale(loss / cfg.grad_accum).backward()
            else:
                _, loss = model(x, y)
                (loss / cfg.grad_accum).backward()
            step_tokens += x.numel()
        last_loss = float(loss.item())

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        completed = step
        tokens_since += step_tokens

        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            tps = tokens_since / dt if dt > 0 else 0.0
            elapsed = time.time() - run_start
            done = step - start_step
            remaining = cfg.steps - step
            eta = (elapsed / done) * remaining if done > 0 else 0.0
            emit(
                {
                    "type": "step",
                    "step": step,
                    "loss": last_loss,
                    "perplexity": _perplexity(last_loss),
                    "lr": lr,
                    "tok_per_sec": tps,
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                }
            )
            t0, tokens_since = time.time(), 0

        if (cfg.eval_interval and step % cfg.eval_interval == 0) or step == cfg.steps:
            metrics = _estimate_loss(model, dataset, cfg)
            emit(
                {
                    "type": "eval",
                    "step": step,
                    **metrics,
                    "val_perplexity": _perplexity(metrics["val"]),
                }
            )

        if tokenizer is not None and cfg.sample_every and (step % cfg.sample_every == 0):
            emit({"type": "sample", "step": step, "text": _quick_sample(model, tokenizer, device)})

        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0 and step != cfg.steps:
            _save("in_progress", completed, last_loss)

    _save(status, completed, last_loss)
    emit(
        {
            "type": "done",
            "out_dir": cfg.out_dir,
            "status": status,
            "completed_steps": completed,
            "final_loss": last_loss,
        }
    )
    return {
        "status": status,
        "completed_steps": completed,
        "final_loss": last_loss,
        "out_dir": cfg.out_dir,
    }


@torch.no_grad()
def _quick_sample(model: GPT, tokenizer, device: str, n: int = 80) -> str:
    model.eval()
    start = tokenizer.encode("The") or [0]
    idx = torch.tensor([start], dtype=torch.long, device=device)
    eos = None
    try:
        from .data import EOT

        eos = tokenizer.token_id(EOT)
    except Exception:
        eos = None
    out = model.generate(idx, max_new_tokens=n, temperature=0.8, top_k=40, eos_token_id=eos)
    model.train()
    return tokenizer.decode(out[0].tolist())
