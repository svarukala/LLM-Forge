# LLM Forge 🔨🧠

**Build your own tiny language model — from scratch, on Windows (or Mac/Linux).**

LLM Forge is a hands-on workshop for teams who want a real, intuitive grounding in how
modern language models work. You'll go the whole way: turn text into tokens, pre-train a
GPT from random weights, watch the loss fall, fine-tune it to follow instructions, and
finally *chat with a model you made yourself*.

No cloud. No accounts. No magic. Everything runs locally on your machine.

> Inspired by the Mac-only *Language Model Builder*, but rebuilt from scratch as
> **original, open, cross-platform code** using **PyTorch** (so it runs great on Windows
> with an NVIDIA GPU, and fine on CPU for the small stuff).

---

## Why this exists

The fastest way to demystify "AI magic" is to build a small version yourself. This repo
is meant to be **run by everyone on the team**: clone it, run it, and come away with the
same mental model of what pre-training and fine-tuning actually *do*.

There are three layers, so people can go as deep as they want:

| Layer | Who it's for | Needs |
|-------|-------------|-------|
| 🟢 **Playground** | Total beginners | Nothing but Python — runs in seconds |
| 🟡 **Pipeline** | Builders | `pip install -r requirements.txt` (PyTorch) |
| 🔵 **Dashboard** | Visual learners / demos | Same, plus a browser |

---

## 🟢 Layer 1: The 60-second playground (zero dependencies)

See a model *learn* right now, with nothing installed but Python:

```powershell
python -m llmforge.playground
```

This trains a tiny character-level model in pure Python on a sample corpus and prints
text getting less random every few steps. It's the "aha" moment — no GPU, no libraries.

---

## 🟡 Layer 2: The real pipeline (PyTorch)

Install once:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then walk the full pipeline with the CLI:

```powershell
# 1. Train a tokenizer on your data (BPE)
python -m llmforge.cli tokenizer --input data/sample/corpus.txt --vocab-size 2048

# 2. Pre-train a GPT from scratch
python -m llmforge.cli pretrain --data data/sample/corpus.txt --steps 500

# 3. Sample from the base model
python -m llmforge.cli sample --prompt "Once upon a time"

# 4. Fine-tune it to follow instructions (SFT)
python -m llmforge.cli finetune --data data/sample/chat.jsonl --steps 300

# 5. Chat with the model you made
python -m llmforge.cli chat
```

Checkpoints are saved in the portable **safetensors** format under `runs/`. Because
optimizer and step state are saved alongside them, you can **stop a pre-training run and
resume it later** with `--resume` (see "Advanced training" below), and share checkpoints
with others.

> ### 🎓 This is an *educational* tiny model, not a general-purpose assistant
> Everything here is sized to run on a laptop and to make the *mechanics* visible. Even the
> largest preset is orders of magnitude smaller than ChatGPT-class models and is trained on a
> tiny slice of data. Expect it to learn grammar, names, and story shape — **not** facts,
> reasoning, or reliable instruction-following. That gap is the whole point: you get to feel
> exactly what scale buys. See the "why scale matters" lesson below.

---

## 📈 Recommended datasets & sized runs (real coherent output)

The bundled `data/sample/` files are tiny — enough to see the *plumbing* work, but too
small to produce real sentences. For output that actually reads like English on a **CPU
devbox**, use **TinyStories** (from [LLMDataHub](https://github.com/Zjh-819/LLMDataHub)) —
a corpus of simple children's stories with a small vocabulary, purpose-built so *tiny*
models become coherent.

### 1. Download a right-sized slice (one command)
```powershell
python scripts/prepare_data.py --pretrain-mb 6 --sft-examples 1200
```
This creates:
- `data/pretrain/tinystories.txt` — ~6 MB pre-training corpus (~7,000 stories)
- `data/finetune/stories_sft.jsonl` — 1,200 in-domain "write a short story" instruction pairs
  (in-domain on purpose, so fine-tuning stays coherent instead of drifting to gibberish)

### 2. Pre-train (~15–20 min on CPU)
```powershell
python -m llmforge.cli pretrain --preset cpu --data data/pretrain/tinystories.txt --out runs/base
```
`--preset cpu` expands to the verified "coherent" knobs below (override any of them
explicitly if you like):
```powershell
python -m llmforge.cli pretrain --data data/pretrain/tinystories.txt `
  --tok-kind bpe --vocab-size 4096 --block-size 128 `
  --n-layer 4 --n-head 4 --n-embd 256 --batch-size 32 --steps 1200 --out runs/base
```

### 3. Fine-tune (~5 min) then chat
```powershell
python -m llmforge.cli finetune --preset cpu --data data/finetune/stories_sft.jsonl --base runs/base --out runs/chat
python -m llmforge.cli chat
```

### What you should see (verified run on this repo)
- **Pre-train loss:** `8.35 → 3.40`; samples go from word-salad to real stories.
- **~1200-step sample:** *"Once upon a time, there was a little boy named Timmy. Timmy loved
  to play with his friends. One day, Timmy went to the store..."*
- **Fine-tuned chat** ("Write a short story for a young child."): *"Once upon a time, there
  was a little girl named Lily. She loved to play in the park with her toys. One day, she
  saw a big box..."*

It's a 4.2M-param model — grammatical and on-topic, though it rambles. Want it sharper?
Train longer (`--steps 4000+`), use a bigger slice (`--pretrain-mb 30`), or a wider model
(`--n-embd 384 --n-layer 6`) — ideally on an NVIDIA GPU.

---

## 🎚️ Presets & the "why scale matters" lesson

Not sure what size model your machine can handle? Ask:

```powershell
python -m llmforge.cli doctor
```

It detects whether you have a CUDA GPU and recommends a **preset** you can pass to both
`pretrain` and `finetune` with a single flag:

```powershell
python -m llmforge.cli pretrain --preset cpu --data data/pretrain/tinystories.txt --out runs/base
python -m llmforge.cli finetune --preset cpu --data data/finetune/stories_sft.jsonl --base runs/base --out runs/chat
```

Explicit flags always override the preset, so `--preset gpu --steps 2000` uses the GPU
architecture but your step count. The **web dashboard does the same detection** — it shows
your hardware in a banner up top and pre-selects the recommended preset in a dropdown.

| Preset | Params | Needs | ~Time | What you get |
|--------|--------|-------|-------|--------------|
| `cpu` | ~4M | any CPU | ~15–20 min | **coherent English**, but *loose topic-following* |
| `gpu` | ~25M | CUDA GPU | ~1–2 hr | follows the prompt topic, **stops cleanly** |
| `gpu-large` | ~85M (GPT-2-small class) | strong CUDA GPU | several hr | best prose |

### The honest scale lesson (the key takeaway)

Ask the `cpu` model to *"write a story about a dog"* and you'll often get a coherent story
about a **bird** or a **girl named Lily** — not a dog. **This is expected, and it's the most
important thing the workshop teaches.** At ~4M parameters the model has learned the *shape*
of a story (grammar, names, dialogue, a little moral) but not the *binding* between your
prompt word and the content. Its validation loss plateaus around ~3.0 — that flat line **is**
the ceiling of this size.

Prompt-following and clean stopping are *emergent* behaviours that show up as you scale
parameters + data + training. Run the same TinyStories pipeline with `--preset gpu` on a
CUDA box and the model starts writing about the noun you actually asked for. You
get to **feel** why the industry spends millions on compute, instead of just reading about it.

### Sizing cheat-sheet
| Preset | Tokenizer | Model | Steps | ~Time | Result |
|--------|-----------|-------|-------|-------|--------|
| `cpu` (Quick taste, `--steps 500`) | bpe 4096 | 4L / 256 | 500 | ~7 min | story-ish text, loss falls |
| **`cpu` (recommended)** | bpe 4096 | 4L / 256 | 1200 | ~15–20 min | real TinyStories sentences, loose topics |
| `gpu` | bpe 8192 | 8L / 512 | 5000 | ~1–2 hr (GPU) | follows topics, stops cleanly |
| `gpu-large` | bpe 8192 | 12L / 768 | 12000 | several hr (GPU) | noticeably more fluent |

---

## ⚙️ Advanced training, resuming & inspecting checkpoints

The pipeline exposes the knobs you need for reproducible, resumable runs:

```powershell
# Resume an interrupted pre-training run (restores optimizer + step + RNG state)
python -m llmforge.cli pretrain --data data/pretrain/tinystories.txt --out runs/base --resume

# Reproducibility: fix the seed, or go fully deterministic (slower)
python -m llmforge.cli pretrain --data data/pretrain/tinystories.txt --seed 1337 --deterministic

# Save a periodic checkpoint every N steps; evaluate every N steps
python -m llmforge.cli pretrain --data data/pretrain/tinystories.txt --checkpoint-every 200 --eval-interval 50

# Gradient accumulation (bigger effective batch) and CUDA mixed precision
python -m llmforge.cli pretrain --data data/pretrain/tinystories.txt --grad-accum 4 --amp

# Reuse an existing tokenizer only when it's compatible; otherwise it's retrained safely
python -m llmforge.cli pretrain --data data/pretrain/tinystories.txt --tok-kind bpe --reuse-tokenizer
```

By default a tokenizer is **retrained** whenever the requested kind/vocab-size or the corpus
fingerprint differs from the cached one — so `--preset cpu` (BPE) can never silently reuse an
incompatible char tokenizer. Pass `--reuse-tokenizer` only when you know it's safe.

### Sampling controls

```powershell
python -m llmforge.cli sample --checkpoint runs/base --prompt "Once upon a time" `
  --temperature 0.8 --top-k 40 --top-p 0.9        # nucleus sampling
python -m llmforge.cli sample --checkpoint runs/base --no-stop   # don't stop at <|endoftext|>
```

### Inspect a checkpoint

```powershell
python -m llmforge.cli checkpoint-info --checkpoint runs/base
```

Prints the schema version, model dimensions, tokenizer kind/vocab, dataset & tokenizer
fingerprints, completed steps, final loss, whether the run completed or was stopped, and
whether it can be resumed.

### One-command smoke test

Verify the whole pipeline end-to-end (tokenizer → pre-train → sample → fine-tune → chat) on a
micro model in a few seconds:

```powershell
python -m llmforge.cli smoke
```

---



```powershell
python -m llmforge.cli serve
# open http://localhost:8000
```

Launch training runs, watch the loss curve live, and chat — all from the browser. It's
fully self-serve: a **hardware banner** up top detects your CPU/GPU and pre-selects the
right **preset**, you can **upload your own corpus** (`.txt`) for pre-training and your own
**chat pairs** (`.jsonl`) for fine-tuning, pick the tokenizer (char/BPE) from a dropdown,
and click **Pre-train → Fine-tune → Chat** in that order. No CLI steps required. Great
for a lunch-and-learn or a team demo.

> ⚠️ **Security:** the dashboard has **no authentication** and binds to `localhost` only by
> default. It will refuse to bind to a public interface unless you pass `--allow-public`
> (which prints a warning). Only expose it on a network you trust. See [`SECURITY.md`](SECURITY.md).

---

## 📚 Guided lessons

Short, friendly explainers in [`lessons/`](lessons/), each linked to the exact code that
implements the idea:

1. [What is a language model?](lessons/01-what-is-a-language-model.md)
2. [Tokenization: turning text into numbers](lessons/02-tokenization.md)
3. [Embeddings & attention](lessons/03-embeddings-and-attention.md)
4. [Pre-training: learning from raw text](lessons/04-pretraining.md)
5. [Fine-tuning: teaching it to be helpful](lessons/05-finetuning.md)
6. [Dashboard field guide: every knob & number explained](lessons/06-dashboard-glossary.md)
7. [Under the hood: what happens when you click Pre-train / Fine-tune](lessons/07-under-the-hood.md)

---

## Requirements & hardware

- **Python 3.10+**
- **Layer 1 (playground):** nothing else — pure standard library.
- **Layer 2/3:** PyTorch. CPU works for the `cpu` preset; an NVIDIA **CUDA** GPU is strongly
  recommended for the `gpu`/`gpu-large` presets. Apple Silicon (**MPS**) works for CPU-class
  sizes.

Install PyTorch for your platform, then the package:

```bash
# CPU-only (Windows/Linux/macOS)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA CUDA (example: CUDA 12.1) — pick the right index for your driver
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon: the default wheel already includes the MPS backend
pip install torch

# then, from the repo root:
pip install -r requirements.txt   # or:  pip install -e ".[dev]"  for the dev tools
```

### Rough memory & time estimates

| Preset | Params | Peak RAM/VRAM* | Device | ~Time |
|--------|--------|----------------|--------|-------|
| `cpu` | ~4M | < 1 GB | any CPU | ~15–20 min (1200 steps) |
| `gpu` | ~25M | ~2–3 GB VRAM | CUDA GPU | ~1–2 hr (5000 steps) |
| `gpu-large` | ~85M | ~6–8 GB VRAM | strong CUDA GPU | several hr (12000 steps) |

\* Ballpark for these settings; actual usage depends on batch size, block size, and dtype.
If you hit an out-of-memory error, lower `--batch-size`, then `--block-size`, then model
width (`--n-embd`, `--n-layer`).

### Troubleshooting

- **`Corpus too small: N tokens ...`** — your dataset has fewer tokens than the context
  window needs for both a train and a validation window. Use a bigger corpus, or lower
  `--block-size`. See `scripts/prepare_data.py` to fetch a right-sized TinyStories slice.
- **`No usable examples found ...` (fine-tuning)** — every chat example was dropped because
  the prompt filled the entire context with no room for a response. Shorten prompts or raise
  `--block-size` (must be ≤ the base model's block size).
- **CUDA out of memory** — reduce `--batch-size` (and use `--grad-accum` to keep the effective
  batch), then `--block-size`, then model size. `--amp` also lowers VRAM on CUDA.
- **Apple Silicon (MPS)** — the `doctor` command detects MPS. Use CPU-class sizes; some ops
  may fall back to CPU. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if you hit an unsupported op.
- **Tokenizer was retrained when I didn't expect it** — that's intentional when the kind,
  vocab size, or corpus fingerprint changed. Pass `--reuse-tokenizer` only when compatible.
- **Windows: `pytest` `PermissionError` scanning temp** — see the note in
  [`CONTRIBUTING.md`](CONTRIBUTING.md) about pointing `--basetemp` at a fresh directory.

- Built and tested on Windows; works on macOS/Linux too.

## Development

Contributions welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). Quality gates (all run in
CI on Python 3.10–3.12):

```bash
pip install -e ".[dev]"
ruff check .            # lint
ruff format --check .   # formatting
pyright                 # static types
pytest -q               # tests
python -m llmforge.cli smoke   # end-to-end pipeline
```

See also [`SECURITY.md`](SECURITY.md) (dashboard has no auth) and
[`CHANGELOG.md`](CHANGELOG.md).

## License

MIT — use it, fork it, teach with it.
