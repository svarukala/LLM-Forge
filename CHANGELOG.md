# Changelog

All notable changes to LLM Forge are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- **Dashboard dataset paths are sandboxed**: `/api/pretrain` and `/api/finetune` now reject
  any `data` path outside the bundled `data/sample/` directory or `runs/uploads/` (HTTP 422),
  so a browser client can no longer make the server train on arbitrary files.
- **Safe resume loading**: `train_state.pt` is loaded with `torch.load(weights_only=True)`,
  so resuming a checkpoint can never execute arbitrary code. RNG state is serialized as plain
  tensors/lists to keep the file pickle-free.
- **Guarded legacy migration**: migrating a pre-v2 checkpoint refuses to copy a tokenizer
  from a path that escapes the checkpoint directory unless migration is explicitly trusted.
- **Aligned sampling validation**: the chat API rejects `temperature <= 0`, `top_k < 1`, and
  `top_p` outside `(0, 1]` with a 422 instead of failing later inside the model.

### Added
- **"Your models" status strip**: the dashboard now shows which stages already exist on disk
  (pre-trained `base`, fine-tuned `chat`) with each checkpoint's size, tokenizer, steps, and
  final loss, via a new `/api/checkpoints` endpoint. Work from a previous session persists and
  is visible after a restart. Non-finite (NaN) losses from stopped runs are reported as null.
- **Tokenizer playground**: a dashboard panel (and `/api/tokenize`) that splits text into
  tokens/IDs live, for both `char` and a trained `bpe` tokenizer, so users can *see* why
  sub-word tokenization is more efficient.
- **Base-model sampling panel**: a dashboard panel (and `/api/sample`, wrapping
  `sample.generate_text`) that generates raw autocomplete from the pre-training-only
  checkpoint, making the difference between pre-training and fine-tuning tangible.
- **Dashboard learning aids**: the Training monitor now shows live **perplexity** and
  **learning-rate** tiles, and **every monitor label has a plain-English hover tooltip**
  (e.g. what `train` loss vs `val` loss mean). The Chat panel gained **temperature / top-k /
  top-p** sampling controls and a **context-usage meter** (`tokens used / block_size`).
  [Lesson 6 · Dashboard field guide](lessons/06-dashboard-glossary.md) documents all of them.
- **Hardened `--resume`**: verifies model block size, vocabulary size, tokenizer kind, and
  corpus fingerprint before resuming, and persists/restores the dataset-local RNG so resumed
  training is reproducible (a 5+5 resumed run matches a straight 10-step run).
- **Resume support**: `pretrain --resume` restores optimizer state, step position, and RNG
  state from a separate `train_state.pt` file (model weights stay in safetensors).
- **`checkpoint-info` command**: prints schema version, model dimensions, tokenizer info,
  training metadata, and whether a checkpoint is resumable.
- **`smoke` command**: a one-command end-to-end self-test (tokenizer → pre-train → sample →
  fine-tune → chat) on a micro model in seconds.
- **EOS-aware generation**: `GPT.generate()` accepts `eos_token_id` and stops cleanly; new
  `top_p` (nucleus) sampling and `sample --no-stop` control.
- **Reproducibility**: consistent seeding of Python/NumPy/PyTorch, opt-in `--deterministic`
  mode, local RNG instances in datasets, and `--seed` flags.
- **Richer training metrics**: perplexity, token throughput, elapsed time, and ETA.
- **SFT train/validation split** so evaluation isn't measured on the exact training examples.
- **Multi-turn chat**: `ChatSession` is now stateful with history and context trimming.
- **Checkpoint schema v2**: self-contained tokenizer copied into each checkpoint, plus
  dataset/tokenizer fingerprints, seed, config, and package version in metadata. Legacy
  (v1) checkpoints auto-migrate on load.
- **Dashboard hardening**: Pydantic request models with strict bounds, thread-safe training
  manager, true per-subscriber SSE fan-out, bounded event history, chat-cache invalidation
  on checkpoint overwrite, a `/api/status` job/elapsed/step/error snapshot, upload size and
  path-traversal guards, and a localhost-only default with an explicit `--allow-public` flag.
- **Engineering quality gates**: `pytest` suite, GitHub Actions CI (Python 3.10–3.12) with
  Ruff lint/format, Pyright type checking, tests, and smoke test. `CONTRIBUTING.md`,
  `SECURITY.md`, and this changelog.

### Fixed
- **SFT context overflow**: long prompts could exceed `block_size` and trigger assertion
  failures. A deterministic truncation policy now always preserves the assistant marker and
  terminal EOT token; unusable examples are dropped and counted.
- **Pretrain corpus validation**: too-small corpora now fail before training with a clear
  message; the random-window bound is corrected so every valid window is sampleable.
- **Tokenizer special tokens**: `<|user|>`, `<|assistant|>`, etc. are now encoded as single
  reserved tokens via longest-match, for both char and BPE tokenizers.
- **Incompatible tokenizer reuse**: `pretrain --preset cpu` can no longer silently reuse an
  incompatible cached tokenizer; reuse is explicit via `--reuse-tokenizer` and guarded by
  kind/vocab/corpus-fingerprint checks.
- **Checkpoint metadata accuracy**: records actual completed steps and final loss, validates
  `steps >= 1`, and distinguishes completed vs user-stopped runs.

## [0.1.0]

### Added
- Initial release: zero-dependency playground, PyTorch GPT pipeline (tokenizer, pre-train,
  sample, fine-tune, chat), FastAPI dashboard, guided lessons, hardware presets and
  `doctor` command, and a TinyStories data-prep script.
