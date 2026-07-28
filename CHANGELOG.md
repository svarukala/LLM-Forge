# Changelog

All notable changes to LLM Forge are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
