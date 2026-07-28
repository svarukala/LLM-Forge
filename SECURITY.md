# Security

LLM Forge is a **local, educational** project. It is not designed to be a hardened,
internet-facing service. Please keep the following in mind.

## The web dashboard has NO authentication

`python -m llmforge.cli serve` starts a FastAPI app that can launch training jobs, read
datasets from disk, and write checkpoints. It has **no login, no authorization, and no
rate limiting**.

- By default it binds to **localhost only** (`127.0.0.1`), so only your machine can reach it.
- Binding to any other interface is refused unless you pass `--allow-public`, which prints a
  prominent warning. **Only do this on a network you fully trust.** Anyone who can reach the
  port can train models, upload data, and read generated text.

```bash
# Safe (default): only your machine
python -m llmforge.cli serve

# UNSAFE: reachable from the network, no auth
python -m llmforge.cli serve --host 0.0.0.0 --allow-public
```

## Input hardening that *is* in place

Even locally, the dashboard validates and constrains inputs so a mistake can't run away:

- All training/generation parameters have strict min/max bounds (Pydantic request models).
- Uploaded datasets are capped in size and written only under `runs/uploads/`, with filenames
  sanitized to block path traversal (`..`, absolute paths, separators are rejected).
- Chat/finetune checkpoint paths are constrained to live inside the `runs/` directory.
- Only one training job runs at a time; concurrent job requests are rejected.

## Checkpoints are self-contained

Each checkpoint stores its own tokenizer inside the checkpoint directory and records a schema
version, model dimensions, and dataset/tokenizer fingerprints. Loading validates these and
refuses tokenizer paths that escape the checkpoint directory. Still, **only load checkpoints
from sources you trust** — `train_state.pt` is a pickled PyTorch file (like any model
artifact) and should be treated with the same caution as any downloaded code.

## Reporting a vulnerability

This is a teaching project maintained on a best-effort basis. If you find a security issue,
please open an issue describing the problem and reproduction steps. Do not include real
secrets or sensitive data in reports.
