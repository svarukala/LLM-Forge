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
- Training dataset paths supplied to the dashboard are restricted to bundled samples
  (`data/sample/`) or uploaded files (`runs/uploads/`); any other path is rejected (HTTP 422),
  so a browser client cannot make the server train on arbitrary files.
- Chat/finetune checkpoint paths are constrained to live inside the `runs/` directory.
- Only one training job runs at a time; concurrent job requests are rejected.

## Checkpoints are self-contained and load safely

Each checkpoint stores its own tokenizer inside the checkpoint directory and records a schema
version, model dimensions, and dataset/tokenizer fingerprints. Loading validates these and
refuses tokenizer paths that escape the checkpoint directory. Migrating an old (pre-v2)
checkpoint refuses to copy a tokenizer from outside the checkpoint directory unless migration
is explicitly trusted.

Model weights are stored as **safetensors** (a non-executable tensor format). The optional
resume file (`train_state.pt`) is loaded with `torch.load(..., weights_only=True)`, so
resuming **cannot execute arbitrary code**, even from an untrusted checkpoint — everything it
contains is a plain tensor or primitive container (optimizer tensors, step counts, RNG
state). As always, prefer checkpoints from sources you trust.

## Reporting a vulnerability

This is a teaching project maintained on a best-effort basis. If you find a security issue,
please open an issue describing the problem and reproduction steps. Do not include real
secrets or sensitive data in reports.
