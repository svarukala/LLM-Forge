# Contributing to LLM Forge

Thanks for helping make LLM Forge a better learning tool! This project values clear,
beginner-friendly code and honest explanations over cleverness.

## Development setup

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate

# Install PyTorch for your platform first (see README "Requirements"), then:
pip install -e ".[dev]"
```

## Quality gates

All of these run in CI (`.github/workflows/ci.yml`) on Python 3.10–3.12 and must pass:

```bash
ruff check .            # lint
ruff format --check .   # formatting
pyright                 # static type check
pytest -q               # test suite
python -m llmforge.cli smoke   # end-to-end tiny pipeline
```

Run `ruff format .` to auto-format before committing.

### Windows note on the test temp directory

If `pytest` fails with `PermissionError` while scanning the system temp directory (a stale,
locked temp folder from a previous run), point pytest at a fresh base temp dir:

```powershell
$env:PYTEST_ADDOPTS = "--basetemp=$env:USERPROFILE\llmforge_pytmp"
Remove-Item -Recurse -Force $env:USERPROFILE\llmforge_pytmp -ErrorAction SilentlyContinue
pytest -q
```

## Testing conventions

- Keep tests tiny and CPU-only: micro models (2 layers, 32-dim), a handful of steps, so the
  whole suite runs in seconds. See `tests/conftest.py` for shared fixtures.
- Every bug fix should come with a regression test.
- Char tokenizers can only encode characters they were trained on — make sure any test string
  is covered by the fixture corpus.

## Coding style

- Prefer clarity over brevity; comment the *why*, not the *what*.
- Public functions get docstrings that a learner can follow.
- Keep the three layers (playground / pipeline / dashboard) working independently.
- Don't break the "no-dependency playground" — it must run on a bare Python install.

## Pull requests

1. Fork and branch from `main`.
2. Make your change with tests and docs updates.
3. Ensure all quality gates pass locally.
4. Open a PR describing what changed and why, and how you verified it.

## Reporting issues

Please include your OS, Python version, whether you're on CPU/CUDA/Apple Silicon, and the
exact command plus error output.
