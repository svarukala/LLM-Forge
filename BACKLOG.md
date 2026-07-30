# LLM Forge — Improvement Backlog

Ideas inspired by a review of [languagemodelbuilder.com](https://languagemodelbuilder.com/),
mapped to our browser-first, non-ML-audience goals. **Implemented so far:** #1–#3
(learning-rate/perplexity readouts, chat sampler controls, chat context-usage meter), plus
**#4** (base-model sampling panel) and **#9** (tokenizer playground). Everything below is
queued for later.

## Medium effort — small backend + UI

### 5. Curated dataset catalog
Replace the bare file-picker with a small gallery of bundled corpora / chat sets, each with a
one-line description and one-click select. Keep "Add your own" as a tile. Friendlier for
non-ML users than an empty file input.
- Bundle 3–4 small corpora + chat sets under `data/sample/` with short descriptions.
- Backend: `/api/datasets` returning the catalog; wire selection into pretrain/finetune `data`.

### 6. Checkpoint browser
**Partially done:** a read-only "Your models" strip now lists `base`/`chat` with metadata
(size, tokenizer kind, steps, final loss, date) via `/api/checkpoints`, using
`checkpoint.py::checkpoint_info()`.
**Remaining:** list *all* checkpoints (not just base/chat), let users pick one to chat with,
and add a delete action.
- Backend: extend `/api/checkpoints`; add a delete endpoint.
- Frontend: per-row actions (select for chat, delete), "compare earlier checkpoints" idea.

### 7. Guided stepper / progressive disclosure
Gray out Step 2 (Fine-tune) and Step 3 (Chat) until prerequisites are met, with helper text
("Finish pre-training to unlock fine-tuning"). Extends the button-locking already in place to
whole-card states plus a visible checklist of progress. (The "Your models" strip from #6 now
provides the underlying state — `has_base` / `has_chat` — to drive this.)

## Higher effort — high wow-factor

### 8. Token X-ray (signature feature)
Color each generated token by its sampled probability and show the top alternative tokens on
hover, in both chat and the base-model sampling panel. Best single feature for *seeing* how a
model decides.
- Backend: extend `GPT.generate()` to optionally return per-token probabilities / top-k
  alternatives; thread through `/api/chat` and `/api/sample`.
- Frontend: render replies as probability-shaded token chips with a hover popover.

### 10. Resume-from-dashboard
Resume is already implemented in the CLI; add a "Resume" button on checkpoint rows that calls a
resume-enabled training start. Requires surfacing training-state files in checkpoint metadata.

## Probably skip (scope)

### DPO preference ballot
Direct Preference Optimization with a side-by-side "this one is better" ballot. A whole new
training mode — large effort and less essential for a first-principles tour. Revisit only if
there's strong demand.
