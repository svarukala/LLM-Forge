# Lesson 7 · Under the hood: what happens when you click Pre-train / Fine-tune

Lessons 1–6 explained the *ideas*. This one is the **step-by-step mechanics** of what your
machine actually does between clicking a button and getting a smarter model — including
what the "random weights" are, and precisely how they improve. Everything here maps to real
lines in [`llmforge/model.py`](../llmforge/model.py) and
[`llmforge/train.py`](../llmforge/train.py).

---

## 1. From button-click to training loop (the plumbing)

When you press **Start pre-training** in the dashboard:

1. `app.js` sends `POST /api/pretrain` with your knobs (steps, layers, n_embd, preset…).
2. The FastAPI server ([`server/app.py`](../llmforge/server/app.py)) hands them to
   `TrainingManager.start_pretrain`, which spawns a **background thread** so the browser
   stays responsive.
3. That thread builds the model + data and calls the one shared `train()` function.
4. As training runs, `train()` **emits events** (`step`, `eval`, `sample`) that stream back
   to your browser over Server-Sent Events and become the live loss curve and samples.

**Fine-tune** is the same path (`/api/finetune` → `_run_finetune`), with one difference:
instead of creating a fresh random model, it **loads your `runs/base` checkpoint** and
continues training on chat pairs. Same loop, warmer starting point.

---

## 2. What "the weights" actually are

A neural network is just a big pile of numbers (**parameters**, a.k.a. weights) arranged
into matrices. In our GPT, `model.num_params()` counts them — ~4.2M for the `cpu` preset.
They live in these components (all in `model.py`):

| Component | Code | What it holds | Role |
|-----------|------|---------------|------|
| **Token embeddings** | `transformer.wte` | `vocab_size × n_embd` table | one learned vector per token — the model's "meaning" of each word-piece |
| **Position embeddings** | `transformer.wpe` | `block_size × n_embd` table | a learned vector per slot, so the model knows word *order* |
| **Attention weights** | `c_attn`, `c_proj` in each `Block` | `n_embd × 3·n_embd` etc. | how tokens decide which earlier tokens to look at and mix |
| **MLP weights** | `c_fc`, `c_proj` in each `Block` | `n_embd × 4·n_embd` | per-token "thinking" after attention |
| **LayerNorm** | `ln_1`, `ln_2`, `ln_f` | small scale vectors | keep activations numerically stable |
| **Output head** | `lm_head` | `n_embd × vocab_size` | turns the final vector into a score for every possible next token |

> **Weight tying** (`model.py`: `self.transformer.wte.weight = self.lm_head.weight`): the
> input embedding table and the output head are literally the *same* matrix, reused. Fewer
> parameters, and it makes intuitive sense — the vector that *represents* "cat" should also
> be what you compare against when *predicting* "cat".

### Where the "random" comes from
When a fresh model is created, every one of those numbers is initialized by `_init_weights`:

```python
nn.init.normal_(module.weight, mean=0.0, std=0.02)   # tiny random Gaussian values
if module.bias is not None:
    nn.init.zeros_(module.bias)                       # biases start at 0
```

So at step 0 the model is ~4.2M small random numbers centered on zero. It knows **nothing**
— ask it to continue "The cat" and every token in the vocabulary is roughly equally likely.
That's why the very first samples are word-salad and the loss starts near `ln(vocab_size)`
(≈ 8.3 for a 4096-token vocab — the entropy of a uniform guess).

---

## 3. One training step, in slow motion

This is the heart of `train()`. It runs once per **step**, thousands of times. Every step
does exactly six things:

```python
lr = _lr_at(step, cfg)                       # (0) pick this step's learning rate
x, y = dataset.get_batch("train", batch)     # (1) sample a batch of text windows
_, loss = model(x, y)                        # (2) FORWARD: predict + measure error
optimizer.zero_grad(set_to_none=True)        # (3) clear last step's gradients
loss.backward()                              # (4) BACKWARD: who caused the error?
torch.nn.utils.clip_grad_norm_(...)          # (5) clip to avoid huge unstable updates
optimizer.step()                             # (6) UPDATE every weight a little
```

Let's unpack the three that matter most.

### (2) Forward pass — make a prediction, measure how wrong
`x` is a batch of token windows; `y` is the same windows shifted by one (the correct
"next token" at every position). The model:
- looks up each token's vector (`wte`) + its position vector (`wpe`),
- passes them through the attention + MLP blocks so each position gathers context,
- produces **logits** — a raw score for every vocabulary token at every position,
- compares those to `y` with **cross-entropy loss** (`F.cross_entropy` in `model.forward`).

**Loss is a single number: the model's average "surprise" at the correct answer.** High loss
= it put low probability on the right token. This is the number you watch fall.

### (4) Backward pass — assign blame (backpropagation)
`loss.backward()` uses calculus (the chain rule) to compute, for **every single weight**, a
**gradient**: "if I nudge this weight up a hair, does the loss go up or down, and how much?"
PyTorch does this automatically (autograd). After this line, every parameter carries a
`.grad` telling it which direction reduces error.

### (6) Optimizer step — nudge the weights
`optimizer.step()` (AdamW, from `configure_optimizer`) moves each weight a **tiny amount** in
the direction its gradient says reduces loss:

```
new_weight ≈ old_weight − learning_rate × (gradient, smoothed by Adam)
```

- **learning_rate** = how big that nudge is. Our `_lr_at` **warms it up** for the first 50
  steps (so early chaos doesn't blow things up), then **cosine-decays** it (so late training
  fine-tunes gently). This is the `lr` you see in the monitor.
- **AdamW** keeps a running average of each weight's gradients (momentum) and scales the
  step per-weight, which makes learning far faster and more stable than plain gradient
  descent.
- **grad clipping** caps the total update size so one freak batch can't destroy progress.

That's the whole magic: **predict → measure surprise → compute blame → nudge.** Repeat a few
thousand times and those 4.2M random numbers slowly organize themselves into something that
encodes grammar, names, and story structure. No rules are ever hand-written — the weights
*discover* them because organized weights predict the next token better than random ones.

---

## 4. Watching it improve (what the numbers mean per step)

- Every 10 steps `train()` emits a **`step`** event → updates **loss** and **tok/s**.
- Every 50 steps it runs `_estimate_loss` on held-out data → emits **`eval`** with
  **train** and **val** loss.
- Every 100 steps `_quick_sample` generates text → the **Samples** panel.

A healthy pre-train: loss `8.3 → ~3.4`, samples morph from `qx the the` into real
TinyStories sentences. When loss flattens, the weights have extracted about all this model
*size* can (that plateau is the "scale wall" from the README).

---

## 5. How fine-tuning differs (same loop, three tweaks)

Fine-tuning reuses `train()` verbatim. The differences:

1. **Warm start, not random.** `_run_finetune` calls `load_checkpoint(runs/base)` — the
   weights are already good at language. We're *adapting* them, not building from scratch,
   which is why it needs far fewer steps.
2. **Different data shape.** `SFTData` formats each row with the chat template
   (`<|user|> … <|assistant|> … <|endoftext|>`) — see [Lesson 5](05-finetuning.md).
3. **Masked loss.** In `_format_example` the prompt tokens are set to `-1`, and
   `F.cross_entropy(..., ignore_index=-1)` **skips them**. So gradients only flow from the
   *assistant's* tokens — the model is rewarded for producing good answers, not for
   re-predicting the user's question. It also learns to emit `<|endoftext|>` to stop.

Because it's the same predict→blame→nudge loop, the weights shift *just enough* to pick up
the instruction-following behaviour while keeping the language skill they already have.

---

## 6. The one-paragraph summary (for sharing)

> Clicking **Pre-train** creates ~4 million random numbers and then, thousands of times,
> shows the model a snippet of text, asks it to predict the next token, measures how wrong
> it was (**loss**), computes how each number contributed to that error (**backpropagation**),
> and nudges every number a hair in the direction that reduces the error (**gradient descent
> via AdamW**). Slowly the random numbers organize into a model of language. **Fine-tune**
> does the exact same thing, but starts from those trained numbers and only grades the
> model's *answers*, teaching it to be a helpful assistant instead of a raw autocomplete.

**Back to:** [README](../README.md) · [Lesson 6 · Dashboard field guide](06-dashboard-glossary.md)
