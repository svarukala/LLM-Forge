# Lesson 4 · Pre-training: learning from raw text

**Pre-training** is where the model learns language from scratch by reading a big pile of
text and repeatedly predicting the next token. Code:
[`llmforge/train.py`](../llmforge/train.py) and [`llmforge/data.py`](../llmforge/data.py).

## The loop (this is the entire algorithm)

1. Grab a random window of tokens from the corpus → inputs `x` and the same window shifted
   by one → targets `y`. (`PretrainData.get_batch`)
2. Run `x` through the model to get predicted next-token probabilities.
3. Compare predictions to `y` with **cross-entropy loss** — one number saying how wrong we
   were.
4. **Backpropagate**: compute how each weight contributed to the error.
5. Nudge every weight a little in the direction that reduces the loss (AdamW optimizer).
6. Repeat thousands of times.

That's it. No labels, no humans — the text *is* its own supervision. This is why it's
called *self-supervised* learning.

## Run it

```powershell
python -m llmforge.cli pretrain --data data/sample/corpus.txt --steps 500
```

You'll see the loss fall and periodic **samples** — text the model generates mid-training.
Early samples are gibberish; later ones start to resemble the corpus.

## Things to watch

- **train vs. val loss** — printed at each eval. If train keeps falling but val stops, the
  model is memorizing (overfitting) rather than generalizing.
- **learning-rate schedule** — we warm up then cosine-decay (`_lr_at`). Too high and
  training diverges; too low and it crawls.
- **checkpoints** — saved to `runs/base/` in safetensors. Stop anytime and resume/share.

## Try this
Point `--data` at your own `.txt` file (team docs, a book you like) and pre-train. The
model's "voice" will shift toward your data. Then generate:

```powershell
python -m llmforge.cli sample --prompt "In the beginning"
```

**Next:** [Lesson 5 · Fine-tuning »](05-finetuning.md)
