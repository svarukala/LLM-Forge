# Lesson 6 · Dashboard field guide (every knob & number explained)

The web dashboard (`python -m llmforge.cli serve`) exposes the same controls as the CLI,
just with buttons. This lesson is a **plain-English glossary** of every input you can set
and every number you see stream by, so nothing on the screen is a mystery. Keep it open in
a second tab during a demo.

---

## Part A · The knobs you set (before training)

### 0 · Your machine — **Preset**
A one-click bundle of all the settings below, tuned for your hardware. The banner runs
`doctor` for you (see [`llmforge/presets.py`](../llmforge/presets.py)) and pre-selects:

- **`cpu`** (~4M params) — coherent English, loose topic-following. Runs anywhere.
- **`gpu`** (~25M params) — follows the prompt topic, stops cleanly. Needs a CUDA GPU.
- **`gpu-large`** (~85M params) — best prose. Needs a strong GPU.

Pick a preset and the fields below auto-fill. Change any field afterward to override just
that one — your manual value always wins.

### 1 · Pre-train knobs

#### **Steps**
How many times we run the training loop (grab a batch → predict → measure error → nudge
weights). **More steps = more learning**, up to a point, and more time. 1 step ≈ 0.8 s on
this CPU. Think of it as "how many flashcards the model studies." Too few → still gibberish;
too many → diminishing returns / overfitting (see **val** below).

#### **Layers** (a.k.a. `n_layer`)
How many stacked **transformer blocks** the model has. Each layer lets the model do another
round of "look at all previous words and reconsider." **More layers = deeper reasoning** and
more parameters, but slower. Playground = 0 real layers; `cpu` preset = 4; GPT-2-small = 12.

#### **Embd width** (a.k.a. `n_embd`, embedding/hidden size)
How many numbers represent each token internally — the "width" of the model's thinking.
Every token becomes a vector of this length (e.g. 256). **Wider = more nuance per token**
(and many more parameters). Layers × width together set the model's capacity: `cpu` = 4×256,
`gpu` = 8×512, `gpu-large` = 12×768.

#### **Tokenizer**
How raw text is chopped into the tokens the model predicts (see
[Lesson 2](02-tokenization.md)):
- **`char`** — one token per character. Tiny vocabulary, zero dependencies, but the model
  must learn spelling from scratch. Good for the 60-second taste.
- **`bpe`** — Byte-Pair Encoding merges common character pairs into sub-word tokens
  ("ing", "the", "story"). Fewer tokens per sentence, more realistic — what real LLMs use.

#### **Corpus (.txt)**
The raw text the model pre-trains on. Upload your own, or leave blank to use the bundled
sample. This is the *only* thing the model knows about the world — its "voice" comes
entirely from here. (TinyStories → it writes children's stories.)

### 2 · Fine-tune knobs

#### **Steps** (fine-tune)
Same idea, but usually far fewer — you're gently *adapting* an already-trained model, not
teaching it language from scratch. Too many and it "forgets" its general fluency.

#### **Chat data (.jsonl)**
Instruction/response pairs that teach the model to *answer* instead of just autocomplete
(see [Lesson 5](05-finetuning.md)). Each line is `{"prompt": "...", "response": "..."}`.

---

## Part B · Hidden knobs the presets set for you

You won't see these in the basic UI, but the presets choose them and they matter:

| Term | Meaning | Why it matters |
|------|---------|----------------|
| **Block size** (`block_size`) | Context length — how many tokens the model sees at once (e.g. 128). | The model's "short-term memory." Longer = it can track longer stories, but costs more compute. |
| **Heads** (`n_head`) | Attention heads per layer. | Each head can focus on a different relationship (subject↔verb, etc.). See [Lesson 3](03-embeddings-and-attention.md). |
| **Vocab size** | Number of distinct tokens the tokenizer knows (e.g. 4096 for BPE). | Bigger vocab = shorter sequences but a larger output layer. |
| **Batch size** | How many text windows we process per step (e.g. 32). | Bigger = smoother, faster learning per step, but more memory. |
| **Learning rate** (`lr`) | How big a step we take when nudging weights. | Too high → training diverges (loss explodes); too low → it crawls. We warm up then decay it. |

---

## Part C · The numbers you watch (during training)

These stream into the **Training monitor** live (code emits them in
[`llmforge/train.py`](../llmforge/train.py); the UI renders them in
[`app.js`](../llmforge/server/static/app.js)).

#### **status**
`idle` → `training` → `done ✓` (or `stopped` / `error`). Just the run's state.

#### **step**
Which training iteration we're on, counting up toward your **Steps** target. Your progress
bar, essentially.

#### **loss** (training loss)
**The single most important number.** It measures how *wrong* the model's next-token
predictions are on the current batch — lower is better. It's the average "surprise"
(cross-entropy). It starts high and should fall:

- Random model over a 4096-token vocab starts near `ln(4096) ≈ 8.3`.
- A coherent TinyStories run lands around `3.0–3.4`.
- **Falling loss = the model is learning.** A flat or rising loss = something's wrong
  (learning rate too high, or it's done learning what it can).

#### **val** (validation loss)
The same loss measured on **held-out text the model never trains on**, printed at each eval.
It's the honesty check:

- **train and val both falling** → genuine learning. 
- **train falling but val flat/rising** → **overfitting**: the model is memorizing the
  training text instead of learning general patterns. Time to stop or add data.

#### **tok/s** (tokens per second)
Throughput — how many tokens the model chews through each second. Purely a *speed* gauge
(hardware/model-size dependent), not a quality signal. GPUs push this 10–100× higher than CPU.

#### **device**
Where training runs: `cpu`, `cuda` (NVIDIA GPU), or `mps` (Apple Silicon). Auto-detected.
If you expected `cuda` but see `cpu`, PyTorch can't see your GPU.

#### **Samples during training**
Every so often the model is asked to generate a bit of text *right now*, so you can watch
quality improve in real time. Early samples are word-salad; later ones read like the corpus.
This is the fun part — loss is abstract, but seeing gibberish turn into stories is visceral.

---

## Putting it together (what a healthy run looks like)

1. Click **Start pre-training**. `status` → `training`, `loss` starts near ~8 and **falls**.
2. `val` tracks `loss` downward; **Samples** go from `xqz the the` → real sentences.
3. `loss` flattens around ~3 — the model has learned what this size *can*. Click stop.
4. **Fine-tune**, then **Chat**. If replies ramble or ignore your topic on the `cpu` preset,
   that's expected — jump back to the ["why scale matters"](../README.md#-presets--the-why-scale-matters-lesson)
   note. Bigger preset on a GPU fixes it.

**Back to:** [README](../README.md) · [Lesson 5 · Fine-tuning](05-finetuning.md)

**Next:** [Lesson 7 · Under the hood »](07-under-the-hood.md)
