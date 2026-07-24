# Lesson 1 · What is a language model?

**Big idea:** A language model is just a machine that, given some text, predicts the
*next token* (roughly, the next word or word-piece). That's it. Everything else —
chatbots, code assistants, summarizers — is built on top of this one trick.

## The whole game: next-token prediction

Given "The little cat sat on the ___", a good model puts high probability on "mat",
"floor", "sofa" and low probability on "photosynthesis". Train it on enough text and this
simple objective forces it to learn grammar, facts, and reasoning patterns — because all
of those help it predict the next token better.

## See it for yourself (60 seconds, no install)

```powershell
python -m llmforge.playground
```

This trains a **neural bigram model** — a table of `P(next char | current char)` — with
gradient descent, in pure Python. Watch the loss fall and the samples drift from noise
toward real-looking text. The code is [`llmforge/playground.py`](../llmforge/playground.py)
and it's short enough to read in one sitting.

## From bigram to GPT

The playground only looks at **one** previous character. A real GPT looks at **hundreds**
of previous tokens and weighs them using *attention*. But the training objective is
identical: predict the next token, measure the error (cross-entropy loss), nudge the
weights. Lessons 2–4 build up to the real thing.

**Next:** [Lesson 2 · Tokenization »](02-tokenization.md)
