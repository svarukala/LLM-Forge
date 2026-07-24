"""Layer 1: the zero-dependency playground.

    python -m llmforge.playground

Trains a *genuine* neural bigram language model — a VxV table of logits updated by
gradient descent — in pure Python (stdlib only: no torch, no numpy). It's intentionally
the simplest thing that actually *learns*: you'll watch the loss fall and the generated
text drift from random noise toward the statistics of the training corpus.

This mirrors, in miniature, exactly what the real GPT in `llmforge/model.py` does — the
only difference is scale and that a GPT looks at many previous tokens instead of one.
"""

from __future__ import annotations

import math
import os
import random

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "corpus.txt")


def load_text() -> str:
    try:
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback so the demo always runs even without the sample file.
        return (
            "the little cat played in the warm sun. the little dog ran in the park. "
            "the sun was warm and the cat was happy. a little bird sang in the tree. "
        ) * 40


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


class BigramModel:
    """P(next_char | current_char) as a table of logits trained with SGD."""

    def __init__(self, vocab_size: int, lr: float = 0.5):
        self.V = vocab_size
        self.lr = lr
        # weights[i][j] = logit for predicting char j after char i
        self.W = [[0.0] * vocab_size for _ in range(vocab_size)]

    def step(self, i: int, target: int) -> float:
        probs = softmax(self.W[i])
        loss = -math.log(max(probs[target], 1e-9))
        # gradient of cross-entropy wrt logits: p - onehot(target)
        row = self.W[i]
        lr = self.lr
        for j in range(self.V):
            grad = probs[j] - (1.0 if j == target else 0.0)
            row[j] -= lr * grad
        return loss

    def sample(self, start: int, length: int) -> list[int]:
        out = [start]
        cur = start
        for _ in range(length):
            probs = softmax(self.W[cur])
            r = random.random()
            acc = 0.0
            nxt = self.V - 1
            for j, p in enumerate(probs):
                acc += p
                if r <= acc:
                    nxt = j
                    break
            out.append(nxt)
            cur = nxt
        return out


def main() -> None:
    random.seed(1337)
    text = load_text()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)

    data = [stoi[c] for c in text]
    pairs = list(zip(data, data[1:]))

    print("=" * 64)
    print("  LLM Forge - Layer 1 Playground (pure Python, zero dependencies)")
    print("=" * 64)
    print(f"  corpus: {len(text)} chars | vocabulary: {V} unique characters")
    print(f"  training a neural bigram model on {len(pairs)} char pairs\n")

    model = BigramModel(V, lr=0.3)

    steps = 4000
    report_every = 500
    running = 0.0
    for step in range(1, steps + 1):
        i, target = random.choice(pairs)
        running += model.step(i, target)
        if step % report_every == 0:
            avg = running / report_every
            running = 0.0
            seed_char = random.choice(chars)
            sample = "".join(itos[t] for t in model.sample(stoi[seed_char], 60))
            sample = sample.replace("\n", " ")
            print(f"  step {step:5d} | loss {avg:.3f} | sample: {sample!r}")

    print("\n  Notice how the loss fell and the samples started to look like the")
    print("  training text. That's it -- that's 'learning'. The real GPT in")
    print("  llmforge/model.py does the same thing, just with attention over many")
    print("  tokens and millions of parameters. Ready for Layer 2? See the README.")


if __name__ == "__main__":
    main()
