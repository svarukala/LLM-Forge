# Lesson 3 · Embeddings & attention

This is the heart of the transformer. Code: [`llmforge/model.py`](../llmforge/model.py).

## Embeddings: giving tokens meaning

A token ID like `41` means nothing by itself. The **embedding table** (`wte` in the code)
maps each ID to a vector of, say, 128 learnable numbers. During training these vectors
arrange themselves so that related tokens end up near each other. We also add a
**position embedding** (`wpe`) so the model knows *where* each token sits in the sequence.

## Attention: letting words read their context

The key insight: the meaning of a word depends on the words around it ("bank" of a river
vs. a "bank" account). **Self-attention** lets each token look at earlier tokens and pull
in the information it needs.

For every token the model computes three vectors:
- **Query** — "what am I looking for?"
- **Key** — "what do I offer?"
- **Value** — "what I'll hand over if you attend to me."

Each token compares its query against every key (a dot product), turns those scores into
weights with softmax, and takes a weighted sum of the values. See
`CausalSelfAttention.forward` in the model.

**Causal** = a token may only attend to itself and *earlier* tokens, never the future.
That mask is what makes next-token prediction honest.

## The transformer block

One block = attention + a small feed-forward network (`MLP`), each wrapped in a
**residual connection** and **layer norm**. Stack `n_layer` of these and you have a GPT.
More layers and wider embeddings = more capacity (and more compute).

## Try this
In `ModelConfig`, bump `n_head` from 4 to 8 (keep `n_embd` divisible by it) and retrain.
More heads = more independent "relationships" the model can track at once.

**Next:** [Lesson 4 · Pre-training »](04-pretraining.md)
