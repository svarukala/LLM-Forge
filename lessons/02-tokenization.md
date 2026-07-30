# Lesson 2 · Tokenization: turning text into numbers

Neural networks only do math on numbers, not letters. **Tokenization** is the step that
converts text into a sequence of integer IDs (and back).

> 💡 **See it live:** the dashboard's **🔤 Tokenizer playground** (top of
> `python -m llmforge.cli serve`) splits whatever you type into tokens/IDs in real time.
> Type a sentence with `char` then `bpe` and compare the `chars/token` number.

## Two tokenizers in LLM Forge

Code: [`llmforge/tokenizer.py`](../llmforge/tokenizer.py)

### Character-level (the simple one)
Every unique character gets an ID. "cat" → `[41, 24, 49]`. No training needed, tiny
vocabulary. Great for demos, but sequences get long and the model can't reuse the idea of
a whole word.

### Byte-Pair Encoding / BPE (the realistic one)
This is what GPT-2 uses. It starts from bytes and **merges the most frequent pairs**
repeatedly, so common chunks like `the`, `ing`, or `_cat` become single tokens. Result: a
fixed vocabulary (say 2048) that balances short sequences against a manageable table size.

Train one on your own corpus:

```powershell
python -m llmforge.cli tokenizer --input data/sample/corpus.txt --kind bpe --vocab-size 2048
```

## Why it matters

- **Vocabulary size** is a model dimension: the embedding table and output layer are both
  `vocab_size` wide (see `ModelConfig` in [`llmforge/config.py`](../llmforge/config.py)).
- **Special tokens** like `<|endoftext|>`, `<|user|>`, `<|assistant|>` let us mark
  boundaries — essential for chat fine-tuning (Lesson 5).

## Try this
Encode a sentence with both tokenizers and compare how many tokens each produces. Fewer
tokens = the model sees more text within its fixed context window.

**Next:** [Lesson 3 · Embeddings & attention »](03-embeddings-and-attention.md)
