# Lesson 5 · Fine-tuning: teaching it to be helpful

A pre-trained model is a brilliant autocomplete, but it doesn't know it's supposed to
*answer questions*. **Fine-tuning** adapts it to a task using a smaller, curated dataset.

## Supervised fine-tuning (SFT)

We show the model example conversations formatted with a **chat template**:

```
<|user|> What color is the sky? \n <|assistant|> On a clear day the sky is blue. <|endoftext|>
```

Code: `_format_example` and `SFTData` in [`llmforge/data.py`](../llmforge/data.py).

The crucial trick: **we mask the prompt in the loss** (set those targets to `-1` so they're
ignored). The model is only scored on the *assistant's* tokens, so it learns to *produce*
answers rather than to parrot questions.

Everything else is identical to pre-training — same loop, same loss — we just start from
the pre-trained weights and feed it chat data:

```powershell
python -m llmforge.cli finetune --data data/sample/chat.jsonl --base runs/base --steps 300
```

## Then chat with what you made

```powershell
python -m llmforge.cli chat
```

The chat code ([`llmforge/chat.py`](../llmforge/chat.py)) wraps your message in the same
template and generates until the model emits `<|endoftext|>`.

## Beyond SFT (where the frontier goes)

Real assistants add a **preference** step (RLHF / DPO): show the model pairs of answers,
tell it which is better, and train it to prefer the better one. That's out of scope for
this tiny workshop, but it's the same idea — more signal about *what a good answer looks
like*. Once you're comfortable here, that's the natural next thing to read about.

## Try this
Add your own rows to `chat.jsonl` (your team's FAQ, your writing style), re-run fine-tuning,
and watch the model pick up your tone.

**Next:** [Lesson 6 · Dashboard field guide »](06-dashboard-glossary.md)
