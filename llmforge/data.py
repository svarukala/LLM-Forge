"""Data loading for pre-training and fine-tuning.

Pre-training: one long stream of token ids; we grab random contiguous windows.
Fine-tuning (SFT): prompt/response pairs formatted into a chat template, where the loss
is only computed on the assistant's response tokens (the model learns to *answer*, not to
parrot the question).

Read alongside lessons/04-pretraining.md and lessons/05-finetuning.md.
"""

from __future__ import annotations

import json
import random

import torch

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOT = "<|endoftext|>"


def load_text_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class PretrainData:
    """A single long token stream; yields random (x, y) windows for next-token prediction.

    Both the training and validation splits must be strictly larger than the context
    window (``block_size + 1``) or there would be no valid window to sample. We validate
    this up front with an actionable error instead of silently training on a short stream.
    """

    def __init__(
        self, ids: list[int], block_size: int, device: str, val_frac: float = 0.1, seed: int = 1337
    ):
        need = block_size + 1
        if len(ids) < 2 * need:
            raise ValueError(
                f"Corpus too small: {len(ids)} tokens, but block_size={block_size} needs at "
                f"least {2 * need} tokens (one full window each for train and validation). "
                f"Use a larger corpus or a smaller --block-size."
            )
        data = torch.tensor(ids, dtype=torch.long)
        n_val = max(need, int(len(data) * val_frac))
        # guarantee the training split also keeps at least one full window
        n_val = min(n_val, len(data) - need)
        self.train = data[:-n_val]
        self.val = data[-n_val:]
        self.block_size = block_size
        self.device = device
        self._gen = torch.Generator().manual_seed(seed)

    def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train if split == "train" else self.val
        # last valid start index is len - block_size - 1 (inclusive); randint high is exclusive
        max_start = len(source) - self.block_size - 1
        ix = torch.randint(0, max_start + 1, (batch_size,), generator=self._gen)
        x = torch.stack([source[i : i + self.block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)


def _format_example(
    tokenizer, prompt: str, response: str, block_size: int
) -> tuple[list[int], list[int]] | None:
    """Build (input_ids, target_ids) for one chat example, masking the prompt in the loss.

    Deterministic truncation policy (so a long prompt can never overflow ``block_size``):
      1. The full sequence length is capped at ``block_size + 1`` tokens.
      2. The assistant marker suffix (``\\n<|assistant|> ``) is ALWAYS preserved.
      3. The terminal ``<|endoftext|>`` token is ALWAYS preserved (space is reserved for it).
      4. If the prompt is too long, the *user* portion is trimmed (keeping the tail nearest
         the assistant marker); the leading ``<|user|>`` prefix is dropped only if needed.
      5. The response body fills whatever room remains, then EOT is appended.
      6. If not even one response token can survive, the example is unusable -> return None
         (the caller drops it and counts it).

    Loss is computed only on the assistant response tokens (and the EOT); prompt targets
    are masked to -1.
    """
    max_total = block_size + 1
    try:
        eot: int | None = tokenizer.token_id(EOT)
    except Exception:
        eot = None
    reserve_eot = 1 if eot is not None else 0

    prefix = tokenizer.encode(f"{USER} ")
    user = tokenizer.encode(prompt)
    suffix = tokenizer.encode(f"\n{ASSISTANT} ")  # contains the assistant marker
    body = tokenizer.encode(response)

    budget = max_total - reserve_eot  # tokens available for prompt + response
    # Need the assistant suffix plus at least one response token to be a usable example.
    if len(suffix) + 1 > budget:
        return None

    max_prompt = budget - 1  # always leave room for >=1 response token
    prompt_ids = prefix + user + suffix
    if len(prompt_ids) > max_prompt:
        keep_non_suffix = max_prompt - len(suffix)
        combined = prefix + user
        combined = combined[len(combined) - keep_non_suffix :] if keep_non_suffix > 0 else []
        prompt_ids = combined + suffix

    avail_resp = budget - len(prompt_ids)
    body = body[: max(0, avail_resp)]
    if not body:
        return None

    ids = prompt_ids + body + ([eot] if eot is not None else [])
    x = ids[:-1]
    y = ids[1:]
    mask_len = min(len(prompt_ids) - 1, len(y))
    y = [-1] * mask_len + y[mask_len:]
    return x, y


class SFTData:
    """Supervised fine-tuning data from a JSONL file of {"prompt", "response"} rows.

    Rows that cannot yield a valid (prompt-masked, response-scored) example after truncation
    are dropped and counted in ``self.dropped`` / ``self.drop_reasons``. A held-out validation
    split is created so evaluation is not measured on the exact training examples.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        block_size: int,
        device: str,
        val_frac: float = 0.1,
        seed: int = 1337,
    ):
        examples: list[tuple[list[int], list[int]]] = []
        dropped = 0
        drop_reasons = {"parse_error": 0, "missing_fields": 0, "too_long_or_empty": 0}

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    drop_reasons["parse_error"] += 1
                    continue
                if "prompt" not in row or "response" not in row:
                    dropped += 1
                    drop_reasons["missing_fields"] += 1
                    continue
                formatted = _format_example(tokenizer, row["prompt"], row["response"], block_size)
                if formatted is None or len(formatted[0]) < 2:
                    dropped += 1
                    drop_reasons["too_long_or_empty"] += 1
                    continue
                examples.append(formatted)

        if not examples:
            raise ValueError(
                f"No usable examples found in {path} (dropped {dropped}: {drop_reasons}). "
                f"Check the JSONL format and that responses fit within block_size={block_size}."
            )

        rng = random.Random(seed)
        rng.shuffle(examples)
        n_val = int(len(examples) * val_frac)
        n_val = min(n_val, max(0, len(examples) - 1))  # keep >=1 training example
        self.val_examples = examples[:n_val]
        self.examples = examples[n_val:] or examples
        self.dropped = dropped
        self.drop_reasons = drop_reasons
        self.block_size = block_size
        self.device = device
        self.pad_id = _safe_pad_id(tokenizer)
        self._rng = random.Random(seed + 1)

    def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        pool = self.val_examples if (split == "val" and self.val_examples) else self.examples
        picks = [self._rng.choice(pool) for _ in range(batch_size)]
        maxlen = max(len(x) for x, _ in picks)
        xs, ys = [], []
        for x, y in picks:
            pad = maxlen - len(x)
            xs.append(x + [self.pad_id] * pad)
            ys.append(y + [-1] * pad)  # padded targets are ignored in the loss
        x = torch.tensor(xs, dtype=torch.long, device=self.device)
        y = torch.tensor(ys, dtype=torch.long, device=self.device)
        return x, y


def _safe_pad_id(tokenizer) -> int:
    try:
        return tokenizer.token_id("<|pad|>")
    except Exception:
        return 0
