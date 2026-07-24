"""Chat with a fine-tuned checkpoint using the same chat template used in training."""

from __future__ import annotations

import torch

from .checkpoint import load_checkpoint
from .tokenizer import load_tokenizer
from .config import pick_device
from .data import USER, ASSISTANT, EOT


class ChatSession:
    def __init__(self, out_dir: str, device: str | None = None):
        self.device = pick_device(device)
        self.model, tok_path, _ = load_checkpoint(out_dir, device=self.device)
        self.tokenizer = load_tokenizer(tok_path)
        try:
            self.eot_id = self.tokenizer.token_id(EOT)
        except Exception:
            self.eot_id = None

    def reply(self, message: str, max_new_tokens: int = 256,
              temperature: float = 0.7, top_k: int = 40) -> str:
        prompt = f"{USER} {message}\n{ASSISTANT} "
        ids = self.tokenizer.encode(prompt) or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        block = self.model.cfg.block_size

        generated = []
        for _ in range(max_new_tokens):
            cond = idx[:, -block:]
            logits, _ = self.model(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tok = int(nxt.item())
            if self.eot_id is not None and tok == self.eot_id:
                break
            generated.append(tok)
            idx = torch.cat((idx, nxt), dim=1)

        return self.tokenizer.decode(generated).strip()


def repl(out_dir: str) -> None:
    session = ChatSession(out_dir)
    print("LLM Forge chat. Type 'exit' to quit.\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in {"exit", "quit"}:
            break
        if not msg:
            continue
        print(f"bot> {session.reply(msg)}\n")
