"""Chat with a fine-tuned checkpoint using the same chat template used in training.

``ChatSession`` is genuinely stateful: it keeps a running conversation history and, on each
turn, rebuilds the prompt from as much recent history as fits in the model's context window
(older turns are trimmed first). Set ``multi_turn=False`` for the original stateless behavior.
"""

from __future__ import annotations

import torch

from .checkpoint import load_checkpoint
from .config import pick_device
from .data import ASSISTANT, EOT, USER
from .tokenizer import load_tokenizer


class ChatSession:
    def __init__(
        self,
        out_dir: str,
        device: str | None = None,
        multi_turn: bool = True,
        max_history_turns: int = 8,
    ):
        self.device = pick_device(device)
        self.model, tok_path, _ = load_checkpoint(out_dir, device=self.device)
        if not tok_path:
            raise ValueError(f"Checkpoint {out_dir} has no tokenizer to load.")
        self.tokenizer = load_tokenizer(tok_path)
        self.multi_turn = multi_turn
        self.max_history_turns = max_history_turns
        self.history: list[tuple[str, str]] = []  # (user_msg, assistant_msg)
        # Populated on each reply() so callers (e.g. the dashboard) can show context usage.
        self.last_context: dict = {"prompt_tokens": 0, "generated_tokens": 0, "block_size": 0}
        try:
            self.eot_id: int | None = self.tokenizer.token_id(EOT)
        except Exception:
            self.eot_id = None

    def reset(self) -> None:
        """Clear conversation history (start a fresh chat)."""
        self.history = []

    def _build_prompt_ids(self, message: str) -> list[int]:
        """Assemble prompt token ids from recent history that fit within the context window."""
        block = self.model.cfg.block_size
        current = self.tokenizer.encode(f"{USER} {message}\n{ASSISTANT} ")
        # reserve room so at least a few response tokens can be generated
        budget = max(1, block - 16)

        if not self.multi_turn or not self.history:
            return current[-budget:]

        turns = self.history[-self.max_history_turns :]
        prefix_ids: list[int] = []
        # add older turns from most-recent backwards until we run out of budget
        for user_msg, bot_msg in reversed(turns):
            turn_ids = self.tokenizer.encode(f"{USER} {user_msg}\n{ASSISTANT} {bot_msg}{EOT}\n")
            if len(turn_ids) + len(current) + len(prefix_ids) > budget:
                break
            prefix_ids = turn_ids + prefix_ids
        return (prefix_ids + current)[-budget:]

    def reply(
        self,
        message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = 40,
        top_p: float | None = None,
    ) -> str:
        ids = self._build_prompt_ids(message) or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.eot_id,
        )
        generated = out[0].tolist()[len(ids) :]
        text = self.tokenizer.decode(generated).strip()
        self.last_context = {
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated),
            "block_size": self.model.cfg.block_size,
        }
        if self.multi_turn:
            self.history.append((message, text))
        return text


def repl(out_dir: str) -> None:
    session = ChatSession(out_dir)
    print("LLM Forge chat. Type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in {"exit", "quit"}:
            break
        if msg.lower() == "reset":
            session.reset()
            print("(history cleared)\n")
            continue
        if not msg:
            continue
        print(f"bot> {session.reply(msg)}\n")
