"""A tiny end-to-end self-test: tokenizer -> pre-train -> sample -> fine-tune -> chat.

Runs entirely on CPU in a temp directory in a few seconds with a micro model. It proves the
whole pipeline is wired correctly (used by `llm-forge smoke` and by the test suite).
"""

from __future__ import annotations

import os
import tempfile

_CORPUS = (
    "the little cat sat on the warm mat. the little dog ran in the green park. "
    "the sun was warm and the cat was happy. a little bird sang in the tall tree. "
    "the dog and the cat played all day in the park near the big house. "
) * 40

_CHAT = [
    {"prompt": "say hello", "response": "hello there, nice to meet you."},
    {"prompt": "talk about a cat", "response": "the cat sat on the mat in the sun."},
    {"prompt": "talk about a dog", "response": "the dog ran in the park all day."},
] * 8


def run_smoke(verbose: bool = True) -> bool:
    from .checkpoint import checkpoint_info, load_checkpoint
    from .config import ModelConfig, TrainConfig
    from .data import PretrainData, SFTData
    from .model import GPT
    from .sample import generate_text
    from .tokenizer import save_tokenizer_meta, text_fingerprint, train_tokenizer
    from .train import train

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.txt")
        with open(corpus, "w", encoding="utf-8") as f:
            f.write(_CORPUS)
        tok_path = os.path.join(tmp, "tokenizer.json")
        base = os.path.join(tmp, "base")
        chat = os.path.join(tmp, "chat")

        log("[1/5] training char tokenizer...")
        tok = train_tokenizer(_CORPUS, kind="char")
        tok.save(tok_path)
        save_tokenizer_meta(tok_path, "char", tok.vocab_size, text_fingerprint(_CORPUS))

        log("[2/5] pre-training a micro model...")
        cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=64)
        model = GPT(cfg)
        ids = tok.encode(_CORPUS)
        ds = PretrainData(ids, block_size=32, device="cpu")
        tcfg = TrainConfig(
            batch_size=8,
            block_size=32,
            steps=20,
            eval_interval=10,
            eval_iters=3,
            sample_every=0,
            device="cpu",
            out_dir=base,
        )
        res = train(model, ds, tcfg, tokenizer=tok, tokenizer_path=tok_path)
        assert res["status"] == "completed" and res["completed_steps"] == 20

        log("[3/5] sampling from base model...")
        txt = generate_text(base, "the", max_new_tokens=16, temperature=0.8, top_k=10)
        assert isinstance(txt, str) and len(txt) > 0

        log("[4/5] fine-tuning on chat pairs...")
        import json

        sft = os.path.join(tmp, "chat.jsonl")
        with open(sft, "w", encoding="utf-8") as f:
            for row in _CHAT:
                f.write(json.dumps(row) + "\n")
        bmodel, btok, _ = load_checkpoint(base, device="cpu")
        from .tokenizer import load_tokenizer

        assert btok is not None
        btokenizer = load_tokenizer(btok)
        sds = SFTData(sft, btokenizer, block_size=bmodel.cfg.block_size, device="cpu")
        ftcfg = TrainConfig(
            batch_size=4,
            block_size=bmodel.cfg.block_size,
            steps=15,
            eval_interval=10,
            eval_iters=2,
            sample_every=0,
            device="cpu",
            out_dir=chat,
        )
        fres = train(bmodel, sds, ftcfg, tokenizer=btokenizer, tokenizer_path=btok)
        assert fres["status"] == "completed"

        log("[5/5] chatting with fine-tuned model...")
        from .chat import ChatSession

        session = ChatSession(chat)
        reply = session.reply("talk about a dog", max_new_tokens=16)
        assert isinstance(reply, str)

        info = checkpoint_info(chat)
        assert info["resumable"] is True

    log("SMOKE TEST PASSED (tokenizer -> pretrain -> sample -> finetune -> chat)")
    return True


if __name__ == "__main__":
    run_smoke()
