"""LLM Forge command-line interface.

    python -m llmforge.cli <command> [options]

Commands:
    tokenizer   Train a tokenizer on a text corpus.
    pretrain    Pre-train a GPT from scratch on a text corpus.
    sample      Generate text from a trained checkpoint.
    finetune    Supervised fine-tuning on chat pairs (JSONL).
    chat        Interactive chat with a fine-tuned checkpoint.
    serve       Launch the local web dashboard.
    play        Run the zero-dependency playground.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import ModelConfig, TrainConfig, RUNS_DIR


def _default_tokenizer_path(kind: str) -> str:
    ext = "json"
    return os.path.join(RUNS_DIR, f"tokenizer.{ext}")


def cmd_tokenizer(args: argparse.Namespace) -> None:
    from .tokenizer import train_tokenizer
    from .data import load_text_file

    text = load_text_file(args.input)
    print(f"Training {args.kind} tokenizer on {len(text)} chars...")
    tok = train_tokenizer(text, kind=args.kind, vocab_size=args.vocab_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"Saved tokenizer ({tok.vocab_size} tokens) -> {args.out}")


def _ensure_tokenizer(args, text: str):
    """Load an existing tokenizer or train one on the fly."""
    from .tokenizer import load_tokenizer, train_tokenizer

    if os.path.exists(args.tokenizer):
        print(f"Using tokenizer {args.tokenizer}")
        return load_tokenizer(args.tokenizer)
    print(f"No tokenizer at {args.tokenizer}; training a {args.tok_kind} one now...")
    tok = train_tokenizer(text, kind=args.tok_kind, vocab_size=args.vocab_size)
    os.makedirs(os.path.dirname(args.tokenizer) or ".", exist_ok=True)
    tok.save(args.tokenizer)
    return tok


def _apply_preset(args: argparse.Namespace, stage: str) -> None:
    """If --preset was given, fill in any arg the user did not set explicitly.

    Explicit CLI flags always win; the preset only supplies values the user left at
    their default sentinel (None for the knobs we route through the preset).
    """
    if not getattr(args, "preset", None):
        return
    from .presets import get_preset

    p = get_preset(args.preset)
    print(f"Using preset '{p.name}': {p.blurb}")

    def fill(attr, value):
        if getattr(args, attr, None) is None:
            setattr(args, attr, value)

    if stage == "pretrain":
        fill("tok_kind", p.tok_kind)
        fill("vocab_size", p.vocab_size)
        fill("block_size", p.block_size)
        fill("n_layer", p.n_layer)
        fill("n_head", p.n_head)
        fill("n_embd", p.n_embd)
        fill("batch_size", p.batch_size)
        fill("steps", p.pretrain_steps)
        fill("lr", p.lr)
    elif stage == "finetune":
        fill("batch_size", p.batch_size)
        fill("steps", p.finetune_steps)
        fill("lr", p.finetune_lr)


# Defaults applied when neither a preset nor an explicit flag set a value.
_PRETRAIN_DEFAULTS = dict(tok_kind="bpe", vocab_size=2048, block_size=128,
                          n_layer=4, n_head=4, n_embd=128, batch_size=16,
                          steps=500, lr=3e-4)
_FINETUNE_DEFAULTS = dict(batch_size=8, steps=300, lr=1e-4)


def _fill_defaults(args, defaults):
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)


def cmd_doctor(args: argparse.Namespace) -> None:
    from .presets import describe
    print(describe())


def cmd_pretrain(args: argparse.Namespace) -> None:
    from .model import GPT
    from .data import load_text_file, PretrainData
    from .train import train
    from .config import pick_device

    _apply_preset(args, "pretrain")
    _fill_defaults(args, _PRETRAIN_DEFAULTS)

    text = load_text_file(args.data)
    tokenizer = _ensure_tokenizer(args, text)

    device = pick_device(args.device)
    mcfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(mcfg)
    print(f"Model: {model.num_params()/1e6:.2f}M params on {device}")

    ids = tokenizer.encode(text)
    dataset = PretrainData(ids, block_size=args.block_size, device=device)

    tcfg = TrainConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        steps=args.steps,
        learning_rate=args.lr,
        device=device,
        out_dir=args.out,
    )
    train(model, dataset, tcfg, tokenizer=tokenizer,
          tokenizer_path=args.tokenizer, on_event=_print_event)
    print(f"\nDone. Checkpoint -> {args.out}")


def cmd_finetune(args: argparse.Namespace) -> None:
    from .data import SFTData
    from .train import train
    from .checkpoint import load_checkpoint
    from .tokenizer import load_tokenizer
    from .config import pick_device

    _apply_preset(args, "finetune")
    _fill_defaults(args, _FINETUNE_DEFAULTS)

    device = pick_device(args.device)
    print(f"Loading base checkpoint {args.base}...")
    model, tok_path, _ = load_checkpoint(args.base, device=device)
    tokenizer = load_tokenizer(tok_path)

    dataset = SFTData(args.data, tokenizer, block_size=model.cfg.block_size, device=device)
    print(f"Loaded {len(dataset.examples)} chat examples")

    tcfg = TrainConfig(
        batch_size=args.batch_size,
        block_size=model.cfg.block_size,
        steps=args.steps,
        learning_rate=args.lr,
        device=device,
        out_dir=args.out,
    )
    train(model, dataset, tcfg, tokenizer=tokenizer,
          tokenizer_path=tok_path, on_event=_print_event)
    print(f"\nDone. Fine-tuned checkpoint -> {args.out}")


def cmd_sample(args: argparse.Namespace) -> None:
    from .sample import generate_text
    text = generate_text(args.checkpoint, args.prompt,
                         max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print(text)


def cmd_chat(args: argparse.Namespace) -> None:
    from .chat import repl
    repl(args.checkpoint)


def cmd_serve(args: argparse.Namespace) -> None:
    from .server.app import run
    run(host=args.host, port=args.port)


def cmd_play(args: argparse.Namespace) -> None:
    from .playground import main
    main()


def _print_event(evt: dict) -> None:
    t = evt.get("type")
    if t == "start":
        print(f"  training {evt['params']/1e6:.2f}M params on {evt['device']} "
              f"for {evt['steps']} steps")
    elif t == "step":
        print(f"  step {evt['step']:5d} | loss {evt['loss']:.3f} "
              f"| lr {evt['lr']:.1e} | {evt['tok_per_sec']:.0f} tok/s")
    elif t == "eval":
        print(f"  -- eval @ {evt['step']}: train {evt['train']:.3f} val {evt['val']:.3f}")
    elif t == "sample":
        print(f"  ~ sample: {evt['text']!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-forge", description="Build a tiny LLM from scratch.")
    sub = p.add_subparsers(dest="command", required=True)

    # tokenizer
    t = sub.add_parser("tokenizer", help="Train a tokenizer on a corpus.")
    t.add_argument("--input", required=True)
    t.add_argument("--kind", choices=["bpe", "char"], default="bpe")
    t.add_argument("--vocab-size", type=int, default=2048)
    t.add_argument("--out", default=os.path.join(RUNS_DIR, "tokenizer.json"))
    t.set_defaults(func=cmd_tokenizer)

    # shared model/training args for pretrain
    pt = sub.add_parser("pretrain", help="Pre-train a GPT from scratch.")
    pt.add_argument("--data", required=True)
    pt.add_argument("--tokenizer", default=os.path.join(RUNS_DIR, "tokenizer.json"))
    pt.add_argument("--preset", choices=["cpu", "gpu", "gpu-large"], default=None,
                    help="Hardware preset (cpu/gpu/gpu-large). Run 'doctor' for a recommendation.")
    pt.add_argument("--tok-kind", choices=["bpe", "char"], default=None)
    pt.add_argument("--vocab-size", type=int, default=None)
    pt.add_argument("--block-size", type=int, default=None)
    pt.add_argument("--n-layer", type=int, default=None)
    pt.add_argument("--n-head", type=int, default=None)
    pt.add_argument("--n-embd", type=int, default=None)
    pt.add_argument("--batch-size", type=int, default=None)
    pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--lr", type=float, default=None)
    pt.add_argument("--device", default=None)
    pt.add_argument("--out", default=os.path.join(RUNS_DIR, "base"))
    pt.set_defaults(func=cmd_pretrain)

    # finetune
    ft = sub.add_parser("finetune", help="Supervised fine-tuning on chat pairs.")
    ft.add_argument("--data", required=True)
    ft.add_argument("--base", default=os.path.join(RUNS_DIR, "base"))
    ft.add_argument("--preset", choices=["cpu", "gpu", "gpu-large"], default=None,
                    help="Hardware preset. Use the same one you pre-trained with.")
    ft.add_argument("--batch-size", type=int, default=None)
    ft.add_argument("--steps", type=int, default=None)
    ft.add_argument("--lr", type=float, default=None)
    ft.add_argument("--device", default=None)
    ft.add_argument("--out", default=os.path.join(RUNS_DIR, "chat"))
    ft.set_defaults(func=cmd_finetune)

    # sample
    s = sub.add_parser("sample", help="Generate text from a checkpoint.")
    s.add_argument("--checkpoint", default=os.path.join(RUNS_DIR, "base"))
    s.add_argument("--prompt", default="Once upon a time")
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.set_defaults(func=cmd_sample)

    # chat
    c = sub.add_parser("chat", help="Chat with a fine-tuned checkpoint.")
    c.add_argument("--checkpoint", default=os.path.join(RUNS_DIR, "chat"))
    c.set_defaults(func=cmd_chat)

    # serve
    sv = sub.add_parser("serve", help="Launch the web dashboard.")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    # play
    pl = sub.add_parser("play", help="Run the zero-dependency playground.")
    pl.set_defaults(func=cmd_play)

    # doctor
    dr = sub.add_parser("doctor", help="Detect hardware and recommend a preset.")
    dr.set_defaults(func=cmd_doctor)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
