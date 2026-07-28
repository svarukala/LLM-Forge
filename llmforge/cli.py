"""LLM Forge command-line interface.

    python -m llmforge.cli <command> [options]

Commands:
    tokenizer        Train a tokenizer on a text corpus.
    pretrain         Pre-train a GPT from scratch on a text corpus.
    sample           Generate text from a trained checkpoint.
    finetune         Supervised fine-tuning on chat pairs (JSONL).
    chat             Interactive chat with a fine-tuned checkpoint.
    serve            Launch the local web dashboard.
    play             Run the zero-dependency playground.
    doctor           Detect hardware and recommend a preset.
    checkpoint-info  Print a checkpoint's metadata.
    smoke            Run a tiny end-to-end CPU train+sample+SFT self-test.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import RUNS_DIR, ModelConfig, TrainConfig


def _default_tokenizer_path() -> str:
    return os.path.join(RUNS_DIR, "tokenizer.json")


def _validate_sampling(
    temperature: float, top_k: int, max_new_tokens: int, top_p: float | None = None
) -> None:
    if max_new_tokens < 1:
        raise SystemExit("error: --max-new-tokens must be >= 1")
    if temperature <= 0:
        raise SystemExit("error: --temperature must be > 0")
    if top_k is not None and top_k < 0:
        raise SystemExit("error: --top-k must be >= 0")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise SystemExit("error: --top-p must be in (0, 1]")


def cmd_tokenizer(args: argparse.Namespace) -> None:
    from .data import load_text_file
    from .tokenizer import save_tokenizer_meta, text_fingerprint, train_tokenizer

    text = load_text_file(args.input)
    print(f"Training {args.kind} tokenizer on {len(text)} chars...")
    tok = train_tokenizer(text, kind=args.kind, vocab_size=args.vocab_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    save_tokenizer_meta(args.out, args.kind, tok.vocab_size, text_fingerprint(text))
    print(f"Saved tokenizer ({tok.vocab_size} tokens) -> {args.out}")


def _ensure_tokenizer(args, text: str):
    """Load a compatible existing tokenizer or train a new one.

    Guards against silently reusing an incompatible tokenizer (e.g. a leftover char
    tokenizer when the preset asked for BPE). Reuse happens only when the saved tokenizer's
    kind/vocab/corpus fingerprint match the request, or when --reuse-tokenizer is given.
    """
    from .tokenizer import (
        load_tokenizer,
        load_tokenizer_meta,
        save_tokenizer_meta,
        text_fingerprint,
        train_tokenizer,
    )

    fingerprint = text_fingerprint(text)
    if os.path.exists(args.tokenizer):
        meta = load_tokenizer_meta(args.tokenizer)
        compatible = (
            meta is not None
            and meta.get("kind") == args.tok_kind
            and (args.tok_kind == "char" or meta.get("vocab_size") == args.vocab_size)
            and meta.get("corpus_fingerprint") == fingerprint
        )
        if compatible:
            print(f"Reusing compatible tokenizer {args.tokenizer}")
            return load_tokenizer(args.tokenizer)
        if getattr(args, "reuse_tokenizer", False):
            print(
                f"WARNING: reusing tokenizer {args.tokenizer} despite mismatch "
                f"(requested kind={args.tok_kind} vocab={args.vocab_size}); --reuse-tokenizer set."
            )
            return load_tokenizer(args.tokenizer)
        why = (
            "no metadata"
            if meta is None
            else (
                f"saved kind={meta.get('kind')} vocab={meta.get('vocab_size')} "
                f"fingerprint_match={meta.get('corpus_fingerprint') == fingerprint}"
            )
        )
        print(
            f"Existing tokenizer at {args.tokenizer} is incompatible ({why}); retraining a "
            f"{args.tok_kind} tokenizer. Pass --reuse-tokenizer to force reuse."
        )

    print(f"Training a {args.tok_kind} tokenizer (vocab={args.vocab_size})...")
    tok = train_tokenizer(text, kind=args.tok_kind, vocab_size=args.vocab_size)
    os.makedirs(os.path.dirname(args.tokenizer) or ".", exist_ok=True)
    tok.save(args.tokenizer)
    save_tokenizer_meta(args.tokenizer, args.tok_kind, tok.vocab_size, fingerprint)
    return tok


def _apply_preset(args: argparse.Namespace, stage: str) -> None:
    """If --preset was given, fill in any arg the user did not set explicitly."""
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


_PRETRAIN_DEFAULTS = dict(
    tok_kind="bpe",
    vocab_size=2048,
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=128,
    batch_size=16,
    steps=500,
    lr=3e-4,
)
_FINETUNE_DEFAULTS = dict(batch_size=8, steps=300, lr=1e-4)


def _fill_defaults(args, defaults):
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)


def cmd_doctor(args: argparse.Namespace) -> None:
    from .presets import describe

    print(describe())


def cmd_checkpoint_info(args: argparse.Namespace) -> None:
    import json

    from .checkpoint import checkpoint_info

    info = checkpoint_info(args.checkpoint)
    print(json.dumps(info, indent=2))


def cmd_pretrain(args: argparse.Namespace) -> None:
    from .config import pick_device
    from .data import PretrainData, load_text_file
    from .model import GPT
    from .tokenizer import text_fingerprint
    from .train import train

    _apply_preset(args, "pretrain")
    _fill_defaults(args, _PRETRAIN_DEFAULTS)

    if args.steps < 1:
        raise SystemExit("error: --steps must be >= 1")

    text = load_text_file(args.data)
    tokenizer = _ensure_tokenizer(args, text)

    device = pick_device(args.device)
    ids = tokenizer.encode(text)
    try:
        dataset = PretrainData(ids, block_size=args.block_size, device=device, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    resume_dir = args.out if getattr(args, "resume", False) else None
    if resume_dir and os.path.exists(os.path.join(resume_dir, "config.json")):
        from .checkpoint import load_checkpoint

        model, _, _ = load_checkpoint(resume_dir, device=device)
        print(f"Resuming from {resume_dir}")
    else:
        resume_dir = None
        mcfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
        )
        model = GPT(mcfg)
    print(f"Model: {model.num_params()/1e6:.2f}M params on {device}")

    tcfg = TrainConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        steps=args.steps,
        eval_interval=args.eval_interval,
        learning_rate=args.lr,
        grad_accum=args.grad_accum,
        seed=args.seed,
        device=device,
        amp=args.amp,
        deterministic=args.deterministic,
        checkpoint_every=args.checkpoint_every,
        out_dir=args.out,
    )
    extra = {
        "tokenizer_kind": tokenizer.kind,
        "vocab_size": tokenizer.vocab_size,
        "corpus_fingerprint": text_fingerprint(text),
        "stage": "pretrain",
    }
    result = train(
        model,
        dataset,
        tcfg,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        on_event=_print_event,
        resume_dir=resume_dir,
        extra_meta=extra,
    )
    print(
        f"\n{result['status']} at step {result['completed_steps']} "
        f"(loss {result['final_loss']:.3f}). Checkpoint -> {args.out}"
    )


def cmd_finetune(args: argparse.Namespace) -> None:
    from .checkpoint import load_checkpoint
    from .config import pick_device
    from .data import SFTData
    from .tokenizer import load_tokenizer
    from .train import train

    _apply_preset(args, "finetune")
    _fill_defaults(args, _FINETUNE_DEFAULTS)

    if args.steps < 1:
        raise SystemExit("error: --steps must be >= 1")

    device = pick_device(args.device)
    if not os.path.exists(os.path.join(args.base, "config.json")):
        raise SystemExit(f"error: no base checkpoint at {args.base}. Pre-train first.")
    print(f"Loading base checkpoint {args.base}...")
    model, tok_path, base_cfg = load_checkpoint(args.base, device=device)
    if not tok_path or not os.path.exists(tok_path):
        raise SystemExit(f"error: base checkpoint {args.base} has no usable tokenizer.")
    tokenizer = load_tokenizer(tok_path)

    try:
        dataset = SFTData(
            args.data, tokenizer, block_size=model.cfg.block_size, device=device, seed=args.seed
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(
        f"Loaded {len(dataset.examples)} train / {len(dataset.val_examples)} val chat examples "
        f"({dataset.dropped} dropped: {dataset.drop_reasons})"
    )

    tcfg = TrainConfig(
        batch_size=args.batch_size,
        block_size=model.cfg.block_size,
        steps=args.steps,
        learning_rate=args.lr,
        seed=args.seed,
        device=device,
        deterministic=args.deterministic,
        out_dir=args.out,
    )
    extra = {
        "tokenizer_kind": tokenizer.kind,
        "vocab_size": tokenizer.vocab_size,
        "stage": "finetune",
        "base_checkpoint": args.base,
        "dropped_examples": dataset.dropped,
    }
    result = train(
        model,
        dataset,
        tcfg,
        tokenizer=tokenizer,
        tokenizer_path=tok_path,
        on_event=_print_event,
        extra_meta=extra,
    )
    print(
        f"\n{result['status']} at step {result['completed_steps']} "
        f"(loss {result['final_loss']:.3f}). Fine-tuned checkpoint -> {args.out}"
    )


def cmd_sample(args: argparse.Namespace) -> None:
    from .sample import generate_text

    _validate_sampling(args.temperature, args.top_k, args.max_new_tokens, args.top_p)
    text = generate_text(
        args.checkpoint,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_at_eot=not args.no_stop,
    )
    print(text)


def cmd_chat(args: argparse.Namespace) -> None:
    from .chat import repl

    repl(args.checkpoint)


def cmd_serve(args: argparse.Namespace) -> None:
    from .server.app import run

    run(host=args.host, port=args.port, allow_public=args.allow_public)


def cmd_play(args: argparse.Namespace) -> None:
    from .playground import main

    main()


def cmd_smoke(args: argparse.Namespace) -> None:
    from .smoke import run_smoke

    ok = run_smoke()
    raise SystemExit(0 if ok else 1)


def _print_event(evt: dict) -> None:
    t = evt.get("type")
    if t == "start":
        print(
            f"  training {evt['params']/1e6:.2f}M params on {evt['device']} "
            f"for {evt['steps']} steps (from step {evt.get('start_step', 0)})"
        )
    elif t == "step":
        eta = evt.get("eta_s", 0)
        print(
            f"  step {evt['step']:5d} | loss {evt['loss']:.3f} | ppl {evt['perplexity']:.1f} "
            f"| lr {evt['lr']:.1e} | {evt['tok_per_sec']:.0f} tok/s | eta {eta:.0f}s"
        )
    elif t == "eval":
        print(
            f"  -- eval @ {evt['step']}: train {evt['train']:.3f} val {evt['val']:.3f} "
            f"(val ppl {evt.get('val_perplexity', float('nan')):.1f})"
        )
    elif t == "sample":
        print(f"  ~ sample: {evt['text']!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-forge", description="Build a tiny LLM from scratch.")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("tokenizer", help="Train a tokenizer on a corpus.")
    t.add_argument("--input", required=True)
    t.add_argument("--kind", choices=["bpe", "char"], default="bpe")
    t.add_argument("--vocab-size", type=int, default=2048)
    t.add_argument("--out", default=_default_tokenizer_path())
    t.set_defaults(func=cmd_tokenizer)

    pt = sub.add_parser("pretrain", help="Pre-train a GPT from scratch.")
    pt.add_argument("--data", required=True)
    pt.add_argument("--tokenizer", default=_default_tokenizer_path())
    pt.add_argument(
        "--preset",
        choices=["cpu", "gpu", "gpu-large"],
        default=None,
        help="Hardware preset (cpu/gpu/gpu-large). Run 'doctor' for a recommendation.",
    )
    pt.add_argument("--tok-kind", choices=["bpe", "char"], default=None)
    pt.add_argument("--vocab-size", type=int, default=None)
    pt.add_argument("--block-size", type=int, default=None)
    pt.add_argument("--n-layer", type=int, default=None)
    pt.add_argument("--n-head", type=int, default=None)
    pt.add_argument("--n-embd", type=int, default=None)
    pt.add_argument("--batch-size", type=int, default=None)
    pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--lr", type=float, default=None)
    pt.add_argument("--grad-accum", type=int, default=1)
    pt.add_argument("--eval-interval", type=int, default=50)
    pt.add_argument("--checkpoint-every", type=int, default=0)
    pt.add_argument("--seed", type=int, default=1337)
    pt.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only).")
    pt.add_argument("--deterministic", action="store_true", help="Fully deterministic (slower).")
    pt.add_argument(
        "--reuse-tokenizer",
        action="store_true",
        help="Reuse an existing tokenizer even if its settings/corpus differ.",
    )
    pt.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint at --out (optimizer + step state).",
    )
    pt.add_argument("--device", default=None)
    pt.add_argument("--out", default=os.path.join(RUNS_DIR, "base"))
    pt.set_defaults(func=cmd_pretrain)

    ft = sub.add_parser("finetune", help="Supervised fine-tuning on chat pairs.")
    ft.add_argument("--data", required=True)
    ft.add_argument("--base", default=os.path.join(RUNS_DIR, "base"))
    ft.add_argument(
        "--preset",
        choices=["cpu", "gpu", "gpu-large"],
        default=None,
        help="Hardware preset. Use the same one you pre-trained with.",
    )
    ft.add_argument("--batch-size", type=int, default=None)
    ft.add_argument("--steps", type=int, default=None)
    ft.add_argument("--lr", type=float, default=None)
    ft.add_argument("--seed", type=int, default=1337)
    ft.add_argument("--deterministic", action="store_true")
    ft.add_argument("--device", default=None)
    ft.add_argument("--out", default=os.path.join(RUNS_DIR, "chat"))
    ft.set_defaults(func=cmd_finetune)

    s = sub.add_parser("sample", help="Generate text from a checkpoint.")
    s.add_argument("--checkpoint", default=os.path.join(RUNS_DIR, "base"))
    s.add_argument("--prompt", default="Once upon a time")
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--top-p", type=float, default=None)
    s.add_argument("--no-stop", action="store_true", help="Do not stop at <|endoftext|>.")
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("chat", help="Chat with a fine-tuned checkpoint.")
    c.add_argument("--checkpoint", default=os.path.join(RUNS_DIR, "chat"))
    c.set_defaults(func=cmd_chat)

    sv = sub.add_parser("serve", help="Launch the web dashboard.")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument(
        "--allow-public",
        action="store_true",
        help="Allow binding beyond localhost (UNSAFE: no authentication).",
    )
    sv.set_defaults(func=cmd_serve)

    pl = sub.add_parser("play", help="Run the zero-dependency playground.")
    pl.set_defaults(func=cmd_play)

    dr = sub.add_parser("doctor", help="Detect hardware and recommend a preset.")
    dr.set_defaults(func=cmd_doctor)

    ci = sub.add_parser("checkpoint-info", help="Print a checkpoint's metadata.")
    ci.add_argument("--checkpoint", required=True)
    ci.set_defaults(func=cmd_checkpoint_info)

    sm = sub.add_parser("smoke", help="Run a tiny end-to-end CPU self-test.")
    sm.set_defaults(func=cmd_smoke)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
