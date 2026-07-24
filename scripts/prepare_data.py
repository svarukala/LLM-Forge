"""Download and size real training data for LLM Forge.

TinyStories (Eldan & Li, 2023) is a corpus of very simple children's stories written with
a small vocabulary. It's the canonical dataset for making *tiny* models produce coherent
English on modest hardware -- exactly what we want on a CPU devbox. Listed under the
pre-training section of https://github.com/Zjh-819/LLMDataHub.

This script:
  1. Streams a size-capped slice of TinyStories for PRE-TRAINING  -> data/pretrain/tinystories.txt
  2. Derives an in-domain instruction set for FINE-TUNING          -> data/finetune/stories_sft.jsonl
     (prompt = "write a short story" style, response = an actual short story), so the
     fine-tuned model stays in the same coherent domain instead of drifting to gibberish.

Usage:
    python scripts/prepare_data.py --pretrain-mb 6 --sft-examples 1200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import urllib.request

VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
SEP = "<|endoftext|>"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAIN_OUT = os.path.join(ROOT, "data", "pretrain", "tinystories.txt")
SFT_OUT = os.path.join(ROOT, "data", "finetune", "stories_sft.jsonl")

INSTRUCTIONS = [
    "Write a short story for a young child.",
    "Tell me a simple bedtime story.",
    "Write a little story about a friend.",
    "Make up a short and happy story.",
    "Write a gentle story for a five year old.",
    "Tell a short story with a nice ending.",
]

# Common TinyStories subjects. If a story contains one, we phrase the instruction to name
# it -- this is what teaches the model to CONDITION on the topic you ask for.
SUBJECTS = [
    "dog", "puppy", "cat", "kitten", "bird", "rabbit", "bunny", "bear", "fish", "frog",
    "duck", "mouse", "fox", "lion", "horse", "pig", "cow", "sheep", "elephant", "monkey",
    "ball", "tree", "car", "truck", "train", "boat", "plane", "flower", "robot", "dragon",
    "teddy", "balloon", "cake", "star", "moon", "boy", "girl", "kite", "boat", "garden",
]

TOPIC_TEMPLATES = [
    "Write a short story about a {subj}.",
    "Tell me a story about a {subj}.",
    "Can you write a little story about a {subj}?",
    "Write a bedtime story about a {subj}.",
    "Make up a short story about a {subj}.",
]


def pick_subject(story: str) -> str | None:
    low = story.lower()
    # earliest-appearing known subject = most likely the story's actual topic
    best, best_pos = None, len(low) + 1
    for s in SUBJECTS:
        m = re.search(rf"\b{re.escape(s)}\b", low)
        if m and m.start() < best_pos:
            best, best_pos = s, m.start()
    return best


def stream_slice(url: str, target_bytes: int) -> str:
    """Download from `url` until we've collected ~target_bytes, cut at a story boundary."""
    print(f"Streaming up to {target_bytes/1e6:.1f} MB from TinyStories...")
    chunks, total = [], 0
    req = urllib.request.Request(url, headers={"User-Agent": "llm-forge/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        while total < target_bytes:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            print(f"  ...{total/1e6:.1f} MB", end="\r")
    print()
    text = b"".join(chunks).decode("utf-8", errors="ignore")
    # trim the trailing partial story so the corpus ends cleanly
    cut = text.rfind(SEP)
    if cut != -1:
        text = text[:cut + len(SEP)]
    return text


def stories_from(text: str) -> list[str]:
    out = []
    for raw in text.split(SEP):
        s = raw.strip()
        if s:
            out.append(s)
    return out


def build_sft(stories: list[str], n: int, min_words: int, max_words: int) -> list[dict]:
    random.seed(1337)
    pool = [s for s in stories if min_words <= len(s.split()) <= max_words]
    random.shuffle(pool)
    examples = []
    for story in pool:
        if len(examples) >= n:
            break
        subj = pick_subject(story)
        if subj:
            prompt = random.choice(TOPIC_TEMPLATES).format(subj=subj)
        else:
            prompt = random.choice(INSTRUCTIONS)
        examples.append({"prompt": prompt, "response": story})
    topical = sum(1 for e in examples if "about a" in e["prompt"])
    print(f"  {topical}/{len(examples)} fine-tune prompts are topic-conditioned")
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-mb", type=float, default=6.0,
                    help="approx MB of story text for pre-training (default 6)")
    ap.add_argument("--sft-examples", type=int, default=1200,
                    help="number of instruction/story pairs for fine-tuning")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=180)
    ap.add_argument("--source", choices=["valid", "train"], default="valid",
                    help="'valid' is smaller and fast; 'train' for a bigger slice")
    ap.add_argument("--reuse-corpus", action="store_true",
                    help="skip download and rebuild the SFT set from the existing corpus")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(PRETRAIN_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(SFT_OUT), exist_ok=True)

    if args.reuse_corpus and os.path.exists(PRETRAIN_OUT):
        with open(PRETRAIN_OUT, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Reusing existing corpus {PRETRAIN_OUT} ({len(text)/1e6:.2f} MB)")
    else:
        url = VALID_URL if args.source == "valid" else TRAIN_URL
        text = stream_slice(url, int(args.pretrain_mb * 1e6))
        with open(PRETRAIN_OUT, "w", encoding="utf-8") as f:
            f.write(text)
    stories = stories_from(text)
    print(f"Pre-train corpus: {len(text)/1e6:.2f} MB, {len(stories)} stories -> {PRETRAIN_OUT}")

    sft = build_sft(stories, args.sft_examples, args.min_words, args.max_words)
    with open(SFT_OUT, "w", encoding="utf-8") as f:
        for ex in sft:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Fine-tune set:   {len(sft)} instruction/story pairs -> {SFT_OUT}")
    print("\nNext: see the 'Recommended run' commands in the README.")


if __name__ == "__main__":
    main()
