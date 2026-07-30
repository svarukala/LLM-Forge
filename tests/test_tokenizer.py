"""Tokenizer behaviour, focusing on reserved special-token handling and round-tripping."""

import pytest

from llmforge.tokenizer import SPECIAL_TOKENS, train_tokenizer

SPECIALS_TEXT = "<|user|> hello <|assistant|> hi there <|endoftext|>"


def _char_text():
    # a char tokenizer can only encode characters it has seen, so cover the alphabet used
    return SPECIALS_TEXT + " abcdefghijklmnopqrstuvwxyz .,!?" + "\n"


def test_char_specials_are_atomic():
    tok = train_tokenizer(_char_text(), kind="char")
    ids = tok.encode(SPECIALS_TEXT)
    # each special string must map to exactly ONE id, not one-per-character
    for sp in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
        assert tok.encode(sp) == [tok.token_id(sp)]
        assert len(tok.encode(sp)) == 1
    # and appear once each in the combined encoding
    for sp in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
        assert ids.count(tok.token_id(sp)) == 1


def test_char_roundtrip_plain_text():
    tok = train_tokenizer(_char_text(), kind="char")
    text = "the cat sat."
    assert tok.decode(tok.encode(text)) == text


def test_bpe_specials_are_atomic():
    tok = train_tokenizer(
        SPECIALS_TEXT * 20 + " the cat sat on the mat", kind="bpe", vocab_size=200
    )
    for sp in ["<|user|>", "<|assistant|>", "<|endoftext|>", "<|pad|>"]:
        enc = tok.encode(sp)
        assert enc == [tok.token_id(sp)], f"{sp} not atomic: {enc}"


def test_bpe_roundtrip_with_specials():
    tok = train_tokenizer(
        SPECIALS_TEXT * 20 + " the cat sat on the mat", kind="bpe", vocab_size=200
    )
    ids = tok.encode(SPECIALS_TEXT)
    for sp in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
        assert tok.token_id(sp) in ids


def test_token_id_missing_raises():
    tok = train_tokenizer(_char_text(), kind="char")
    with pytest.raises(KeyError):
        tok.token_id("<|not-a-real-special|>")


def test_all_special_tokens_present():
    tok = train_tokenizer(_char_text(), kind="char")
    for sp in SPECIAL_TOKENS:
        assert isinstance(tok.token_id(sp), int)


def test_char_pieces_match_encoding():
    tok = train_tokenizer(_char_text(), kind="char")
    pieces = tok.pieces("hello")
    assert [p["piece"] for p in pieces] == list("hello")
    assert [p["id"] for p in pieces] == tok.encode("hello")


def test_bpe_pieces_reconstruct_text():
    tok = train_tokenizer("the cat sat on the mat. " * 40, kind="bpe", vocab_size=200)
    text = "the cat sat"
    pieces = tok.pieces(text)
    assert [p["id"] for p in pieces] == tok.encode(text)
    # Joining the (space-restored) pieces reproduces the original text.
    assert "".join(p["piece"] for p in pieces) == text
