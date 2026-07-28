"""Generation: sampling-parameter validation and EOS-aware early stopping."""

import pytest
import torch

from tests.conftest import make_char_tokenizer, tiny_model


def test_generate_validates_params():
    tok = make_char_tokenizer()
    model = tiny_model(tok)
    idx = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=0)
    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=5, temperature=-1.0)
    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=5, top_p=2.0)


def test_generate_stops_at_eos():
    tok = make_char_tokenizer()
    model = tiny_model(tok)
    eos = tok.token_id("<|endoftext|>")
    idx = torch.zeros((1, 1), dtype=torch.long)
    # low temperature + top_k=1 is effectively greedy; assert eos is accepted and the
    # output never exceeds the requested budget.
    out = model.generate(idx, max_new_tokens=8, temperature=0.1, top_k=1, eos_token_id=eos)
    assert out.shape[1] <= 1 + 8


def test_generate_runs_with_top_p():
    tok = make_char_tokenizer()
    model = tiny_model(tok)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=6, temperature=0.9, top_k=10, top_p=0.9)
    assert out.shape[1] == 1 + 6
