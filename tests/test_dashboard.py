"""Dashboard API: request validation, upload guards, concurrency, SSE fan-out, cache."""

import os

import pytest
from fastapi.testclient import TestClient

from llmforge.server.app import (
    TrainingManager,
    _fit_block_size,
    _safe_data_path,
    create_app,
)


def client():
    return TestClient(create_app())


def test_status_and_hardware_ok():
    c = client()
    assert c.get("/api/status").status_code == 200
    assert c.get("/api/hardware").status_code == 200


def test_pretrain_rejects_invalid_config():
    c = client()
    assert c.post("/api/pretrain", json={"steps": -1}).status_code == 422
    assert c.post("/api/pretrain", json={"n_layer": 9999}).status_code == 422
    assert c.post("/api/pretrain", json={"preset": "does-not-exist"}).status_code == 422
    assert c.post("/api/pretrain", json={"tok_kind": "weird"}).status_code == 422


def test_pretrain_rejects_incompatible_head_embd_synchronously():
    """n_embd must be divisible by n_head and n_head <= n_embd. Bad combos must be rejected
    with a 422 up front, never spawning a background job that fails later."""
    from llmforge.server.app import _validate_model_config

    c = client()
    mgr = c.app.state.manager

    # n_embd not divisible by n_head
    r = c.post("/api/pretrain", json={"steps": 5, "n_embd": 30, "n_head": 4})
    assert r.status_code == 422
    assert "divisible" in r.json()["error"]
    assert mgr.running is False  # no job was started

    # n_head larger than n_embd
    r = c.post("/api/pretrain", json={"steps": 5, "n_embd": 8, "n_head": 16})
    assert r.status_code == 422
    assert mgr.running is False

    # the helper accepts a valid geometry without raising
    _validate_model_config({"n_embd": 32, "n_head": 4})
    _validate_model_config({})  # defaults (128 / 4) are valid
    with pytest.raises(ValueError):
        _validate_model_config({"n_embd": 30, "n_head": 4})


def test_upload_rejects_traversal_and_oversize():
    c = client()
    assert (
        c.post("/api/upload", json={"filename": "../evil.txt", "content": "x"}).status_code == 422
    )
    assert c.post("/api/upload", json={"filename": "a/b.txt", "content": "x"}).status_code == 422
    big = "x" * (26 * 1024 * 1024)
    assert c.post("/api/upload", json={"filename": "ok.txt", "content": big}).status_code == 422
    ok = c.post("/api/upload", json={"filename": "ok.txt", "content": "hello"})
    assert ok.status_code == 200
    assert ok.json()["bytes"] == 5


def test_upload_requires_matching_extension():
    c = client()
    # corpus must be .txt
    assert (
        c.post(
            "/api/upload", json={"kind": "corpus", "filename": "data.jsonl", "content": "x"}
        ).status_code
        == 422
    )
    # chat must be .jsonl
    assert (
        c.post(
            "/api/upload", json={"kind": "chat", "filename": "pairs.txt", "content": "x"}
        ).status_code
        == 422
    )
    # matching extensions are accepted
    assert (
        c.post(
            "/api/upload", json={"kind": "corpus", "filename": "c.txt", "content": "hi"}
        ).status_code
        == 200
    )
    assert (
        c.post(
            "/api/upload",
            json={
                "kind": "chat",
                "filename": "p.jsonl",
                "content": '{"prompt":"a","response":"b"}',
            },
        ).status_code
        == 200
    )


def test_chat_rejects_path_traversal():
    c = client()
    r = c.post("/api/chat", json={"checkpoint": "../../etc/passwd", "message": "hi"})
    assert r.status_code == 422


def test_chat_rejects_out_of_range_sampling():
    """API sampling bounds must mirror GPT.generate() so bad values 422 instead of crashing."""
    c = client()
    assert c.post("/api/chat", json={"message": "hi", "temperature": 0}).status_code == 422
    assert c.post("/api/chat", json={"message": "hi", "top_k": 0}).status_code == 422
    assert c.post("/api/chat", json={"message": "hi", "top_p": 0}).status_code == 422
    assert c.post("/api/chat", json={"message": "hi", "top_p": 1.5}).status_code == 422


def test_sample_rejects_out_of_range_sampling():
    """/api/sample shares GPT.generate()'s bounds — bad values must 422, not crash later."""
    c = client()
    assert c.post("/api/sample", json={"temperature": 0}).status_code == 422
    assert c.post("/api/sample", json={"top_k": 0}).status_code == 422
    assert c.post("/api/sample", json={"top_p": 1.5}).status_code == 422


def test_sample_without_base_checkpoint_returns_400(monkeypatch, tmp_path):
    from llmforge.server import app as app_module

    monkeypatch.setattr(app_module, "RUNS_DIR", str(tmp_path))
    c = client()
    r = c.post("/api/sample", json={"prompt": "hi"})
    assert r.status_code == 400
    assert "Pre-train first" in r.json()["error"]


def test_sample_generates_from_base(monkeypatch, tmp_path, base_checkpoint):
    """Happy path: with a base checkpoint present, /api/sample returns generated text."""
    from llmforge.server import app as app_module

    # base_checkpoint writes to tmp_path/base; point the server's RUNS_DIR at tmp_path.
    monkeypatch.setattr(app_module, "RUNS_DIR", str(tmp_path))
    c = client()
    r = c.post("/api/sample", json={"prompt": "the", "max_new_tokens": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["text"], str)
    assert body["checkpoint"] == "base"


def test_sample_rejects_arbitrary_checkpoint_path():
    c = client()
    r = c.post("/api/sample", json={"checkpoint": "../../etc/passwd", "prompt": "hi"})
    assert r.status_code == 422


def test_tokenize_char_splits_each_character():
    c = client()
    r = c.post("/api/tokenize", json={"text": "hello", "kind": "char"})
    assert r.status_code == 200
    d = r.json()
    assert d["kind"] == "char"
    assert d["count"] == 5
    assert [t["piece"] for t in d["tokens"]] == list("hello")


def test_tokenize_rejects_unknown_kind():
    c = client()
    assert c.post("/api/tokenize", json={"text": "hi", "kind": "nope"}).status_code == 422


def test_tokenize_bpe_without_checkpoint_returns_400(monkeypatch, tmp_path):
    from llmforge.server import app as app_module

    monkeypatch.setattr(app_module, "RUNS_DIR", str(tmp_path))
    c = client()
    r = c.post("/api/tokenize", json={"text": "hello", "kind": "bpe"})
    assert r.status_code == 400
    assert "bpe" in r.json()["error"].lower()


def test_tokenize_bpe_uses_checkpoint_tokenizer(monkeypatch, tmp_path):
    from llmforge.server import app as app_module
    from llmforge.tokenizer import save_tokenizer_meta, text_fingerprint, train_tokenizer

    monkeypatch.setattr(app_module, "RUNS_DIR", str(tmp_path))
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    tok = train_tokenizer("the cat sat on the mat. " * 40, kind="bpe", vocab_size=200)
    tok_path = str(base / "tokenizer.json")
    tok.save(tok_path)
    save_tokenizer_meta(tok_path, "bpe", tok.vocab_size, text_fingerprint("x"))

    c = client()
    r = c.post("/api/tokenize", json={"text": "the cat", "kind": "bpe"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["kind"] == "bpe"
    assert d["count"] >= 1
    assert "base" in d["source"]


def test_checkpoints_empty_when_no_runs(monkeypatch, tmp_path):
    from llmforge.server import app as app_module

    monkeypatch.setattr(app_module, "RUNS_DIR", str(tmp_path))
    c = client()
    d = c.get("/api/checkpoints").json()
    assert d["checkpoints"] == []
    assert d["has_base"] is False
    assert d["has_chat"] is False


def test_checkpoints_reports_base_summary(monkeypatch, tmp_path, base_checkpoint):
    """With a trained base checkpoint present, /api/checkpoints surfaces its metadata."""
    from llmforge.server import app as app_module

    monkeypatch.setattr(app_module, "RUNS_DIR", str(tmp_path))
    c = client()
    d = c.get("/api/checkpoints").json()
    assert d["has_base"] is True
    assert d["has_chat"] is False
    base = next(s for s in d["checkpoints"] if s["name"] == "base")
    assert base["role"] == "Pre-trained base"
    assert base["n_embd"] == 32
    assert base["tokenizer_kind"] == "char"
    assert base["steps"] == 10
    assert isinstance(base["final_loss"], float)


def test_pretrain_rejects_arbitrary_data_path(tmp_path):
    c = client()
    for bad in ("/etc/passwd", "..\\..\\secrets.txt", "C:\\Windows\\win.ini", "runs\\base"):
        r = c.post("/api/pretrain", json={"data": bad, "steps": 5})
        assert r.status_code == 422, bad


def test_finetune_rejects_arbitrary_data_path():
    c = client()
    for bad in ("/etc/passwd", "../../secrets.jsonl", "runs\\chat\\config.json"):
        r = c.post("/api/finetune", json={"data": bad, "steps": 5})
        assert r.status_code == 422, bad


def test_safe_data_path_accepts_sample_and_uploads():
    import os

    from llmforge.config import RUNS_DIR

    ok_sample = _safe_data_path(os.path.join("data", "sample", "corpus.txt"))
    assert ok_sample.endswith(os.path.join("data", "sample", "corpus.txt"))
    # An uploaded file is accepted only once it actually exists on disk.
    updir = os.path.join(RUNS_DIR, "uploads")
    os.makedirs(updir, exist_ok=True)
    mine = os.path.join(updir, "mine.txt")
    with open(mine, "w", encoding="utf-8") as f:
        f.write("hello")
    try:
        ok_upload = _safe_data_path(mine)
        assert ok_upload.endswith(os.path.join("uploads", "mine.txt"))
    finally:
        os.remove(mine)


def test_safe_data_path_rejects_missing_file_and_directory():
    import os

    from llmforge.config import RUNS_DIR

    # A path inside an allowed root that does not exist is rejected.
    with pytest.raises(ValueError, match="not found or not a regular file"):
        _safe_data_path(os.path.join(RUNS_DIR, "uploads", "does-not-exist.txt"))
    # A directory (even the allowed sample dir itself) is not a regular file.
    with pytest.raises(ValueError):
        _safe_data_path(os.path.join("data", "sample"))


def test_safe_data_path_rejects_outside():
    with pytest.raises(ValueError):
        _safe_data_path(os.path.join("data", "secret.txt"))
    with pytest.raises(ValueError):
        _safe_data_path(os.path.join("runs", "base", "config.json"))


def test_under_runs_default_base_not_double_joined(monkeypatch, tmp_path):
    """Regression: the fine-tune default base must resolve to <RUNS_DIR>/base, not
    <RUNS_DIR>/runs/base, when RUNS_DIR is the default *relative* 'runs' path."""
    from llmforge.server import app as app_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module, "RUNS_DIR", "runs")
    expected = os.path.abspath(os.path.join("runs", "base"))  # where pretrain writes
    # The default the fine-tune endpoint computes:
    assert app_module._under_runs(os.path.join("runs", "base")) == expected
    # A bare checkpoint name is still interpreted relative to RUNS_DIR:
    assert app_module._under_runs("base") == expected
    # An absolute path inside RUNS_DIR round-trips too:
    assert app_module._under_runs(expected) == expected


def test_concurrent_job_rejected():
    app = create_app()
    c = TestClient(app)
    app.state.manager.running = True  # simulate a job already in flight
    r = c.post("/api/pretrain", json={"steps": 5})
    assert r.status_code == 409
    r2 = c.post("/api/finetune", json={"steps": 5})
    assert r2.status_code == 409


def test_multi_subscriber_fanout():
    mgr = TrainingManager()
    a = mgr.subscribe()
    b = mgr.subscribe()
    mgr._emit({"type": "step", "step": 1, "loss": 0.5})
    ea = a.get_nowait()
    eb = b.get_nowait()
    assert ea == eb == {"type": "step", "step": 1, "loss": 0.5}
    # unsubscribing one must not affect the other
    mgr.unsubscribe(a)
    mgr._emit({"type": "step", "step": 2, "loss": 0.4})
    assert b.get_nowait()["step"] == 2
    assert a.empty()


def test_history_is_bounded():
    mgr = TrainingManager()
    mgr.MAX_HISTORY = 10
    for i in range(50):
        mgr._emit({"type": "step", "step": i})
    assert len(mgr.history) <= 10


def test_cache_invalidation_clears_session():
    app = create_app()
    cache = app.state.chat_cache
    mgr = app.state.manager
    key = "C:\\fake\\runs\\chat"
    import os

    cache[os.path.abspath(key)] = object()
    mgr._invalidate_checkpoint(key)
    assert os.path.abspath(key) not in cache


def test_fit_block_size_shrinks_for_small_corpus():
    # Large corpus: keep the requested window, no shrink.
    assert _fit_block_size(10_000, 128) == (128, False)
    # Small corpus (126 tokens): largest fitting window is 126//2 - 1 = 62, and it shrank.
    assert _fit_block_size(126, 128) == (62, True)
    # Exactly enough for the requested window: no shrink.
    assert _fit_block_size(2 * (128 + 1), 128) == (128, False)
    # Tinier than the floor still fits: PretrainData will raise its clear error, not us.
    bs, shrunk = _fit_block_size(10, 128, floor=8)
    assert bs == 8 and shrunk is False
