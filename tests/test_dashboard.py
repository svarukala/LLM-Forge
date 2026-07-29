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
    ok_upload = _safe_data_path(os.path.join(RUNS_DIR, "uploads", "mine.txt"))
    assert ok_upload.endswith(os.path.join("uploads", "mine.txt"))


def test_safe_data_path_rejects_outside():
    with pytest.raises(ValueError):
        _safe_data_path(os.path.join("data", "secret.txt"))
    with pytest.raises(ValueError):
        _safe_data_path(os.path.join("runs", "base", "config.json"))


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
