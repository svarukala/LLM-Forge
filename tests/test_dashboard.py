"""Dashboard API: request validation, upload guards, concurrency, SSE fan-out, cache."""

from fastapi.testclient import TestClient

from llmforge.server.app import TrainingManager, create_app


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
