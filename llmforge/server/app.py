"""LLM Forge web dashboard.

A small FastAPI app that lets you launch pre-training / fine-tuning runs, watch the loss
curve and samples stream in live, and chat with the model — all from the browser.

Security note: this dashboard has **no authentication**. It binds to localhost (127.0.0.1)
by default. Do not expose it to a network you do not trust. Binding to a public interface
requires an explicit, clearly-unsafe flag (see ``run(allow_public=...)``).

Run with:  python -m llmforge.cli serve
"""

import json
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable

from pydantic import BaseModel, Field, field_validator

from ..config import RUNS_DIR, ModelConfig, TrainConfig, pick_device
from ..train import train

# ---------------------------------------------------------------------------
# Request models with strict bounds (reject nonsense before it reaches training)
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB is plenty for an educational corpus
_VALID_PRESETS = {"cpu", "gpu", "gpu-large"}
SAMPLE_DIR = os.path.join("data", "sample")  # bundled example datasets


class PretrainRequest(BaseModel):
    preset: str | None = None
    data: str | None = None
    tok_kind: str | None = None
    vocab_size: int | None = Field(default=None, ge=16, le=50000)
    block_size: int | None = Field(default=None, ge=8, le=2048)
    n_layer: int | None = Field(default=None, ge=1, le=48)
    n_head: int | None = Field(default=None, ge=1, le=32)
    n_embd: int | None = Field(default=None, ge=8, le=4096)
    batch_size: int | None = Field(default=None, ge=1, le=1024)
    steps: int | None = Field(default=None, ge=1, le=200000)
    reuse_tokenizer: bool = False

    @field_validator("preset")
    @classmethod
    def _preset_known(cls, v):
        if v not in (None, "", "auto") and v not in _VALID_PRESETS:
            raise ValueError(f"unknown preset {v!r}; choose one of {sorted(_VALID_PRESETS)}")
        return v

    @field_validator("tok_kind")
    @classmethod
    def _kind_known(cls, v):
        if v not in (None, "", "auto", "char", "bpe"):
            raise ValueError("tok_kind must be 'char' or 'bpe'")
        return v


class FinetuneRequest(BaseModel):
    preset: str | None = None
    data: str | None = None
    base: str | None = None
    batch_size: int | None = Field(default=None, ge=1, le=1024)
    steps: int | None = Field(default=None, ge=1, le=200000)

    @field_validator("preset")
    @classmethod
    def _preset_known(cls, v):
        if v not in (None, "", "auto") and v not in _VALID_PRESETS:
            raise ValueError(f"unknown preset {v!r}; choose one of {sorted(_VALID_PRESETS)}")
        return v


class UploadRequest(BaseModel):
    kind: str = "corpus"
    filename: str = "upload.txt"
    content: str = ""

    @field_validator("kind")
    @classmethod
    def _kind_ok(cls, v):
        if v not in ("corpus", "chat"):
            raise ValueError("kind must be 'corpus' or 'chat'")
        return v

    @field_validator("content")
    @classmethod
    def _size_ok(cls, v):
        if len(v.encode("utf-8")) > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
        return v


class ChatRequest(BaseModel):
    checkpoint: str | None = None
    message: str = Field(default="", max_length=8000)
    max_new_tokens: int = Field(default=200, ge=1, le=2000)
    # These bounds mirror GPT.generate()'s own validation so bad values are rejected with a
    # 422 at the API boundary instead of blowing up inside the model at generation time.
    temperature: float = Field(default=0.8, gt=0.0, le=5.0)
    top_k: int | None = Field(default=40, ge=1, le=100000)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


def _resolve_preset(params: dict, stage: str) -> dict:
    """Fill in any missing training knob from a named preset, if one was requested.

    Explicit values sent by the browser always win; the preset only supplies keys the
    request left unset. Mirrors the CLI's --preset behaviour so both front doors agree.
    """
    name = params.get("preset")
    if not name or name == "auto":
        return params
    try:
        from ..presets import get_preset

        p = get_preset(name)
    except Exception:
        return params

    if stage == "pretrain":
        mapping = {
            "tok_kind": p.tok_kind,
            "vocab_size": p.vocab_size,
            "block_size": p.block_size,
            "n_layer": p.n_layer,
            "n_head": p.n_head,
            "n_embd": p.n_embd,
            "batch_size": p.batch_size,
            "steps": p.pretrain_steps,
        }
    else:
        mapping = {"batch_size": p.batch_size, "steps": p.finetune_steps}

    merged = dict(params)
    for k, v in mapping.items():
        if merged.get(k) in (None, "", "auto"):
            merged[k] = v
    return merged


def _safe_run_path(*parts: str) -> str:
    """Join under RUNS_DIR and refuse anything that escapes it (path traversal guard)."""
    root = os.path.abspath(RUNS_DIR)
    candidate = os.path.abspath(os.path.join(root, *parts))
    if candidate != root and os.path.commonpath([candidate, root]) != root:
        raise ValueError("path escapes the runs directory")
    return candidate


def _under_runs(path: str) -> str:
    """Normalise an incoming checkpoint/base path to a safe path inside RUNS_DIR."""
    if os.path.isabs(path):
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(RUNS_DIR))
        return _safe_run_path(rel)
    return _safe_run_path(path)


def _safe_data_path(path: str) -> str:
    """Restrict a browser-supplied dataset path to trusted locations.

    A dashboard client must never be able to point training at an arbitrary server file.
    Only two roots are allowed: the bundled ``data/sample`` datasets and user uploads under
    ``runs/uploads`` (which the /api/upload endpoint writes with sanitized filenames). The
    path may arrive relative or absolute (uploads echo back an absolute path), but it must
    resolve *inside* one of those roots. Absolute paths elsewhere, ``..`` traversal, and any
    other location raise ValueError, which the endpoints translate to HTTP 422.
    """
    candidate = os.path.abspath(path)
    allowed_roots = [
        os.path.abspath(SAMPLE_DIR),
        os.path.abspath(os.path.join(RUNS_DIR, "uploads")),
    ]
    for root in allowed_roots:
        try:
            if candidate != root and os.path.commonpath([candidate, root]) == root:
                return candidate
        except ValueError:
            continue  # e.g. different drive on Windows -> not under this root
    raise ValueError(
        "data path must be a bundled sample under data/sample/ or an uploaded file "
        "under runs/uploads/"
    )


class TrainingManager:
    """Runs one training job at a time in a background thread and fans out events.

    Thread-safety: ``_lock`` guards all start/stop/status transitions. Live events are
    delivered to each SSE subscriber through its *own* bounded queue (true fan-out), and a
    bounded history buffer lets a late-joining browser catch up without unbounded growth.
    """

    MAX_HISTORY = 2000
    MAX_QUEUE = 4000

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._subscribers: list[queue.Queue[dict]] = []
        self.running = False
        self.history: list[dict] = []
        self.job_id: str | None = None
        self.job_kind: str | None = None
        self.started_at: float | None = None
        self.current_step: int = 0
        self.total_steps: int = 0
        self.error: str | None = None
        self.last_status: str = "idle"
        self._on_checkpoint_written: Callable[[str], None] | None = None

    # -- subscription / fan-out --------------------------------------------
    def subscribe(self) -> "queue.Queue[dict]":
        q: queue.Queue[dict] = queue.Queue(maxsize=self.MAX_QUEUE)
        with self._lock:
            self._subscribers.append(q)
            snapshot = list(self.history)
        for evt in snapshot:
            try:
                q.put_nowait(evt)
            except queue.Full:
                break
        return q

    def unsubscribe(self, q: "queue.Queue[dict]") -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def is_running(self) -> bool:
        with self._lock:
            return self.running

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            elapsed = (time.time() - self.started_at) if self.started_at else 0.0
            return {
                "running": self.running,
                "job_id": self.job_id,
                "job_kind": self.job_kind,
                "status": self.last_status,
                "elapsed_sec": round(elapsed, 1),
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "error": self.error,
                "checkpoints": _list_checkpoints(),
            }

    def _emit(self, evt: dict) -> None:
        with self._lock:
            self.history.append(evt)
            if len(self.history) > self.MAX_HISTORY:
                del self.history[: len(self.history) - self.MAX_HISTORY]
            if evt.get("type") == "step":
                self.current_step = int(evt.get("step", self.current_step))
            elif evt.get("type") == "error":
                self.error = evt.get("message")
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(evt)
            except queue.Full:
                pass  # slow consumer: drop rather than block the trainer

    def _begin(self, kind: str, total_steps: int) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._stop.clear()
            self.history.clear()
            self.running = True
            self.job_id = job_id
            self.job_kind = kind
            self.started_at = time.time()
            self.current_step = 0
            self.total_steps = total_steps
            self.error = None
            self.last_status = "running"
        return job_id

    def _finish(self, status: str) -> None:
        with self._lock:
            self.running = False
            self.last_status = status
            job_id = self.job_id
        self._emit({"type": "job_end", "status": status, "job_id": job_id})

    def start_pretrain(self, params: dict) -> str | None:
        with self._lock:
            if self.running:
                return None
            merged = _resolve_preset(params, "pretrain")
            total = int(merged.get("steps") or 0)
            job_id = self._begin("pretrain", total)
            self._thread = threading.Thread(target=self._run_pretrain, args=(merged,), daemon=True)
            self._thread.start()
            return job_id

    def start_finetune(self, params: dict) -> str | None:
        with self._lock:
            if self.running:
                return None
            merged = _resolve_preset(params, "finetune")
            total = int(merged.get("steps") or 0)
            job_id = self._begin("finetune", total)
            self._thread = threading.Thread(target=self._run_finetune, args=(merged,), daemon=True)
            self._thread.start()
            return job_id

    def _run_pretrain(self, params: dict) -> None:
        status = "error"
        try:
            from ..data import PretrainData, load_text_file
            from ..model import GPT
            from ..tokenizer import (
                load_tokenizer,
                load_tokenizer_meta,
                save_tokenizer_meta,
                text_fingerprint,
                train_tokenizer,
            )

            data_path = params.get("data") or os.path.join("data", "sample", "corpus.txt")
            text = load_text_file(data_path)
            device = pick_device(None)
            tok_kind = params.get("tok_kind") or "char"
            vocab_size = int(params.get("vocab_size") or 2048)
            fingerprint = text_fingerprint(text)

            os.makedirs(RUNS_DIR, exist_ok=True)
            tok_path = os.path.join(RUNS_DIR, "tokenizer.json")
            existing = load_tokenizer_meta(tok_path)
            reuse = bool(params.get("reuse_tokenizer"))
            compatible = (
                existing is not None
                and existing.get("kind") == tok_kind
                and existing.get("corpus_fingerprint") == fingerprint
                and (tok_kind == "char" or existing.get("vocab_size") == vocab_size)
            )
            if reuse and compatible:
                tokenizer = load_tokenizer(tok_path)
                self._emit({"type": "info", "message": "Reusing existing compatible tokenizer."})
            else:
                if reuse and existing is not None and not compatible:
                    self._emit(
                        {
                            "type": "info",
                            "message": "Requested tokenizer differs from cached one; retraining it.",
                        }
                    )
                tokenizer = train_tokenizer(text, kind=tok_kind, vocab_size=vocab_size)
                tokenizer.save(tok_path)
                save_tokenizer_meta(tok_path, tok_kind, tokenizer.vocab_size, fingerprint)

            mcfg = ModelConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=int(params.get("block_size") or 128),
                n_layer=int(params.get("n_layer") or 4),
                n_head=int(params.get("n_head") or 4),
                n_embd=int(params.get("n_embd") or 128),
            )
            model = GPT(mcfg)
            ids = tokenizer.encode(text)
            dataset = PretrainData(ids, block_size=mcfg.block_size, device=device)
            out_dir = os.path.join(RUNS_DIR, "base")
            tcfg = TrainConfig(
                batch_size=int(params.get("batch_size") or 16),
                block_size=mcfg.block_size,
                steps=int(params.get("steps") or 500),
                device=device,
                out_dir=out_dir,
            )
            extra = {
                "tokenizer_kind": tok_kind,
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "corpus_fingerprint": fingerprint,
                "dataset": os.path.basename(data_path),
            }
            result = train(
                model,
                dataset,
                tcfg,
                tokenizer=tokenizer,
                tokenizer_path=tok_path,
                on_event=self._emit,
                stop_flag=self._stop.is_set,
                extra_meta=extra,
            )
            status = result.get("status", "completed")
            self._invalidate_checkpoint(out_dir)
        except Exception as exc:  # surface errors to the UI instead of dying silently
            self._emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self._finish(status)

    def _run_finetune(self, params: dict) -> None:
        status = "error"
        try:
            from ..checkpoint import load_checkpoint
            from ..data import SFTData
            from ..tokenizer import load_tokenizer

            device = pick_device(None)
            base = _under_runs(params.get("base") or os.path.join(RUNS_DIR, "base"))
            if not os.path.exists(os.path.join(base, "config.json")):
                raise FileNotFoundError("No base checkpoint yet. Pre-train first.")
            model, tok_path, _ = load_checkpoint(base, device=device)
            if not tok_path:
                raise ValueError("Base checkpoint has no tokenizer.")
            tokenizer = load_tokenizer(tok_path)
            data_path = params.get("data") or os.path.join("data", "sample", "chat.jsonl")
            dataset = SFTData(data_path, tokenizer, block_size=model.cfg.block_size, device=device)
            if getattr(dataset, "dropped", 0):
                self._emit(
                    {
                        "type": "info",
                        "message": f"Dropped {dataset.dropped} example(s) that could not fit the context.",
                    }
                )
            out_dir = os.path.join(RUNS_DIR, "chat")
            tcfg = TrainConfig(
                batch_size=int(params.get("batch_size") or 8),
                block_size=model.cfg.block_size,
                steps=int(params.get("steps") or 300),
                device=device,
                out_dir=out_dir,
            )
            result = train(
                model,
                dataset,
                tcfg,
                tokenizer=tokenizer,
                tokenizer_path=tok_path,
                on_event=self._emit,
                stop_flag=self._stop.is_set,
            )
            status = result.get("status", "completed")
            self._invalidate_checkpoint(out_dir)
        except Exception as exc:
            self._emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self._finish(status)

    def _invalidate_checkpoint(self, out_dir: str) -> None:
        if self._on_checkpoint_written is not None:
            try:
                self._on_checkpoint_written(out_dir)
            except Exception:
                pass


def create_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import ValidationError

    app = FastAPI(title="LLM Forge")
    manager = TrainingManager()
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    chat_cache: dict = {}
    chat_cache_lock = threading.Lock()

    def _invalidate(out_dir: str) -> None:
        key = os.path.abspath(out_dir)
        with chat_cache_lock:
            chat_cache.pop(key, None)

    manager._on_checkpoint_written = _invalidate
    app.state.manager = manager
    app.state.chat_cache = chat_cache

    @app.get("/api/status")
    def status():
        return manager.status()

    @app.get("/api/hardware")
    def hardware():
        from ..presets import PRESETS, detect_hardware, recommend_preset

        hw = detect_hardware()
        return {
            "hardware": hw.as_dict(),
            "recommended": recommend_preset(hw),
            "presets": {name: p.as_dict() for name, p in PRESETS.items()},
        }

    @app.post("/api/pretrain")
    async def pretrain(request: Request):
        try:
            req = PretrainRequest(**await _json(request))
        except ValidationError as e:
            return JSONResponse({"error": json.loads(e.json())}, status_code=422)
        params = req.model_dump()
        if params.get("data"):
            try:
                params["data"] = _safe_data_path(params["data"])
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=422)
        job_id = manager.start_pretrain(params)
        if job_id is None:
            return JSONResponse({"error": "A training job is already running."}, status_code=409)
        return JSONResponse({"started": True, "job_id": job_id})

    @app.post("/api/finetune")
    async def finetune(request: Request):
        try:
            req = FinetuneRequest(**await _json(request))
        except ValidationError as e:
            return JSONResponse({"error": json.loads(e.json())}, status_code=422)
        params = req.model_dump()
        if params.get("data"):
            try:
                params["data"] = _safe_data_path(params["data"])
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=422)
        job_id = manager.start_finetune(params)
        if job_id is None:
            return JSONResponse({"error": "A training job is already running."}, status_code=409)
        return JSONResponse({"started": True, "job_id": job_id})

    @app.post("/api/stop")
    def stop():
        manager.stop()
        return {"stopping": True}

    @app.post("/api/upload")
    async def upload(request: Request):
        """Save an uploaded dataset (corpus .txt or chat .jsonl) sent as JSON text."""
        try:
            req = UploadRequest(**await _json(request))
        except ValidationError as e:
            return JSONResponse({"error": json.loads(e.json())}, status_code=422)
        raw_name = req.filename or ""
        if (
            (".." in raw_name)
            or ("/" in raw_name)
            or ("\\" in raw_name)
            or os.path.basename(raw_name) != raw_name.strip()
        ):
            return JSONResponse({"error": "invalid filename (no paths allowed)"}, status_code=422)
        name = os.path.basename(raw_name).strip() or "upload.txt"
        if name.startswith("."):
            return JSONResponse({"error": "invalid filename"}, status_code=422)
        try:
            updir = _safe_run_path("uploads")
            os.makedirs(updir, exist_ok=True)
            path = _safe_run_path("uploads", name)
        except ValueError:
            return JSONResponse({"error": "invalid upload path"}, status_code=422)
        with open(path, "w", encoding="utf-8") as f:
            f.write(req.content)
        return {"path": path, "bytes": len(req.content.encode("utf-8"))}

    @app.get("/api/events")
    def events():
        return StreamingResponse(_event_stream(manager), media_type="text/event-stream")

    @app.post("/api/chat")
    async def chat(request: Request):
        try:
            req = ChatRequest(**await _json(request))
        except ValidationError as e:
            return JSONResponse({"error": json.loads(e.json())}, status_code=422)
        raw = req.checkpoint or os.path.join(RUNS_DIR, "chat")
        try:
            checkpoint = _under_runs(raw)
        except ValueError:
            return JSONResponse({"error": "invalid checkpoint path"}, status_code=422)
        if not os.path.exists(os.path.join(checkpoint, "config.json")):
            return JSONResponse(
                {"error": "No fine-tuned checkpoint yet. Pre-train then fine-tune first."},
                status_code=400,
            )
        from ..chat import ChatSession

        key = os.path.abspath(checkpoint)
        with chat_cache_lock:
            session = chat_cache.get(key)
            if session is None:
                session = ChatSession(checkpoint)
                chat_cache[key] = session
        reply = session.reply(
            req.message,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        return {"reply": reply}

    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


def _event_stream(manager: "TrainingManager"):
    """Per-subscriber SSE generator: each client gets its own queue and independent stream."""
    q = manager.subscribe()
    try:
        while True:
            try:
                evt = q.get(timeout=1.0)
                yield f"data: {json.dumps(evt)}\n\n"
            except queue.Empty:
                yield ": keep-alive\n\n"
    finally:
        manager.unsubscribe(q)


async def _json(request) -> dict:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _list_checkpoints() -> list[str]:
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for name in os.listdir(RUNS_DIR):
        if os.path.exists(os.path.join(RUNS_DIR, name, "config.json")):
            out.append(name)
    return sorted(out)


def run(host: str = "127.0.0.1", port: int = 8000, allow_public: bool = False) -> None:
    import uvicorn

    is_loopback = host in ("127.0.0.1", "localhost", "::1")
    if not is_loopback and not allow_public:
        raise SystemExit(
            f"Refusing to bind to {host!r}: the dashboard has NO authentication.\n"
            f"Re-run with --allow-public if you really intend to expose it (UNSAFE)."
        )
    if not is_loopback and allow_public:
        print("*" * 70)
        print("WARNING: binding beyond localhost with NO authentication.")
        print("Anyone who can reach this host can train models and read your data.")
        print("*" * 70)
    print(f"LLM Forge dashboard -> http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
