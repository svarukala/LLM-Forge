"""LLM Forge web dashboard (Layer 3).

A small FastAPI app that lets you launch pre-training / fine-tuning runs, watch the loss
curve and samples stream in live, and chat with the model — all from the browser.

Run with:  python -m llmforge.cli serve
"""

import json
import os
import queue
import threading
from typing import Optional

from ..config import ModelConfig, TrainConfig, RUNS_DIR, pick_device


def _resolve_preset(params: dict, stage: str) -> dict:
    """Fill in any missing training knob from a named preset, if one was requested.

    Explicit values sent by the browser always win; the preset only supplies keys the
    request left unset. Mirrors the CLI's --preset behaviour so both front doors agree.
    """
    name = params.get("preset")
    if not name:
        return params
    try:
        from ..presets import get_preset
        p = get_preset(name)
    except Exception:
        return params

    if stage == "pretrain":
        mapping = {
            "tok_kind": p.tok_kind, "vocab_size": p.vocab_size,
            "block_size": p.block_size, "n_layer": p.n_layer, "n_head": p.n_head,
            "n_embd": p.n_embd, "batch_size": p.batch_size, "steps": p.pretrain_steps,
        }
    else:
        mapping = {"batch_size": p.batch_size, "steps": p.finetune_steps}

    merged = dict(params)
    for k, v in mapping.items():
        if merged.get(k) in (None, "", "auto"):
            merged[k] = v
    return merged


class TrainingManager:
    """Runs one training job at a time in a background thread and fans out events."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self.running = False
        self.history: list[dict] = []

    def is_running(self) -> bool:
        return self.running

    def stop(self) -> None:
        self._stop.set()

    def _emit(self, evt: dict) -> None:
        self.history.append(evt)
        self._queue.put(evt)

    def events(self):
        """Generator yielding queued events as Server-Sent Events."""
        # replay recent history so a late-joining browser catches up
        for evt in list(self.history):
            yield f"data: {json.dumps(evt)}\n\n"
        while self.running or not self._queue.empty():
            try:
                evt = self._queue.get(timeout=1.0)
                yield f"data: {json.dumps(evt)}\n\n"
            except queue.Empty:
                yield ": keep-alive\n\n"

    def start_pretrain(self, params: dict) -> bool:
        if self.running:
            return False
        self._stop.clear()
        self.history.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run_pretrain, args=(params,), daemon=True)
        self._thread.start()
        return True

    def start_finetune(self, params: dict) -> bool:
        if self.running:
            return False
        self._stop.clear()
        self.history.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run_finetune, args=(params,), daemon=True)
        self._thread.start()
        return True

    def _run_pretrain(self, params: dict) -> None:
        try:
            from ..model import GPT
            from ..data import load_text_file, PretrainData
            from ..tokenizer import train_tokenizer
            from ..train import train

            params = _resolve_preset(params, "pretrain")
            data_path = params.get("data", os.path.join("data", "sample", "corpus.txt"))
            text = load_text_file(data_path)
            device = pick_device(None)

            # Always (re)train the tokenizer to match the chosen dataset + kind, so the
            # dashboard's tokenizer dropdown and any uploaded corpus actually take effect.
            tok_path = os.path.join(RUNS_DIR, "tokenizer.json")
            tokenizer = train_tokenizer(text, kind=params.get("tok_kind", "char"),
                                        vocab_size=int(params.get("vocab_size", 2048)))
            tokenizer.save(tok_path)

            mcfg = ModelConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=int(params.get("block_size", 128)),
                n_layer=int(params.get("n_layer", 4)),
                n_head=int(params.get("n_head", 4)),
                n_embd=int(params.get("n_embd", 128)),
            )
            model = GPT(mcfg)
            ids = tokenizer.encode(text)
            dataset = PretrainData(ids, block_size=mcfg.block_size, device=device)
            tcfg = TrainConfig(
                batch_size=int(params.get("batch_size", 16)),
                block_size=mcfg.block_size,
                steps=int(params.get("steps", 500)),
                device=device,
                out_dir=os.path.join(RUNS_DIR, "base"),
            )
            train(model, dataset, tcfg, tokenizer=tokenizer, tokenizer_path=tok_path,
                  on_event=self._emit, stop_flag=self._stop.is_set)
        except Exception as exc:  # surface errors to the UI instead of dying silently
            self._emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.running = False

    def _run_finetune(self, params: dict) -> None:
        try:
            from ..data import SFTData
            from ..checkpoint import load_checkpoint
            from ..tokenizer import load_tokenizer
            from ..train import train

            params = _resolve_preset(params, "finetune")
            device = pick_device(None)
            base = params.get("base", os.path.join(RUNS_DIR, "base"))
            model, tok_path, _ = load_checkpoint(base, device=device)
            tokenizer = load_tokenizer(tok_path)
            data_path = params.get("data", os.path.join("data", "sample", "chat.jsonl"))
            dataset = SFTData(data_path, tokenizer, block_size=model.cfg.block_size, device=device)
            tcfg = TrainConfig(
                batch_size=int(params.get("batch_size", 8)),
                block_size=model.cfg.block_size,
                steps=int(params.get("steps", 300)),
                device=device,
                out_dir=os.path.join(RUNS_DIR, "chat"),
            )
            train(model, dataset, tcfg, tokenizer=tokenizer, tokenizer_path=tok_path,
                  on_event=self._emit, stop_flag=self._stop.is_set)
        except Exception as exc:
            self._emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.running = False


def create_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="LLM Forge")
    manager = TrainingManager()
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    @app.get("/api/status")
    def status():
        return {"running": manager.is_running(), "checkpoints": _list_checkpoints()}

    @app.get("/api/hardware")
    def hardware():
        from ..presets import detect_hardware, recommend_preset, PRESETS
        hw = detect_hardware()
        return {
            "hardware": hw.as_dict(),
            "recommended": recommend_preset(hw),
            "presets": {name: p.as_dict() for name, p in PRESETS.items()},
        }

    @app.post("/api/pretrain")
    async def pretrain(request: Request):
        params = await _json(request)
        ok = manager.start_pretrain(params)
        return JSONResponse({"started": ok}, status_code=200 if ok else 409)

    @app.post("/api/finetune")
    async def finetune(request: Request):
        params = await _json(request)
        ok = manager.start_finetune(params)
        return JSONResponse({"started": ok}, status_code=200 if ok else 409)

    @app.post("/api/stop")
    def stop():
        manager.stop()
        return {"stopping": True}

    @app.post("/api/upload")
    async def upload(request: Request):
        """Save an uploaded dataset (corpus .txt or chat .jsonl) sent as JSON text.

        Body: {"kind": "corpus"|"chat", "filename": "...", "content": "..."}
        Returns the server-side path to pass back in a pretrain/finetune request.
        """
        params = await _json(request)
        content = params.get("content", "")
        name = os.path.basename(params.get("filename", "") or "").strip() or "upload.txt"
        updir = os.path.join(RUNS_DIR, "uploads")
        os.makedirs(updir, exist_ok=True)
        path = os.path.join(updir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"path": path, "bytes": len(content.encode("utf-8"))}

    @app.get("/api/events")
    def events():
        return StreamingResponse(manager.events(), media_type="text/event-stream")

    @app.post("/api/chat")
    async def chat(request: Request):
        params = await _json(request)
        checkpoint = params.get("checkpoint", os.path.join(RUNS_DIR, "chat"))
        message = params.get("message", "")
        if not os.path.exists(os.path.join(checkpoint, "config.json")):
            return JSONResponse({"error": "No fine-tuned checkpoint yet. Pre-train then fine-tune first."},
                                status_code=400)
        from ..chat import ChatSession
        session = _chat_cache.get(checkpoint) or ChatSession(checkpoint)
        _chat_cache[checkpoint] = session
        return {"reply": session.reply(message)}

    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


_chat_cache: dict = {}


async def _json(request) -> dict:
    try:
        return await request.json()
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


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    print(f"LLM Forge dashboard -> http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
