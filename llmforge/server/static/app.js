// LLM Forge dashboard — talks to the FastAPI backend over JSON + Server-Sent Events.

const $ = (id) => document.getElementById(id);
const lossPoints = [];   // {step, train, val}
let PRESETS = {};        // filled from /api/hardware
let totalSteps = 0;      // total steps for the running job (for the progress bar)

function setStat(id, val) { $(id).textContent = val; }

// ---- Loss chart (tiny hand-rolled canvas plot, no libraries) ----
function drawChart() {
  const c = $("loss-chart");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 30;
  ctx.clearRect(0, 0, W, H);
  if (lossPoints.length < 2) return;

  const steps = lossPoints.map(p => p.step);
  const all = lossPoints.flatMap(p => [p.train, p.val].filter(v => v != null));
  const minX = Math.min(...steps), maxX = Math.max(...steps);
  const minY = Math.min(...all), maxY = Math.max(...all);
  const x = s => pad + (W - 2 * pad) * (s - minX) / Math.max(1, maxX - minX);
  const y = v => H - pad - (H - 2 * pad) * (v - minY) / Math.max(1e-6, maxY - minY);

  // axes
  ctx.strokeStyle = "#2a323c"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke();

  const line = (key, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    for (const p of lossPoints) {
      if (p[key] == null) continue;
      const px = x(p.step), py = y(p[key]);
      started ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      started = true;
    }
    ctx.stroke();
  };
  line("train", "#6ea8fe");
  line("val", "#7ee0c0");

  ctx.fillStyle = "#8a97a6"; ctx.font = "11px sans-serif";
  ctx.fillText(`loss ${maxY.toFixed(2)}`, 4, pad + 4);
  ctx.fillText(minY.toFixed(2), 4, H - pad);
  ctx.fillStyle = "#6ea8fe"; ctx.fillText("train", W - 90, pad + 4);
  ctx.fillStyle = "#7ee0c0"; ctx.fillText("val", W - 45, pad + 4);
}

// ---- Hardware detection + presets ----
async function loadHardware() {
  try {
    const r = await fetch("/api/hardware");
    const d = await r.json();
    PRESETS = d.presets || {};
    const hw = d.hardware || {};
    const rec = d.recommended;

    let line;
    if (hw.device === "cuda") {
      const mem = hw.gpu_mem_gb ? `${hw.gpu_mem_gb} GB` : "unknown mem";
      line = `✅ CUDA GPU detected: ${hw.gpu_name} (${mem}). You can run the GPU presets.`;
    } else if (hw.device === "mps") {
      line = `🍎 Apple Silicon (MPS) detected: ${hw.gpu_name}. CPU preset recommended for reliability.`;
    } else {
      line = "💻 No GPU detected (CPU only). The CPU preset is coherent but topic-following is loose — that's the scale lesson.";
    }
    $("hw-line").textContent = line;

    const sel = $("preset");
    for (const [name, p] of Object.entries(PRESETS)) {
      const opt = document.createElement("option");
      const star = name === rec ? " ★ recommended" : "";
      const gated = p.tier === "gpu" && hw.device !== "cuda" ? " (needs CUDA)" : "";
      opt.value = name;
      opt.textContent = `${name} · ~${Math.round(p.approx_params_m)}M params · ${p.est_time}${star}${gated}`;
      sel.appendChild(opt);
    }
    if (rec) { sel.value = rec; applyPreset(rec); }
  } catch {
    $("hw-line").textContent = "Could not detect hardware; using custom knobs.";
  }
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) { $("preset-blurb").textContent = ""; return; }
  $("pt-steps").value = p.pretrain_steps;
  $("pt-layers").value = p.n_layer;
  $("pt-embd").value = p.n_embd;
  $("pt-tok").value = p.tok_kind;
  $("ft-steps").value = p.finetune_steps;
  $("preset-blurb").textContent = p.blurb;
}

$("preset").onchange = () => applyPreset($("preset").value);

// ---- Training UI state (button locking + progress bar) ----
function setTraining(running) {
  // While a job runs, both "start" buttons are disabled so you can't launch a second one;
  // the Stop button is only usable while something is actually running.
  $("btn-pretrain").disabled = running;
  $("btn-finetune").disabled = running;
  $("btn-stop").disabled = !running;
}

function formatEta(sec) {
  if (sec == null || !isFinite(sec) || sec < 0) return "—";
  sec = Math.round(sec);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

function formatLr(lr) {
  if (lr == null || !isFinite(lr)) return "—";
  return lr.toExponential(1);  // e.g. 3.0e-4
}

function showProgress(show) {
  $("progress-wrap").hidden = !show;
}

function showError(message) {
  $("error-message").textContent = message || "Unknown error.";
  $("error-banner").hidden = false;
  $("error-banner").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideError() {
  $("error-banner").hidden = true;
  $("error-message").textContent = "";
}

function showInfo(message) {
  $("info-message").textContent = message || "";
  $("info-banner").hidden = false;
}

function hideInfo() {
  $("info-banner").hidden = true;
  $("info-message").textContent = "";
}

function setProgress(step, total) {
  const bar = $("progress-bar");
  if (!total || total <= 0) {
    // We don't know the length yet — show an animated indeterminate bar.
    bar.classList.add("indeterminate");
    $("progress-label").textContent = step ? `step ${step}` : "starting…";
    return;
  }
  bar.classList.remove("indeterminate");
  const pct = Math.max(0, Math.min(100, (step / total) * 100));
  bar.style.width = `${pct}%`;
  $("progress-label").textContent = `${Math.round(pct)}%  (${step}/${total})`;
}

// ---- Event stream ----
function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = (e) => {
    let evt;
    try { evt = JSON.parse(e.data); } catch { return; }
    handleEvent(evt);
  };
}

function handleEvent(evt) {
  switch (evt.type) {
    case "start":
      setStat("stat-status", "training");
      setStat("stat-device", evt.device);
      setStat("stat-eta", "—");
      setStat("stat-ppl", "—");
      setStat("stat-lr", "—");
      hideError();
      lossPoints.length = 0;
      $("samples").textContent = "";
      totalSteps = evt.steps || 0;
      setTraining(true);
      showProgress(true);
      setProgress(evt.start_step || 0, totalSteps);
      break;
    case "step":
      setStat("stat-step", evt.step);
      setStat("stat-loss", evt.loss.toFixed(3));
      if (evt.perplexity != null) setStat("stat-ppl", evt.perplexity.toFixed(1));
      if (evt.lr != null) setStat("stat-lr", formatLr(evt.lr));
      setStat("stat-tps", Math.round(evt.tok_per_sec));
      setStat("stat-eta", formatEta(evt.eta_s));
      setProgress(evt.step, totalSteps);
      pushPoint(evt.step, evt.loss, null);
      break;
    case "eval":
      setStat("stat-val", evt.val.toFixed(3));
      if (evt.val_perplexity != null) setStat("stat-ppl", evt.val_perplexity.toFixed(1));
      pushPoint(evt.step, evt.train, evt.val);
      break;
    case "sample":
      $("samples").textContent += `[step ${evt.step}] ${evt.text}\n\n`;
      $("samples").scrollTop = $("samples").scrollHeight;
      break;
    case "done":
      setStat("stat-status", "done ✓");
      setStat("stat-eta", "—");
      setProgress(totalSteps, totalSteps);
      setTraining(false);
      refreshStatus();
      refreshModels();
      break;
    case "stopped":
      setStat("stat-status", "stopped");
      setStat("stat-eta", "—");
      $("progress-bar").classList.remove("indeterminate");
      setTraining(false);
      refreshStatus();
      refreshModels();
      break;
    case "error":
      setStat("stat-status", "error");
      setStat("stat-eta", "—");
      $("progress-bar").classList.remove("indeterminate");
      showProgress(false);
      setTraining(false);
      showError(evt.message);
      $("samples").textContent += `\n[ERROR] ${evt.message}\n`;
      break;
    case "info":
      showInfo(evt.message);
      break;
  }
}

function pushPoint(step, train, val) {
  const existing = lossPoints.find(p => p.step === step);
  if (existing) {
    if (train != null) existing.train = train;
    if (val != null) existing.val = val;
  } else {
    lossPoints.push({ step, train, val });
  }
  drawChart();
}

// ---- Actions ----
async function post(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r;
}

// ---- Dataset upload ----
let uploadedCorpus = null;   // server-side path once a .txt is uploaded
let uploadedChat = null;     // server-side path once a .jsonl is uploaded

async function uploadFile(fileInput) {
  const file = fileInput.files[0];
  if (!file) return null;
  const content = await file.text();
  const r = await post("/api/upload", { filename: file.name, content });
  const data = await r.json();
  return data.path;
}

$("pt-file").onchange = async () => {
  const el = $("pt-file-name");
  el.textContent = "Uploading…";
  uploadedCorpus = await uploadFile($("pt-file"));
  el.textContent = uploadedCorpus
    ? `Using your corpus: ${$("pt-file").files[0].name}`
    : "Using built-in sample corpus.";
};

$("ft-file").onchange = async () => {
  const el = $("ft-file-name");
  el.textContent = "Uploading…";
  uploadedChat = await uploadFile($("ft-file"));
  el.textContent = uploadedChat
    ? `Using your chat data: ${$("ft-file").files[0].name}`
    : "Using built-in sample chat pairs.";
};

$("btn-pretrain").onclick = async () => {
  const body = {
    steps: +$("pt-steps").value,
    n_layer: +$("pt-layers").value,
    n_embd: +$("pt-embd").value,
    tok_kind: $("pt-tok").value,
  };
  const preset = $("preset").value;
  if (preset) body.preset = preset;
  if (uploadedCorpus) body.data = uploadedCorpus;
  hideError();
  hideInfo();
  setTraining(true);                 // lock immediately so a fast double-click can't double-start
  showProgress(true);
  setProgress(0, +$("pt-steps").value);
  const r = await post("/api/pretrain", body);
  if (!r.ok) { await reportStartError(r); setTraining(false); }
};

$("btn-finetune").onclick = async () => {
  const body = { steps: +$("ft-steps").value };
  const preset = $("preset").value;
  if (preset) body.preset = preset;
  if (uploadedChat) body.data = uploadedChat;
  hideError();
  hideInfo();
  setTraining(true);
  showProgress(true);
  setProgress(0, +$("ft-steps").value);
  const r = await post("/api/finetune", body);
  if (!r.ok) { await reportStartError(r); setTraining(false); }
};

async function reportStartError(r) {
  showProgress(false);
  if (r.status === 409) { showError("A training job is already running. Stop it before starting another."); return; }
  let detail = `Could not start (HTTP ${r.status}).`;
  try {
    const d = await r.json();
    const raw = d && (d.error ?? d.detail);
    if (raw != null) detail = typeof raw === "string" ? raw : JSON.stringify(raw);
  } catch {}
  showError(detail);
}

$("btn-stop").onclick = () => post("/api/stop", {});

$("chat-form").onsubmit = async (e) => {
  e.preventDefault();
  const input = $("chat-input");
  const msg = input.value.trim();
  if (!msg) return;
  addMsg("you", msg);
  input.value = "";
  const body = { message: msg };
  const temp = parseFloat($("chat-temp").value);
  if (isFinite(temp) && temp > 0) body.temperature = temp;
  const topk = parseInt($("chat-topk").value, 10);
  if (Number.isInteger(topk) && topk >= 1) body.top_k = topk;
  const topp = parseFloat($("chat-topp").value);
  if (isFinite(topp) && topp > 0 && topp <= 1) body.top_p = topp;
  const r = await post("/api/chat", body);
  const data = await r.json();
  addMsg("bot", r.ok ? (data.reply || "(no reply)") : (data.error || "error"));
  if (r.ok && data.context) updateContext(data.context);
};

function updateContext(ctx) {
  const block = ctx.block_size || 0;
  const used = (ctx.prompt_tokens || 0) + (ctx.generated_tokens || 0);
  if (!block) return;
  const pct = Math.min(100, Math.round((used / block) * 100));
  const wrap = $("chat-context");
  const bar = $("context-bar");
  bar.style.width = `${pct}%`;
  bar.classList.toggle("full", pct >= 90);
  $("context-label").textContent =
    `context: ${used} / ${block} tokens (${pct}%) — prompt ${ctx.prompt_tokens}, reply ${ctx.generated_tokens}`;
  wrap.hidden = false;
}

function addMsg(who, text) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  $("chat-log").appendChild(div);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    setTraining(s.running);
    if (s.running) {
      // A job is already in flight (e.g. after a page reload) — resync the progress bar.
      setStat("stat-status", "training");
      totalSteps = s.total_steps || totalSteps;
      showProgress(true);
      setProgress(s.current_step || 0, totalSteps);
    } else if (s.status === "error" && s.error) {
      // A job failed while we were away — surface it on reload.
      setStat("stat-status", "error");
      showError(s.error);
    } else if ($("stat-status").textContent === "idle") {
      setStat("stat-status", s.checkpoints.length ? "idle" : "no checkpoints yet");
    }
  } catch {}
}

// ---- Base-model sampling (#4) ----
$("samp-form").onsubmit = async (e) => {
  e.preventDefault();
  const btn = $("samp-btn");
  const out = $("samp-output");
  const body = {
    prompt: $("samp-prompt").value,
    max_new_tokens: +$("samp-len").value || 200,
  };
  const temp = parseFloat($("samp-temp").value);
  if (isFinite(temp) && temp > 0) body.temperature = temp;
  const topk = parseInt($("samp-topk").value, 10);
  if (Number.isInteger(topk) && topk >= 1) body.top_k = topk;
  const topp = parseFloat($("samp-topp").value);
  if (isFinite(topp) && topp > 0 && topp <= 1) body.top_p = topp;
  btn.disabled = true;
  out.textContent = "Generating…";
  try {
    const r = await post("/api/sample", body);
    const data = await r.json();
    out.textContent = r.ok
      ? (data.text || "(empty)")
      : (typeof data.error === "string" ? data.error : JSON.stringify(data.error));
  } catch {
    out.textContent = "Request failed.";
  } finally {
    btn.disabled = false;
  }
};

// ---- Tokenizer playground (#9) ----
let tokTimer = null;
async function runTokenize() {
  const text = $("tok-input").value;
  const kind = $("tok-kind").value;
  const r = await post("/api/tokenize", { text, kind });
  const data = await r.json();
  const out = $("tok-output");
  out.innerHTML = "";
  if (!r.ok) {
    $("tok-count").textContent = "";
    $("tok-source").textContent = "";
    out.textContent = typeof data.error === "string" ? data.error : "Could not tokenize.";
    return;
  }
  $("tok-source").textContent = data.source || "";
  const chars = text.length;
  $("tok-count").textContent =
    `${data.count} tokens · ${chars} characters · ${(chars / Math.max(1, data.count)).toFixed(1)} chars/token · vocab ${data.vocab_size}`;
  for (const t of data.tokens) {
    const chip = document.createElement("span");
    chip.className = "tok-chip";
    const piece = document.createElement("span");
    piece.className = "tok-piece";
    piece.textContent = displayPiece(t.piece);
    const id = document.createElement("span");
    id.className = "tok-id";
    id.textContent = t.id;
    chip.append(piece, id);
    out.appendChild(chip);
  }
}
function displayPiece(p) {
  if (p === " ") return "␣";
  if (p === "\n") return "⏎";
  return p.replace(/ /g, "␣").replace(/\n/g, "⏎");
}
function scheduleTokenize() {
  clearTimeout(tokTimer);
  tokTimer = setTimeout(runTokenize, 200);
}
$("tok-input").addEventListener("input", scheduleTokenize);
$("tok-kind").addEventListener("change", runTokenize);

// ---- Your models (checkpoint status strip) ----
async function refreshModels() {
  try {
    const r = await fetch("/api/checkpoints");
    const d = await r.json();
    const byName = {};
    for (const s of d.checkpoints || []) byName[s.name] = s;
    renderStage("stage-base", "stage-base-detail", byName.base,
      "Not yet — run Step 1 to create it.");
    renderStage("stage-chat", "stage-chat-detail", byName.chat,
      "Not yet — run Step 2 (needs a base first).");
    // Any extra checkpoints beyond base/chat.
    const extras = (d.checkpoints || []).filter(s => s.name !== "base" && s.name !== "chat");
    const box = $("models-extra");
    box.innerHTML = "";
    if (extras.length) {
      box.textContent = "Other checkpoints: " +
        extras.map(s => `${s.name} (${describeModel(s)})`).join(", ");
    }
  } catch {}
}

function renderStage(stageId, detailId, s, missingText) {
  const stage = $(stageId);
  const detail = $(detailId);
  if (s) {
    stage.classList.add("ready");
    stage.classList.remove("missing");
    detail.textContent = `✓ ready — ${describeModel(s)}${formatWhen(s.modified)}`;
    detail.title = `runs/${s.name}`;
  } else {
    stage.classList.add("missing");
    stage.classList.remove("ready");
    detail.textContent = missingText;
    detail.title = "";
  }
}

function describeModel(s) {
  const parts = [];
  if (s.n_layer != null && s.n_embd != null) parts.push(`${s.n_layer}×${s.n_embd}`);
  if (s.tokenizer_kind) parts.push(s.tokenizer_kind);
  if (s.steps != null) parts.push(`${s.steps} steps`);
  if (s.final_loss != null) parts.push(`loss ${s.final_loss.toFixed(2)}`);
  return parts.join(" · ") || "checkpoint";
}

function formatWhen(mtime) {
  if (!mtime) return "";
  const d = new Date(mtime * 1000);
  return ` · ${d.toLocaleString()}`;
}

setTraining(false);   // stop disabled until a job is actually running
connectEvents();
refreshStatus();
refreshModels();
loadHardware();
runTokenize();        // tokenize the placeholder text on load
