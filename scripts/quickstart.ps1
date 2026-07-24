# LLM Forge — one-shot quickstart for Windows PowerShell.
# Sets up a venv, installs deps, and runs the whole pipeline on the sample data.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "== LLM Forge quickstart ==" -ForegroundColor Cyan

# 0. Zero-dependency taste of "it learns" (works even before installing anything)
Write-Host "`n[0/5] Playground (pure Python)..." -ForegroundColor Yellow
python -m llmforge.playground

# 1. Virtual environment + dependencies
if (-not (Test-Path ".venv")) {
    Write-Host "`n[setup] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
Write-Host "[setup] Installing dependencies (CPU PyTorch)..." -ForegroundColor Yellow
pip install --quiet torch tokenizers safetensors numpy fastapi uvicorn `
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# 2. Tokenizer (char is instant and needs no extra downloads)
Write-Host "`n[1/5] Training tokenizer..." -ForegroundColor Yellow
python -m llmforge.cli tokenizer --input data/sample/corpus.txt --kind char --out runs/tokenizer.json

# 3. Pre-train
Write-Host "`n[2/5] Pre-training..." -ForegroundColor Yellow
python -m llmforge.cli pretrain --data data/sample/corpus.txt --tokenizer runs/tokenizer.json --steps 500

# 4. Sample
Write-Host "`n[3/5] Sampling from the base model..." -ForegroundColor Yellow
python -m llmforge.cli sample --prompt "The little"

# 5. Fine-tune + chat
Write-Host "`n[4/5] Fine-tuning on chat pairs..." -ForegroundColor Yellow
python -m llmforge.cli finetune --data data/sample/chat.jsonl --base runs/base --steps 300

Write-Host "`n[5/5] Done! Try:  python -m llmforge.cli chat" -ForegroundColor Green
Write-Host "Or launch the dashboard:  python -m llmforge.cli serve" -ForegroundColor Green
