#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_PATH="$ROOT_DIR/bootstrap.log"

cd "$ROOT_DIR"

{
  echo "[spec103] bootstrap start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[spec103] root: $ROOT_DIR"

  # v8/v7 train from scratch -- no HF downloads, no LoRA, no bitsandbytes.
  if [ -f uv.lock ]; then
    uv sync --frozen
  else
    uv sync
  fi

  echo "[spec103] nvidia-smi:"
  nvidia-smi || echo "[spec103] WARNING: nvidia-smi not found"

  echo "[spec103] bootstrap complete"
} 2>&1 | tee "$LOG_PATH"
