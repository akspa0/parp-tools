#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_PATH="$ROOT_DIR/bootstrap.log"
HF_HOME="${HF_HOME:-/runpod-volume/hf_cache}"
export HF_HOME

cd "$ROOT_DIR"

{
  echo "[v23] bootstrap start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[v23] root: $ROOT_DIR"
  echo "[v23] HF_HOME: $HF_HOME"
  if [ -f uv.lock ]; then
    uv sync --frozen
  else
    uv sync
  fi
  uv pip install transformers peft bitsandbytes
  uv run python -c "from transformers import AutoModelForDepthEstimation; AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')"
  echo "[v23] bootstrap complete"
} 2>&1 | tee "$LOG_PATH"
