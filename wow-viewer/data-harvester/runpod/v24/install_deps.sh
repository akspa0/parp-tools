#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_PATH="$ROOT_DIR/bootstrap.log"
HF_HOME="${HF_HOME:-/runpod-volume/hf_cache}"
export HF_HOME

cd "$ROOT_DIR"

{
  echo "[v24] bootstrap start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[v24] root: $ROOT_DIR"
  echo "[v24] HF_HOME: $HF_HOME"

  if [ -f uv.lock ]; then
    uv sync --frozen
  else
    uv sync
  fi

  # V24.1 DA-V2 deps: transformers + peft for LoRA, bitsandbytes for 8-bit optimizer.
  uv pip install transformers peft bitsandbytes scipy Pillow

  # Pre-download DA-V2-Small weights so training doesn't stall on first epoch.
  # The model is ~100MB; cache it to HF_HOME for reuse across Pod restarts.
  mkdir -p "$HF_HOME"
  echo "[v24] downloading DA-V2-Small pretrained weights..."
  uv run python -c "
from transformers import AutoModelForDepthEstimation
AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
print('DA-V2-Small weights cached successfully')
"

  echo "[v24] bootstrap complete"
} 2>&1 | tee "$LOG_PATH"