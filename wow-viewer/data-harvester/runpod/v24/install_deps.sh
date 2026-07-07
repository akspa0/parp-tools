#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_PATH="$ROOT_DIR/bootstrap.log"

cd "$ROOT_DIR"

{
  echo "[v24] bootstrap start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[v24] root: $ROOT_DIR"
  if [ -f uv.lock ]; then
    uv sync --frozen
  else
    uv sync
  fi
  echo "[v24] bootstrap complete (no pretrained weights to fetch: V24 trains from scratch)"
} 2>&1 | tee "$LOG_PATH"
