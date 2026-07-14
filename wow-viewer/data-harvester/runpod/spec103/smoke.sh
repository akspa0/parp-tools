#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-smoke_spec103}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/models/spec103/runs/$RUN_NAME}"
ARCH="${ARCH:-v8}"

cd "$ROOT_DIR"

STORE_DEST="$(uv run python -c "
import json
print(json.load(open('manifest.json'))['store']['dest'])
")"
VAL_KEY="${VAL_KEY:-map}"
VAL_VALUE="${VAL_VALUE:-Azeroth}"

echo "[spec103] smoke: arch=$ARCH store=$STORE_DEST device=$DEVICE"

# 2 epochs, 16 train tiles: proves the pod (CUDA, deps, data) works before the real run.
uv run python scripts/train_spec103_v7.py \
  --store "$STORE_DEST" \
  --curation-manifest data/curation/curation_manifest.parquet \
  --output "$OUTPUT_DIR" \
  --val-key "$VAL_KEY" --val-value "$VAL_VALUE" \
  --arch "$ARCH" \
  --epochs 2 --limit 16 --batch 4 --patience 999 --workers 2

test -f "$OUTPUT_DIR/checkpoint_best.pt"
echo "[spec103] smoke passed"
