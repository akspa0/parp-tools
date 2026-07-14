#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCH="${ARCH:-v8}"
RUN_NAME="${RUN_NAME:-spec103_${ARCH}_real_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/models/spec103/runs/$RUN_NAME}"
VAL_KEY="${VAL_KEY:-map}"
VAL_VALUE="${VAL_VALUE:-Azeroth}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-20}"
BATCH="${BATCH:-24}"
LR="${LR:-2e-4}"
WDL_PRIOR_DROPOUT="${WDL_PRIOR_DROPOUT:-0.25}"
OUTPUT_HEAD_MODE="${OUTPUT_HEAD_MODE:-legacy_clamped}"
WORKERS="${WORKERS:-4}"

cd "$ROOT_DIR"

STORE_DEST="$(uv run python -c "
import json
print(json.load(open('manifest.json'))['store']['dest'])
")"

echo "[spec103] train: arch=$ARCH store=$STORE_DEST holdout=$VAL_KEY=$VAL_VALUE"
echo "[spec103] epochs=$EPOCHS batch=$BATCH lr=$LR wdl_prior_dropout=$WDL_PRIOR_DROPOUT output_head_mode=$OUTPUT_HEAD_MODE"

uv run python scripts/train_spec103_v7.py \
  --store "$STORE_DEST" \
  --curation-manifest data/curation/curation_manifest.parquet \
  --output "$OUTPUT_DIR" \
  --val-key "$VAL_KEY" --val-value "$VAL_VALUE" \
  --arch "$ARCH" \
  --epochs "$EPOCHS" --patience "$PATIENCE" --batch "$BATCH" --lr "$LR" \
  --wdl-prior-dropout "$WDL_PRIOR_DROPOUT" \
  --output-head-mode "$OUTPUT_HEAD_MODE" \
  --workers "$WORKERS" \
  --resume

echo "[spec103] done: $OUTPUT_DIR/checkpoint_best.pt"
