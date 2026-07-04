#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/v22}"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-smoke_v23}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/models/v23/height/runs/$RUN_NAME}"

cd "$ROOT_DIR"

if [ ! -d "$DATASET_DIR" ]; then
  echo "dataset dir not found: $DATASET_DIR" >&2
  exit 1
fi

mapfile -t STORE_DIRS < <(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d -name "*.zarr" | sort)
if [ "${#STORE_DIRS[@]}" -eq 0 ]; then
  echo "no bundled V22 stores found under $DATASET_DIR" >&2
  exit 1
fi

BUILD_NAME="$(basename "${STORE_DIRS[0]}" .zarr)"

uv run python scripts/train_v23_height.py \
  --dataset-dir "$DATASET_DIR" \
  --builds "$BUILD_NAME" \
  --epochs 2 \
  --train-max-tiles 4 \
  --val-max-tiles 2 \
  --device "$DEVICE" \
  --target-vram-gb 22 \
  --batch-size 4 \
  --gpct-K 4 \
  --gpct-weight 0.1 \
  --bias-free-mask-ratio 0.15 \
  --deterministic \
  --seed 42 \
  --run-name "$RUN_NAME" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

test -f "$OUTPUT_DIR/checkpoints/v23_height_last.pt"
test -f "$OUTPUT_DIR/val_preview_2/tile_0.png"
