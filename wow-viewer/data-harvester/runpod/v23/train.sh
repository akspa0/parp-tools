#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -gt 0 ]; then
  exec uv run python scripts/train_v23_height.py "$@"
fi

DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/v22}"
TILESET_PRUNE_TABLE="${TILESET_PRUNE_TABLE:-$ROOT_DIR/config/tileset_prune_table.json}"
CURATION_MANIFEST="${CURATION_MANIFEST:-$ROOT_DIR/config/curation_manifest.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/models/v23/height/runs/${RUN_NAME:-v23_curated_2k_keymaps_runpod}}"
RUN_NAME="${RUN_NAME:-v23_curated_2k_keymaps_runpod}"
BUILDS="${BUILDS:-0_5_3_3368 3_3_5_12340}"
MAPS="${MAPS:-Azeroth Kalimdor Kalidar PVPZone01 PVPZone02 Northrend Expansion01}"
EPOCHS="${EPOCHS:-2}"
TRAIN_MAX_TILES="${TRAIN_MAX_TILES:-2000}"
VAL_MAX_TILES="${VAL_MAX_TILES:-256}"
VAL_INTERVAL="${VAL_INTERVAL:-2}"
VAL_PREVIEW_INTERVAL="${VAL_PREVIEW_INTERVAL:-2}"
TARGET_VRAM_GB="${TARGET_VRAM_GB:-22}"
MEMORY_PROFILE="${MEMORY_PROFILE:-24gb}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
AUTOTUNE_BATCH_CANDIDATES="${AUTOTUNE_BATCH_CANDIDATES:-1 2 4 8 12 16 24 32 40 48 64 80 96}"
AUTOTUNE_SAFETY_FACTOR="${AUTOTUNE_SAFETY_FACTOR:-0.85}"
GPCT_K="${GPCT_K:-2}"
GPCT_WEIGHT="${GPCT_WEIGHT:-0.1}"
SDC_WEIGHT="${SDC_WEIGHT:-0.1}"
BIAS_FREE_MASK_RATIO="${BIAS_FREE_MASK_RATIO:-0.15}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SEED="${SEED:-42}"

if [ ! -d "$DATASET_DIR" ]; then
  echo "dataset dir not found: $DATASET_DIR" >&2
  exit 1
fi

if [ ! -f "$TILESET_PRUNE_TABLE" ]; then
  echo "tileset prune table not found: $TILESET_PRUNE_TABLE" >&2
  exit 1
fi

if [ ! -f "$CURATION_MANIFEST" ]; then
  echo "curation manifest not found: $CURATION_MANIFEST" >&2
  echo "Package it with scripts/package_v23_runpod.py --curation-manifest <kept_tiles.parquet>, or set CURATION_MANIFEST." >&2
  exit 1
fi

read -r -a BUILD_ARGS <<< "$BUILDS"
read -r -a MAP_ARGS <<< "$MAPS"
read -r -a AUTOTUNE_ARGS <<< "$AUTOTUNE_BATCH_CANDIDATES"

echo "[v23] curated RunPod train"
echo "[v23] dataset_dir=$DATASET_DIR"
echo "[v23] builds=${BUILD_ARGS[*]}"
echo "[v23] maps=${MAP_ARGS[*]}"
echo "[v23] curation_manifest=$CURATION_MANIFEST"
echo "[v23] tileset_prune_table=$TILESET_PRUNE_TABLE"
echo "[v23] run_name=$RUN_NAME output_dir=$OUTPUT_DIR"

exec uv run python scripts/train_v23_height.py \
  --dataset-dir "$DATASET_DIR" \
  --builds "${BUILD_ARGS[@]}" \
  --maps "${MAP_ARGS[@]}" \
  --curation-manifest "$CURATION_MANIFEST" \
  --tileset-prune-table "$TILESET_PRUNE_TABLE" \
  --epochs "$EPOCHS" \
  --train-max-tiles "$TRAIN_MAX_TILES" \
  --val-max-tiles "$VAL_MAX_TILES" \
  --val-interval "$VAL_INTERVAL" \
  --val-preview-interval "$VAL_PREVIEW_INTERVAL" \
  --device cuda \
  --target-vram-gb "$TARGET_VRAM_GB" \
  --memory-profile "$MEMORY_PROFILE" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --autotune-batch-size \
  --autotune-batch-candidates "${AUTOTUNE_ARGS[@]}" \
  --autotune-safety-factor "$AUTOTUNE_SAFETY_FACTOR" \
  --gpct-K "$GPCT_K" \
  --gpct-weight "$GPCT_WEIGHT" \
  --sdc-weight "$SDC_WEIGHT" \
  --bias-free-mask-ratio "$BIAS_FREE_MASK_RATIO" \
  --log-interval "$LOG_INTERVAL" \
  --deterministic \
  --seed "$SEED" \
  --run-name "$RUN_NAME" \
  --output-dir "$OUTPUT_DIR"
