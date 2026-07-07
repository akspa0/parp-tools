#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# manifest.json's v24_stores[]/v18_subsets[] are index-paired (build i's V24
# store goes with build i's V18 subset); their "dest" fields may not share a
# stem, so pair them from the manifest rather than guessing from a file glob.
mapfile -t V24_STORES < <(uv run python -c "
import json
m = json.load(open('manifest.json'))
for s in m['v24_stores']:
    print(s['dest'])
")
mapfile -t V18_STORES < <(uv run python -c "
import json
m = json.load(open('manifest.json'))
for s in m['v18_subsets']:
    print(s['dest'])
")
if [ "${#V24_STORES[@]}" -eq 0 ]; then
  echo "no V24 stores listed in manifest.json" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-v24_runpod_train}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/models/v24/runs/$RUN_NAME}"
EPOCHS_A="${EPOCHS_A:-200}"
EPOCHS_B="${EPOCHS_B:-1000}"
PATIENCE_A="${PATIENCE_A:-40}"
PATIENCE_B="${PATIENCE_B:-100}"
SYNTH_DROPOUT="${SYNTH_DROPOUT:-0.5}"
BATCH_SIZE_A="${BATCH_SIZE_A:-64}"
BATCH_SIZE_B="${BATCH_SIZE_B:-24}"
AUTOTUNE_CANDIDATES_A="${AUTOTUNE_CANDIDATES_A:-16 32 64 96 128 192 256 384 512}"
AUTOTUNE_CANDIDATES_B="${AUTOTUNE_CANDIDATES_B:-2 4 8 12 16 24 32 48 64 96}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SEED="${SEED:-94}"

read -r -a AUTOTUNE_A_ARGS <<< "$AUTOTUNE_CANDIDATES_A"
read -r -a AUTOTUNE_B_ARGS <<< "$AUTOTUNE_CANDIDATES_B"

echo "[v24] RunPod train: v24_stores=${V24_STORES[*]} v18_stores=${V18_STORES[*]}"
echo "[v24] run_name=$RUN_NAME output_dir=$OUTPUT_DIR"

echo "[v24] Stage A: epochs=$EPOCHS_A patience=$PATIENCE_A batch_size=$BATCH_SIZE_A (autotuned)"
uv run python scripts/train_v24_stage_a.py \
  --v24-store "${V24_STORES[@]}" \
  --v18-store "${V18_STORES[@]}" \
  --output "$OUTPUT_DIR/stage_a" \
  --epochs "$EPOCHS_A" \
  --patience "$PATIENCE_A" \
  --synth-dropout "$SYNTH_DROPOUT" \
  --batch-size "$BATCH_SIZE_A" \
  --autotune-batch-size \
  --autotune-batch-candidates "${AUTOTUNE_A_ARGS[@]}" \
  --amp-dtype "$AMP_DTYPE" \
  --device cuda \
  --log-interval "$LOG_INTERVAL" \
  --seed "$SEED"

echo "[v24] Stage B: epochs=$EPOCHS_B patience=$PATIENCE_B batch_size=$BATCH_SIZE_B (autotuned)"
uv run python scripts/train_v24_stage_b.py \
  --v24-store "${V24_STORES[@]}" \
  --v18-store "${V18_STORES[@]}" \
  --stage-a-checkpoint "$OUTPUT_DIR/stage_a/stage_a.pt" \
  --output "$OUTPUT_DIR/stage_b" \
  --epochs "$EPOCHS_B" \
  --patience "$PATIENCE_B" \
  --batch-size "$BATCH_SIZE_B" \
  --autotune-batch-size \
  --autotune-batch-candidates "${AUTOTUNE_B_ARGS[@]}" \
  --amp-dtype "$AMP_DTYPE" \
  --device cuda \
  --log-interval "$LOG_INTERVAL" \
  --seed "$SEED"

echo "[v24] done: $OUTPUT_DIR/stage_a/stage_a.pt + $OUTPUT_DIR/stage_b/stage_b.pt"
