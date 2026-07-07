#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-smoke_v24}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/models/v24/runs/$RUN_NAME}"

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

echo "[v24] smoke: v24_stores=${V24_STORES[*]} v18_stores=${V18_STORES[*]} device=$DEVICE"

uv run python scripts/train_v24_stage_a.py \
  --v24-store "${V24_STORES[@]}" \
  --v18-store "${V18_STORES[@]}" \
  --output "$OUTPUT_DIR/stage_a" \
  --epochs 2 \
  --limit 16 \
  --device "$DEVICE" \
  --log-interval 1 \
  --seed 42 \
  "$@"

test -f "$OUTPUT_DIR/stage_a/stage_a.pt"

uv run python scripts/train_v24_stage_b.py \
  --v24-store "${V24_STORES[@]}" \
  --v18-store "${V18_STORES[@]}" \
  --stage-a-checkpoint "$OUTPUT_DIR/stage_a/stage_a.pt" \
  --output "$OUTPUT_DIR/stage_b" \
  --epochs 2 \
  --limit 16 \
  --device "$DEVICE" \
  --log-interval 1 \
  --seed 42 \
  "$@"

test -f "$OUTPUT_DIR/stage_b/stage_b.pt"
echo "[v24] smoke passed"
