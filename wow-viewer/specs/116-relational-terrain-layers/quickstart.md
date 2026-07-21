# Quickstart: Relational Terrain Layer Reconstruction

**Feature**: 116-relational-terrain-layers | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

This is the operator runbook. **The user runs every training and heavy step** (FR-018); the
assistant prepares the scripts and hands off these exact invocations. Every command is dry-run /
print-plan by default and refuses to write or train without an explicit flag (FR-015).

All commands run from `wow-viewer/data-harvester/` via `uv run`. The input is the existing v50
curriculum Zarr store (build 0.5.3.3368, Kalimdor + Azeroth) — **no new harvest is required**.

## 0. Prerequisites

- The v50 curriculum store is already built (Spec 109/112/114). It lives one directory **above**
  `data-harvester/`, not under it. Verified on disk (2,990 rows: 1,629 authored + 1,361
  synthetic):
  ```bash
  STORE="../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr"
  ```
- Texture-name dumps from `WowViewer.Tool.Harvest dump-texture-names`, one file per map (Spec 115
  already requires these; reuse the same files). The dual curriculum spans Kalimdor + Azeroth, so
  pass both:
  ```bash
  DUMPS=("../output/v50/v50.1/texture-names/Kalimdor.json" "../output/v50/v50.1/texture-names/Azeroth.json")
  ```
- Output root for this feature (operator-configurable; kept alongside the real dataset tree
  rather than inside `data-harvester/output/`, which has no `datasets/` folder of its own):
  ```bash
  OUT="../output/datasets/spec116"
  ```

## 1. US1 — family→slot consistency (decides the output vocabulary; no model)

```bash
uv run python scripts/spec116_family_slot_consistency.py \
  --store "$STORE" --dumps "${DUMPS[@]}" \
  --threshold 0.70 \
  --output "$OUT/family-slot-consistency-0_5_3_3368-v1.json" --write
```
- **What it does**: extracts layer-entry rows, joins each slot's texture to a surface family, and
  reports the per-family slot distribution + a summary consistency score + a `slot_keyed`/
  `family_keyed` recommendation.
- **Time**: <1 min, CPU. **Memory**: <1 GB.
- **Reads the result**: the `recommendation` field is the durable vocabulary decision US3 consumes.

## 2. US2 — shape→coverage coupling (decides if structure is derivable from geometry; no model)

```bash
uv run python scripts/spec116_shape_coverage_coupling.py \
  --store "$STORE" \
  --output "$OUT/shape-coverage-coupling-0_5_3_3368-v1.json" --write
```
- **What it does**: per tile/layer, fits a non-linear `{elevation, slope} → coverage` model,
  reports explained variance, a dip-test for bimodality, and whether a high-coupling population
  exists.
- **Time**: a few min, CPU (per-tile `GradientBoostingRegressor`). **Memory**: <2 GB.

## 3. US4 — spatially-isolated held-out split (makes evaluation trustworthy)

```bash
uv run python scripts/spec116_build_held_out_split.py \
  --store "$STORE" \
  --buffer-rings 1 --seed 116 \
  --output "$OUT/spec116-held-out-0_5_3_3368-v1" --write
```
- **What it does**: builds a split where no held-out tile shares an edge **or corner** with a
  training tile; prints `verified_violation_count` (must be 0).
- **Time**: seconds, CPU. **Memory**: <1 GB.
- **Important**: rebuilding this split **invalidates absolute comparison** with all prior results
  (FR-017). The report names the baseline requiring re-run.

### 3b. Re-score an existing model, stratified by relief (immediate value)

RGB-only baseline checkpoint (e.g. `direct_cnn_v112` or an un-deconfounded `mit_b0_regression`
run):
```bash
uv run python scripts/spec116_train_structure.py \
  --store "$STORE" --split "$OUT/spec116-held-out-0_5_3_3368-v1" \
  --rescore-checkpoint <existing Spec 114 geometry checkpoint.pt> \
  --rescore-output "$OUT/geometry-rescore-v1.json" --print-only
```

A Spec 115 deconfounded checkpoint (trained with `--feature-store`, RGB + 5 generated
terrain-feature classes = 8 input channels) needs that same feature-map store reconstructed at
rescore time, **and** `--rescore-source` restricted to whatever row domain the feature-map store
actually covers (an authored-only feature map does not cover the dual curriculum's synthetic
rows):
```bash
uv run python scripts/spec116_train_structure.py \
  --store "$STORE" --split "$OUT/spec116-held-out-0_5_3_3368-v1" \
  --rescore-checkpoint <deconfounded checkpoint.pt> \
  --feature-store <v115-feature-map-v1 store used for training> \
  --rescore-source authored \
  --rescore-output "$OUT/geometry-rescore-v1.json" --print-only
```
- Reports flat vs relief-bearing MAE + the trivial (tile-mean) baseline per stratum, and whether
  the checkpoint beats the trivial baseline on relief-bearing regions. No training; read-only.
  MAE is reported in raw world-height units (the model output is decoded via each tile's own
  min/max), not the normalized `[0,1]` units earlier training runs reported -- the numbers are not
  directly comparable to previously recorded MAE figures for the same checkpoint, only to each
  other within this same rescore run.
- `--rescore-in-channels` is auto-derived (`3`, or `3 + --feature-store`'s `class_count`) and
  should not normally be set by hand; it exists as an override only.
- If `--feature-store` is missing held-out rows, the error names the row count and suggests the
  matching `--rescore-source` value -- this is a domain mismatch (feature map built from fewer
  rows than the split covers), not a bug.
- `--print-only` (the default even without it, unless `--rescore-output` is also given) never
  writes a file; drop `--print-only` while keeping `--rescore-output <path.json>` to persist it.

## 4. US3 — structure prediction training (USER-RUN; dry-run first, then confirm)

### 4a. Dry run (prints the plan, trains nothing)
```bash
uv run python scripts/spec116_train_structure.py \
  --store "$STORE" --split "$OUT/spec116-held-out-0_5_3_3368-v1" --dumps "${DUMPS[@]}" \
  --slot 1 --vocabulary "$OUT/family-slot-consistency-0_5_3_3368-v1.json" \
  --epochs 100 --batch 16 --lr 1e-3 --max-class-weight 15.0 \
  --output "$OUT/structure-slot1-run1"
```
- Prints the full plan, the exact time/memory estimate, and exits **without training**.
- `--vocabulary` takes the US1 report JSON written in step 1 (its `recommendation` field is read
  automatically) -- not the literal string `slot_keyed`/`family_keyed`.

### 4b. Real run (user adds `--confirm-run`, optionally `--device cuda`)
```bash
uv run python scripts/spec116_train_structure.py \
  --store "$STORE" --split "$OUT/spec116-held-out-0_5_3_3368-v1" --dumps "${DUMPS[@]}" \
  --slot 1 --vocabulary "$OUT/family-slot-consistency-0_5_3_3368-v1.json" \
  --epochs 100 --batch 16 --lr 1e-3 --max-class-weight 15.0 \
  --output "$OUT/structure-slot1-run1" \
  --device cuda --confirm-run
```
- **What it writes**: `checkpoint_best.pt` + `structure-run.json` (schema `v50-structure-run-v1`).
- **Time/Mem (estimate)**: ~1.5M params, ~1.4k rows → ~1–2 min/epoch on one GPU; <2 GB GPU memory.
  The dry run prints the exact figure.
- **Gate**: per-class IoU/recall; `promotion_verdict` starts `pending`. Repeat for `--slot 2` and
  `--slot 3` (three independent checkpoints — D-04).

### 4c. Inference + legality audit (incl. an OOD hand-painted image)
```bash
uv run python scripts/spec116_infer_structure.py \
  --checkpoint "$OUT/structure-slot1-run1/checkpoint_best.pt" \
  --inputs <tile.png> <hand-painted.png> \
  --tile-table <MTEX table json> \
  --slot 1 \
  --output "$OUT/structure-infer-run1/audit.json" --write
```
- For the hand-painted image, omit `--tile-table` entirely; the run sets
  `legal_table_available=false` and emits an auditable record without fabricating references
  (SC-009). `--tile-table` is one shared MTEX table (a JSON list of texture names, or
  `{"texture_names": [...]}`) applied to every `--inputs` image in the run.
- `--inputs` accepts loose 256x256 PNG/JPEG files and/or folders (mixed) -- no v50 store is
  required, so this runs unchanged on an image with no client backing at all.
- Batch mode over an existing v50 store (legality resolved per tile via its own texture-name
  dump) is also available: swap `--inputs .../--tile-table ...` for `--store "$STORE" --dumps
  "${DUMPS[@]}"`.

## 5. US5 — feed predicted structure into geometry (the payoff)

### 5a. Materialize the frozen structure checkpoint's output
```bash
uv run python scripts/spec116_materialize_structure.py \
  --store "$STORE" \
  --dumps "${DUMPS[@]}" \
  --checkpoint "$OUT/structure-slot1-run1/checkpoint_best.pt" \
  --slot 1 \
  --output "$OUT/spec116-structure-slot1-v1" --write
```
- **What it does**: runs the frozen slot-1 checkpoint over every tile in the source store and
  writes a derived Zarr store with `structure_family` (int8 16x16), `structure_confidence`
  (float16 16x16), `structure_legal` (bool 16x16), and a row-aligned `index.parquet`.
- **Time**: ~1 min, CPU (or GPU with `--device cuda`). **Memory**: <2 GB.
- **Source store is never mutated**; the derived store is immutable once written.

### 5b. Paired geometry comparison (USER-RUN, via the existing Spec 114 geometry trainer)

The geometry trainer is `scripts/v50_train_direct_geometry.py`; its `--feature-store` was built
for Spec 115's per-pixel `v115-feature-map-v1` shape, not Spec 116's per-chunk
`v116-structure-store-v1`. Bridge the derived structure store first (dry run, then `--write`):
```bash
uv run python scripts/spec116_structure_to_feature_map.py \
  --structure-store "$OUT/spec116-structure-slot1-v1" \
  --output "$OUT/spec116-structure-slot1-feature-map-v1" --write
```
- Upsamples each 16x16 chunk's predicted family + confidence to a 256x256 per-pixel soft class
  distribution -- the exact shape/schema `--feature-store` already validates. No trainer change.

Then run the geometry trainer **twice** on the **same** held-out split — once without the bridged
structure store, once with it — and compare relief-region error. `--held-out-split` consumes the
Spec 116 split directly (read-only; `--store` is never mutated), overriding the trainer's own
`--val-key`/`--val-value` column.

**Run A — without structure (baseline):**
```bash
uv run python scripts/v50_train_direct_geometry.py \
  --store "$STORE" --architecture mit_b0_regression --source authored \
  --held-out-split "$OUT/spec116-held-out-0_5_3_3368-v1" \
  --output "$OUT/geometry-without-structure-run1" --run-id geometry-without-structure-run1 \
  --confirm-run
```

**Run B — with predicted structure:**
```bash
uv run python scripts/v50_train_direct_geometry.py \
  --store "$STORE" --architecture mit_b0_regression --source authored \
  --held-out-split "$OUT/spec116-held-out-0_5_3_3368-v1" \
  --feature-store "$OUT/spec116-structure-slot1-feature-map-v1" \
  --output "$OUT/geometry-with-structure-run1" --run-id geometry-with-structure-run1 \
  --confirm-run
```
- `v50_train_direct_geometry.py` has **no `--device` flag** -- it hardcodes CUDA once
  `--confirm-run` is given and refuses to run on CPU.
- `--feature-store` requires `--architecture mit_b0_regression` (the RGB-only `direct_cnn_v112`
  baseline cannot take the extra input channels).
- Re-score each resulting checkpoint with `spec116_train_structure.py --rescore-checkpoint`
  (section 3b) to get the relief-stratified MAE for the comparison record below: use
  `--rescore-in-channels 3` for Run A and `--rescore-in-channels <3 + class_count>` for Run B
  (`class_count` is printed by the bridge step and recorded in the feature-map store's attrs).

**Record the comparison** in `v50-structure-geometry-comparison-v1`:
```json
{
  "schema": "v50-structure-geometry-comparison-v1",
  "held_out_split": {"path": "...", "sha256": "..."},
  "without_structure": {
    "checkpoint_sha256": "...",
    "relief_mae": <float>,
    "flat_mae": <float>,
    "trivial_baseline_relief_mae": <float>
  },
  "with_structure": {
    "checkpoint_sha256": "...",
    "relief_mae": <float>,
    "flat_mae": <float>,
    "trivial_baseline_relief_mae": <float>
  },
  "sc007_beats_trivial_on_relief": <bool>,
  "absolute_comparison_to_prior_runs_invalid": true
}
```
- **SC-007**: the honest bar — does the structure-augmented model beat the trivial (tile-mean)
  baseline on relief-bearing regions? If yes, predicted structure helps. If no, record an honest
  negative finding.
- **`absolute_comparison_to_prior_runs_invalid=true`**: rebuilding the held-out split
  invalidates absolute comparison with all prior runs (FR-017). Only relative comparison
  (with vs without structure, same split) is valid.
- The assistant prepares this record template; the user fills in the MAE values from the two
  training runs and validates with `validate_structure_geometry_comparison`.

## Validation commands (assistant-run, lightweight)

```bash
# Focused spec116 tests
uv run python -m pytest tests/spec116/ -q

# Full data-harvester suite when a shared module changes
uv run python -m pytest -q

# Lint + compile
uv run ruff check src/harvester/spec116 scripts/spec116_*.py
uv run python -m py_compile src/harvester/spec116/*.py
```

## What the assistant will never launch

- Any `--confirm-run` training pass (FR-018 / Rule 0).
- Any GPU job, full-corpus rebuild, or long-running build.
- The assistant prepares the script, states what it writes and how long it takes, and hands the
  exact CLI to the user. Only the user presses go.