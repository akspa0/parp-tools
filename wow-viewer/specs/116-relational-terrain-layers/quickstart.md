# Quickstart: Relational Terrain Layer Reconstruction

**Feature**: 116-relational-terrain-layers | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

This is the operator runbook. **The user runs every training and heavy step** (FR-018); the
assistant prepares the scripts and hands off these exact invocations. Every command is dry-run /
print-plan by default and refuses to write or train without an explicit flag (FR-015).

All commands run from `wow-viewer/data-harvester/` via `uv run`. The input is the existing v50
curriculum Zarr store (build 0.5.3.3368, Kalimdor + Azeroth) — **no new harvest is required**.

## 0. Prerequisites

- The v50 curriculum store is already built (Spec 109/112/114). Set its path:
  ```bash
  STORE="output/datasets/curriculum-0_5_3_3368-dual_v1.zarr"
  ```
- A texture-name dump from `WowViewer.Tool.Harvest dump-texture-names` for the same build
  (Spec 115 already requires this; reuse the same file(s)):
  ```bash
  DUMPS="output/datasets/texture-names-0_5_3_3368.json"
  ```
- Output root for this feature (operator-configurable; default):
  ```bash
  OUT="output/datasets/spec116"
  ```

## 1. US1 — family→slot consistency (decides the output vocabulary; no model)

```bash
uv run python scripts/spec116_family_slot_consistency.py \
  --store "$STORE" --dumps "$DUMPS" \
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

```bash
uv run python scripts/spec116_train_structure.py \
  --store "$STORE" --split "$OUT/spec116-held-out-0_5_3_3368-v1" \
  --rescore-checkpoint <existing geometry checkpoint.pt> --print-only
```
- Reports flat vs relief-bearing error + the trivial (tile-mean) baseline per stratum. No training.

## 4. US3 — structure prediction training (USER-RUN; dry-run first, then confirm)

### 4a. Dry run (prints the plan, trains nothing)
```bash
uv run python scripts/spec116_train_structure.py \
  --store "$STORE" --split "$OUT/spec116-held-out-0_5_3_3368-v1" --dumps "$DUMPS" \
  --slot 1 --vocabulary <US1 recommendation> \
  --epochs 100 --batch-size 16 --lr 1e-3 --max-class-weight 15.0 \
  --output "$OUT/structure-slot1-run1"
```
- Prints the full plan, the exact time/memory estimate, and exits **without training**.

### 4b. Real run (user adds `--confirm-run`, optionally `--device cuda`)
```bash
uv run python scripts/spec116_train_structure.py \
  --store "$STORE" --split "$OUT/spec116-held-out-0_5_3_3368-v1" --dumps "$DUMPS" \
  --slot 1 --vocabulary <US1 recommendation> \
  --epochs 100 --batch-size 16 --lr 1e-3 --max-class-weight 15.0 \
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
  --output "$OUT/structure-infer-run1"
```
- For the hand-painted image, omit `--tile-table`; the run sets `legal_table_available=false` and
  emits an auditable record without fabricating references (SC-009).

## 5. US5 — feed predicted structure into geometry (the payoff)

### 5a. Materialize the frozen structure checkpoint's output
```bash
uv run python scripts/spec116_materialize_structure.py \
  --store "$STORE" --split "$OUT/spec116-held-out-0_5_3_3368-v1" \
  --checkpoint "$OUT/structure-slot1-run1/checkpoint_best.pt" \
  --output "$OUT/spec116-structure-<hash>-v1" --write
```

### 5b. Paired geometry runs (USER-RUN, via the existing Spec 114 geometry trainer)
Run the Spec 114 geometry trainer **without** the structure store, then **with**
`--feature-store "$OUT/spec116-structure-<hash>-v1"`, on the **same** held-out split. Compare
relief-region error; record the result in `v50-structure-geometry-comparison-v1`. SC-007 is the
honest bar: the first model to beat the trivial baseline on relief-bearing regions — or an honest
negative finding.

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