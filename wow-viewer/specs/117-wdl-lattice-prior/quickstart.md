# Quickstart: WDL-Lattice Coarse Prior for Terrain Geometry

**Feature**: 117-wdl-lattice-prior | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

This is the operator runbook. **The user runs every training and heavy step**; the assistant
prepares the scripts and hands off exact invocations. Every command is dry-run / print-plan by
default. Not implemented yet — this quickstart is written ahead of code, per this project's
plan-before-code discipline, and will be exercised for real as each phase lands.

All commands run from `wow-viewer/data-harvester/` via `uv run`.

## 0. Prerequisites

- The corrected v50 dual curriculum store (this session's synthetic-lighting refresh) is already
  built. Once US1 lands, it — or its successor — must also carry the four `wdl_lattice_*` arrays;
  until then this store does not yet have them and none of the steps below can run for real.
  ```bash
  STORE="../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v2.zarr"
  ```
- The existing Spec 116 spatially-isolated held-out split (reused verbatim, D-03):
  ```bash
  SPLIT="../output/datasets/spec116/spec116-held-out-0_5_3_3368-dual_v2"
  ```
- Output root for this feature:
  ```bash
  OUT="../output/datasets/spec117"
  ```

## 1. US1 — verify the signal export (no model)

No new CLI: re-run whatever store build/finalize path already produces `$STORE` for this build, and
confirm the new arrays exist:

```bash
uv run python -c "
import zarr
g = zarr.open_group('$STORE', mode='r')
for name in ('wdl_lattice_outer17','wdl_lattice_inner16','wdl_lattice_outer_present','wdl_lattice_inner_present'):
    print(name, name in g, g[name].shape if name in g else None)
"
```
- **What it does**: confirms the four new arrays exist with the expected shapes
  `(N,17,17)`/`(N,16,16)` before any Python training code is written against them.
- **Time**: seconds, CPU.

## 2. US2 — standalone lattice predictor (USER-RUN; dry-run first, then confirm)

### 2a. Dry run
```bash
uv run python scripts/spec117_train_lattice.py \
  --store $STORE --held-out-split $SPLIT \
  --epochs 100 --lr-schedule onecycle \
  --output "$OUT/lattice-run1"
```
- Prints the full plan and exits without training.

### 2b. Real run (user adds `--confirm-run`)
```bash
uv run python scripts/spec117_train_lattice.py \
  --store $STORE --held-out-split $SPLIT \
  --epochs 100 --lr-schedule onecycle \
  --output "$OUT/lattice-run1" \
  --confirm-run
```
- **What it writes**: `checkpoint_best.pt` + `model_stage_run.json` (`v50-model-stage-run-v1`,
  `stage="lattice_prior"`).
- **Read the result**: `metrics.best_val_mae` against `baselines.tile_mean.val_mae` (D-02). If the
  predictor does not beat the trivial baseline, US3 does not proceed without an explicit override
  (spec US2 acceptance 3) — this is the same honesty gate that made tonight's other results
  trustworthy.

## 3. US3(i) — bridge the frozen predictor into the existing feature-store shape

```bash
uv run python scripts/spec117_lattice_to_feature_map.py \
  --store $STORE \
  --checkpoint "$OUT/lattice-run1/checkpoint_best.pt" \
  --output "$OUT/lattice-feature-map-v1" --write
```
- **What it does**: runs the frozen predictor over every tile, writes a `(N,1,256,256)`
  `feature_map` array under `schema="v115-feature-map-v1"`, `class_count=1` — the exact shape the
  existing coarse/detailer trainers already validate (D-01). **No new trainer flags.**
- **Time**: ~1 min, CPU (or GPU with a device flag if the predictor script exposes one, mirroring
  `spec116_structure_to_feature_map.py`).

## 4. US3(ii) — paired chain-integration comparison (USER-RUN, existing trainers)

Run the **existing, unmodified** trainers with and without the bridged store, on the identical held-
out split, and compare against the pre-existing real baseline (this session's structure-augmented
detailer run):

**Coarse, without the lattice prior (baseline — already exists, no need to re-run if reused):**
```bash
# Reuse an existing coarse run's model_stage_run.json rather than retraining if one already
# covers this exact store/split/source combination.
```

**Coarse, with the lattice prior:**
```bash
uv run python scripts/v50_train_direct_geometry.py \
  --store $STORE --architecture mit_b0_regression --source authored \
  --held-out-split $SPLIT \
  --feature-store "$OUT/lattice-feature-map-v1" \
  --output "$OUT/coarse-with-lattice-run1" --run-id coarse-with-lattice-run1 \
  --confirm-run
```

**Detailer, with the lattice prior stacked on top of the already-proven structure prior:**
```bash
uv run python scripts/v50_train_geometry_detailer.py \
  --store $STORE --coarse-store <materialized coarse store matching the run above> \
  --source authored --held-out-split $SPLIT \
  --feature-store "$OUT/lattice-feature-map-v1" \
  --frequency-2d-weight 0.1 --laplacian-weight 0.1 --edge-weight 0.1 \
  --transition-focus-weight 0.5 --band-lf-weight 0.05 --band-hf-weight 0.05 \
  --lr-schedule onecycle \
  --output "$OUT/detailer-with-lattice-run1" --run-id detailer-with-lattice-run1 \
  --confirm-run
```
- **Read the result**: compare `metrics.best_val_mae` (and, better, an honest relief-stratified
  rescore via the same `spec116_train_structure.py --rescore-checkpoint` machinery used this
  session) against the existing structure-augmented detailer's real result. Report which feed point
  helped, per spec US3 acceptance scenario 3 — a null result is a valid, reportable outcome.

## Validation commands (assistant-run, lightweight)

```bash
uv run python -m pytest tests/spec117/ -q
uv run ruff check src/harvester/spec117 scripts/spec117_*.py
uv run python -m py_compile src/harvester/spec117/*.py
```

## What the assistant will never launch

- Any `--confirm-run` training pass.
- Any GPU job or full-corpus rebuild.
- The assistant prepares the script, states what it writes and how long it takes, and hands the
  exact CLI to the user. Only the user presses go.
