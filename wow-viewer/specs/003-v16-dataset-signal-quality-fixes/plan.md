# Implementation Plan: V16 Dataset Signal Quality Fixes

**Spec**: `003-v16-dataset-signal-quality-fixes/spec.md`
**Created**: 2026-05-20

## Phase 1: Archive Existing Stores & Clean Rebuild (Validation Gate)

**Goal**: Move potentially-poisoned existing stores aside, rebuild all 6 builds from scratch with new C# binary and raw format, validate.

### Step 1.1 — Archive existing datasets
Move `output/datasets/v16/*.zarr/` to `output/datasets/v16/archive_before_003/`.
**Validation**: Confirm old stores are moved, v16/ is empty.

### Step 1.2 — Rebuild 3_3_5_12340 (reference build)
Run `build_v16_dataset.py build --build 3_3_5_12340`.
**Validation**: Check signal_validation.json passes, tile_x/tile_y are correct (not all 0).

### Step 1.3 — Rebuild remaining builds
In parallel: `0_5_3_3368`, `0_5_5_3494`, `0_7_0_3694`, `3_0_1_8303`, `4_0_0_11927`.
**Validation**: Same checks per build.

## Phase 2: Human Validation (Visual Gate)

**Goal**: Generate comparison images for each build, visually verify liquid data, object masks, tile coordinates.

### Step 2.1 — Generate validation images
Run `inspect_v16_dataset.py --write-overview --write-images` on each build.
**Validation**: Open each `{build}.validation_audit_overview.png`, verify:
- tile_x/tile_y are real coordinates (not -1 or all 0)
- liquid mask shows water where expected
- filtered object mask excludes trees
- mddf/modf masks are separate

### Step 2.2 — Spot-check ocean tiles
For 3_3_5 and 3_0_1, find coastal/ocean tiles and verify liquid is present.
**Validation**: Human confirms liquid data at expected locations.

## Phase 3: Training Readiness (Smoke Gate)

**Goal**: Confirm the training stack can consume the rebuilt stores.

### Step 3.1 — Run training-readiness validator
`validate_v16_training_ready.py --build 3_3_5_12340`
**Validation**: `overall_ok=true`, `issues=0`.

### Step 3.2 — Smoke training run
`train_v16.py --builds 3_3_5_12340 --epochs 1 --batch-size 2 --device cpu --train-max-tiles 8 --val-max-tiles 4`
**Validation**: Completes without errors, validation images show filtered weight mask.

## Phase 4: Doc Sync

Update `wow-viewer/docs/architecture/v16-terrain-model-spec-2026-05-16.md` to reflect new arrays (`mcnk_flags_16`, `mddf_mask`, `modf_mask`, `object_filtered_mask`).
Update memory bank files.
