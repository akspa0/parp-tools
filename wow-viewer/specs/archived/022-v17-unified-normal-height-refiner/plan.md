# Implementation Plan: V17 Unified Normal Trainer

**Branch**: `022-v17-unified-normal-height-refiner` | **Date**: 2026-05-24 | **Spec**: `specs/022-v17-unified-normal-height-refiner/spec.md`

## Summary

Unify the V16.1 normal workflow into one explicit variant (`v17_hybrid`) that combines V16.1.2 refiner/distill behavior with V16.1.3 height-channel input. Add strict variant resolution, fail-fast CLI behavior, and small curated defaults (50 epochs, 80 train tiles) for fast, high-signal runs.

## Technical Context

**Language/Version**: Python 3.11+, PyTorch 2.x

**Primary Dependencies**: torch, numpy, pyarrow, zarr

**Storage**: Existing `wow-viewer/output/datasets/v16/*.zarr` and curation manifests

**Testing**: 1-epoch sanity run + bounded 50-epoch curated run

**Target Platform**: CUDA GPU (4070 Ti SUPER class), no-compile path acceptable for sanity

## Project Structure

```
wow-viewer/specs/022-v17-unified-normal-height-refiner/
├── spec.md
├── plan.md
└── tasks.md

wow-viewer/data-harvester/
├── scripts/train_v16_1_common.py     # variant resolver + hybrid wiring + defaults
└── src/harvester/v16_1_dataset.py     # checkerboard interpolation already applied
```

## Constitution Check

- Uses existing `wow-viewer` paths only; no cross-repo source references.
- Keeps training/data logic in `wow-viewer/data-harvester`.
- Uses staged dataset paths under `wow-viewer/output/datasets/v16`.
- Adds explicit validation evidence via run config/log and pool summary files.

## Implementation Phases

### Phase 1: Explicit Variant Contract

1. Add `--normal-variant` to `train_v16_1_common.py` with explicit options:
   - `v16_1_1_base`
   - `v16_1_2_refiner`
   - `v16_1_3_height`
   - `v17_hybrid`
2. Add a resolver that maps variant -> toggles (`height_channel`, `refiner_enabled`) and rejects conflicting manual flags.
3. Print resolved variant/toggles before training and persist them in `config.json`.

### Phase 2: V17 Hybrid Behavior

4. Enable the V16.1.2 refiner/distillation path while height-channel input is active when variant is `v17_hybrid`.
5. Ensure model selection, dataset input wiring, and refiner activation logic follow resolved variant instead of ad-hoc flag checks.

### Phase 3: Curated Small-Pool Defaults

6. For `v17_hybrid`, set safe defaults if user omits values:
   - `epochs=50`
   - `train_max_tiles=80`
   - `val_max_tiles=10`
7. Require or strongly validate curation-manifest presence for `v17_hybrid` and emit clear startup diagnostics.

### Phase 4: Validation

8. Run a 1-epoch sanity command and verify resolved variant proof in logs/config.
9. Run or stage a 50-epoch 80-tile curated run command for execution.
10. Confirm `normal_gt` preview no longer exhibits checkerboard halftone artifact.
