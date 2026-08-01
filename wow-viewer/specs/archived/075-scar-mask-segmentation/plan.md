# Implementation Plan: V21 Scar Mask Segmentation Model

**Branch**: `075-scar-mask-segmentation` | **Date**: 2026-06-23 | **Spec**: [spec.md](spec.md)

> Deprecated as primary plan (2026-06-23): Keep this lane only as a coarse diagnostic/checkpoint-resume baseline. Do not continue Phase 2/3 as the active brush-family route unless explicitly reopened. Active work moves to `076-full-map-fractal-brush-library`.

## Summary

Build the first useful brush-aware model from the spec 074 scar library: a V21-era single-output binary segmentation model that predicts alpha-scar presence from the minimap. This model is intentionally not a 263k-class scar classifier. It learns where authored alpha brushwork exists, preserving the 256x256 coordinate space needed for later connected-component and scar-family retrieval.

The dataset root remains the patched V18 Zarr corpus because that is the current on-disk substrate. The model/training lane is named V21 to match the active model generation.

## Technical Context

**Language**: Python 3.11+

**Dependencies**: PyTorch, NumPy, Zarr, PyArrow, Pillow, pytest.

**Input Data**: Existing V18 Zarr stores under `wow-viewer/output/datasets/v18/`.

**Target**: `scar_mask_256 = max(alpha_256[..., layers] > threshold)` with default layers `1,2,3` and threshold `0.05`.

**Output**: One logits tensor `(B, 1, 256, 256)`.

**Validation**: Unit tests plus smoke training against local V18 data.

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | Pass | Code stays under `wow-viewer/data-harvester/`. |
| Library-First | Pass | Dataset/model live in `src/harvester`; CLI is a wrapper. |
| Real-Data Validation | Pass | Smoke training uses V18 Zarr stores generated from staged data. |
| Residual Model Chain | Pass | One model, one output, one checkpoint. |
| Training Script Validation | Pass | New training script has documented loss and smoke validation. |

## Project Structure

```text
wow-viewer/specs/075-scar-mask-segmentation/
├── spec.md
├── plan.md
└── tasks.md

wow-viewer/data-harvester/src/harvester/
├── v21_scar_dataset.py
├── v21_scar_model.py
└── test_v21_scar_mask.py

wow-viewer/data-harvester/scripts/
└── train_v21_scar_mask.py

wow-viewer/docs/architecture/
└── v21-scar-mask-segmentation-2026-06-23.md
```

## Phases

### Phase 1 — Dataset, Model, And Smoke Trainer

**Goal**: Create a minimal trainable scar-mask model lane and validate it with tests plus a real-data smoke run.

**Approach**:
1. Implement `V75ScarMaskDataset` reading V18 Zarr minimaps and alpha masks.
2. Implement `V75ScarMaskModel` with one logits head.
3. Implement BCE + Dice loss and preview helpers in a standalone training script.
4. Add tests for target construction, model shape, and loss behavior.
5. Run smoke training with bounded steps.

**Validation Gate**: `pytest src/harvester/test_v75_scar_mask.py` passes and `train_v75_scar_mask.py --max-steps 2 --val-max-steps 1` writes outputs.

### Phase 2 — Inference And Component Extraction

**Goal**: Convert model probabilities into connected components compatible with the 074 scar catalog.

**Blocking**: Phase 1 smoke model must exist.

**Approach**: Implement inference script, thresholding, connected-component extraction, and JSONL output.

### Phase 3 — Scar-Family Retrieval Spec

**Goal**: Use predicted components to retrieve nearest scar families from 074 dedupe outputs.

**Blocking**: Phase 2 component extraction must be validated.

## Open Questions

1. Should L1-L3 remain the default target, or should layer-specific masks become separate future models?
2. Should the first production run include all V18 builds or only `0_5_3_3368` + `3_3_5_12340`?
3. What IoU/F1 threshold is good enough to justify the next scar-family retrieval model?
