# Feature Specification: Scar Mask Segmentation Model

**Feature Branch**: `075-scar-mask-segmentation`

**Created**: 2026-06-23

**Status**: Deprecated as primary direction; retained as coarse diagnostic baseline only

> Deprecation note (2026-06-23): This model predicts whole-tile alpha-scar presence and does not segment reusable brush/fractal/paste units. It should not guide the primary terrain decomposition plan. Future training targets must come from `076-full-map-fractal-brush-library` after full-map canvas and curated-library validation.

**Input**: User request: use Spec Kit to define and build the first useful model from the isolated alpha-brush/scar signals.

## Summary

Spec 074 isolated MCAL alpha-mask micro-patterns (“scars”) and showed that exact reuse exists, but most components are local hand-edited variants. The first useful trainable model should not classify every exact scar. It should learn the **scar-presence level** of the terrain: where authored alpha brushwork exists in the minimap.

Model naming follows the current V21 terrain-model line. The training data still comes from the patched V18 Zarr stores because those are the current shared dataset substrate with newer signals layered in.

This feature trains one single-output segmentation model:

```text
minimap_rgb_256 -> alpha_scar_mask_256
```

The target is the union of thresholded alpha layers L1-L3 from V18 `alpha_256`. This is the first model in the brush-aware chain. Later specs can train separate models for layer roles, scar-family retrieval, or multi-tile prefab grouping.

## User Stories & Testing

### User Story 1 — Train A First Brush-Aware Segmentation Baseline (Priority: P1)

As a terrain reconstruction researcher, I want a model that segments alpha-scar regions from a minimap tile, so I can verify whether minimap pixels contain enough signal to recover the authored brushwork layer.

**Independent Test**: Run a short smoke training session on existing V18 Zarr data and produce a checkpoint plus preview image showing minimap, target scar mask, predicted scar mask, and error.

**Acceptance Scenarios**:

1. **Given** V18 Zarr stores with `minimap_rgb` and `alpha_256`, **When** the dataset loads a tile, **Then** it returns a 3-channel minimap input and a single-channel binary scar mask target derived from alpha layers L1-L3 at threshold `0.05`.
2. **Given** a batch from the dataset, **When** the model runs forward, **Then** it produces one `(B, 1, 256, 256)` logits tensor and no other output heads.
3. **Given** a smoke training command with a small step limit, **When** it completes, **Then** it writes a checkpoint, metrics JSON, and preview PNG.

### User Story 2 — Keep The Path Open For Scar-Family Retrieval (Priority: P2)

As a researcher, I want the segmentation output to be compatible with later component/family assignment, so that this model can become the first stage of a brush-aware reconstruction pipeline.

**Independent Test**: The output mask is compatible with connected-component postprocessing at 256x256 resolution.

**Acceptance Scenarios**:

1. **Given** the model output probabilities, **When** thresholded at `0.5`, **Then** connected components can be extracted in the same coordinate space as the 074 scar catalog bboxes.
2. **Given** the future scar-family model, **When** it consumes predicted components, **Then** no resizing between tile-local component coordinates and the predicted scar mask is required.

## Functional Requirements

- **FR-001**: Dataset MUST read only existing V18 Zarr stores under `wow-viewer/output/datasets/v18/` by default.
- **FR-002**: Dataset MUST derive a single target mask from `alpha_256` using configurable layer indices, default `1,2,3`, and configurable threshold, default `0.05`.
- **FR-003**: Model MUST have exactly one output tensor: scar-mask logits with shape `(B, 1, 256, 256)`.
- **FR-004**: Training MUST use a documented binary segmentation loss: BCE-with-logits plus soft Dice loss.
- **FR-005**: Training script MUST support a bounded smoke run with `--max-steps` and `--val-max-steps`.
- **FR-006**: Training script MUST write metrics JSON, best checkpoint, latest checkpoint, and preview PNG.
- **FR-007**: No training script may change existing V18/V20/V21 height/normal/liquid/holes/texcomp training behavior.

## Success Criteria

- **SC-001**: Unit tests verify dataset target construction, model output shape, and loss behavior.
- **SC-002**: Smoke training runs on local V18 data for at least 2 train steps and 1 validation step without crashing.
- **SC-003**: Smoke output includes a preview image showing target vs prediction.
- **SC-004**: The model remains single-output and single-checkpoint.

## Assumptions

- `alpha_256` contains useful brush/scar supervision for L1-L3.
- A binary scar-presence mask is the safest first model because it avoids a sparse 263k-way scar classifier.
- Later models can learn layer role and scar-family identity after this binary segmentation baseline proves signal exists.

## Out Of Scope

- Predicting exact scar pattern IDs.
- Predicting multi-tile prefab/paste identity.
- Multi-head layer-role segmentation.
- Height reconstruction.
- Viewer integration.
