# Feature Specification: V16.1.3 Height-Channel Normal Model

**Feature Branch**: `016-v16-1-3-height-channel-normal-model`

**Created**: 2026-05-24

**Status**: Draft

**Input**: User observation that heightmap data contains the interpolation structure (17×17→257×257) that the normal ground truth shares, but the model's per-pixel Tanh head never learns this. Adding height as an input channel gives the model the spatial-frequency prior it needs.

## Problem Statement

The V16.1.1 normal model (`minimap_rgb_256 → normal_xyz`) produces a full-resolution raster where every pixel is independently computed. The ground truth normals from ADT MCNR chunks are interpolated 17×17 vertex data — smooth gradients, not per-pixel independent values. The model has no structural prior for this interpolation.

V16.1.2 attempted a separate refiner model that took `(pred_normals, height)` and refined them. This failed because the refiner was a detached computation graph — no gradient flow back to the main model.

**V16.1.3 fix**: Add height as a built-in input channel to the normal model itself. The model sees `cat(minimap_rgb, height_norm)` = 4ch input and predicts normals. This is a single model with clean gradient flow — no separate refiner, no detached branches, no distillation loops.

## Architecture

```
V16.1.1: minimap_rgb (3ch) → U-Net → head → Tanh (3ch normals)
V16.1.3: cat(minimap_rgb, height_norm) (4ch) → U-Net → head → Tanh (3ch normals)
```

The height channel carries the same 17×17→257×257 interpolation structure as the normal ground truth. The model learns: "where height changes smoothly, normals should also change smoothly."

## User Scenarios & Testing

### User Story 1 — Height Channel Improves Normal Prediction (Priority: P1)

A terrain researcher trains the normal model with height as an additional input channel. The model's validation loss is lower than the baseline V16.1.1 model on the same pool, and validation images show smoother, more interpolation-consistent normal output.

**Why this priority**: This is the core value — does adding height actually help?

**Independent Test**: Compare a bounded V16.1.3 smoke run against the V16.1.1 baseline on the same 400-tile pool. V16.1.3 should achieve lower val_loss and show smoother normal_pred panels.

**Acceptance Scenarios**:

1. **Given** a V16.1.3 model with 4ch input, **When** training completes on a bounded pool, **Then** val_loss is lower than the V16.1.1 baseline at the same epoch count.
2. **Given** the same validation tiles, **When** comparing normal_pred panels, **Then** V16.1.3 output shows fewer isolated pixel outliers and more coherent gradient structure.

---

### User Story 2 — Proper Tile Randomization (Priority: P1)

All tiles and validation tiles are properly randomized each epoch so training sees diverse terrain every pass.

**Why this priority**: The user suspects the current randomization may be insufficient. Proper shuffle ensures the model doesn't overfit to a fixed tile order.

**Independent Test**: Run a smoke with `--train-epoch-tiles 128 --train-max-tiles 400` and verify `train_epoch_orders.jsonl` shows different tile selections across epochs.

**Acceptance Scenarios**:

1. **Given** a training run with 400 pool tiles and 128 per-epoch tiles, **When** 5 epochs complete, **Then** `train_epoch_orders.jsonl` contains 5 entries with different selected_positions.
2. **Given** the same run, **When** reviewing validation, **Then** val tiles are a fixed, non-overlapping subset of the pool.

---

### User Story 3 — 1000-Epoch Long Run with Autotune (Priority: P2)

A long training run (1000 epochs) with VRAM autotune targeting 12GB produces a model checkpoint that can be evaluated for real terrain quality.

**Why this priority**: Short scouting runs prove the architecture works; the long run proves it converges.

**Independent Test**: Launch a 1000-epoch run with `--autotune-batch-size --target-vram-gb 12`. Verify it starts, autotune selects a batch size, and the first few epochs complete cleanly.

**Acceptance Scenarios**:

1. **Given** `--target-vram-gb 12 --autotune-batch-size`, **When** the run starts, **Then** autotune selects a batch size that fits within 12GB.
2. **Given** the 1000-epoch run, **When** epochs 1–3 complete, **Then** training loss decreases and val_loss is logged.

---

## Requirements

### Functional Requirements

- **FR-001**: A new model class `V161NormalHeightModel` MUST accept 4ch input: `cat(minimap_rgb, height_norm)`.
- **FR-002**: The model architecture MUST use the same U-Net backbone as `V161NormalModel` but with the first conv layer changed from `Conv2d(3, 64)` to `Conv2d(4, 64)`.
- **FR-003**: The model MUST output 3ch normals with Tanh activation, identical to `V161NormalModel`.
- **FR-004**: The dataset loader MUST provide `height_norm` as an additional input channel when the normal task requests it.
- **FR-005**: The training script MUST wire the new model class into the `normal` task when `--height-channel` is enabled.
- **FR-006**: The run directory MUST use a `v16_1_3_` prefix when height-channel mode is active.
- **FR-007**: The existing V16.1.1 normal model and training path MUST remain unchanged and runnable.
- **FR-008**: The long run MUST support `--epochs 1000 --target-vram-gb 12 --autotune-batch-size`.

### Key Entities

- **Height-Channel Normal Model**: Normal model with 4ch input (minimap + height_norm) predicting 3ch normals.
- **Height-Normalized Input**: `height_norm = (height_raw - height_mean) / height_std`, same normalization used in V16.1 height training.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A bounded V16.1.3 smoke run completes with val_loss lower than the V16.1.1 baseline at the same epoch.
- **SC-002**: The 1000-epoch run starts, autotune selects a batch size for 12GB, and training loss decreases over the first 10 epochs.
- **SC-003**: Validation preview images show smoother normal_pred panels compared to V16.1.1.

## Assumptions

- The existing V16.1 dataset already carries `height_257`, `height_mean`, and `height_std` — no dataset changes needed.
- The height normalization convention (`(raw - mean) / std`) is consistent with what the model sees during training.
- Adding one input channel increases parameter count negligibly (~200 extra params in the first conv).
- The 4070 Ti SUPER can handle the slightly larger model within 12GB VRAM with autotune.
