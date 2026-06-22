# Feature Specification: V19 Minimal-Signal Height Regressor

**Feature Branch**: `066-v19-height-regressor`

**Created**: 2026-06-18

**Status**: Draft

**Input**: User description: "Build V19 — a minimal-input height regressor using V18's curated dataset, modernized V7 architecture, and no WDL prior. Fix the MCNR checkerboard interpolation bug first."

## Problem Statement

Two blocking issues prevent progress on terrain reconstruction:

1. **MCNR checkerboard double-interpolation bug**: The C# `AssembleNormals` method in `AdtTensorPackBuilder.cs` fills MCNR checkerboard gaps before writing to Zarr. The Python `normal_mask` is then computed from already-interpolated data (all True), making `_interpolate_checkerboard_normals` a no-op. Models trained on this data learn interpolation artifacts instead of real surface geometry.

2. **No minimal-input height regressor exists**: V7 used 11 channels including WDL priors, object masks, and brush masks — none of which are available at inference time (user only has images). V18 uses per-task models with fewer channels but doesn't produce a standalone height regressor from minimap alone.

V19 solves both: fix the normals, then train a minimal-input height model (minimap RGB ± normals → heightmap) using V18's curated dataset infrastructure.

## User Scenarios & Testing

### User Story 1 - MCNR Checkerboard Fix (Priority: P1)

As a dataset operator, I can generate Zarr shards where MCNR normals are stored raw (with zeros at checkerboard gap positions) alongside a proper validity mask, so that Python-side interpolation produces clean normals without checkerboard artifacts.

**Why this priority**: This is a blocking bug. Every model trained on the current data learns interpolation artifacts. All downstream work (V19, V18 refinements) depends on clean normal data.

**Independent Test**: Generate a Zarr shard from a staged 3.3.5 client tile. Verify that `normal_mask` has ~50% False positions (checkerboard pattern) and that interpolated normals have no visible checkerboard artifacts in visualization.

**Acceptance Scenarios**:

1. **Given** a staged 3.3.5 client ADT, **When** the harvester generates a Zarr shard, **Then** the `mcnr_normal_xyz` array contains raw MCNR values with zeros at gap positions (x%2 != y%2).
2. **Given** a generated Zarr shard, **When** the `normal_mask` array is inspected, **Then** approximately 50% of positions are False (the checkerboard gap positions).
3. **Given** raw normals and a valid mask, **When** `_interpolate_checkerboard_normals` runs in Python, **Then** gap positions are filled by averaging cardinal neighbors and renormalizing, with no checkerboard pattern visible in the output.
4. **Given** the fix is applied, **When** an existing V16.1 model is trained on the corrected data, **Then** normal-related loss metrics improve and visualizations show no checkerboard artifacts.

---

### User Story 2 - V19 Dataset Adapter (Priority: P1)

As a ML engineer, I can load V18 Zarr data through a V19 dataset adapter that provides minimap RGB (3ch) + optional normals (3ch) as input and height_257 as target, with liquid height override for water pixels, so I can train a minimal-input height regressor.

**Why this priority**: The dataset adapter defines the input/output contract for V19. Without it, no training can happen.

**Independent Test**: Instantiate the V19 dataset on a V18 Zarr store. Verify input tensor shape is [6, 256, 256] (or [3, 256, 256] without normals), target is [1, 257, 257], and water pixels in the target are set to liquid_height values.

**Acceptance Scenarios**:

1. **Given** a V18 Zarr store with valid minimap and normal data, **When** V19Dataset loads a tile, **Then** the input tensor contains minimap RGB (channels 0-2) and normal XYZ (channels 3-5) concatenated along the channel dimension.
2. **Given** a V18 Zarr store where a tile has liquid_mask > 0, **When** V19Dataset loads that tile, **Then** the height target at liquid pixels is set to the liquid_height value from the Zarr store (not the MCVT height).
3. **Given** a V18 Zarr store where a tile has no normal data, **When** V19Dataset loads that tile, **Then** the normal channels are zero-filled and the model receives a 3ch minimap-only input.
4. **Given** curation manifests are available, **When** V19Dataset loads with curation filtering, **Then** whiteplate tiles are rejected and tiles are weighted by curation_quality_score.

---

### User Story 3 - V19 Model Architecture (Priority: P1)

As a ML engineer, I can instantiate a V19HeightModel that takes 3-6 channel input (minimap ± normals) and outputs a 257×257 heightmap plus height bounds, using a modernized V7-style U-Net with residual blocks, GroupNorm, and bilinear upsampling.

**Why this priority**: The model architecture is the core of V19. It must be defined before training can begin.

**Independent Test**: Instantiate V19HeightModel with in_channels=6. Verify forward pass produces output shape [B, 1, 257, 257] and bounds shape [B, 4]. Verify parameter count is ~20M.

**Acceptance Scenarios**:

1. **Given** V19HeightModel is instantiated with in_channels=6, **When** a random [B, 6, 256, 256] tensor is passed through, **Then** the output heightmap shape is [B, 1, 257, 257] clamped to [0, 1] and bounds shape is [B, 4].
2. **Given** V19HeightModel is instantiated with in_channels=3, **When** a random [B, 3, 256, 256] tensor is passed through, **Then** the model runs successfully with ~20M parameters.
3. **Given** the model architecture, **When** inspected, **Then** it uses ResConvBlock with GroupNorm, BilinearUp (no ConvTranspose2d), skip connections, and residual connections in conv blocks.

---

### User Story 4 - V19 Training Script (Priority: P1)

As a ML engineer, I can run `train_v19.py` to train the V19 height regressor on V18 Zarr data with multi-component loss (L1 + normal consistency + edge), curation-weighted sampling, grouped validation splits, and early stopping — without any cross-repo imports.

**Why this priority**: The training script is the entrypoint for producing a usable model checkpoint.

**Independent Test**: Run `train_v19.py` for 2 epochs on a small V18 Zarr subset. Verify it completes without errors, produces a checkpoint file, and logs train/val loss.

**Acceptance Scenarios**:

1. **Given** a V18 Zarr store, **When** `train_v19.py` is run with `--epochs 2`, **Then** training completes, a checkpoint is saved, and train/val loss are logged.
2. **Given** the training script, **When** inspected for imports, **Then** it has zero references to `gillijimproject_refactor` and all loss functions are defined locally or imported from `wow-viewer/data-harvester/src/harvester/`.
3. **Given** the training script, **When** run with `--input-channels 3`, **Then** the model trains with minimap-only input (no normals).
4. **Given** the training script, **When** run with `--input-channels 6`, **Then** the model trains with minimap + normals input.

---

### User Story 5 - V19 Baseline Training Run (Priority: P2)

As a ML engineer, I can produce a trained V19 checkpoint from a baseline training run and compare its height MAE against ground truth heightmaps, establishing a quality baseline for future iterations.

**Why this priority**: Without a baseline, we can't measure improvement from future changes.

**Independent Test**: Train V19 for 50+ epochs on V18 data. Report height MAE on validation set. Compare against V7 baseline if available.

**Acceptance Scenarios**:

1. **Given** a trained V19 checkpoint, **When** evaluated on the validation set, **Then** height MAE is reported and logged.
2. **Given** validation predictions, **When** saved as preview images, **Then** visual inspection confirms no checkerboard artifacts and reasonable terrain reconstruction.

---

### Edge Cases

- What happens when a tile has no minimap data? The dataset should skip it.
- What happens when a tile has no normal data? The model receives 3ch input (minimap only) with zero-filled normals.
- What happens when liquid_mask covers the entire tile? The height target is entirely liquid_height values; terrain_valid_mask zeroes out the loss for those pixels.
- What happens when curation manifest rejects all tiles? The dataset raises an error with a clear message.
- What happens when the Zarr store has old-format normals (already interpolated)? The mask will be all True; interpolation is a no-op (safe fallback).

## Requirements

### Functional Requirements

- **FR-001**: The C# `AssembleNormals` method MUST write raw MCNR normals to the tensor pack without interpolation, leaving zero values at checkerboard gap positions.
- **FR-002**: The C# `AssembleNormals` method MUST produce a `McnrMask257` boolean mask where true = position had original MCNR data, false = gap position.
- **FR-003**: The `TerrainTileTensorPack` MUST include a `McnrMask257` field.
- **FR-004**: The `NpzTileSerializer` and Zarr serializer MUST write `mcnr_mask_257` to the output store.
- **FR-005**: The Python `_interpolate_checkerboard_normals` function MUST correctly fill gap positions using cardinal neighbor averaging when given raw normals + real mask.
- **FR-006**: The V19Dataset MUST provide input tensors of shape [C, 256, 256] where C is 3 (minimap only) or 6 (minimap + normals).
- **FR-007**: The V19Dataset MUST set height targets to liquid_height values where liquid_mask > 0.
- **FR-008**: The V19Dataset MUST support curation manifest filtering (whiteplate rejection, quality scoring).
- **FR-009**: The V19HeightModel MUST use ResConvBlock with GroupNorm, BilinearUp, skip connections, and residual connections.
- **FR-010**: The V19HeightModel MUST output heightmap shape [B, 1, 257, 257] clamped to [0, 1] and bounds shape [B, 4].
- **FR-011**: The training script MUST have zero references to `gillijimproject_refactor`.
- **FR-012**: The training script MUST support `--input-channels 3` and `--input-channels 6` modes.
- **FR-013**: The training script MUST use grouped validation splits (no data leakage between train/val tiles of the same map).
- **FR-014**: The training script MUST use curation-weighted sampling and early stopping with plateau-based LR scheduling.

### Key Entities

- **V19HeightModel**: Minimal-input height regressor. Input: minimap RGB ± normals. Output: heightmap + bounds.
- **V19Dataset**: Dataset adapter over V18 Zarr stores. Provides V19 input/output contract.
- **McnrMask257**: Boolean mask indicating which positions in the normal map have original MCNR data.
- **TerrainTileTensorPack**: Extended with McnrMask257 field for raw normal storage.

## Success Criteria

### Measurable Outcomes

- **SC-001**: 100% of newly generated Zarr shards contain raw MCNR normals with proper checkerboard mask (no pre-interpolation).
- **SC-002**: Visual inspection of interpolated normals from corrected shards shows zero checkerboard artifacts.
- **SC-003**: V19HeightModel trains to completion without errors on V18 Zarr data.
- **SC-004**: V19 baseline height MAE is measurable and logged.
- **SC-005**: Zero cross-repo imports in V19 training code.

## Assumptions

- V18 Zarr stores are available and contain valid minimap and normal data.
- The existing `_interpolate_checkerboard_normals` Python function is correct once given raw data + real mask.
- V7's loss formulation (L1 + normal consistency + edge) is a reasonable starting point for V19.
- Curation manifests exist for the target Zarr stores.
- The model trains from scratch (no pretrained backbone).

## Architecture Notes

### V19 Model Architecture

```
V19HeightModel
├── _UNetBackbone (3 or 6 input channels)
│   ├── ResConvBlock(64) + GroupNorm + reflect-pad + residual
│   ├── ResConvBlock(128) + MaxPool
│   ├── ResConvBlock(256) + MaxPool
│   ├── ResConvBlock(512) + MaxPool
│   ├── ResConvBlock(1024) + MaxPool
│   ├── ResConvBlock(2048) + MaxPool
│   ├── ResConvBlock(2048) (bottleneck)
│   ├── BilinearUp + skip concat × 6 levels
│   └── AdaptiveAvgPool2d(1) → bounds_fc → height bounds (4 values)
├── out_conv(64 → 2)  # global + local heightmap
└── clamp(0, 1)
```

~20M parameters. No ConvTranspose2d (no checkerboard artifacts). GroupNorm for stable training.

### MCNR Checkerboard Pattern

MCNR stores 145 normals per chunk in a checkerboard pattern:
- Even rows (0, 2, 4, ...): 9 vertices at even column positions (0, 2, 4, ... 16)
- Odd rows (1, 3, 5, ...): 8 vertices at odd column positions (1, 3, 5, ... 15)
- Valid positions: x%2 == y%2
- Gap positions: x%2 != y%2 (zeros in raw data)

### Liquid Height Override

Where `liquid_mask > 0`, the height target is overridden with `liquid_height` from the Zarr store. This teaches the model to predict the actual water surface elevation instead of hallucinating terrain under water.

### Loss Weighting

- `terrain_valid_mask_257`: normal_mask × (1 - object_presence) × (1 - 0.85×liquid) × roof_weight
- Curation quality scores used as sample weights
- Bucket sampling: up-weight hard tiles, down-weight easy
