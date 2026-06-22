# Feature Specification: V20 Multi-Modal Chained Terrain Intent Reconstruction

**Feature Branch**: `067-v20-multimodal-terrain-intent`

**Created**: 2026-06-19

**Status**: Draft

**Input**: User description: "v20 multi-model chained solution for terrain intent, liquid types, alpha blending, and object placement. Extract original terrain intent underneath everything, map liquid semantic classes, predict precise object placement parameters from ADT metadata, and avoid brute-force zeroing of the loss."

## Problem Statement

Our current terrain height prediction approach (V19 and prior) suffers from three architectural limitations:

1. **Brute-Force Loss Zeroing**: When we mask out buildings/objects using `terrain_valid_mask` (zeroing out gradients), the network receives no gradient feedback in object areas. Consequently, it learns nothing about how to naturally interpolate or reconstruct smooth terrain under structures, resulting in messy height profiles and sharp height spikes at roof boundaries (roof-ridge vs. canyon confusion).
2. **Terrain Intent Loss**: The model is unaware of the "digital canvas" (the Terrain Intent) underneath human-built and natural occlusions (structures, trees, water surfaces). It does not explicitly learn the boundaries of oceans, rivers, slimes, and lava, nor how they shape the ground geometry.
3. **Untapped ADT Placement Metadata**: The original ADT files contain precise placement logs (WMO/M2 filenames, translations, scales, rotations). Currently, this is ignored at training time, even though we have an "Object Roof Library" that can render exact top-down masks for every model.

V20 solves this by decomposing terrain reconstruction into a chain of "Tiny Machines" (focused, modular networks) that explicitly segment the visual canvas, reconstruct the smooth ground-intent under structures via targeted inpainting supervision, and predict object placements.

---

## User Scenarios & Testing

### User Story 1 - Multi-Modal Semantic Segmentation (V20-MSS) (Priority: P1)

As a pipeline operator, I can run the visual segmentation network (V20-MSS) on a single minimap tile to produce a multi-layer semantic mask (Liquid Class, Object Footprint, and Alpha Blend maps) with high spatial accuracy.

**Why this priority**: This model forms the semantic front-end of the pipeline. Its output feeds as inputs/priors to the height inpainting and placement networks.

**Independent Test**: Run inference on Elwynn Forest and Stranglethorn Vale minimap tiles. Verify output liquid mask identifies ocean/river segments and object mask outlines buildings with F1-score > 0.85.

**Acceptance Scenarios**:
1. **Given** a minimap RGB tile, **When** passed through V20-MSS, **Then** the output contains a liquid class layer predicting 5 channels (none, ocean, river, magma, slime) and an object presence mask (1ch).
2. **Given** a tile with alpha-blended textures, **When** passed through V20-MSS, **Then** the output predicts a 4-channel alpha blend weight map matching the MCAL texture layouts.

---

### User Story 2 - Terrain Intent Inpainter (V20-TII) (Priority: P1)

As a terrain editor, I can predict a smooth ground heightmap that naturally interpolates underneath buildings and water bodies from the minimap and predicted semantic masks, preventing height spikes or canyon-like artifacts.

**Why this priority**: Reconstructing the clean ground canvas ("Terrain Intent") is the primary goal of terrain restoration.

**Independent Test**: Train the model using ground-truth heightmaps where building footprints have been inpainted with smooth interpolation. Verify the model predicts smooth, clean terrain profiles under Goldshire Inn and other large structures.

**Acceptance Scenarios**:
1. **Given** a tile with multiple structures, **When** passed through V20-TII, **Then** the output heightmap predicts smooth terrain under the structures instead of trying to map the structure roofs as hills.
2. **Given** an ocean or river boundary, **When** passed through V20-TII, **Then** the predicted heights reflect the correct slope and basin shape for the respective liquid type (deep trenches for ocean, shallow channels for rivers).

---

### User Story 3 - Terrain Fingerprint Classifier (V20-TFC) (Priority: P2)

As a terrain reconstructor, I can match regions of the minimap and predicted alpha blending layouts to a pre-defined library of 3D terrain brush fingerprints, providing a high-frequency height prior for ground intent reconstruction.

**Why this priority**: Repeating "prefab" terrain sculpting patterns are a major feature of WoW's terrain design. Recognizing these patterns directly unlocks exact 3D geometric relationships.

**Independent Test**: Classify a visual tile segment containing a known copy-paste mountain ridge into the correct brush ID with classification accuracy > 80%.

**Acceptance Scenarios**:
1. **Given** a minimap segment with copy-paste terrain geometry, **When** processed by V20-TFC, **Then** the model predicts the correct brush library ID and its translation, rotation, and scale offsets.
2. **Given** a predicted brush ID and offset, **When** rendered as a prior heightmap, **Then** it aligns with the unoccluded heightmap segments.

---

### User Story 4 - Object Placement Restorer (V20-OPR) (Priority: P2)

As a scene reconstructor, I can extract the list of model placements (WMO/M2 assets, coordinates, scales, rotations) from the minimap image and object mask, allowing complete recreation of the scene layout.

**Why this priority**: Instead of predicting building heights as messy heightmap pixels, this model maps visual footprints back to concrete 3D assets in our Object Roof Library.

**Independent Test**: Predict WMO placements from a Staged client map tile. Compare output MODF coordinates and model IDs against original ADT placements.

**Acceptance Scenarios**:
1. **Given** a minimap segment identified as a structure, **When** processed by V20-OPR, **Then** the model outputs a predicted WMO model ID and a placement matrix (xyz, rotation, scale).
2. **Given** predicted placements, **When** written to a PM4/ADT placement chunk, **Then** the reconstructed tile matches the layout of the original stage tile.

---

## Edge Cases

- **Occluded Terrain intent**: If a building completely covers a hill, the inpainting algorithm must predict a reasonable, smooth terrain pass-through using the surrounding terrain contours.
- **Unclassified Structures**: If a custom/non-library building is seen, the segmentation network flags it as `unknown_object`, and the terrain inpainter still successfully smooths it out.
- **Overlapping Liquids**: If river water meets the ocean, the segmentation class boundary should gracefully transition, and the terrain height should step down smoothly.
- **Novel Terrain Relief**: If a terrain patch does not match any known brush fingerprint, the classifier outputs a low confidence, and the inpainter falls back on general visual texture clues.

---

## Functional Requirements

- **FR-001**: The dataset patching script MUST construct a `ground_intent_height_257` array where building footprints (`object_precise_mask > 0`) have their height values replaced with bi-harmonic/spring-interpolated ground heights from the surrounding boundary.
- **FR-002**: The dataset patching script MUST add `liquid_type_256` by block-broadcasting 16x16 `liquid_type_16` (decoded from MCNK flags) and masking with the high-resolution `liquid_mask`.
- **FR-003**: The `V20Dataset` MUST provide `ground_intent_height` as the height target for supervision instead of the raw building-covered heightmap.
- **FR-004**: The `V20-MSS` segmentation model MUST be a multi-head U-Net outputting liquid segmentation, object footprints, and alpha blend weights.
- **FR-005**: The `V20-TFC` classifier MUST detect and classify sub-tile regions into a library of ~150-200 terrain brush fingerprints derived from the mined paste catalog.
- **FR-006**: The `V20-TII` inpainting model MUST accept the minimap RGB, predicted liquid profiles, predicted object footprints, and predicted terrain brush height priors to reconstruct the terrain intent.
- **FR-007**: The training loss for `V20-TII` MUST include a gradient consistency term across the boundary of the `object_mask` to ensure seamless transition between real and inpainted terrain.

---

## Success Criteria

- **SC-001**: 100% of structures in validation tiles have their roofs smoothed out in the predicted terrain intent heightmap (zero canyon/roof-ridge confusion).
- **SC-002**: Liquid type classification accuracy on ocean vs river vs lava pixels is > 90%.
- **SC-003**: Average height MAE inside object boundaries drops by > 30% compared to the V19 baseline (which didn't supervise under objects).
- **SC-004**: Terrain fingerprint classifier correctly identifies known repeating brushes with accuracy > 80%.
- **SC-005**: Placement reconstruction successfully maps visually distinct structures (like Town Halls, towers, and bridges) to their correct model IDs and positions.

---

## Training & Processing Instructions

### Navigating to Environment
Always navigate to the `data-harvester` directory first:
```powershell
cd wow-viewer/data-harvester
```

### Step 0: Dataset Patching (Liquid Type & Ground Intent Inpainting)
Before starting training, run the patching script to generate continuous ground height targets inpainted under visual structural roofs:
```powershell
uv run python scripts/patch_v20_signals.py --workers 4
```

### Model 1: Minimap Semantic Segmentor (V20-MSS)
To start training Model 1 to predict liquid semantic classes, object footprints, and texture weight layers:
```powershell
uv run python scripts/train_v20_segmentor.py `
  --dataset-dir ../output/datasets/v18 `
  --builds 0_5_3_3368 3_3_5_12340 `
  --epochs 50 `
  --batch-size 16 `
  --lr 2e-4 `
  --workers 4 `
  --device cuda `
  --out-dir ../output/ml-training/v20_segmentor
```
Checkpoints, training parameters, and training metrics/histories will be written to `wow-viewer/output/ml-training/v20_segmentor`.

