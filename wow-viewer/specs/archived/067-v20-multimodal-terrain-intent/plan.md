# Implementation Plan: V20 Multi-Modal Chained Terrain Intent Reconstruction

**Branch**: `067-v20-multimodal-terrain-intent` | **Spec**: `specs/067-v20-multimodal-terrain-intent/spec.md`

## Summary

Build a chained model pipeline to reconstruct the "Terrain Intent" (smooth ground canvas underneath buildings and water) and object placement files from a single minimap tile input. This plan outlines the database additions, C#/Python dataset inpainting pipelines, U-Net segmentor architecture, and placement mapping resolver.

---

## Technical Context

**Language/Version**: C# (.NET 10) + Python 3.11+

**Primary Dependencies**: PyTorch, Zarr, SciPy (for inpainting/interpolation), NumPy

**Storage**: Zarr v3 stores (terrain-intent arrays) + Parquet/JSON for object placement metadata libraries.

---

### Proposed Architecture

### 1. Dataset Prep: Zarr Corpus Patching
To avoid rebuilding the entire dataset, we will write a direct Zarr patching utility `patch_v20_signals.py` to append the necessary semantic and target layers to our staged 0.5.3 and 3.3.5 stores:
* **Liquid Type Map (`liquid_type_256`)**:
  - Read `mcnk_flags_16` from Zarr and decode the liquid type classes `[none: 0, ocean: 1, river: 2, magma: 3, slime: 4]`.
  - Block-broadcast the 16x16 flags grid to 256x256 using nearest-neighbor scaling.
  - Mask the high-resolution map with the high-resolution binary `liquid_mask` (zeroing out anywhere `liquid_mask == 0`).
  - Write `liquid_type_256` back to Zarr.
* **Inpainted Ground target (`ground_intent_height_257`)**:
  - Read `height_257` and `object_precise_mask`.
  - Mask out the height pixels where `object_precise_mask > 0`.
  - Perform fast bi-harmonic interpolation using SciPy's griddata or thin-plate spline solver from the surrounding unoccluded heights.
  - Write the resulting smooth, continuous target `ground_intent_height_257` back to Zarr.

### 2. Model 1: Minimap Semantic Segmentor (V20-MSS)
* **Architecture**: A U-Net backbone with multiple decoder heads:
  - **Head 1 (Liquid Type)**: Conv2D(32 -> 5 channels) for `[none, ocean, river, magma, slime]` segmentation.
  - **Head 2 (Object Footprint)**: Conv2D(32 -> 1 channel) Sigmoid for `[object_presence]` footprint.
  - **Head 3 (Alpha Blending)**: Conv2D(32 -> 4 channels) Sigmoid for `[terrain_alpha_layers]` mapping.
* **Inputs**: Minimap RGB (3ch).
* **Loss**: Multi-class Cross-Entropy (Liquids) + Binary Cross-Entropy (Object presence) + L1 (Alpha blending weights).

### 3. Model 2: Terrain Fingerprint Classifier (V20-TFC)
* **Architecture**: A ResNet classification network or ConvNeXt backbone with regression heads.
* **Workflow**:
  - Crop regions from the minimap and predicted alpha masks where high-relief terrain is detected.
  - Classify the region into one of the top ~150-200 canonical 3D terrain brushes from our mined paste library (`v18_full_corpus_v5`).
  - Regress the relative 2D offset, rotation, and scale parameters.
  - Render the matched brush profile to generate a `predicted_brush_height_prior_257` map.

### 4. Model 3: Terrain Intent Inpainter (V20-TII)
* **Architecture**: An encoder-decoder U-Net with skip connections.
* **Inputs**: 
  - Minimap RGB (3ch)
  - Predicted Liquid Mask (5ch from Model 1)
  - Predicted Object Mask (1ch from Model 1)
  - Predicted Terrain Brush Height Prior (1ch from Model 2)
* **Outputs**: `pred_ground_height` (1ch, 257x257).
* **Supervision & Loss**:
  - L1 loss against `ground_intent_height_257` (computed globally).
  - Normal consistency loss against normals calculated from `ground_intent_height_257`.
  - **Boundary gradient penalty**: L1 loss on the first-order height differences at the edge of the object mask to guarantee smooth blending between real and inpainted terrain.

### 5. Model 5: Object Placement Restorer (V20-OPR)
* **Architecture**: A ResNet backbone with classification and regression heads.
* **Workflow**:
  - For each connected component in the predicted object mask, crop the minimap patch.
  - Predict the WMO/M2 Model ID and 3D placement parameters.
  - Snaps the object's Z-height onto the reconstructed ground intent heightmap from Model 3 to prevent floating/buried buildings.

---

## Implementation Phases

### Phase 1: Ground-Intent & Liquid-Type Dataset Patching
* **Goal**: Patch the 0.5.3 and 3.3.5 Zarr datasets with inpainted ground heights and liquid type maps.
* **Tasks**:
  - Write `patch_v20_signals.py` dataset utility.
  - Implement block-broadcasting of liquid flags and SciPy bi-harmonic/spring inpainting.
  - Verify visually that inpainted heights are smooth and natural under buildings (no spikes).

### Phase 2: Segmentation Network (V20-MSS)
* **Goal**: Train the front-end segmentation model to identify liquid types, object boundaries, and alpha texture weights.
* **Tasks**:
  - Create `v20_models.py` containing U-Net definition.
  - Create `v20_dataset.py` to expose the new targets.
  - Train Model 1 and output predicted semantic segmentations for validation.

### Phase 3: Terrain Fingerprint Classifier (V20-TFC)
* **Goal**: Build the canonical brush catalog from the paste library and train the fingerprint classifier.
* **Tasks**:
  - Build the top 200 brush library from `paste_library_catalog.json`.
  - Train V20-TFC to recognize these brushes and regress their scale/rotation/offset.
  - Verify classification accuracy on validation tiles.

### Phase 4: Terrain Intent Inpainter (V20-TII)
* **Goal**: Train the inpainting model using predicted semantic masks and brush priors.
* **Tasks**:
  - Train Model 3 on `ground_intent_height_257` targets.
  - Implement boundary gradient loss.
  - Verify that the network successfully infers logical terrain contours under buildings.

### Phase 5: Placement Resolver (V20-OPR)
* **Goal**: Train the OPR network to predict model IDs and placements relative to reconstructed ground height.
* **Tasks**:
  - Train OPR on cropped building footprints.
  - Snap Z-height using Model 3 heightmap.
  - Verify placement accuracy against original ADT placements.

---

## Verification Plan

### Automated Tests
- `pytest tests/test_v20_patching.py`: Verify that Zarr patching generates arrays with correct shapes and no NaN values.
- `pytest tests/test_v20_dataset.py`: Verify shapes and targets (inpainted ground).
- `pytest tests/test_v20_models.py`: Verify forward passes and parameters.

### Manual Verification
- Visual inspection of the inpainted height maps compared to raw height maps.
- Validation previews showing predicted Ground Intent side-by-side with ground-truth terrain intent.
