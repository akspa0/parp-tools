# Tasks: V20 Multi-Modal Chained Terrain Intent Reconstruction

**Spec**: `specs/067-v20-multimodal-terrain-intent/spec.md`
**Plan**: `specs/067-v20-multimodal-terrain-intent/plan.md`

## Phase 1: Ground-Intent & Liquid-Type Dataset Patching
- [x] **1.1** Write `patch_v20_signals.py` dataset utility.
- [x] **1.2** Implement block-broadcasting of `mcnk_flags_16` to `liquid_type_256`, masked by `liquid_mask`.
- [x] **1.3** Implement SciPy bi-harmonic interpolation under structure footprints to compute `ground_intent_height_257`.
- [x] **1.4** Write direct patching execution and verify Zarr array shapes/values.
- [x] **1.5** Verify visually that inpainted targets look smooth and natural (no spikes under Goldshire Inn, etc.).

## Phase 2: Segmentation Network (V20-MSS)
- [x] **2.1** Create `v20_models.py` containing the multi-head segmentation U-Net definition.
- [x] **2.2** Create `v20_dataset.py` supporting `liquid_type_256`, `object_precise_mask_256`, and `alpha_256` targets.
- [x] **2.3** Write `train_v20_segmentor.py` using the fully optimized training script structure (autotuning, worker persistent pool, deterministic sampler).
- [ ] **2.4** Train Model 1 and verify segmentation F1-score > 0.85 on validation tiles.

## Phase 3: Terrain Fingerprint Classifier (V20-TFC)
- [ ] **3.1** Build the canonical brush catalog (top 150-200 clusters) from `paste_library_catalog.json`.
- [ ] **3.2** Define `V20FingerprintClassifier` in `v20_models.py`.
- [ ] **3.3** Write `train_v20_tfc.py` to train Model 2 classification and offset/rotation/scale regression.
- [ ] **3.4** Verify classification accuracy > 80% on validation tiles.

## Phase 4: Terrain Intent Inpainter (V20-TII)
- [ ] **4.1** Create `v20_inpainter.py` taking RGB, predicted semantic masks, and predicted brush priors.
- [ ] **4.2** Formulate boundary gradient penalty loss to enforce seamless ground transition at object boundaries.
- [ ] **4.3** Write `train_v20_inpainter.py` and train Model 3 on `ground_intent_height_257` targets.
- [ ] **4.4** Verify that predicted heightmaps have smooth contours under Goldshire Inn and other complex building clusters.

## Phase 5: Placement Resolver (V20-OPR)
- [ ] **5.1** Build structural crop utility to extract individual building images from the minimap using object footprints.
- [ ] **5.2** Train Model 4 to classify cropped models against the Object Roof Library and predict 3D translation/scale/rotation values, snapping Z-height to reconstructed ground height.
- [ ] **5.3** Verify placement accuracy against original ADT placements in validation tiles.
