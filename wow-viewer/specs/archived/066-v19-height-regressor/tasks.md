# Tasks: V19 Minimal-Signal Height Regressor

**Spec**: `specs/066-v19-height-regressor/spec.md`
**Plan**: `specs/066-v19-height-regressor/plan.md`

## Phase 1: MCNR Checkerboard Fix

- [x] **1.1** Add `McnrMask257` field to `TerrainTileTensorPack.cs`
- [x] **1.2** Modify `AssembleNormals` in `AdtTensorPackBuilder.cs`: remove interpolation loop, compute mask from MCNR sample positions
- [x] **1.3** Add `mcnr_mask_257` to `NpzTileSerializer.cs`
- [x] **1.4** Add `mcnr_mask_257` to `RawArraySerializer.cs` (Zarr path)
- [x] **1.5** Build solution and run existing tests
- [x] **1.6** Verify Python `_interpolate_checkerboard_normals` works with raw data + real mask
- [x] **1.7** Patch V18 Zarr stores (0.5.3 + 3.3.5) with mcnr_mask_257 + zeroed gap positions

## Phase 2: V19 Dataset Adapter

- [x] **2.1** Rewrite `v19_dataset.py` to wrap V161Dataset with input channel selection
- [x] **2.2** Implement liquid_height target override (where liquid_mask > 0, use liquid_height)
- [x] **2.3** Integrate curation manifest filtering (whiteplate rejection, quality scoring)
- [x] **2.4** Test V19Dataset instantiation and verify shapes

## Phase 3: V19 Model Architecture

- [x] **3.1** Clean up `v19_models.py` (verify architecture, fix any issues)
- [x] **3.2** Verify forward pass shapes and parameter count (51M params)

## Phase 4: V19 Training Script

- [x] **4.1** Create `v19_losses.py` with L1 + normal consistency + edge loss (no cross-repo imports)
- [x] **4.2** Rewrite `train_v19.py` using V18 infrastructure patterns
- [x] **4.3** Run smoke test (forward + backward pass verified)

## Phase 5: Baseline Training

- [ ] **5.1** Train V19 on full V18 corpus (50+ epochs)
- [ ] **5.2** Evaluate height MAE on validation set
- [ ] **5.3** Save preview images for visual inspection
