# Tasks: V25 Terrain Convergence Model (Spec 102)

This document contains the step-by-step tasks for the V25 SegFormer Decompiler and Terrain-Texture Convergence model.

---

## Phase 1: SegFormer Frontend & Decompiler Modules

- [x] **T001 [US2] Author `data-harvester/src/harvester/v25/segformer.py` - Custom Segformer**:
  - Load `nvidia/mit-b0` or `nvidia/mit-b1` model using Hugging Face `transformers` library.
  - Implement semantic segmentation logits mapping.
  - Implement `TerrainInpaintHead` to map the raw RGB minimap + predicted object masks to $3\times256\times256$ clean terrain-shadow maps.
- [x] **T002 [US1] Author `data-harvester/src/harvester/v25/segformer.py` - Placements**:
  - Implement `ObjectPlacementHead` to regress object classes, translations, and 3D rotations from the encoder feature layers.
- [x] **T003 [US1/US2] Author `data-harvester/tests/v25/test_segformer.py`**:
  - Test SegFormer loading (offline-compatible), output shape checks, and classifier head forward paths.
- [x] **T004 [US1/US2] Checkpoint Phase 1**:
  - Run `uv run python -m pytest tests/v25/test_segformer.py` and verify all pass.

---

## Phase 2: Height Solver & Sylvester Math

- [x] **T005 [US2] Author `data-harvester/src/harvester/v25/solver.py`**:
  - Implement `BatchedSylvesterSolver` solving $(I + \gamma_c L_c) X + X (\gamma_r L_r) = Y$ via GPU eigendecomposition (`torch.linalg.eigh`).
- [x] **T006 [US2] Author `data-harvester/src/harvester/v25/lapnet.py`**:
  - Implement progressive `V25StageBPredictor` scaling heights progressively ($33 \rightarrow 65 \rightarrow 129 \rightarrow 257$) using the Sylvester solver guided by features from the clean minimap, zeroing out edge affinities inside the predicted object mask.
- [x] **T007 [US2] Author `data-harvester/tests/v25/test_solver.py` and `test_lapnet.py`**:
  - Test Sylvester math correctness vs. CPU Scipy solver and progressive output shapes.
- [x] **T008 [US2] Checkpoint Phase 2**:
  - Run `uv run python -m pytest tests/v25/test_solver.py tests/v25/test_lapnet.py` and verify all pass.

---

## Phase 3: WDL Downsampler

- [x] **T009 [US1] Author `data-harvester/src/harvester/v25/prior.py`**:
  - Implement `WdlDownsampler` mapping $(257, 257) \rightarrow (33, 33)$ prior coordinates via mathematical average pooling.
- [x] **T010 [US1] Author `data-harvester/tests/v25/test_prior.py`**:
  - Test WDL prior downsampler output coordinates and shape alignment.
- [x] **T011 [US1] Checkpoint Phase 3**:
  - Run `uv run python -m pytest tests/v25/test_prior.py` and verify all pass.

---

## Phase 4: Decoupled PM4 Post-Processing Handler

- [x] **T012 [US1] Author `data-harvester/src/harvester/v25/pm4_guide.py`**:
  - Implement `V25Pm4GuideHandler` as a standalone post-processing class.
  - Implement snapping logic to align predicted object coordinates to pre-parsed PM4 segment centroids loaded from the database.
  - Integrate `harvester.pm4_asset_matching.scorer` to resolve WMO/M2 counterparts from segment bounds.
- [x] **T013 [US1] Author `data-harvester/tests/v25/test_pm4_guide.py`**:
  - Test coordinate snapping, asset match verification, and out-of-bounds predicted mask rejection.
- [x] **T014 [US1] Checkpoint Phase 4**:
  - Run `uv run python -m pytest tests/v25/test_pm4_guide.py` and verify all pass.

---

## Phase 5: Differentiable Fractal Generator and Parameter Head

- [x] **T015 [US1] Author `data-harvester/src/harvester/v25/fractal.py` - Generator**:
  - Implement `DifferentiableFractalGenerator` in PyTorch generating multi-octave Perlin/Simplex noise on a $256\times256$ grid.
- [x] **T016 [US1] Author `data-harvester/src/harvester/v25/fractal.py` - Parameter Head**:
  - Implement `FractalParameterHead` predicting translation seed $(S_x, S_y)$, frequency $f$, amplitude $A$, persistence $p$, and soft paint boundary mask $M$ ($256\times256$) per active layer.
- [x] **T017 [US1] Author `data-harvester/tests/v25/test_fractal.py`**:
  - Test fractal parameter extraction, output gradients, and numerical stability.
- [x] **T018 [US1] Checkpoint Phase 5**:
  - Run `uv run python -m pytest tests/v25/test_fractal.py` and verify all pass.

---

## Phase 6: Terrain Texture Decoders (MCLY/MTEX) & Losses

- [x] **T019 [US1] Author `data-harvester/src/harvester/v25/texture.py`**:
  - Implement `MtexPredictor` mapping visual features to multi-hot texture index probability vectors.
  - Implement `MclyDecoder` predicting active layers over MCNK grids (`mcly_tileset_ids` of shape `16x16x4`).
- [x] **T020 [US1/US2] Author `data-harvester/src/harvester/v25/losses.py`**:
  - Implement `V25UnifiedLoss` combining SegFormer CE, height L1, progressive height L1/SiLog, MCAL fractal parameter MSE, MCLY CE, MTEX CE, and object placement losses.
  - Write tests under `tests/v25/test_losses.py`.
- [x] **T021 [US1/US2] Checkpoint Phase 6**:
  - Verify model routing and loss gradients propagate: `uv run python -m pytest tests/v25/`

---

## Phase 7: Training and Zarr Dataset Integration

- [ ] **T022 [US1] Author `data-harvester/scripts/train_v25_decompiler.py`**:
  - Build trainer loading raw minimaps, target clean minimaps, object precise masks, object metadata, target heightmaps, and target texture arrays.
  - Configure training optimizations (`--gradient-checkpointing`, `--8bit-optimizer`, and `TileSource.preload()`).
- [ ] **T023 [US1] Author `data-harvester/scripts/validate_v25.py` and `infer_v25_decompiler.py`**:
  - Implement validation evaluation metrics.
  - Implement inference script that exports predicted heights ($257\times257$ & $33\times33$), objects, and textures directly into a structured Zarr group store with Blosc LZ4 level 1 compression.
  - Snaps predicted object placements against database PM4 segment records when requested.
- [ ] **T024 [US1] Checkpoint Phase 7**:
  - Run a 1-epoch training smoke pass and export a test prediction to the Zarr store.
