# Tasks: Spec 265 Two-Stage Cascaded Residual Elevation Refinement

## Phase 1: Residual Dataset Generation & Diagnostic Error Analysis
- [x] **T001**: Implement `harvester.v60.residual_elevation_dataset` extracting $Z_{\text{initial}}$ from Stage 1 checkpoint and computing $\Delta Z = Z_{\text{gt}} - Z_{\text{initial}}$ across all 270 tiles.
- [x] **T002**: Generate and cache `output/datasets/residual_elevation_corpus.npz` (230 train / 40 held-out validation).
- [x] **T003**: Verify residual statistics: compute mean, standard deviation, and spatial error breakdown.

## Phase 2: Stage 2 Architecture & Unit Test Suite
- [x] **T004**: Implement `ResidualElevationRefiner` in `harvester.v60.residual_elevation_model` consuming 6 channels `(RGB [3], Z_initial [1], Normals [2])` and predicting $\Delta Z$ (yards).
- [x] **T005**: Implement multi-scale residual composite loss with normal orientation alignment.
- [x] **T006**: Create unit tests in `tests/v60/test_residual_elevation.py` verifying tensor shapes, gradient flow, and dataset retrieval (3/3 passed in 3.78s).

## Phase 3: Operator GPU Training & Convergence Verification
- [x] **T007**: Implement `scripts/v60_train_residual_refiner.py` with AdamW, CosineAnnealingLR, and held-out validation scoring.
- [ ] **T008**: Operator-owned GPU training run on RTX 4070 Ti SUPER.
- [ ] **T009**: Verify Gate A: Combined $Z_{\text{final}} = Z_{\text{initial}} + \widehat{\Delta Z}$ reduces validation MAE significantly below baseline 81 yds.

## Phase 4: Pipeline Integration & 3D Mesh Inspection
- [x] **T010**: Update `scripts/v60_reconstruct_minimap.py` to support two-stage inference via `--refiner-model`.
- [ ] **T011**: Re-export `development_16_33_reconstructed.obj` and `development_0_0_reconstructed.obj` with restored mountain relief.
- [ ] **T012**: Verify Gate B: Measure land-only Pearson correlation and vertical relief span on `development_16_33`.
- [ ] **T013**: Write verification receipt in `evidence/receipt-spec265.md`.
