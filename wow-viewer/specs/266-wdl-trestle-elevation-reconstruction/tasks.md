# Tasks: Spec 266 WDL Trestle Elevation Reconstruction

## Phase 1: Lineage WDL Prior Synthesis (AC-001)

- [x] **T266-01**: Implement `trestle_wdl_synthesizer.py` in `harvester/v60/` merging native development WDL with transferred Northrend 3.0.1.8303 WDL lattices based on visual match confidence ($\ge 0.85$).
  - *Receipt*: Created `harvester/v60/trestle_wdl_synthesizer.py` supporting `TrestleWdlSynthesizer` with 17x17 extraction and 257x257 bicubic resampling.
- [x] **T266-02**: Run WDL synthesis and generate `output/development_synthesized_wdl.npz`. Verify 17x17 lattice availability across all minimap-supported tiles.
  - *Receipt*: Synthesized 2,035 development tiles into `output/development_synthesized_wdl.npz` (sim $\ge 0.85$). `development_16_33` verified with 603.0 yd vertical span (842.0 to 1445.0 yds).

## Phase 2: Lean Trestle Model Architecture & Unit Tests (AC-002, AC-005)

- [x] **T266-03**: Implement `TrestleElevationUNet` and `TrestleLoss` in `harvester/v60/trestle_elevation_model.py` with 6 input channels, dual residual ($\Delta Z$) and bounds ($[Z_{\min}, Z_{\max}]$) heads, and $\le 10$M parameters.
  - *Receipt*: Created `harvester/v60/trestle_elevation_model.py`. Measured parameter count: 4,698,387 parameters (< 10M constraint; vs >100M in historical v7).
- [x] **T266-04**: Write unit tests in `tests/v60/test_trestle_elevation.py` asserting forward pass tensor shapes, residual addition, gradient backprop, parameter count $<10$M, and OBJ/GLB CCW normal orientation. Run `pytest` to pass.
  - *Receipt*: Ran `uv run pytest tests/v60/test_trestle_elevation.py -v`. 6/6 tests passed in 4.41s.

## Phase 3: Dataset Builder & Model Training (AC-003, AC-004)

- [x] **T266-05**: Implement `trestle_dataset.py` in `harvester/v60/` to prepare training/validation batches of `(RGB, Z_trestle, Normals)` mapped to `(Z_gt, ΔZ, bounds)`.
  - *Receipt*: Created `harvester/v60/trestle_dataset.py` and generated `output/datasets/trestle_elevation_corpus.npz` (230 train, 40 val samples) with authentic Blizzard MARE 17x17 outer lattices downsampled from ground truth.
- [x] **T266-06**: Implement training CLI `scripts/v60_train_trestle_model.py` with AdamW, mixed precision AMP, and evaluation metrics.
  - *Receipt*: Created `scripts/v60_train_trestle_model.py`.
- [x] **T266-07**: Execute model training on RTX 4070 Ti SUPER to save checkpoint `output/models/trestle_elevation_v1.pt`. Record convergence receipt meeting AC-004 (MAE $\le 25.0$ yds, $r \ge 0.75$).
  - *Receipt*: Executed 30 epochs on NVIDIA GeForce RTX 4070 Ti SUPER. Best Validation MAE: 5.20 yards (Target $\le 25.0$ yds). Best Validation Pearson $r$: 0.8754 (Target $\ge 0.75$). Span Ratio: 1.05. Checkpoint saved to `output/models/trestle_elevation_v1.pt`.

## Phase 4: Full Pipeline Reconstruction & Mesh Validation (AC-003, AC-006)

- [x] **T266-08**: Integrate `TrestleElevationUNet` and synthesized WDL lookup into `scripts/v60_reconstruct_minimap.py`.
  - *Receipt*: Added `--trestle-model` and `--synthesized-wdl` arguments; wired 6-channel input inference into Stage 5.
- [x] **T266-09**: Reconstruct `development_16_33` and verify vertical relief span $\ge 250$ yards (AC-003) and upward-pointing CCW face normals (AC-005).
  - *Receipt*: Reconstructed `development_16_33`. Trestle macro anchor span: 427.00 yards. Final reconstructed relief span: 424.75 yards (Target $\ge 250$ yds). Normal verification: 131,072 / 131,072 faces (100.0%) have strictly upward (+Z) normals. Zero inverted faces.
- [x] **T266-10**: Reconstruct diagnostic quilts and export verified 3D meshes (OBJ/GLB) with full macro mountain relief.
  - *Receipt*: Reconstructed `development_16_33` and `development_0_0` meshes and quilts. Verified in `output/reconstructions_trestle/`.
