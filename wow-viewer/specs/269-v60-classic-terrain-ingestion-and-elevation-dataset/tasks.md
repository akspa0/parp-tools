# Tasks: Spec 269 v60 Classic Terrain Zarr Ingestion & Felwood Clean Elevation Training

Receipt: [evidence/receipt-spec269.md](file:///I:/parp/parp-tools/wow-viewer/specs/269-v60-classic-terrain-ingestion-and-elevation-dataset/evidence/receipt-spec269.md)

## Phase 1: Tooling Enhancement (`v60_build_dataset.py`)
- [x] **T269-01**: Refactor `_discover_clients` in `data-harvester/scripts/v60_build_dataset.py` to support direct client roots (paths containing `Data/` or `WTF/`), nested client roots, and normalized build identifiers.
- [x] **T269-02**: Add CLI options `--maps`, `--build-id`, `--tile-workers`, and `--limit` to `v60_build_dataset.py`.
- [x] **T269-03**: Ensure `_stream_tiles` decodes and propagates `tile_x` and `tile_y` from `_metadata` into tile dicts and `index.parquet`.

## Phase 2: Tooling Verification & Unit Tests
- [x] **T269-04**: Author unit tests in `data-harvester/tests/v60/test_v60_build_dataset.py` covering client discovery, build normalization, map filtering, and tile coordinate preservation.
- [x] **T269-05**: Run pytest to verify all tests in `test_v60_build_dataset.py` pass.

## Phase 3: Zarr Dataset Loader & Felwood Indentation Supervision
- [x] **T269-06**: Implement `ZarrElevationDataset` and `build_or_load_zarr_elevation_corpus` in `data-harvester/src/harvester/v60/zarr_elevation_dataset.py` with 6-channel V7 tensors, coarse WDL trestle extraction, and Felwood region detection ($X \in [25, 29], Y \in [19, 26]$).
- [x] **T269-07**: Author unit tests in `data-harvester/tests/v60/test_zarr_elevation_dataset.py` verifying tensor dimensions $(6, 256, 256)$, residual calculation $\Delta Z$, and Felwood sampling.
- [x] **T269-08**: Run pytest to verify all tests in `test_zarr_elevation_dataset.py` pass.

## Phase 4: Training Script Integration
- [x] **T269-09**: Update `data-harvester/scripts/v60_train_trestle_model.py` with `--zarr-store` and `--focus-felwood` CLI flags to supervise `TrestleElevationUNet` on classic terrain indentations.
- [x] **T269-10**: Verify `v60_train_trestle_model.py --help` and dataset instantiation in dry-run mode.

## Phase 5: Spec Status & Registry Hygiene
- [x] **T269-11**: Register Spec 269 in `wow-viewer/specs/STATUS.md` and update `wow-viewer/memory-bank/activeContext.md`.
- [x] **T269-12**: Prepare exact PowerShell 7 execution commands for the operator to stream Kalimdor and Azeroth and launch training.
