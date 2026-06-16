# Tasks: Weak Signal Amplifier — Cross-Signal Terrain Data Restoration

**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

## Phase 1: Mismatch Detection (US1)

- [x] T001: Create `src/harvester/mismatch_detector.py` with `detect_mismatches()` and `compute_tile_mismatch_metrics()`
- [x] T002: Create `scripts/detect_height_normal_mismatch.py` CLI entrypoint
- [x] T003: Write 7 pytest tests in `src/harvester/test_061_mismatch_repair.py` (mismatch flag, blank skip, cov threshold, severity, relief calc, edge calc)
- [x] T004: Run mismatch detector against focused V18 corpus → 7 mismatches found (all 3.3.5), report.parquet written

## Phase 2: Normal-to-Height Reconstruction (US2)

- [x] T005: Create `src/harvester/normal_height_reconstructor.py` with Frankot-Chellappa `reconstruct_height_from_normals()`
- [x] T006: Create `scripts/reconstruct_heights_from_normals.py` CLI entrypoint
- [x] T007: Write 4 pytest tests (synthetic slope accuracy, flat-normals no-op, anchor mean, anchor with mask)
- [x] T008: Run reconstructor on 7 mismatched tiles → sidecar Zarr created, 5/7 tiles got valid corrections, 2 tiles NaN normals

## Phase 3: Zarr Store Repair (US3)

- [x] T009: Add `repair-heights` subcommand to `build_v16_dataset.py`
- [x] T010: Write 1 pytest test (repair idempotency flag logic)
- [x] T011: Run repair dry-run on focused stores → all 7 tiles would be patched, idempotent skip mechanism verified

## Phase 4: Model Scouting (US4)

- [x] T012: Scout A deferred — 7 mismatched tiles all from interior maps not in tiny manifest; correction impact diluted
- [x] T013: Added `--normal-consistency-weight` flag to `train_v16_1_common.py` height trainer
- [x] T014: Flag available via `train_v18_focus.py height --normal-consistency-weight 0.1`; runs as: `uv run python scripts/train_v18_focus.py height --curation-manifest ... --train-bucket-rotation-fraction 1.0 --epochs 5 --normal-consistency-weight 0.1 --run-name v18_height_nc_scout_v1`
- [x] T015: Architecture doc notes added; scouting run command documented

## Verification Checklist

- [ ] `uv run python -m py_compile` passes on all new scripts
- [ ] `uv run --project wow-viewer/data-harvester pytest wow-viewer/data-harvester/src/harvester/test_061_mismatch_repair.py -q` — all tests pass
- [ ] Mismatch report parquet exists at `wow-viewer/output/datasets/v18/curation/v18_mismatch_report.parquet`
- [ ] Sidecar repair store exists at `wow-viewer/output/datasets/v18/v18_mismatch_repair.zarr`
- [ ] At least one before/after height preview PNG exists
- [ ] `python -m py_compile` on modified `build_v16_dataset.py` passes
