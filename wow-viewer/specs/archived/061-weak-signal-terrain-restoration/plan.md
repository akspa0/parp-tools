# Implementation Plan: Weak Signal Amplifier — Cross-Signal Terrain Data Restoration

**Branch**: `061-weak-signal-terrain-restoration` | **Date**: 2026-06-13 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/061-weak-signal-terrain-restoration/spec.md`

## Summary

Detect tiles in the V18 Zarr terrain corpus where normal vectors encode significant terrain variation but height data is suspiciously flat (mismatched supervision). Reconstruct plausible heights from normals via Frankot-Chellappa Fourier-domain integration. Patch corrected heights into the Zarr stores. Scout whether corrected supervision or normal-consistency auxiliary loss improves height model training.

## Technical Context

**Language/Version**: Python 3.11+ (uv-managed), C# 10.0 (repair subcommand in build_v16_dataset.py only)

**Primary Dependencies**: numpy, scipy (FFT for Frankot-Chellappa), pyarrow + zarr (store I/O), pytorch (scouting only)

**Storage**: Zarr v3 stores at `wow-viewer/output/datasets/v18/`, sidecar repair store, parquet mismatch reports

**Testing**: pytest via `uv run --project wow-viewer/data-harvester pytest`

**Target Platform**: Windows desktop (8 GB VRAM for scouting)

**Project Type**: Python library + CLI scripts + data pipeline subcommand

**Performance Goals**: Mismatch detection < 30s for full 4096-tile focused corpus; reconstruction < 2s per tile

**Constraints**: Must not modify Zarr store arrays in-place without backup; must be idempotent on re-run

**Scale/Scope**: Focused V18 corpus (~4096 kept tiles across 2 builds)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status |
|-----------|--------|
| I. Repo independence — all code in `data-harvester/` | PASS |
| II. Library-first — core logic in `src/harvester/`, scripts thin wrappers | PASS |
| III. Real-data validation — staged-client-backed Zarr stores are the data source | PASS |
| IV. Residual model chain — scouting is measurement-only, no architectural commitment | PASS |
| V. Streaming-first — operates on existing Zarr stores directly | PASS |
| VI. No game client path assumptions — all paths under `wow-viewer/output/datasets/` | PASS |

No violations.

## Project Structure

### Documentation (this feature)

```text
specs/061-weak-signal-terrain-restoration/
├── spec.md              # Feature spec (written)
├── plan.md              # This file
├── tasks.md             # Task breakdown
└── [no research/data-model/quickstart — feature is small and self-contained]
```

### Source Code (repository root)

```text
wow-viewer/data-harvester/
├── src/harvester/
│   ├── mismatch_detector.py          # NEW — height-normal mismatch detection
│   ├── normal_height_reconstructor.py # NEW — Frankot-Chellappa normal integration
│   └── test_061_mismatch_repair.py   # NEW — pytest tests
├── scripts/
│   ├── detect_height_normal_mismatch.py  # NEW — CLI for US1
│   ├── reconstruct_heights_from_normals.py # NEW — CLI for US2
│   └── build_v16_dataset.py              # MODIFIED — add repair-heights subcommand (US3)
```

## Implementation Phases

### Phase 1: Mismatch Detection (US1)

**Goal**: Scan V18 Zarr stores and write a mismatch report parquet file.

**Approach**:
- New library module `src/harvester/mismatch_detector.py` with a `detect_mismatches()` function
- Reads `height_257`, `normal_xyz`, `normal_mask` arrays per tile_id from Zarr
- Computes `normal_relief_mean` and `height_range` per tile
- Flags tiles where `normal_relief_mean >= 0.02` AND `height_range < 3.0` AND `normal_cov >= 0.10`
- CLI script `detect_height_normal_mismatch.py` reads curation manifest, iterates kept tiles, calls detector, writes parquet
- 3 pytest tests: flags synthetic mismatch, skips blank tile, respects normal_cov threshold

### Phase 2: Normal-to-Height Reconstruction (US2)

**Goal**: For mismatched tiles, reconstruct height_257 from normal_xyz via Fourier-domain integration.

**Approach**:
- New library module `src/harvester/normal_height_reconstructor.py` with `reconstruct_height_from_normals()`
- Compute surface gradients: `dz/dx = -nx/nz` (clamped), `dz/dy = -ny/nz` (clamped)
- Mask out pixels where `nz < 0.05` or `normal_mask == False`
- Apply Hann window to gradient field to suppress boundary ringing
- Frankot-Chellappa: `H(u,v)·p(u,v) + H(u,v)·q(u,v)` → frequency-domain integration → IFFT
- Anchor result to original height's mean Z
- CLI script `reconstruct_heights_from_normals.py` reads mismatch report, reconstructs flagged tiles, writes sidecar Zarr store
- 2 pytest tests: synthetic slope reconstruction within tolerance, flat normals produce no-op

### Phase 3: Zarr Store Repair (US3)

**Goal**: Patch corrected heights into V18 stores with backup and idempotency.

**Approach**:
- Add `repair-heights` subcommand to `build_v16_dataset.py`
- For each mismatched tile in the report: copy current `height_257[tile_id]` to `height_uncorrected_257[tile_id]` (if not already backed up), then overwrite `height_257[tile_id]` from sidecar
- Update `index.parquet` with `height_corrected` boolean column
- Idempotency: skip tiles where `height_corrected` is already True
- 1 pytest test: idempotent re-run produces no changes on second pass

### Phase 4: Model Scouting (US4)

**Goal**: Measure whether corrected data or normal-consistency loss improves height training.

**Approach**:
- Two independent 5-epoch tiny-manifest scout runs
- Scout A: `train_v18_focus.py height` against corrected-height store, compare val_loss to baseline
- Scout B: Add `_height_normal_consistency_loss()` to `train_v16_1_common.py` (optional, gated behind `--normal-consistency-weight`), scout on uncorrected data
- Write results to `wow-viewer/models/v18/height/runs/v18_height_repair_scout_*`
- No model architecture changes committed unless scout shows clear improvement

## Complexity Tracking

No constitution violations. No complexity justifications needed.
