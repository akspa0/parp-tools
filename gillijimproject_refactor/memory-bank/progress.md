# PROGRESS — wow-viewer

## Position
- V16 dataset generation + training is the primary active workflow.
- Harvest-first is canonical:
  - `WowViewer.Tool.Harvest`
  - staged clients under `output/tmp/wowarchive-clients/`
- Old converter-side `dataset-scan` / `dataset-audit` / `dataset-build-cache` flows are not the primary terrain-AI path.

## Validated Now

### V16 Corpus
- Finalized stores built for:
  - `0_5_3_3368`
  - `0_5_5_3494`
  - `0_7_0_3694`
  - `3_0_1_8303`
  - `3_3_5_12340`
  - `4_0_0_11927`
- All six current `signal_validation.json` files pass.
- Human-eye QA artifacts exist for all six under `wow-viewer/output/datasets/v16/inspection/`.
- `0_7_0_3694` carries the expected allowed warning for zero `has_holes_16` coverage.

### V16 Recovery / Build Surfaces
- In-memory archive harvest path is landed.
- Lean `ARRY` stream profile is landed.
- Map-level resume / `_resume_state.json` is landed.
- Completed-store skip guards and `--rebuild-existing` behavior are landed.
- `repair-index` is landed for coordinate-only fixes.
- `patch-liquids` is landed for in-place liquid rewrites.
- Signal validation gate is landed and passing on the current finalized corpus.
- Dataset inspection / summary / visual QA tooling is landed.
- Default Zarr compression is now `lz4` / `1` / `shuffle`.

### Critical V16 Fixes
- Mixed Cata `_tex0` fallback fixed:
  - `ReadTextureDataFromBytes(...)` now falls back to inline root `MCLY` / `MCAL`.
  - Focused repro on staged `4_0_0_11927 / AhnQiraj / (27,46)` restored alpha/MCLY truth.
- Alpha placeholder `map=memory` metadata fix is landed.
- Liquid presence-mask fix for valid type-`0` water is landed.
- Object instance mask + `placements.parquet` are landed.

### V16 Training Surfaces
- `V16Dataset` is the live loader.
- `V15Model` is the current V16 terrain model host.
- `validate_v16_training_ready.py` passed on staged `3_3_5_12340`.
- Current terrain lane supervises:
  - height
  - normals
  - alpha
  - holes
  - liquid mask
  - MCLY
- `liquid_height` remains in the dataset but is deferred from the current terrain model.
- Short-lived terrain-lane `liquid_height` supervision was superseded.

### Alpha / LK Conversion Lane
- `AlphaToLk` and `LkToAlpha` are both landed in shared `wow-viewer` surfaces.
- Real-data `LkToAlpha` proof exists:
  - `4_0_0_11927 / Azeroth`
  - `839/839` tiles
  - terrain + WMOs rendered in MdxViewer
- Current shared alphaWDT rules that still matter:
  - `MAIN` is row-major
  - always emit all `256` MCNKs
  - `MCRF` stays FourCC-wrapped
  - top-level chunks are contiguous
  - doodads use single-owner chunk routing
  - shared placement rotation stays in raw-file convention

## In Progress
- First real V16 training run.
- WL* partial chunk-fill semantics in the loader / trainer.
- Object segmentation Model A.
- Global asset vocabulary for instance/asset follow-up work.
- PM4 cross-reference / object mapping follow-up.

## High-Value Open Gaps
- Forward `AlphaToLk` AreaID wiring.
- Exact doodad-border ownership for large cross-chunk placements.
- Full chunk-preservation closure is still open:
  - `MFBO`
  - `MCCV`
  - `MCLV`
  - `MTXF`
  - higher-fidelity `MH2O`

## Not Yet
- Production V16 training run with curated corpus.
- Asset-attribute model / PM4 cross-ref workflow.
- Broader chunk-for-chunk terrain conversion closure.
