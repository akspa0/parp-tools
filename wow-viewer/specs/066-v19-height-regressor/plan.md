# Implementation Plan: V19 Minimal-Signal Height Regressor

**Branch**: `066-v19-height-regressor` | **Date**: 2026-06-18 | **Spec**: `specs/066-v19-height-regressor/spec.md`

**Input**: Feature specification from `/specs/066-v19-height-regressor/spec.md`

## Summary

Fix the MCNR checkerboard interpolation bug in the C# harvester, then build V19: a minimal-input height regressor (minimap RGB ± normals → heightmap) using V18's curated Zarr dataset, modernized V7 architecture, and no WDL prior.

## Technical Context

**Language/Version**: C# (.NET 10) + Python 3.11+

**Primary Dependencies**: PyTorch, Zarr, NumPy, PyArrow, PIL, torchvision

**Storage**: Zarr v3 stores (one per client build), Parquet index files

**Testing**: dotnet test (C#), pytest (Python), manual training validation

**Target Platform**: Windows (development), cross-platform (inference)

**Project Type**: Library (C# harvester) + CLI (Python training script)

**Scale/Scope**: ~20M param model, ~100K tiles in V18 corpus

## Constitution Check

- **Repo Independence**: All new code in `wow-viewer/`. Zero references to `gillijimproject_refactor`. ✅
- **Library-First**: C# changes in `WowViewer.Core.IO` (shared library). Python in `data-harvester/src/harvester/`. ✅
- **Real-Data Validation**: Validation against staged 3.3.5 client data. ✅
- **Residual Model Chain**: V19 is a standalone height predictor, not a residual model in the V14 chain. Acceptable as a standalone experiment. ✅
- **No Game Client Paths**: No `H:\CLIENTS` references. ✅

## Project Structure

```text
wow-viewer/
├── src/core/
│   ├── WowViewer.Core/Maps/TerrainTileTensorPack.cs  # Add McnrMask257
│   └── WowViewer.Core.IO/Maps/
│       ├── AdtTensorPackBuilder.cs                    # Fix AssembleNormals
│       ├── NpzTileSerializer.cs                       # Write mcnr_mask_257
│       └── RawArraySerializer.cs                      # Write mcnr_mask_257
├── data-harvester/
│   ├── src/harvester/
│   │   ├── v19_models.py      # V19HeightModel
│   │   ├── v19_dataset.py     # V19Dataset
│   │   └── v16_1_dataset.py   # Verify _interpolate_checkerboard_normals
│   └── scripts/
│       └── train_v19.py       # Training script
└── specs/066-v19-height-regressor/
    ├── spec.md
    └── plan.md
```

## Implementation Phases

### Phase 1: MCNR Checkerboard Fix (C# + Python verification)

**Goal**: Write raw MCNR normals + proper checkerboard mask to Zarr. Verify Python interpolation works.

**Approach**:
1. Add `McnrMask257` field to `TerrainTileTensorPack`
2. Modify `AssembleNormals` to remove interpolation, compute mask from MCNR sample positions
3. Add `mcnr_mask_257` to NPZ and Zarr serializers
4. Verify Python `_interpolate_checkerboard_normals` works correctly with raw data

**Files changed**:
- `src/core/WowViewer.Core/Maps/TerrainTileTensorPack.cs`
- `src/core/WowViewer.Core.IO/Maps/AdtTensorPackBuilder.cs`
- `src/core/WowViewer.Core.IO/Maps/NpzTileSerializer.cs`
- `src/core/WowViewer.Core.IO/Maps/RawArraySerializer.cs`

**Validation**: Build solution, run tests, generate test shard, verify mask pattern.

---

### Phase 2: V19 Dataset Adapter (Python)

**Goal**: V19Dataset provides V19 input/output contract from V18 Zarr stores.

**Approach**:
1. Create `v19_dataset.py` wrapping V161Dataset
2. Implement input channel selection (3ch minimap only, 6ch minimap + normals)
3. Implement liquid_height target override
4. Integrate curation manifest filtering

**Files changed**:
- `data-harvester/src/harvester/v19_dataset.py` (rewrite)
- `data-harvester/src/harvester/v16_1_dataset.py` (minor: verify interpolation)

**Validation**: Instantiate V19Dataset, verify shapes, verify liquid override.

---

### Phase 3: V19 Model Architecture (Python)

**Goal**: V19HeightModel with modernized V7 architecture.

**Approach**:
1. Clean up `v19_models.py` (mostly good, minor fixes)
2. Verify ResConvBlock + GroupNorm + BilinearUp architecture
3. Verify output shapes and parameter count

**Files changed**:
- `data-harvester/src/harvester/v19_models.py` (cleanup)

**Validation**: Forward pass test, parameter count verification.

---

### Phase 4: V19 Training Script (Python)

**Goal**: Clean training script with no cross-repo imports.

**Approach**:
1. Port V7 loss functions into `wow-viewer/data-harvester/src/harvester/v19_losses.py`
2. Rewrite `train_v19.py` using V18 infrastructure patterns
3. Implement grouped validation, curation-weighted sampling, early stopping

**Files changed**:
- `data-harvester/src/harvester/v19_losses.py` (new)
- `data-harvester/scripts/train_v19.py` (rewrite)

**Validation**: Run 2 epochs on small subset, verify checkpoint output.

---

### Phase 5: Baseline Training Run

**Goal**: Produce trained checkpoint and establish quality baseline.

**Approach**:
1. Train V19 on full V18 corpus
2. Evaluate height MAE on validation set
3. Save preview images for visual inspection

**Files changed**: None (training run only)

**Validation**: Height MAE reported, visual inspection passes.

## Complexity Tracking

No constitution violations. All phases are straightforward additions.
