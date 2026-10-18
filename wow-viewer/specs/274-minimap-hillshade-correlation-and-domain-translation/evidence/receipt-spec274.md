# Verification Receipt: Spec 274 Minimap Hillshade Correlation & Domain Translation

**Date**: 2026-10-16  
**Branch**: `v0.6.1-dev`  
**Status**: Implemented & Verified (Gates A, B, C Passed)  

---

## 1. Files Created & Modified

### New Pipeline & Model Implementations
- `data-harvester/src/harvester/v60/dxt1_parity.py`: Vectorized BC1 / DXT1 4x4 block quantization parity codec matching `Dxt1TileCodec.cs`
- `data-harvester/src/harvester/v60/batch_export_clean_pairs.py`: Headless batch-export engine scanning 37 Zarr v3 map stores and emitting clean paired datasets (`real_minimap_rgb`, `synthetic_hillshade_dxt1`, `wdl_lattice_33`, `height_257`, `index.parquet`)
- `data-harvester/scripts/v60_batch_export_clean_pairs.py`: CLI driver with `--dry-run`, `--maps`, `--val-map`, and `--confirm-build`
- `data-harvester/src/harvester/v60/clean_pair_dataset.py`: PyTorch `Dataset` with train/val split loading and synchronous geometric/photometric augmentations
- `data-harvester/src/harvester/v60/domain_translation_model.py`: `DomainTranslationUNet` (5.5M parameters, strictly $\le 10\text{M}$ budget) and composite loss ($L_1$ + Sobel edge gradient + SSIM)
- `data-harvester/scripts/v60_train_domain_translation.py`: Training CLI harness supporting CUDA/CPU, AdamW, Cosine Annealing, checkpointing, and evaluation
- `data-harvester/src/harvester/v60/minimap_hillshade_correlator.py`: Correlation bridge and metric evaluator (Pearson $r$, MAE in yards, RMSE)

### Test Suites
- `data-harvester/tests/v60/test_batch_export_clean_pairs.py`: 4 tests (DXT1 parity, 4-color block quantization, WDL lattice combination, end-to-end dry-run & real build)
- `data-harvester/tests/v60/test_clean_pair_dataset.py`: 2 tests (split filtering, tensor shapes, value ranges, DataLoader batching)
- `data-harvester/tests/v60/test_domain_translation_model.py`: 4 tests (parameter budget $\le 10\text{M}$, forward pass, composite loss calculation, backward gradient flow)
- `data-harvester/tests/v60/test_minimap_hillshade_correlator.py`: 2 tests (elevation metric calculation and correlation validation)

### Governance & Registry
- `specs/274-minimap-hillshade-correlation-and-domain-translation/spec.md`: User stories, acceptance criteria, architectural bounds
- `specs/274-minimap-hillshade-correlation-and-domain-translation/plan.md`: Technical architecture, data flow diagram, and phase breakdown
- `specs/274-minimap-hillshade-correlation-and-domain-translation/tasks.md`: Phased task ledger
- `specs/STATUS.md`: Spec 274 registered in Active specs table
- `memory-bank/activeContext.md`: Dashboard updated with Spec 274 status

---

## 2. Verification Commands & Exit Status

| Stage | Command | Exit Code | Results |
|---|---|---|---|
| **Phase 1: Batch Export & Codec** | `uv run pytest tests/v60/test_batch_export_clean_pairs.py -v` | 0 | 4 passed in 4.39s |
| **Phase 2: PyTorch Dataset** | `uv run pytest tests/v60/test_clean_pair_dataset.py -v` | 0 | 2 passed in 5.72s |
| **Phase 3: Model & Training** | `uv run pytest tests/v60/test_domain_translation_model.py -v` | 0 | 4 passed in 3.56s |
| **Phase 4: Correlator Metrics** | `uv run pytest tests/v60/test_minimap_hillshade_correlator.py -v` | 0 | 2 passed in 0.98s |
| **All Test Suites** | `uv run pytest tests/v60/test_batch_export_clean_pairs.py tests/v60/test_clean_pair_dataset.py tests/v60/test_domain_translation_model.py tests/v60/test_minimap_hillshade_correlator.py -v` | 0 | **12 passed in 12.04s** |
| **Real Store Dry-Run** | `uv run python scripts/v60_batch_export_clean_pairs.py --dry-run` | 0 | Discovered 37 Zarr map stores, **4,448 candidate tiles** |
| **Training Dry-Run** | `uv run python scripts/v60_train_domain_translation.py --dry-run` | 0 | CUDA detected, 5,547,267 parameters, forward/backward pass verified |
| **Solution Integrity** | `dotnet build WowViewer.slnx -c Debug` | 0 | 288 warnings, 0 errors |

---

## 3. Criterion $\to$ Evidence Mapping

| Spec Criterion | Requirement | Verified Evidence | Status |
|---|---|---|---|
| **AC-001** | Headless Batch-Export Command | `scripts/v60_batch_export_clean_pairs.py` scans 37 Zarr v3 stores across `output/datasets/v60/` and discovers 4,448 complete tiles | **PASS** |
| **AC-002** | DXT1 Parity in Synthetic Targets | `dxt1_parity.py` quantizes synthetic hillshade targets through 4x4 block RGB565 endpoints and 4-color palette, eliminating the codec confound (`test_dxt1_block_quantization_effect`) | **PASS** |
| **AC-003** | Macro & Micro Elevation Alignment | `batch_export_clean_pairs.py` interlaces WDL outer 17x17 + inner 16x16 into (33, 33) lattice and aligns with (257, 257) ADT elevation (`test_export_clean_pairs_dry_run_and_build`) | **PASS** |
| **AC-004** | Lightweight Domain Translation Model | `DomainTranslationUNet` measured at 5,547,267 parameters (well below $\le 10\text{M}$ budget; checkpoint size $\approx 22\text{ MB}$) | **PASS** |
| **AC-005** | Object & Texture Normalization Fidelity | `DomainTranslationLoss` incorporates multi-scale $L_1$ + Sobel horizontal/vertical edge gradients + structural SSIM (`test_domain_translation_loss_components`) | **PASS** |
| **AC-006** | Terrain Elevation Correlation Benchmark | `compute_elevation_metrics` computes Pearson $r$, MAE, RMSE, and max error in yards without bias (`test_compute_elevation_metrics_exact`) | **PASS** |

---

## 4. Operator-Owned Execution Commands (PowerShell 7)

Per AGENTS.md §3 and §5, massive dataset materialization and multi-epoch GPU training are operator-owned. The exact, validated PowerShell 7 commands are:

### Step 1: Batch-Export Clean Pairs Across All 37 Maps
```powershell
# Navigate to data-harvester
Set-Location -Path "I:\parp\parp-tools\wow-viewer\data-harvester"

# Export clean paired dataset (4,448 tiles) to output/datasets/v60_clean_pairs.zarr
uv run python scripts/v60_batch_export_clean_pairs.py `
    --source-root "..\output\datasets\v60" `
    --output "..\output\datasets\v60_clean_pairs.zarr" `
    --val-map "Kalimdor" `
    --confirm-build
```

### Step 2: Train Domain Translation Style-Transfer Network (RTX GPU)
```powershell
# Navigate to data-harvester
Set-Location -Path "I:\parp\parp-tools\wow-viewer\data-harvester"

# Train 60 epochs with batch size 8 on CUDA
uv run python scripts/v60_train_domain_translation.py `
    --dataset "..\output\datasets\v60_clean_pairs.zarr" `
    --epochs 60 `
    --batch-size 8 `
    --lr 2e-4 `
    --checkpoint-dir "models/domain_translation" `
    --device "cuda"
```
