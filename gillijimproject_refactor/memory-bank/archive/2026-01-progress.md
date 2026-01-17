# Archived Progress Details - January 2026

## Session Jan 16, 2026 (Late) - Global Normalization Fix ✅

### Critical Bug: Per-Tile Normalization Seams
**Problem**: Each tile normalized to its own min/max caused visible brightness discontinuities at tile boundaries when stitched.
**Root Cause**: `VlmDatasetExporter.GenerateHeightmap` calculated per-tile min/max instead of global.

### Solution: `regenerate_heightmaps_global.py` ✅
New Python script that generates **BOTH** heightmap types:
1. **Global normalized** (`*_heightmap.png`) - For ML training, seamless stitching
2. **Per-tile normalized** (`*_heightmap_local.png`) - Reveals hidden terrain details

**Algorithm**:
1. Pass 1: Scan all tile JSONs to find global height range
2. Pass 2: Regenerate all heightmaps using global range
3. Pass 3: Stitch both types into full-map PNGs

**Azeroth v10 Results**:
- Global range: -524.58 to 896.78 (1421.36 units)
- 685 tiles processed
- Stitched output: 16384 × 15872 px
- **No more tile seams!**

### Dataset Exports Completed
| Map | Tiles | Output |
|-----|-------|--------|
| Azeroth | 685 | `053_azeroth_v10/` |
| Kalidar | 56 | `053_kalidar_v1/` |
| RazorfenKraulInstance | 6 | `053_razorfen_v1/` |

### Pending Exports
- Kalimdor → `053_kalimdor_v6/`
- DeadminesInstance → `053_deadmines_v1/`

### Files Created/Updated
- `scripts/regenerate_heightmaps_global.py` - New dual-mode heightmap generator
- `scripts/stitch_heightmaps.py` - PNG stitching utility

## Session Jan 16, 2026 (Earlier) - Alpha MCVT Format Fix ✅

### Critical Discovery: Alpha MCVT Format
**Alpha ADT uses a DIFFERENT MCVT layout than later WoW versions:**
- **Alpha**: 81 outer (9×9) FIRST, then 64 inner (8×8) = 145 total
- **Standard (LK+)**: Interleaved pattern (9 outer, 8 inner, 9 outer, 8 inner...)
- This was the root cause of all diagonal heightmap artifacts

### Bugs Fixed in VlmDatasetExporter.GenerateHeightmap ✅
| Bug | Fix |
|-----|-----|
| Wrong MCVT indexing | Changed from interleaved to Alpha format (81 outer + 64 inner) |
| Fixed normalization | Changed from hardcoded -2000/2000 to per-tile min/max |
| Heightmap quality | Now 100% accurate terrain representation |

### Files Updated
- `VlmDatasetExporter.cs` - Fixed GenerateHeightmap + added StitchHeightmapsToWebP
- `HeightmapBakeService.cs` - Updated to use Alpha MCVT format
- `render_heightmaps.py` - Updated to use Alpha MCVT format
- `render_normalmaps.py` - Updated to use Alpha MCNR format

## Session Jan 15, 2026 (Late Evening) - Critical Bug Fixes ✅

### Bugs Fixed
| Bug | Fix |
|-----|-----|
| MCVT layout | Initially thought interleaved; later discovered Alpha uses sequential 81+64 |
| MCNR layout | Same fix as MCVT |
| Resolution 129→256 | Now uses ALL 145 vertices per chunk (4× data) |
| MCNR padding (448 vs 435) | Truncate to 435 for both Alpha and LK |

### Files Updated
- `render_heightmaps.py` - 256×256 full resolution
- `render_normalmaps.py` - 256×256 full resolution
- `train_height_regressor_v5_multichannel.py` - OUTPUT_SIZE = 256

### 🚫 UNSOLVED: Height Semantics Problem

**Model doesn't understand what heights mean:**
- Normalmap shows slopes but not absolute elevation
- Per-tile normalization loses global height context
- No way to tell model "water = low, mountains = high"

**Proposed Solutions for Next Session**:
1. **Absolute height bounds** - include min/max in training, predict range
2. **WDL as hint** - 17×17 low-res height grid as additional input
3. **Multi-tile training** - adjacent tiles for continuity
4. **PM4 rasterization** - use pathfinding mesh for ground truth

## Session Jan 15, 2026 (Evening) - V5 Model ✅

### V5 Multi-Channel Model
- 6-channel input (minimap+normalmap) → heightmap
- Best val_loss: 0.1558
- Checkpoint: `J:\vlm_output\wow_height_regressor_v5_multichannel\best_model.pt`

### OBJ/MTL Export
- Fixed Z scale, face winding, UV clamping
- Batch processing to single timestamped folder

## Session Jan 15, 2026 (Earlier) - V4-DPT Training ✅

### V4-DPT Training Script
Created `train_height_regressor_v4_dpt.py` using Hugging Face DPT:

| Component | Status |
|-----------|--------|
| DPT model integration | ✅ `Intel/dpt-large` |
| Multi-loss function | ✅ L1 + Gradient + SSIM + Edge + Scale-Invariant |
| Best val_loss | ✅ 0.3705 at Epoch 4 |

## Session Jan 14, 2026 - Image-to-Height & Texture Pipeline ✅

### Image-to-Height Regression
- **Tiny ViT Model**: Configured `ViTForImageClassification` for 145-float regression.
- **Dataset Generation**: `generate_height_regression_dataset.py` creates 4,096 samples from VLM exports.
- **Inference**: `img2mesh.py` generates 3D OBJ meshes with UV textures from minimap crops.
- **Status**: Training is currently running (~35% complete).

### Terrain Prefab Analysis
- **Canonical Hashing**: Rotation/Mirror invariant hashing for geometry chunks.
- **Seed-and-Grow**: Algorithm for detecting large multi-chunk patterns (Macro-Prefabs).
- **Library**: `prefab_instances.json` created for the world.

### C# Texture Pipeline (Pivot)
- **MinimapBakeService.cs**: New C# service for high-res compositing.
- **CLI Integration**: Added `vlm-bake` command.
- **Status**: Code implemented; awaiting dependency resolution for `SixLabors.ImageSharp`.

## Session Jan 14, 2026 - (Earlier) VLM Training & GGUF

### VLM Pipeline Complete ✅
Full end-to-end pipeline established for training AI on WoW terrain:
1. **Extraction**: `vlm-export` generates rich JSON + Images (Minimap/Shadows/Alpha).
2. **Curation**: `vlm_curate.py` formats data for Qwen2-VL.
3. **Training**: `train_local.py` finetunes model (Unsloth 4-bit LoRA).
4. **Export**: `export_gguf.py` converts trained adapters to `q4_k_m.gguf` for Inference.

### Documentation
- `docs/VLM_Training_Guide.md` - Complete guide to the pipeline.

## Session Jan 3-4, 2026 - WMO v14→v17 Conversion Fixes

### WMO Lighting Fix ✅
Resolved dark/black WMO rendering in Noggit:

| Component | Status |
|-----------|--------|
| MOLV parsing | ✅ Lightmap UVs per face-vertex |
| MOLM parsing | ✅ Lightmap metadata (offset, width, height) |
| MOLD parsing | ✅ Raw lightmap pixel data |
| `GenerateMocvFromLightmaps()` | ✅ Samples lightmaps → MOCV |
| Neutral gray fallback | ✅ RGB=128 if no lightmap data |
| DiffuseColor fix | ✅ Replace black (L<32) with gray |
| MOCV for all groups | ✅ Exterior groups now get vertex colors |

### MOBA Batch Bounding Box ✅
Fixed WMO placement failures by calculating `unknown_box`:
- `CalculateBatchBoundingBoxes()` computes min/max from vertex positions
- Applied to both rebuilt and native batches
- Batches now write proper bx,by,bz,tx,ty,tz values

### v14 Index Handling ✅
- Regenerates sequential indices when MOIN mismatches expected count
- Handles case where v14 MOVT contains `nFaces * 3` sequential vertices

### Texture Extraction ✅
- `ExtractWmoTextures()` copies BLP files from `--wmo-dir` to output

### ⚠️ Unresolved: Geometry Drop-outs
Complex WMOs (Ironforge) show random geometry drop-outs:
- Sections of geometry missing or not rendering
- Drop-outs appear in different places on each conversion run
- **Attempted fixes that made things worse:**
  - UV V-flip (1-Y) - broke texture alignment further
  - Forced batch rebuilding - caused more drop-outs
  - ValidateAndRepairGroupData - disabled (made both issues worse)

### Root Cause Analysis Needed
The v14 WMO parsing appears to have fundamental issues:
- Batch/face/vertex relationship handling may be incorrect
- Portal geometry not being processed correctly
- Need to study Ghidra v14 client code or MirrorMachine exporter
