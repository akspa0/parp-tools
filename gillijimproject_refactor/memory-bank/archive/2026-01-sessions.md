# Archived Sessions - January 2026

## Session Jan 19, 2026 - Minimap TRS Path Resolution Fix ✅

### Root Cause: TRS Column Order + Coordinate Padding
**Problem**: 3.0.1/4.0.0 VLM export only finding 2/342 minimap tiles despite md5translate.trs loading 13,531 entries.

**Issues Found via Ghidra RE + wowdev.wiki/TRS.md**:
1. **Column Order Reversed**: Code assumed `<hash> <plain>` but TRS format is `<plain>\t<hash>`
2. **Coordinate Padding Wrong**: Code used `D2` for both x,y but TRS uses `map_%d_%02d.blp` (x not padded)

**Files Modified**:
- `Md5TranslateResolver.cs` - Swapped column order in parser (lines 163-177)
- `VlmDatasetExporter.cs` - Added correct TRS format candidates (lines 1537-1575)

**Result**: 50/50 tiles now found (100% success rate)

### Remaining Issues (LK/Cata ADT)
- **Heightmaps**: ❌ Values corrupted - likely MCVT format difference from Alpha
- **Normal Maps**: ❌ Incorrect data - likely MCNR offset issue

## Session Jan 16, 2026 (Late) - Global Normalization Fix ✅

### Root Cause: Per-Tile Normalization Seams
**Problem**: Each tile was normalized to its own min/max, causing brightness discontinuities at tile boundaries when stitched.
**Solution**: Created Python script that:
1. **Pass 1**: Scans ALL tile JSONs to find global height range (e.g., Azeroth: -524.58 to 896.78)
2. **Pass 2**: Regenerates heightmaps using that single global range
3. **Pass 3**: Stitches into seamless full-map PNG

### New Script: `regenerate_heightmaps_global.py` ✅
Generates **BOTH** heightmap types:
- `*_heightmap.png` - Global normalization (for ML training, seamless)
- `*_heightmap_local.png` - Per-tile normalization (reveals hidden terrain details)

## Session Jan 16, 2026 (Earlier) - Alpha MCVT Format Fix

### Key Files Updated
- `VlmDatasetExporter.cs` - Fixed GenerateHeightmap + added StitchHeightmapsToWebP
- `HeightmapBakeService.cs` - Updated to use Alpha MCVT format
- `render_heightmaps.py` - Updated to use Alpha MCVT format
- `render_normalmaps.py` - Updated to use Alpha MCNR format

## Session Jan 15, 2026 (Late Evening) - Bug Fixes & Resolution Upgrade

### Critical Bugs Fixed ✅
1. **MCVT/MCNR Layout Bug**: Fixed in both `render_heightmaps.py` and `render_normalmaps.py`
2. **Resolution Upgraded 129×129 → 256×256**: Old: Only outer vertices (9×9), New: ALL 145 vertices (4× more data)
3. **MCNR Padding Handled**: Alpha exports: 448 bytes, LK exports: 435 bytes - Fixed: Truncate to 435
4. **V5 Model Updated**: `OUTPUT_SIZE` changed from 129 to 256

### UNSOLVED PROBLEM: Height Semantics 🚫
**The model doesn't know what heights mean.**
Current approach: minimap + normalmap → heightmap
- Normalmap encodes surface orientation (slopes)
- But **what is "up" vs "down"?** Model only sees relative gradients.
- No absolute height reference in the training data
- Tiles are normalized per-tile, losing global height context

**Potential Solutions**:
1. Include absolute height bounds in training
2. Use WDL as height reference (17×17 low-res samples)
3. Multi-scale training (adjacent tiles together)
4. PM4 geometry as ground truth

## Session Jan 14-15, 2026 - Minimap Baking & AI Smoothing

### Completed ✅
1. **MinimapBakeService.cs - WoW Weighted Blend Algorithm**:
   - Fixed texture layer compositing to match WoW's `adt.fragment.shader` blending
   - Added shadow overlay/debake functionality with configurable intensity
   - Added `--export-layers` flag to export individual texture layers

2. **img2mesh.py - Heightmap Smoothing**:
   - Added post-processing smoothing to reduce AI-generated heightmap noise
   - Smoothing methods: `gaussian`, `laplacian`, `bilateral`, `median`

3. **Ghidra Analysis - WoW Lighting Defaults**:
   - Analyzed `CGxLight` constructor in WoWClient.exe
   - Default direction: `(0, 0, 1)` Z-up
   - Directional color: `RGB(255, 255, 255)` pure white
   - Ambient color: `RGB(0, 0, 0)` black

## Session Jan 14, 2026 - VLM Training & GGUF Export

### Completed ✅
1. **VLM Training Setup (`train_local.py`)**:
   - Configured Unsloth `SFTTrainer` for Qwen2-VL-8B
   - Implemented custom `UnslothVisionDataCollator` and dataset loading
   - Resolved VLM detection issues by passing `processing_class`

2. **GGUF Export Pipeline (`export_gguf.py`)**:
   - Manual Pipeline: Merges LoRA adapters → Exports to F16 GGUF → Quantizes to Q4_K_M
   - Robustness: Explicitly handles PEFT model wrapping for merging
   - Dependencies: Forces `gguf` library install from Git

## Session Jan 13, 2026 - VLM Terrain Data Export

### Completed ✅
1. **Custom ADT Parser (`WoWRollback.Core`)**:
   - Implemented `AdtParser.cs` to handle `MVER`, `MHDR`, `MCIN`, `MTEX`, `MMDX`, `MDDF`, `MODF`, and `MCNK` chunks
   - Created lightweight `AdtData` models to replace external library types

2. **Visual Texture Decompilation**:
   - Created `AlphaMapGenerator`: Converts raw `MCAL` (alpha map) data into grayscale PNG images
   - Captures "pressure/opacity" data per texture layer

3. **VlmDatasetExporter Improvements**:
   - Updated to use the new `AdtParser`
   - Exports rich JSON dataset with obj_content, layer_masks, textures, objects

## Session Jan 10, 2026 - WMO Debugging & Fixes

### Completed ✅
1. **Resolved Corrupt Groups**: Identified that `ParseMogp` was unconditionally skipping 128 bytes for MOGP header, but `Ironforge.wmo` on disk likely uses 68-byte headers. Implemented **Adaptive Skipping**.

2. **Resolved UV Orientation**: Confirmed via `x64dbg-mcp` that v14 UVs in memory are standard `0-1` floats. The "Upside Down" issue was caused by an unnecessary `1.0 - V` flip in the converter.

3. **WMO v14 Format Analysis**:
   - Confirmed Chunk Order: `MOPY -> MOVT -> MONR -> MOTV -> MOIN -> MOBA`
   - Confirmed `MOIN` token usage (instead of `MOVI`) in v14 groups
   - Validated `MOBA` batch structure handling

4. **Refactored WMO Batch Logic**:
   - Identified that storing `FirstFace` (index / 3) caused precision loss
   - Updated `WmoBatch` to store `FirstIndex` and `IndexCount` directly

## Session Jan 3-4, 2026 - WMO v14→v17 Fixes

### Completed ✅
1. **Lighting Fixed** - MOCV generation from lightmaps + neutral gray fallback
2. **DiffuseColor Fix** - Replace black materials with neutral gray
3. **MOBA Bounding Boxes** - `CalculateBatchBoundingBoxes()` for all batches
4. **MOCV for All Groups** - Exterior groups now get vertex colors
5. **v14 Index Regeneration** - Sequential indices when MOIN mismatches
6. **Texture Extraction** - BLP files copied to output
