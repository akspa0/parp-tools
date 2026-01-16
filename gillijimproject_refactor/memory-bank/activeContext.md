# Active Context

## Current Focus: VLM Dataset Generation for V6 Training (Jan 16, 2026)

### Status Summary
**CRITICAL FIX**: Per-tile normalization was causing visible seams when tiles were stitched. Created `regenerate_heightmaps_global.py` to generate BOTH per-tile (shows hidden details) AND global (proper training data) heightmaps.

### Session Jan 16, 2026 (Late) - Global Normalization Fix

#### Root Cause: Per-Tile Normalization Seams ✅
**Problem**: Each tile was normalized to its own min/max, causing brightness discontinuities at tile boundaries when stitched.
**Solution**: Created Python script that:
1. **Pass 1**: Scans ALL tile JSONs to find global height range (e.g., Azeroth: -524.58 to 896.78)
2. **Pass 2**: Regenerates heightmaps using that single global range
3. **Pass 3**: Stitches into seamless full-map PNG

#### New Script: `regenerate_heightmaps_global.py` ✅
Generates **BOTH** heightmap types:
- `*_heightmap.png` - Global normalization (for ML training, seamless)
- `*_heightmap_local.png` - Per-tile normalization (reveals hidden terrain details)

**Usage**:
```bash
python regenerate_heightmaps_global.py <dataset_dir> [--no-stitch] [--max-size 16384]
```

**Outputs**:
- `images/*_heightmap.png` - Per-tile global normalized
- `images/*_heightmap_local.png` - Per-tile local normalized  
- `stitched/*_full_heightmap_global.png` - Stitched global
- `stitched/*_full_heightmap_local.png` - Stitched local
- `global_height_bounds.json` - Height range metadata

#### Dataset Exports In Progress
| Map | Status | Tiles | Output |
|-----|--------|-------|--------|
| Azeroth | ✅ Done | 685 | `053_azeroth_v10/` |
| Kalimdor | 🔧 Pending | ~951 | `053_kalimdor_v6/` |
| Kalidar | ✅ Done | 56 | `053_kalidar_v1/` |
| DeadminesInstance | 🔧 Pending | ? | `053_deadmines_v1/` |
| RazorfenKraulInstance | ✅ Done | 6 | `053_razorfen_v1/` |

**Client Path**: `H:\053-client\`

### Session Jan 16, 2026 (Earlier) - Alpha MCVT Format Fix

#### Key Files Updated
- `VlmDatasetExporter.cs` - Fixed GenerateHeightmap + added StitchHeightmapsToWebP
- `HeightmapBakeService.cs` - Updated to use Alpha MCVT format
- `render_heightmaps.py` - Updated to use Alpha MCVT format
- `render_normalmaps.py` - Updated to use Alpha MCNR format

### Previous Session Jan 15, 2026 (Late Evening) - Bug Fixes & Resolution Upgrade

#### Critical Bugs Fixed ✅
1. **MCVT/MCNR Layout Bug** (initially thought interleaved, later found Alpha uses sequential):
   - Fixed in both `render_heightmaps.py` and `render_normalmaps.py`

2. **Resolution Upgraded 129×129 → 256×256**:
   - Old: Only used outer vertices (9×9 per chunk = 129×129 total)
   - New: Uses ALL 145 vertices (outer at even pixels, inner at odd pixels)
   - 16 pixels per chunk edge = 256 pixels per tile = **4× more data**

3. **MCNR Padding Handled**:
   - Alpha exports: 448 bytes (with padding)
   - LK exports: 435 bytes (145×3 exact)
   - Fixed: Truncate to 435 to handle both

4. **V5 Model Updated**:
   - `OUTPUT_SIZE` changed from 129 to 256
   - Model architecture unchanged (U-Net)

#### Files Updated
- `render_heightmaps.py` - 256×256 with all vertices + gap interpolation
- `render_normalmaps.py` - 256×256 with all normals + gap interpolation  
- `train_height_regressor_v5_multichannel.py` - OUTPUT_SIZE = 256

#### UNSOLVED PROBLEM: Height Semantics 🚫

**The model doesn't know what heights mean.**

Current approach: minimap + normalmap → heightmap
- Normalmap encodes surface orientation (slopes)
- But **what is "up" vs "down"?** Model only sees relative gradients.
- No absolute height reference in the training data
- Tiles are normalized per-tile, losing global height context

**What the model needs to learn**:
- Water is typically at low elevation
- Mountains have certain textures at certain heights
- Transition zones (beach→grass→rock→snow) correlate with elevation
- But these are **WoW-specific** patterns, not universal

**Potential Solutions for Next Session**:

1. **Include absolute height bounds in training**:
   - Add min_height/max_height to training data
   - Train model to predict both heightmap AND height range
   - Reconstruct absolute heights during inference

2. **Use WDL as height reference**:
   - WDL contains 17×17 low-res height samples per tile
   - Provides absolute height context
   - Model learns: minimap + normalmap + WDL_hint → detailed heightmap

3. **Multi-scale training**:
   - Train on multiple adjacent tiles together
   - Model sees height continuity across tile boundaries
   - Learns global elevation patterns

4. **PM4 geometry as ground truth**:
   - PM4 pathfinding meshes have absolute world positions
   - Rasterize PM4 → heightmap for tiles without ADT data
   - Provides true height values for training

#### Next Steps (Fresh Chat)
1. Re-render all heightmaps and normalmaps at 256×256
2. Re-train V5 model with correct data
3. Implement absolute height encoding strategy
4. Test on tiles not in training set

### Previous Session Summary (Jan 15, 2026 - Earlier)

### Session Jan 14-15, 2026 - Minimap Baking & AI Smoothing (Previous)

#### Completed ✅
1.  **MinimapBakeService.cs - WoW Weighted Blend Algorithm**:
    - Fixed texture layer compositing to match WoW's `adt.fragment.shader` blending:
      - Layer 0 weight = `1.0 - sum(layer1..N alphas)`
      - Layer N weight = `alpha[N]`
      - Final color = weighted sum normalized
    - Added shadow overlay/debake functionality with configurable intensity.
    - Added `--export-layers` flag to export individual texture layers for debugging.

2.  **img2mesh.py - Heightmap Smoothing**:
    - Added post-processing smoothing to reduce AI-generated heightmap noise.
    - Smoothing methods: `gaussian`, `laplacian`, `bilateral`, `median`.
    - Added `--output-dir` option (default: `test_output/rebaked_minimaps`).
    - Updated argparse CLI with full options.

3.  **Ghidra Analysis - WoW Lighting Defaults**:
    - Analyzed `CGxLight` constructor in WoWClient.exe:
      - Default direction: `(0, 0, 1)` Z-up
      - Directional color: `RGB(255, 255, 255)` pure white
      - Ambient color: `RGB(0, 0, 0)` black
      - Intensities: `1.0` for both ambient and directional
    - Terrain lighting is data-driven from `Light.dbc` entries.

4.  **Documentation Updated**:
    - `scripts/README.md`: Comprehensive docs for all AI terrain tools.
    - `img2mesh.py` docstring: Updated with all new options and examples.

#### Previous Session (Jan 14) ✅
1.  **Tiny ViT Regressor (`train_tiny_regressor.py`)**:
    - Trained a lightweight Vision Transformer (ViT) to predict 145-float height arrays from 64x64 minimap crops.
    - Achieved in-memory RAM caching and normalization for fast training (~1hr per epoch on consumer GPU).
    - **Status**: Training is **Complete**. Model saved to `j:\vlm_output\wow_tiny_vit_regressor`.
2.  **Terrain Prefab Analysis (`terrain_librarian.py`)**:
    - Developed a tool to detect recurring geometry and alpha patterns (Prefabs).
    - Implemented a "Seed-and-Grow" algorithm to find multi-chunk macro prefabs.
    - Exported `prefab_instances.json` mapping all patterns to the global world grid.
3.  **C# Texture Pipeline Integration**:
    - **MinimapBakeService.cs**: Implemented a C# service for 4096x4096px high-res minimap reconstruction.
    - Composites 256x256 texture layers with per-chunk alpha masks using `SixLabors.ImageSharp`.
    - Integrated `vlm-bake` command into `WoWMapConverter.Cli`.

### Session Jan 14, 2026 - (Earlier) VLM Training & GGUF Export

#### Completed ✅
1.  **VLM Training Setup (`train_local.py`)**:
    - Configured Unsloth `SFTTrainer` for Qwen2-VL-8B.
    - Implemented custom `UnslothVisionDataCollator` and dataset loading.
    - Resolved VLM detection issues by passing `processing_class`.
2.  **GGUF Export Pipeline (`export_gguf.py`)**:
    - **Manual Pipeline**: Merges LoRA adapters -> Exports to F16 GGUF -> Quantizes to Q4_K_M.
    - **Robustness**: Explicitly handles PEFT model wrapping for merging.
    - **Dependencies**: Forces `gguf` library install from Git to match conversion script.
3.  **Dataset Enhancements**:
    - **Atlas Stitching**: `TileStitchingService` now stitches Shadow and Alpha maps into full-world atlases.
    - **Large Map Safety**: Added checks to skip stitching for maps > 16k pixels to prevent crashes.
4.  **Documentation**:
    - Created `docs/VLM_Training_Guide.md` detailing the entire workflow.

### Session Jan 13, 2026 - VLM Terrain Data Export

#### Completed ✅
1.  **Custom ADT Parser (`WoWRollback.Core`)**:
    - Implemented `AdtParser.cs` to handle `MVER`, `MHDR`, `MCIN`, `MTEX`, `MMDX`, `MDDF`, `MODF`, and `MCNK` chunks.
    - Created lightweight `AdtData` models to replace external library types.
2.  **Visual Texture Decompilation**:
    - Created `AlphaMapGenerator`: Converts raw `MCAL` (alpha map) data into grayscale PNG images.
    - Captures "pressure/opacity" data per texture layer, critical for VLM correlation.
3.  **VlmDatasetExporter Improvements**:
    - Updated to use the new `AdtParser`.
    - Exports rich JSON dataset:
        - `obj_content` / `mtl_content` (Mesh)
        - `layer_masks` (List of PNG paths for MCAL)
        - `textures` (List of BLP paths)
        - `objects` (WMO/M2 placements)
    - Verified build of `vlm-export` CLI command.

### Session Jan 10, 2026 - WMO Debugging & Fixes


#### Completed ✅
1.  **Resolved Corrupt Groups**: Identified that `ParseMogp` was unconditionally skipping 128 bytes for MOGP header (matching Client RAM logic), but `Ironforge.wmo` on disk likely uses 68-byte headers. Implemented **Adaptive Skipping** (peeking for `MOPY/MOVI/MOVT` magic) to handle both formats correctly.
2.  **Resolved UV Orientation**: Confirmed via `x64dbg-mcp` that v14 UVs in memory are standard `0-1` floats. The "Upside Down" issue was caused by an unnecessary `1.0 - V` flip in the converter. Removed this flip for a clean pass-through, preserving correct orientation.
3.  **WMO v14 Format Analysis**:
    - Confirmed Chunk Order: `MOPY -> MOVT -> MONR -> MOTV -> MOIN -> MOBA`.
    - Confirmed `MOIN` token usage (instead of `MOVI`) in v14 groups.
    - Validated `MOBA` batch structure handling.
4.  **Refactored WMO Batch Logic**:
    - Identified that storing `FirstFace` (index / 3) caused precision loss for batches starting on non-divisible indices (e.g. `100 / 3 * 3 = 99`).
    - Updated `WmoBatch` to store `FirstIndex` and `IndexCount` directly, eliminating geometry drop-outs caused by misalignment.

### Session Jan 3-4, 2026 - WMO v14→v17 Fixes (Previous)

#### Completed ✅
1. **Lighting Fixed** - MOCV generation from lightmaps + neutral gray fallback
2. **DiffuseColor Fix** - Replace black materials with neutral gray
3. **MOBA Bounding Boxes** - `CalculateBatchBoundingBoxes()` for all batches  
4. **MOCV for All Groups** - Exterior groups now get vertex colors
5. **v14 Index Regeneration** - Sequential indices when MOIN mismatches
6. **Texture Extraction** - BLP files copied to output


### Next Steps (Priority Order)
1. **Fix C# Build Errors**: Resolve dependencies for `MinimapBakeService` in `WoWMapConverter.Core`.
2. **Evaluate Tiny ViT Results**: Run `img2mesh.py` on diverse terrain to check height prediction accuracy.
3. **High-Res De-Baking**: Subtract known layers from minimaps to isolate unknown textures.
4. **Macro-Prefab Gallery**: Finalize visualization of detected 2x2 and 4x4 terrain patterns.

### Additional Modules to Integrate

#### BlpResizer (`BlpResizer/`)
- **Purpose**: Downscale BLP textures for Alpha client (max 256x256)
- **Features**: CASC extraction, batch processing, 7956 tilesets processed from 12.x
- **Dependencies**: SereniaBLPLib, CascLib, BCnEncoder.Net, ImageSharp
- **Key files**: `Program.cs`, `BlpWriter.cs`

#### MinimapModule (`WoWRollback/WoWRollback.MinimapModule/`)
- **Purpose**: Minimap extraction and conversion
- **Services**:
  - `BlpConverter.cs` - BLP format conversion
  - `MinimapExportService.cs` - Minimap tile extraction
  - `AdtMetadataExtractor.cs` - ADT metadata for minimap alignment
  - `SyntheticAdtGenerator.cs` - Generate ADTs from minimaps
- **Dependencies**: SereniaBLPLib, CascLib, ImageSharp, WoWRollback.Core

### Key Reference Sources
- **wow.export** (`lib/wow.export/src/js/3D/`) - Modern format loaders + WebGL renderers
- **WebWowViewerCpp** (`lib/WebWowViewerCpp/`) - C++ WoW viewer (powers wow.tools/mv/)
- **wowdev.wiki ADT/v18** - Split file format documentation
- **Ghidra analysis** (`WoWRollback/docs/`) - Ground-truth for Alpha formats

### Viewer Implementation References
Build C# viewer using these as reference:
- **wow.export renderers** (`lib/wow.export/src/js/3D/renderers/`):
  - `M2RendererGL.js` / `M2LegacyRendererGL.js` - M2 model rendering
  - `MDXRendererGL.js` - Alpha MDX model rendering
  - `WMORendererGL.js` / `WMOLegacyRendererGL.js` - WMO rendering
  - `M3RendererGL.js` - Legion+ M3 model rendering
- **WebWowViewerCpp** - Full C++ implementation with CASC support, map/model viewing

### Version Support Matrix
| Version | ADT | WMO | Models | Status |
|---------|-----|-----|--------|--------|
| Alpha 0.5.3 | Mono WDT | v14 | MDX | ✅ Full |
| Classic-WotLK | v18 | v17 | M2 | ✅ Full |
| Cata+ | Split | v17 | M2 | ✅ Read |
| Legion+ | Split+lod | v17+ | M2/M3 | 🔧 Planned |
