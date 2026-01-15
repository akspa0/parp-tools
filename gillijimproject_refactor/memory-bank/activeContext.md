# Active Context

## Current Focus: VLM Terrain Data Export (Jan 13, 2026)

### Status Summary
Successfully implemented a robust VLM dataset exporter, configured Unsloth training on Windows, and established a manual GGUF export pipeline.
- **VLM Pipeline Complete**: End-to-end workflow from WOW client -> VLM Training -> GGUF Model.
- **GGUF Export**: Created manual merge/convert/quantize pipeline to bypass Unsloth/Windows issues.
- **Documentation**: Comprehensive VLM Training Guide created.

### Session Jan 14, 2026 - Image-to-Height & Texture Pipeline

#### Completed ✅
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

#### In Progress 🚧
- Super-Resolution Baking: Debugging C# build errors (likely missing `SixLabors.ImageSharp` dependency).
- Tiny ViT Training: Evaluating results via `img2mesh.py` once weights are finalized.

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
