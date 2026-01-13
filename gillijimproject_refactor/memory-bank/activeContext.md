# Active Context

## Current Focus: VLM Terrain Data Export (Jan 13, 2026)

### Status Summary
Successfully implemented a robust VLM dataset exporter that correlates terrain meshes, minimaps, and "decompiled" texture data.
- **Custom ADT Parsing**: Ported full ADT parsing to `WoWRollback.Core`, removing `Warcraft.NET` dependency.
- **Visual Data extraction**: Implemented `AlphaMapGenerator` to convert MCAL alpha maps into per-layer PNG masks, allowing the VLM to "see" texture distribution.
- **Mesh Export**: Integrated `TerrainMeshExporter` to generate aligned OBJ/MTL data.

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
1. **Deep dive into v14 WMO format** - Compare parsing with client code
2. **Study MirrorMachine exporter** - Reference implementation
3. **Investigate portal/group relationships** - May be causing drop-outs
4. Test with simpler WMOs to isolate the issue

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
