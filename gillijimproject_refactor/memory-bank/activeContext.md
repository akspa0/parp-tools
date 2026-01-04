# Active Context

## Current Focus: WMO v14→v17 Geometry Issues (Jan 4, 2026)

### Status Summary
WMO conversion has **lighting working** but **geometry drop-outs remain unresolved**. Complex WMOs like Ironforge show random missing sections. Quick fixes (UV flip, batch rebuilding) made things worse. Need deeper investigation into v14 WMO parsing.

### Session Dec 28, 2025 - Major Progress

#### Completed
1. **Documentation Diff**: Created `Alpha-Documentation-Diff.md` comparing wowdev.wiki vs Ghidra ground-truth
2. **v3 Project Structure**: Created `src/WoWMapConverter/` with Core and Cli projects
3. **Alpha Parsers Migrated**: WdtAlpha, MphdAlpha, MainAlpha, Mdnm, Monm
4. **DBC Integration**: DbcReader and AreaIdMapper (no external DBCTool needed)
5. **WMO v14→v17 Converter**: Full conversion with Ghidra-verified structure sizes
6. **MDX→M2 Converter**: Alpha models to WotLK M2 + .skin files
7. **Classic ADT v18**: Full parser with MH2O liquid support
8. **Split ADT (Cata+)**: Root/_tex0/_obj0/_lod file support
9. **Format Detector**: Auto-detect file types and WoW versions

#### v3 Architecture
```
src/WoWMapConverter/
├── WoWMapConverter.Core/
│   ├── Formats/
│   │   ├── Alpha/       ✅ WdtAlpha, MphdAlpha, MainAlpha, Mdnm, Monm
│   │   ├── Classic/     ✅ AdtV18 (with MH2O)
│   │   ├── Cataclysm/   ✅ SplitAdt (root/_tex0/_obj0/_lod)
│   │   ├── Modern/      📁 Created (Legion+ pending)
│   │   ├── Wmo/         📁 Created (v17+ pending)
│   │   ├── Models/      📁 Created (M2/M3 pending)
│   │   └── Shared/      ✅ Chunk, Mhdr, Mcin
│   ├── Converters/
│   │   ├── AlphaToLkConverter.cs    ✅ WDT orchestrator
│   │   ├── WmoV14ToV17Converter.cs  ✅ Full v14→v17
│   │   └── MdxToM2Converter.cs      ✅ MDX→M2 + .skin
│   ├── Dbc/
│   │   ├── DbcReader.cs             ✅ DBC/DB2 parser
│   │   └── AreaIdMapper.cs          ✅ AreaTable crosswalk
│   ├── Pm4/                         📁 Created (pending)
│   └── Services/
│       ├── ListfileService.cs       ✅ Asset path resolution
│       └── AreaIdCrosswalk.cs       ✅ CSV-based mapping
│
├── WoWMapConverter.Cli/             ✅ CLI with convert/convert-wmo/convert-mdx
└── WoWMapConverter.Gui/             📁 Planned (Avalonia)
```

### Completed (Dec 28, 2025 - Session 2)
1. **Wired AlphaToLkConverter** - Full Alpha→LK conversion using gillijimproject-csharp library
2. **AreaID Crosswalk** - `AdtAreaIdPatcher` patches MCNK AreaIDs post-conversion
3. **BlpService** - Full BLP read/resize/write with SereniaBLPLib + BCnEncoder
4. **MinimapService** - Stub for MCCV vertex color extraction

### Completed (Dec 28, 2025 - Session 3)
1. **MCLQ↔MH2O Converter** - Bidirectional liquid conversion based on Noggit3 + wowdev.wiki
   - `Formats/Liquids/MclqChunk.cs` - Full MCLQ structure (vertices, tiles, flow vectors)
   - `Formats/Liquids/Mh2oChunk.cs` - Full MH2O structure (headers, instances, attributes, LVF cases 0-3)
   - `Formats/Liquids/LiquidConverter.cs` - Bidirectional conversion with type mapping
2. **AdtMeshExporter** - OBJ export with minimap textures for terrain visualization
   - Based on proven `WoWRollback.AnalysisModule.AdtMeshExtractor`
   - Full GLB export available via AnalysisModule (SharpGLTF)
3. **WL* Liquid File Support** - Loose water level files for liquid recovery
   - `Formats/Liquids/WlFile.cs` - Unified WLW/WLM/WLQ reader (360-byte blocks, 4x4 vertex grid)
   - `Formats/Liquids/WlToLiquidConverter.cs` - WL* → MCLQ/MH2O with bilinear upscaling
4. **LkToAlphaModule Review** - Reviewed existing LK→Alpha conversion path
   - `AlphaWdtMonolithicWriter.cs` (126KB) - Main LK→Alpha monolithic WDT writer
   - `AlphaMcnkBuilder.cs` - MCNK chunk builder with MH2O→MCLQ conversion

### Critical Discovery: Alpha MCNK Positioning
**Alpha MCNK header has NO position fields** (unlike LK which has PosX/PosY/PosZ):
```c
struct SMChunk {  // Alpha 0.5.3
    uint32_t flags;
    uint32_t indexX;   // Chunk index 0-15, NOT world position
    uint32_t indexY;   // Chunk index 0-15, NOT world position
    float radius;      // Bounding sphere radius
    // ... rest of header (no position fields!)
};
```
The Alpha client **calculates** world positions from:
- Tile coordinates (from MAIN chunk offset in WDT)
- Chunk indices (indexX, indexY in MCNK header)
- Formula: `worldPos = tileOrigin + (chunkIndex * chunkSize)`

**Next: Use Ghidra MCP to verify exact position calculation formula.**

### Completed (Dec 28, 2025 - Session 5)
1. **LkToAlphaConverter** - Full LK→Alpha monolithic WDT conversion
   - `Converters/LkToAlphaConverter.cs` - Main converter orchestrator
   - `Builders/AlphaMhdrBuilder.cs` - Alpha MHDR chunk builder
   - `Builders/AlphaMainBuilder.cs` - Alpha MAIN chunk builder (4096 tiles)
   - `Builders/AlphaMcnkBuilder.cs` - Alpha MCNK builder with MH2O→MCLQ
2. **MH2O→MCLQ Integration** - Automatic liquid conversion in MCNK builder
3. **CLI Command** - `convert-lk-to-alpha` with options:
   - `--wdt`, `--map-dir`, `--output`, `--verbose`
   - `--skip-m2`, `--skip-wmo`, `--no-liquids`

### Key Coordinate Discovery
**LK and Alpha use the SAME world coordinate system for MDDF/MODF placements.**
No coordinate transformation needed - positions are written as-is.
```csharp
// Both LK and Alpha: X/Y plane, Z height
// MDDF/MODF positions: (posX, posZ, posY) where posZ is height
const float TileSize = 533.33333f;
const float WorldBase = 32f * TileSize;
```

### Session Jan 3-4, 2026 - WMO v14→v17 Fixes

#### Completed ✅
1. **Lighting Fixed** - MOCV generation from lightmaps + neutral gray fallback
2. **DiffuseColor Fix** - Replace black materials with neutral gray
3. **MOBA Bounding Boxes** - `CalculateBatchBoundingBoxes()` for all batches  
4. **MOCV for All Groups** - Exterior groups now get vertex colors
5. **v14 Index Regeneration** - Sequential indices when MOIN mismatches
6. **Texture Extraction** - BLP files copied to output

#### ⚠️ Unresolved: Geometry Drop-outs
Complex WMOs show random geometry drop-outs:
- Sections missing/not rendering in Ironforge
- Different parts drop out on each conversion run
- **Attempted fixes that made things WORSE:**
  - UV V-flip (broke textures further)
  - Forced batch rebuild (more drop-outs)
  - ValidateAndRepairGroupData (disabled)

#### Root Cause Analysis Needed
- v14 WMO parsing may have fundamental issues
- Batch/face/vertex relationships not processed correctly
- Need Ghidra analysis or MirrorMachine reference

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
