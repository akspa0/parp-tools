# Progress

## ✅ Working

### Input Parsers (Standardized)
- **Alpha WDT/ADT**: Monolithic format, MCLQ liquids, reversed FourCC handling
- **LK 3.3.5 ADT**: Split format (root + _obj0 + _tex0), MH2O liquids
- **WMO v14/v17**: Both directions implemented
- **M2/MDX**: Framework ready (needs testing)
- **BLP**: BlpResizer complete — 7956 tilesets processed from WoW 12.x

### Standalone Tools
- **BlpResizer**: ✅ Production-ready, CASC extraction works
- **AlphaWdtInspector**: ✅ Diagnostics CLI functional
- **DBCTool.V2**: ✅ Crosswalk CSV generation works
- **vlm-export**: ✅ Extracts ADT/WDT to JSON dataset + Stitched Atlases
- **train_local.py**: ✅ Unsloth Qwen2-VL training script (Windows compatible)
- **export_gguf.py**: ✅ Manual GGUF export (Merge -> Convert -> Quantize)
- **train_tiny_regressor.py**:  [/] Tiny ViT Image-to-Height training (In Progress)
- **terrain_librarian.py**: ✅ Canonical geometry/alpha prefab detection
- **MinimapBakeService.cs**: 🚧 C# Super-Resolution baker (Build Errors)

### Data Generation
- **WDL→ADT**: ✅ Generates terrain from WDL heights
- **MCCV Painting**: ✅ `MccvPainter.cs` generates vertex colors from minimap PNGs (interleaved layout fixed)
- **PM4 MODF Reconstruction (test12)**: ✅ 1101 entries in `pm4-adt-test12/modf_reconstruction/`
- **PM4 MODF Reconstruction (test13)**: ✅ `pm4-reconstruct-modf` CLI wraps `Pm4ModfReconstructor` over `pm4-adt-test13/wmo_flags` → `pm4-adt-test13/modf_reconstruction/`

### PM4 Pipeline Components
- **`Pm4ModfReconstructor`**: ✅ Matches PM4 objects to WMO library, generates MODF entries
- **`Pm4WmoGeometryMatcher`**: ✅ Geometry-based WMO matching using principal extents
- **`wmo-batch-extract`**: ✅ Extracts WMO collision geometry into per-WMO folders with one OBJ per group/flag (`pm4-adt-test13/wmo_flags/`)
- **`pm4-reconstruct-modf` CLI**: ✅ Uses `Pm4ModfReconstructor` + `AdtModfInjector.ServerToAdtPosition` to write `modf_entries.csv`, `mwmo_names.csv`, and `placement_verification.json`
- **`wmo_library.json` (legacy)**: ✅ 352 WMO entries with pre-computed geometry stats (test12 snapshot)

## ⚠️ Partial / Broken

### AdtModfInjector - BROKEN
- **Problem**: Appends MWMO/MODF chunks to end of file
- **Result**: Corrupted ADTs that Noggit cannot read
- **Root cause**: ADT chunks must be in specific order with correct MHDR/MCIN offsets

### Warcraft.NET Terrain.Serialize() - BROKEN
- **Problem**: Corrupts MCNK data during parse→serialize roundtrip
- **Evidence**: MCNK loses ~2,048 bytes after roundtrip
- **Result**: Noggit crashes on load
- **DO NOT USE** for ADT serialization

### Split ADT Merging - ABANDONED
- Custom `AdtPatcher.MergeSplitAdt()` produces corrupted output
- **Decision**: Use WoWMuseum ADTs as base instead of merging split files
 
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


## Next Steps (WoWMapConverter v3)
1. **Test LK→Alpha conversion** - Validate with real LK ADT data
2. **Port WMO v17+ loader** - from wow.export
3. **Port M2 MD21 loader** - from wow.export  
4. **Integrate PM4 pipeline** - from WoWRollback.PM4Module
5. **Add GUI** - Avalonia-based with 3D viewer
6. Test patched ADTs in WoW 3.3.5 client

## Session Dec 28, 2025 - Session 5: LK→Alpha Converter

### LkToAlphaConverter Implementation
Added complete LK→Alpha monolithic WDT conversion to WoWMapConverter.Core:

| File | Purpose |
|------|---------|
| `Converters/LkToAlphaConverter.cs` | Main converter orchestrator |
| `Builders/AlphaMhdrBuilder.cs` | Alpha MHDR chunk builder |
| `Builders/AlphaMainBuilder.cs` | Alpha MAIN chunk builder (4096 tiles) |
| `Builders/AlphaMcnkBuilder.cs` | Alpha MCNK builder with MH2O→MCLQ |

### CLI Command Added
```bash
wowmapconverter convert-lk-to-alpha <wdt> [options]
  --wdt <path>       Input LK WDT file
  --map-dir <dir>    Directory containing LK ADT files
  --output, -o       Output Alpha WDT path
  --skip-m2          Skip M2 doodad placements
  --skip-wmo         Skip WMO placements
  --no-liquids       Disable MH2O → MCLQ conversion
  --verbose, -v      Verbose output
```

### Key Discovery: Coordinate Systems
**LK and Alpha use the SAME world coordinate system for MDDF/MODF placements.**
No coordinate transformation needed - positions are written as-is.
- Both use X/Y plane with Z as height
- MDDF/MODF store positions as (posX, posZ, posY) where posZ is height
- TileSize = 533.33333f, WorldBase = 32 * TileSize

## Session Dec 28, 2025 - Session 4: WL* Support + LkToAlpha Review

### WL* Liquid File Support
Added support for loose "Water Level" files (WLW/WLM/WLQ) for recovering missing water planes:

| File | Purpose |
|------|--------|
| `Formats/Liquids/WlFile.cs` | Unified WL* reader (360-byte blocks, 4x4 vertex grid) |
| `Formats/Liquids/WlToLiquidConverter.cs` | WL* → MCLQ/MH2O with bilinear 4x4→9x9 upscaling |

**WL* File Types:**
- WLW: Water Level Water (`*QIL` magic)
- WLM: Water Level Magma (always magma type)
- WLQ: Water Level alternate (`2QIL` magic, WMO-style types)

**Block Structure (360 bytes):**
- 16 vertices (4x4 grid, z-up) = 192 bytes
- Coord (2 floats) = 8 bytes
- Data (80 ushorts) = 160 bytes

### LkToAlphaModule Review
Reviewed existing LK→Alpha conversion path:
- `AlphaWdtMonolithicWriter.cs` (126KB) - Main monolithic WDT writer
- `AlphaMcnkBuilder.cs` - MCNK builder with MH2O→MCLQ conversion
- `McnkHeader` / `McnkAlphaHeader` structs in `gillijimproject-csharp`

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

**Implication**: Alpha client calculates world positions from tile + chunk indices.
**Next**: Use Ghidra MCP to verify exact formula.

### Ghidra MCP Functions Found
- `CMapChunk` @ `0x00698510` - Chunk constructor
- `AddMapChunk` @ `0x0066b050` - Scene chunk addition
- `CMapArea` @ `0x006aa880` - Area (tile) management
- `CMapArea::AsyncCallback` @ `0x006ab280` - Async tile loading

## Session Dec 28, 2025 - Session 3: MCLQ↔MH2O + Mesh Export

### Liquid Conversion (MCLQ↔MH2O)
Implemented proper bidirectional liquid converter based on:
- **Noggit3** implementation (`liquid_layer.cpp`, `liquid_chunk.cpp`)
- **wowdev.wiki** ADT_v18 documentation

| File | Purpose |
|------|--------|
| `Formats/Liquids/MclqChunk.cs` | MCLQ structure: 9x9 vertices, 8x8 tiles, flow vectors |
| `Formats/Liquids/Mh2oChunk.cs` | MH2O structure: headers, instances, LVF cases 0-3 |
| `Formats/Liquids/LiquidConverter.cs` | Bidirectional conversion with type mapping |

**MCLQ Vertex Types:**
- Water/Ocean: depth, flow0Pct, flow1Pct, height
- Magma: s, t (UV coords), height

**MH2O Vertex Formats (LVF):**
- Case 0: float heightmap + byte depthmap
- Case 1: float heightmap + ushort[2] uvmap
- Case 2: byte depthmap only (height = 0)
- Case 3: float heightmap + ushort[2] uvmap + byte depthmap

### Terrain Mesh Export
Added `AdtMeshExporter` service for OBJ export with minimap textures.
- Based on proven `WoWRollback.AnalysisModule.AdtMeshExtractor`
- Full GLB export available via AnalysisModule (requires SharpGLTF)

### WoWMapConverter.Core Services Summary
| Service | Purpose |
|---------|--------|
| `AdtAreaIdPatcher` | Post-conversion MCNK AreaID patching |
| `BlpService` | BLP read/resize/write (SereniaBLPLib + BCnEncoder) |
| `MinimapService` | MCCV vertex color extraction |
| `AdtMeshExporter` | ADT→OBJ export with minimap textures |
| `AreaIdCrosswalk` | CSV-based Alpha→LK AreaID mapping |
| `ListfileService` | Asset path resolution |

## Session Dec 28, 2025 - WoWMapConverter v3 ADT Pipeline Wired

### Core Pipeline Complete
- **AlphaToLkConverter** now uses `gillijimproject-csharp` library via ProjectReference
- Full Alpha WDT → LK WDT + ADT conversion working
- MCLQ liquids passthrough working
- MDDF/MODF placements with Y/Z coordinate swap working

### New Services Added
| Service | Location | Purpose |
|---------|----------|---------|
| `AdtAreaIdPatcher` | `Services/` | Post-conversion MCNK AreaID patching |
| `BlpService` | `Services/` | BLP read/resize/write (SereniaBLPLib + BCnEncoder) |
| `MinimapService` | `Services/` | MCCV vertex color extraction stub |

### Dependencies Added to Core
- `gillijimproject-csharp` (ProjectReference) - Alpha/LK format library
- `SereniaBLPLib` (ProjectReference) - BLP file reading
- `BCnEncoder.Net` (NuGet) - DXT compression
- `SixLabors.ImageSharp` (NuGet) - Image processing

### Modules to Integrate
| Module | Location | Purpose |
|--------|----------|---------|
| BlpResizer | `BlpResizer/` | BLP texture resizing (max 256x256 for Alpha) |
| MinimapModule | `WoWRollback/WoWRollback.MinimapModule/` | Minimap extraction/conversion |
| PM4Module | `WoWRollback/WoWRollback.PM4Module/` | PM4 pathfinding mesh → MODF reconstruction |

## Session Dec 28, 2025 - WoWMapConverter v3 Major Build

### Documentation
- Created `Alpha-Documentation-Diff.md` - wowdev.wiki vs Ghidra ground-truth comparison
- ~70% wiki accuracy, critical errors in WMO v14 structure sizes (MOGI 40 not 32, MOPY 4 not 2)

### v3 Project Created (`src/WoWMapConverter/`)
- **WoWMapConverter.Core**: Library with Formats/Converters/Dbc/Services
- **WoWMapConverter.Cli**: Unified CLI (convert, convert-wmo, convert-mdx)

### Format Support Implemented
| Format | Location | Status |
|--------|----------|--------|
| Alpha WDT/ADT | `Formats/Alpha/` | ✅ WdtAlpha, MphdAlpha, MainAlpha, Mdnm, Monm |
| Classic ADT v18 | `Formats/Classic/` | ✅ AdtV18 with MH2O liquid |
| Split ADT (Cata+) | `Formats/Cataclysm/` | ✅ SplitAdt (root/_tex0/_obj0/_lod) |
| Format Detection | `Formats/FormatDetector.cs` | ✅ Auto-detect types/versions |

### Converters Implemented
| Converter | Status |
|-----------|--------|
| WMO v14→v17 | ✅ `WmoV14ToV17Converter.cs` - Ghidra-verified sizes |
| MDX→M2 | ✅ `MdxToM2Converter.cs` - With .skin file generation |
| Alpha→LK | ✅ `AlphaToLkConverter.cs` - Orchestrator (needs wiring) |

### DBC Integration (No External Tool Needed)
- `DbcReader.cs` - DBC/DB2 file parser
- `AreaIdMapper.cs` - AreaTable crosswalk (0.5.3 → 3.3.5)

### Earlier: WMO→Q3 BSP Converter
- `--q3v2`: Direct BSP output with proper IBSP format
- `--map`: GtkRadiant .map output (2065 brushes for castle01)
- `--split-groups`: Large WMO segmentation (Karazhan → 101 files)
- Fixed triangle winding order (CCW), VisData headers, lump alignment
- **Ghidra Analysis**: WMO v14 and MDX formats documented

## Session Dec 11, 2025 - PM4 Verification Complete
- PM4→WMO matching **proven working**: 1101 placements, 351 WMOs, 163 tiles
- Verification JSON: `pm4_full_verification.json`
- New CLI commands: `verify-pm4-data`, `csv-to-json`
- Fixed paths documented in `.windsurf/rules/data-paths.md`

## Key Files

| File | Status |
|------|--------|
| `WoWRollback.PM4Module/Pm4AdtPatcher.cs` | ⚠️ Needs update to add WMO names |
| `WoWRollback.PM4Module/MccvPainter.cs` | ✅ Fixed interleaved vertex layout |
| `WoWRollback.PM4Module/Program.cs` | ✅ Has `inject-modf` command (needs fix) |
| `WoWRollback.Core/Services/PM4/AdtModfInjector.cs` | ❌ BROKEN - appends chunks incorrectly |
| `WoWRollback.Core/Services/PM4/Pm4ModfReconstructor.cs` | ✅ Works - generates MODF from PM4 |
| `WoWRollback.AdtModule/` | ✅ Known-good LK ADT write path (Alpha→LK) using WowFiles, reference for MuseumAdtPatcher offsets/structure |
| `WoWRollback.LkToAlphaModule/` | ⚠️ LK↔Alpha ADT/WDT models and writers; placements coord system still being tuned but useful for MODF/MDDF and liquids wiring |

## Data Inventory

| Data | Location | Count |
|------|----------|-------|
| PM4 files | `test_data/development/World/Maps/development/*.pm4` | 616 |
| Split Cata ADTs | `test_data/development/World/Maps/development/*.adt` | 466 root |
| WoWMuseum ADTs | `test_data/WoWMuseum/335-dev/World/Maps/development/*.adt` | 2303 |
| Minimap PNGs | `test_data/minimaps/development/*.png` | 2252 |
| MODF entries (test12) | `pm4-adt-test12/modf_reconstruction/modf_entries.csv` | 1101 |
| WMO names (test12) | `pm4-adt-test12/modf_reconstruction/mwmo_names.csv` | 352 |
| WMO collision (per-group/flag) | `pm4-adt-test13/wmo_flags/` | many OBJ files |
| MODF entries (test13, current) | `pm4-adt-test13/modf_reconstruction/modf_entries.csv` | _current run_ |
| WMO names (test13, current) | `pm4-adt-test13/modf_reconstruction/mwmo_names.csv` | _current run_ |
