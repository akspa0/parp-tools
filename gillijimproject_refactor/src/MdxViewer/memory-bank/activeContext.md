# Active Context — MdxViewer Renderer Reimplementation

## Current Focus

**Fixing 0.6.0 MPQ file extraction.** WMO v16 root files and ADT tiles are found in MPQ hash tables but decompression fails. Compression type `0x08` (PKWARE DCL implode) returns invalid dict size bits. Need correct PKWARE DCL implementation — StormLib reference code in `lib/StormLib/src/pklib/explode.c` is the ground truth. Ghidra RE prompts written for 0.5.3 (with PDB) and 0.6.0 (without PDB) to verify from the binary.

## Immediate Next Steps

1. **Fix PKWARE DCL decompression** — Port StormLib's `explode.c` faithfully to C#, OR diagnose if the data read from MPQ is at wrong offset (block table decryption issue).
2. **Test v16 WMO root loading** — Once decompression works, root files (e.g., Big_Keep.wmo, 472 bytes) should extract and parse.
3. **Test 0.6.0 ADT loading** — ADTs in terrain.MPQ may also use PKWARE compression for small tiles.
4. **Run Ghidra prompts** — `prompt-053-mpq.md` (0.5.3 with PDB, best starting point) and `prompt-060-mpq.md` (0.6.0) to verify MPQ decompression from binary.
5. **M2/WMOv17 readers** — For Standard WDT object rendering (lower priority until MPQ extraction is fixed).

## Session 2026-02-09 Summary

### WMO v16 Root File Loading Investigation
- **Symptom**: WMO v16 root files (e.g., `Big_Keep.wmo`) fail to load with "Failed to read" — group files load but without textures/lighting
- **Root cause chain**: `MpqDataSource.ReadFile` → `NativeMpqService.ReadFile` → `FindFileInArchive` succeeds → `ReadFileFromArchive` returns null
- **Block info**: offset=435912, size=318 (compressed), fileSize=472 (decompressed), flags=0x80000200 (EXISTS|COMPRESSED)
- **Decompression failure**: Compression type byte = `0x08` (PKWARE DCL), but remaining data has dictShift=0 (expected 4/5/6)
- **0.6.0 MPQ structure**: All files in standard MPQ archives (`wmo.MPQ`, `terrain.MPQ`, etc.) — NOT loose files, NOT per-asset `.ext.MPQ` wrappers

### Key Findings About 0.6.0 MPQs
- 11 MPQ archives: base, dbc, fonts, interface, misc, model, sound, speech, terrain(2331), texture(33520), wmo(4603)
- All have internal listfiles (56573 total files extracted)
- Zlib (0x02) works fine for large files (groups extract correctly)
- PKWARE DCL (0x08) fails for small files (root WMOs, possibly some ADTs)
- `FLAG_COMPRESSED (0x200)` = per-sector compression with type byte prefix
- `FLAG_IMPLODED (0x100)` = whole-file PKWARE without type byte (not seen in these archives)

### StormLib Reference Code Available
- `lib/StormLib/src/pklib/explode.c` — Complete PKWARE DCL explode implementation
- `lib/StormLib/src/pklib/pklib.h` — Data structures (`TDcmpStruct`, lookup tables)
- `lib/StormLib/src/SCompression.cpp` — Decompression dispatch (`Decompress_PKLIB`, `SCompDecompress`)
- Key: `explode()` reads bytes 0,1 as ctype/dsize_bits, byte 2 as initial bit buffer, position starts at 3

### WMO Liquid Rendering Added
- MLIQ chunk now parsed in `ParseMogp` sub-chunk switch
- `WmoRenderer` has liquid mesh building + semi-transparent water surface rendering
- Diagnostic logging added for failed material textures

### Ghidra RE Prompts Written
- `specifications/ghidra/prompt-053-mpq.md` — 0.5.3 MPQ implementation (HAS PDB — best starting point)
- `specifications/ghidra/prompt-060-mpq.md` — 0.6.0 MPQ decompression (no PDB, use string refs)

### Files Modified This Session
- `NativeMpqService.cs` — Added diagnostic logging throughout ReadFile/ReadFileFromArchive/ReadFileData/DecompressData
- `MpqDataSource.cs` — Added diagnostic logging to ReadFile and TryResolveLoosePath
- `WmoV14ToV17Converter.cs` — Added diagnostic logging to ParseWmoV14Internal
- `WmoRenderer.cs` — Added WMO liquid rendering, material texture diagnostics
- `PkwareExplode.cs` — New file, PKWARE DCL decompression (needs fixing — current impl fails)
- `AlphaMpqReader.cs` — Wired up PkwareExplode for 0x08 compression
- `StandardTerrainAdapter.cs` — Added ADT loading diagnostics

## Session 2026-02-08 (Late Evening) Summary

### Standard WDT+ADT Support
- **ITerrainAdapter interface** — New common contract for all terrain adapters
- **StandardTerrainAdapter** — Reads LK/Cata WDT (MAIN/MPHD) + split ADT files from MPQ via IDataSource
- **TerrainManager refactored** — Accepts `ITerrainAdapter` (was hardcoded to `AlphaTerrainAdapter`)
- **WorldScene refactored** — New constructor accepts pre-built `TerrainManager`
- **ViewerApp detection** — File size ≥64KB → Alpha WDT, <64KB → Standard WDT (requires MPQ data source)

### Format Specifications Written
- `specifications/alpha-053-terrain.md` — Definitive WDT/ADT/MCNK/MCVT/MCNR/MCLY/MCAL/MCSH/MDDF/MODF spec
- `specifications/alpha-053-coordinates.md` — Complete coordinate system documentation
- `specifications/unknowns.md` — 13 prioritized format unknowns needing Ghidra investigation

### Ghidra LLM Prompts Created
- `specifications/ghidra/prompt-053.md` — 0.5.3 (HAS PDB! Best starting point)
- `specifications/ghidra/prompt-055.md` — 0.5.5 (diff against 0.5.3)
- `specifications/ghidra/prompt-060.md` — 0.6.0 (transitional format detection)
- `specifications/ghidra/prompt-335.md` — 3.3.5 LK (reference build, well-documented)
- `specifications/ghidra/prompt-400.md` — 4.0.0 Cata (split ADT introduction)

### Converter Master Plan
- `memory-bank/converter_plan.md` — 4-phase plan: LK model reading → format conversion → PM4 tiles → unified project

## Session 2026-02-08 (Evening) Summary

### What Was Fixed

#### MCSH Shadow Blending (TerrainRenderer.cs)
- **Problem**: Shadow map (MCSH) was only applied on the base terrain layer. Alpha-blended overlay texture layers drawn on top would cover/wash out the shadows.
- **Root cause**: Both the C# render code and GLSL shader had `isBaseLayer` guards on shadow binding/application.
- **Fix**: Removed `isBaseLayer` condition from both:
  - C# `RenderChunkPass()`: Changed `bool hasShadow = isBaseLayer && chunk.ShadowTexture != 0` → `bool hasShadow = chunk.ShadowTexture != 0`
  - GLSL fragment shader: Changed `if (uShowShadowMap == 1 && uIsBaseLayer == 1 && uHasShadowMap == 1)` → `if (uShowShadowMap == 1 && uHasShadowMap == 1)`
- **Result**: Shadows now darken all texture layers consistently.

#### MDX Bounding Box Pivot Offset (WorldScene.cs, WorldAssetManager.cs)
- **Problem**: MDX model geometry is offset from origin (0,0,0). The MODL bounding box describes where geometry actually sits. MDDF placement position targets origin, but geometry center is elsewhere, causing models to appear displaced.
- **Fix**: Pre-translate geometry by negative bounding box center before scale/rotation/translation:
  - Added `WorldAssetManager.TryGetMdxPivotOffset()` — returns `(BoundsMin + BoundsMax) * 0.5f`
  - Transform chain: `pivotCorrection * mirrorX * scale * rotX * rotY * rotZ * translation`
  - `pivotCorrection = Matrix4x4.CreateTranslation(-pivot)`
  - Applied in both `BuildInstances()` and `OnTileLoaded()` in WorldScene.cs
- **WMO models**: Do NOT need pivot correction — their geometry is already correctly positioned relative to origin.

#### VLM Terrain Rendering (Previous session, 2026-02-08 afternoon)
- **GLSL shader em-dash**: Replaced unicode em-dash with ASCII hyphen in shader comment.
- **NullReferenceException**: Fixed null-conditional access in `DrawTerrainControls`.
- **VLM coordinate conversion**: Fixed `WorldPosition` in `VlmProjectLoader.cs` — swapped posX/posY, removed MapOrigin subtraction.
- **Minimap for VLM projects**: Refactored `DrawMinimap()` to work with either `_terrainManager` or `_vlmTerrainManager`. Added `IsTileLoaded()` to `VlmTerrainManager`.

#### Async Tile Streaming (TerrainManager.cs, VlmTerrainManager.cs)
- Both terrain managers now queue tile parsing to `ThreadPool` background threads.
- Parsed `TileLoadResult` objects enqueued to `ConcurrentQueue`.
- `SubmitPendingTiles()` runs on render thread each frame, uploading max 2 tiles/frame to avoid GPU stalls.
- `_disposed` flag prevents background threads from accessing disposed resources.

#### Thread Safety (VlmProjectLoader.cs, AlphaTerrainAdapter.cs, TerrainRenderer.cs)
- `TileTextures` → `ConcurrentDictionary` in both adapters.
- `_placementLock` protects dedup sets (`_seenMddfIds`, `_seenModfIds`) and placement lists in both adapters.
- `TerrainRenderer.AddChunks()` parameter widened from `Dictionary` to `IDictionary` to accept both.

#### VLM Dataset Generator (ViewerApp.cs)
- New menu item: `File > Generate VLM Dataset...`
- Dialog UI: client path (folder picker), map name, output dir, tile limit, progress log.
- Runs `VlmDatasetExporter.ExportMapAsync()` on `ThreadPool` with `IProgress<string>` feeding real-time log.
- "Open in Viewer" button after export completes.

### Key Technical Decisions
- **Coordinate system**: Renderer X = WoW Y, Renderer Y = WoW X, Z = height. MapOrigin = 17066.66666f, ChunkSize = 533.33333f.
- **MDX pivot**: Bounding box center, NOT PIVT chunk (PIVT is for per-bone skeletal animation pivots).
- **Shadow blending**: Apply to ALL layers, not just base. Overlay layers must also be darkened.
- **Thread safety**: `ConcurrentDictionary` for shared tile data, `lock` for placement dedup sets.

## What Works

| Feature | Status |
|---------|--------|
| Alpha WDT terrain rendering + AOI | ✅ |
| Standard WDT+ADT terrain (LK/Cata) | ✅ (terrain only, needs M2/WMOv17 readers for objects) |
| Terrain MCSH shadow maps | ✅ (all layers, not just base) |
| Terrain alpha map debug view | ✅ (Show Alpha Masks toggle) |
| Async tile streaming | ✅ (background parse, render-thread GPU upload) |
| Standalone MDX rendering | ✅ (MirrorX, front-facing) |
| MDX pivot offset correction | ✅ (bounding box center pre-translation) |
| MDX doodads in WorldScene | ⚠️ Position fixed, textures still broken |
| WMO v14 rendering + textures | ✅ (BLP per-batch) |
| WMO v17 rendering | ❌ Not implemented yet |
| M2 model rendering | ❌ Not implemented yet |
| WMO rotation/facing in WorldScene | ✅ |
| WMO doodad sets | ✅ |
| MDDF/MODF placements | ✅ (position + pivot correct) |
| Bounding boxes | ✅ (actual MODF extents) |
| VLM terrain loading | ✅ (JSON dataset → renderer) |
| VLM minimap | ✅ |
| VLM dataset generator | ✅ (File > Generate VLM Dataset) |
| Live minimap + click-to-teleport | ✅ (WDT + VLM) |
| AreaPOI system | ✅ |
| GLB export (Z-up → Y-up) | ✅ |
| Object picking/selection | ✅ |
| Format specifications | ✅ (specifications/ folder) |
| WMO liquid rendering (MLIQ) | ✅ (semi-transparent water surfaces) |
| Object picking/selection | ✅ (ray-AABB, highlight, info) |
| Camera world coordinates | ✅ (WoW coords in status bar) |
| Left/right sidebar layout | ✅ (docked panels) |
| Ghidra RE prompts (5+2 versions) | ✅ (specifications/ghidra/) |
| 0.6.0 MPQ file extraction | ❌ PKWARE DCL (0x08) decompression fails |

## Key Files

- `Terrain/WorldScene.cs` — Object instance building, pivot offset, rotation transforms, rendering loop
- `Terrain/WorldAssetManager.cs` — Model loading, bounding box/pivot queries
- `Terrain/AlphaTerrainAdapter.cs` — MDDF/MODF parsing, coordinate conversion, thread-safe placement dedup
- `Terrain/VlmProjectLoader.cs` — VLM JSON tile loading, thread-safe TileTextures/placements
- `Terrain/VlmTerrainManager.cs` — VLM terrain AOI, async streaming
- `Terrain/TerrainManager.cs` — WDT terrain AOI, async streaming
- `Terrain/TerrainRenderer.cs` — Terrain shader, shadow maps on all layers, alpha maps, debug views
- `Rendering/WmoRenderer.cs` — WMO geometry, textures, doodad sets
- `Rendering/ModelRenderer.cs` — MDX rendering, MirrorX, blend modes, textures
- `ViewerApp.cs` — Main app, UI, DBC loading, minimap, VLM export dialog
- `Export/GlbExporter.cs` — GLB export with Z-up → Y-up conversion

## Dependencies (all already integrated)

- `MdxLTool` — MDX file parser
- `WoWMapConverter.Core` → `gillijimproject-csharp` — Alpha WDT/ADT/MCNK parsers, WMO v14 parser, VLM dataset export
- `SereniaBLPLib` — BLP texture loading
- `Silk.NET` — OpenGL + windowing + input
- `ImGuiNET` — UI overlay
- `DBCD` — DBC database access
