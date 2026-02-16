# Active Context — MdxViewer / AlphaWoW Viewer

## Current Focus

**v0.4.0 Release — 0.5.3 Rendering Improvements + Initial 3.3.5 Groundwork** — Major rendering improvements for Alpha 0.5.3 (lighting, particles, geoset animations). Initial 3.3.5 WotLK support scaffolding added but **NOT ready for use** — MH2O liquid and terrain texturing are broken. Only client versions 0.5.3 through 0.12 are currently usable.

## 3.3.5 WotLK Status: IN PROGRESS (NOT USABLE)

**Known broken:**
- MH2O liquid rendering — parsing exists but rendering is broken
- Terrain texturing — alpha map decode not working correctly for LK format
- These must be fixed before 3.3.5 data can be used

## Immediate Next Steps

1. **Fix 3.3.5 MH2O liquid rendering** — Parsing exists but output is broken
2. **Fix 3.3.5 terrain texturing** — Alpha map decode for LK format not working
3. **3.3.5 terrain alpha maps** — Current LK path uses basic Mcal decode; needs full `AlphaMapService` integration without breaking 0.5.3
4. **Light.dbc / LightData.dbc integration** — Replace hardcoded TerrainLighting values with real game lighting data per zone
5. **Skybox rendering** — Not yet implemented; needed for proper atmosphere
6. **Ribbon emitters (RIBB)** — Parsed but no rendering code yet
7. **M2 particle emitters** — WarcraftNetM2Adapter doesn't map PRE2/particles to MdxFile format yet

## Session 2026-02-13 Summary — WDL/WL/WMO Fixes

### Completed

1. **WDL parser correctness**
   - Strict chunk parsing (`MVER`/`MAOF`/`MARE`) with version `0x12` validation
   - Proper `MARE` chunk header handling before height reads

2. **WDL terrain scale + overlay behavior improvements**
   - WDL cell size corrected to `WoWConstants.TileSize` (8533.3333), not chunk size
   - Existing ADT-loaded tiles hidden from WDL at load-time
   - Polygon offset added to reduce z-fighting with real terrain
   - UI toggle added to fully disable WDL rendering for testing

3. **WDL preview reliability**
   - `.wdl.mpq` fallback path and error propagation (`LastError`)
   - Preview dialog now displays failure reason instead of closing silently

4. **WMO intermittent non-rendering fix**
   - Converted WMO main + liquid shader programs to shared static programs with ref-counted lifetime
   - Prevents per-instance shader deletion race (same class of bug previously fixed in MDX renderer)

5. **WL liquids transform tooling**
   - Replaced hardcoded axis swap with configurable matrix transform (rotation + translation)
   - Added `WL Transform Tuning` controls in UI and `Apply + Reload WL`
   - Added `WorldScene.ReloadWlLiquids()` for fast iteration

## MDX Particle System — IMPLEMENTED (2026-02-15)

Previously deferred issue now resolved. ParticleRenderer rewritten with per-particle uniforms, texture atlas support, and per-emitter blend modes. Wired into MdxRenderer — emitters created from PRE2 data, updated each frame with bone-following transforms, rendered during transparent pass. Fire, glow, and spell effects now visible.

## Session 2026-02-15 Summary — Multi-Version Support + Lighting/Particle Overhaul

### Completed

1. **Partial WotLK 3.3.5 terrain scaffolding** (StandardTerrainAdapter) — **NOT USABLE**
   - Split ADT file loading via MPQ data source
   - MPHD flags detection for `bigAlpha` (0x4)
   - MH2O liquid chunk parsing — **BROKEN, not rendering correctly**
   - LK alpha maps via `hasLkFlags` detection — **texturing BROKEN**
   - Surgical revert of shared renderer code was needed to restore 0.5.3

2. **M2 (MD20) model loading** (WarcraftNetM2Adapter)
   - Converts MD20 format models to MdxFile runtime format
   - Maps render flags (Unshaded, Unfogged, TwoSided), blend modes
   - Texture loading from M2 texture definitions
   - Bone/animation data mapping

3. **Terrain regression fix** (surgical revert)
   - Commit e172907 broke 0.5.3 terrain rendering (grid pattern artifacts)
   - Root cause: `AlphaTextures.ContainsKey` guard skipping overlay layers + edge fix removal in TerrainRenderer.cs
   - Plus StandardTerrainAdapter ExtractAlphaMaps rewrite with broken `spanSuggestsPacked` logic
   - Surgical revert restored 0.5.3 terrain while preserving M2/WMO improvements

4. **Lighting improvements** (TerrainLighting, ModelRenderer, WmoRenderer)
   - Raised ambient values: day (0.4→0.55), night (0.08→0.25) — no more pitch black
   - Half-Lambert diffuse shading: `dot * 0.5 + 0.5` squared — wraps light around surfaces
   - WMO shader: replaced lossy scalar lighting `(r+g+b)/3.0` with proper `vec3` lighting
   - MDX shader: half-Lambert + reduced specular (0.3→0.15)
   - Moderated day directional light (1.0→0.8) to avoid blow-out with higher ambient

5. **Particle system wired into pipeline** (ParticleRenderer, ModelRenderer)
   - Rewrote ParticleRenderer: per-particle uniforms, texture atlas (rows×columns), per-emitter blend mode
   - MdxRenderer creates ParticleEmitter instances from MdxFile.ParticleEmitters2
   - Emitter transforms follow parent bone matrices when animated
   - Particles rendered during transparent pass after geosets
   - Supports Additive, Blend, Modulate, AlphaKey filter modes

6. **Geoset animation alpha** (ModelRenderer)
   - `UpdateGeosetAnimationAlpha()` evaluates ATSQ alpha keyframe tracks per frame
   - Alpha multiplied into layer alpha during RenderGeosets
   - Geosets with alpha ≈ 0 skipped entirely
   - Supports global sequences and linear interpolation

7. **WMO fixes from 3.3.5 work** (preserved)
   - Multi-MOTV/MOCV chunk handling for ICC-style WMOs
   - Strict WMO validation preventing Northrend loading hangs
   - WMO liquid rotation fixes

### Files Modified
- `TerrainRenderer.cs` — Reverted edge fix + ContainsKey guard
- `StandardTerrainAdapter.cs` — Reverted ExtractAlphaMaps to clean hasLkFlags path
- `TerrainLighting.cs` — Raised ambient/light values, better night visibility
- `ModelRenderer.cs` — Half-Lambert shader, particle wiring, geoset animation alpha
- `WmoRenderer.cs` — vec3 lighting instead of scalar, half-Lambert diffuse
- `ParticleRenderer.cs` — Complete rewrite with working per-particle rendering
- `WarcraftNetM2Adapter.cs` — MD20→MdxFile adapter (from e172907, preserved)
- `WorldAssetManager.cs` — MD20 detection + adapter routing (from e172907, preserved)

## Session 2026-02-13 Summary — MDX Animation System Complete

### Three Bugs Fixed

1. **KGRT Compressed Quaternion Parsing** (`MdxFile.cs`, `MdxTypes.cs`)
   - Rotation keys use `C4QuaternionCompressed` (8 bytes packed), not float4 (16 bytes)
   - Ghidra-verified decompression: 21-bit signed components, W reconstructed from unit norm
   - Added `C4QuaternionCompressed` struct with `Decompress()` method

2. **Animation Never Updated** (`ModelRenderer.cs`, `ViewerApp.cs`)
   - `ViewerApp` called `RenderWithTransform()` directly, bypassing `Render()` which was the only place `_animator.Update()` was called
   - Fix: Extracted `UpdateAnimation()` as public method, called from ViewerApp before render

3. **PIVT Chunk Order — All Pivots Were (0,0,0)** (`MdxFile.cs`)
   - PIVT chunk comes AFTER BONE in MDX files. Inline pivot assignment during `ReadBone()` found 0 pivots
   - Fix: Deferred pivot assignment in `MdxFile.Load()` after all chunks are parsed
   - This caused "horror movie" deformation — bones rotating around world origin instead of joints

### Terrain Animation Added (`WorldScene.cs`)
- Added `UpdateAnimation()` calls for all unique MDX renderers before opaque/transparent render passes
- Uses `HashSet<string>` to ensure each renderer is updated exactly once per frame

### Other Improvements
- `MdxAnimator`: `_objectIdToListIndex` dictionary replaces O(n) `IndexOf` calls
- `GNDX`/`MTGC` chunks now stored in `MdlGeoset` for vertex-to-bone skinning
- MATS values remapped from ObjectIds to bone list indices via dictionary lookup

### Key Architecture (MDX Animation)
- `MdxAnimator` — Evaluates bone hierarchy per-frame, stores matrices in `_boneMatrices[]` by list position
- `ModelRenderer.UpdateAnimation()` — Public method to advance animation clock
- `BuildBoneWeights()` — Converts GNDX/MTGC/MATS to 4-bone skinning format
- Bone transform: `T(-pivot) * S * R * T(pivot) * T(translation) * parentWorld`
- Shader: `uBones[128]` uniform array, vertex attributes for bone indices + weights

### Files Modified
- `MdxTypes.cs` — Added `C4QuaternionCompressed` struct
- `MdxFile.cs` — Fixed `ReadQuatTrack`, stored GNDX/MTGC, deferred pivot assignment
- `MdxAnimator.cs` — `_objectIdToListIndex` dict, cleaned diagnostics
- `ModelRenderer.cs` — Extracted `UpdateAnimation()`, ObjectId→listIndex remapping in `BuildBoneWeights`
- `ViewerApp.cs` — Added `mdxR.UpdateAnimation()` before standalone MDX render
- `WorldScene.cs` — Added per-frame animation update for unique MDX doodad renderers

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
| **Standard WDT+ADT terrain (WotLK 3.3.5)** | ✅ Partial — terrain + M2 models + WMO loading |
| Terrain MCSH shadow maps | ✅ (all layers, not just base) |
| Terrain alpha map debug view | ✅ (Show Alpha Masks toggle) |
| Async tile streaming | ✅ (background parse, render-thread GPU upload) |
| Standalone MDX rendering | ✅ (MirrorX, front-facing) |
| MDX skeletal animation | ✅ (standalone + terrain, compressed quats, GPU skinning) |
| MDX pivot offset correction | ✅ (bounding box center pre-translation) |
| MDX doodads in WorldScene | ✅ Position + animation + particles working |
| WMO v14 rendering + textures | ✅ (BLP per-batch) |
| WMO v17 rendering | ✅ Partial (groups + textures, multi-MOTV/MOCV) |
| M2 model rendering | ✅ MD20→MdxFile adapter (WarcraftNetM2Adapter) |
| Particle effects (PRE2) | ✅ Billboard quads, texture atlas, bone-following |
| Geoset animation alpha (ATSQ) | ✅ Per-frame keyframe evaluation |
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
| Half-Lambert lighting | ✅ Softer shading on MDX + WMO models |
| Improved ambient lighting | ✅ Day/night cycle with WoW-like brightness |

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
