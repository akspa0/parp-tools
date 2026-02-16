# MdxViewer - Memory Bank

## Project Overview

**Purpose**: World of Warcraft model/world viewer. Fully supports Alpha 0.5.3 through 0.12 client data. WotLK 3.3.5 support is in progress (MH2O + texturing broken, not usable yet).

**Key Technologies**:
- .NET 10.0 Windows
- Silk.NET for OpenGL 3.3 rendering
- SixLabors.ImageSharp for image processing
- DBCD for DBC database access
- SereniaBLPLib for BLP texture loading
- ImGuiNET for UI overlay

## Architecture

### Data Sources
- `MpqDataSource` - Main data source using MPQ archives
- Uses `NativeMpqService` for MPQ file reading
- Builds file list from MPQ internal listfiles
- Supports loose files on disk
- Custom scanning for nested WMO MPQ archives

### Rendering Pipeline
- OpenGL via Silk.NET
- Model rendering with vertex buffers + GPU skinning (bone matrices)
- Texture support for BLP format
- Skeletal animation system (MdxAnimator — keyframe interpolation, bone hierarchy, compressed quaternions)

## Key File Formats

### MDX (Alpha Model Format)
- Binary 3D model format used in Alpha 0.5.3 (MDLX magic)
- Contains vertices, faces, textures, animations
- PRE2 particle emitters with texture atlas + segment colors
- Supports geoset animations (ATSQ alpha/color tracks)
- Skeletal animation with compressed quaternion rotation keys

### M2/MD20 (Standard Model Format)
- Binary 3D model format used in WotLK 3.3.5+ (MD20 magic)
- Loaded via WarcraftNetM2Adapter which converts to MdxFile runtime format
- Supports render flags, blend modes, textures, bones

### WMO (World Map Object)
- Indoor/outdoor building models
- Supports BSP collision
- Uses portal system for visibility

### BLP (Blizzard Texture)
- Custom texture format with mipmaps
- JPEG or DXT compression
- Palette-based for older versions

### ADT/WDT (Terrain)
- Alpha client: monolithic WDT with embedded terrain data
- Standard (WotLK): MPHD/MAIN WDT + split ADT files per tile
- 16x16 chunks (MCNK) with alpha maps, heightmaps
- MH2O liquid chunks (WotLK format)

### WDL (Low-Detail Far Terrain)
- Optional low-detail terrain layer (64x64 tile grid)
- Parsed as strict `MVER -> MAOF -> MARE` chunk sequence
- `MVER` version expected: `0x12`
- One WDL cell equals one ADT tile (`WoWConstants.TileSize` = 8533.3333)

## Key Classes

| Class | Purpose |
|-------|---------|
| `ViewerApp` | Main application entry point |
| `MpqDataSource` | MPQ file access and file list building |
| `FileBrowser` | File navigation UI |
| `ModelRenderer` | 3D model rendering + GPU skinning + particles + geoset anim |
| `MdxAnimator` | Skeletal animation (bone hierarchy, keyframe interpolation) |
| `ParticleRenderer` | Billboard particle rendering with texture atlas + blend modes |
| `WarcraftNetM2Adapter` | MD20→MdxFile converter for WotLK M2 models |
| `TerrainLighting` | Day/night cycle with half-Lambert lighting |

## Known Issues & Solutions

### MPQ File Discovery
- **Symptom**: Only BLP/WMO files show, no MDX/M2
- **Fix**: Call `_mpq.GetAllKnownFiles()` and add to `_fileSet`

### Case Sensitivity
- **Symptom**: Files not found despite existing
- **Cause**: Alpha uses uppercase extensions (.MPQ)
- **Fix**: Use `StringComparer.OrdinalIgnoreCase`

### WMO Nested Archives
- **Symptom**: WMO files won't load
- **Cause**: WMO data stored in `.wmo.MPQ` archives
- **Fix**: Use `ScanWmoMpqArchives()` for nested scanning

### WDL Parsing / Rendering Alignment
- **Symptom**: Incorrect WDL terrain or bad positioning
- **Causes fixed**:
  - Chunk parsing not strict enough (`MVER`/`MAOF`/`MARE`)
  - Missing `MVER` version validation
  - `MARE` heights read without consuming per-chunk header first
  - WDL using chunk size instead of tile size
- **Fixes**:
  - Strict parser with version `0x12` check
  - Proper `MARE` header handling before 545 height samples
  - WDL renderer/preview use `WoWConstants.TileSize`

### WDL Overlap / Testing Controls
- **Symptom**: WDL low-res mesh visible where detailed ADT is expected
- **Fixes**:
  - Hide WDL cells for already-loaded ADT tiles at WDL load time
  - Keep hide/show sync on tile load/unload events
  - Apply polygon offset to reduce z-fighting
  - Add UI toggle (`ShowWdlTerrain`) to disable WDL layer during testing

### WMO Intermittent Non-Rendering
- **Symptom**: Some WMOs randomly disappear/reappear
- **Cause**: Per-instance shader program lifetime; one renderer disposing could delete a program another instance still used
- **Fix**: Shared static shader programs (main + liquid) with reference-counted lifetime

### WL Loose Liquids Alignment
- **Context**: WLW/WLQ/WLM are editor-side files; no client runtime loader for authoritative transform behavior
- **Approach**: Matrix-based 3D transform tuning in UI (rotation + translation), then lock final values
- **Tools added**:
  - `WlLiquidLoader.TransformSettings`
  - `WorldScene.ReloadWlLiquids()`
  - `WL Transform Tuning` UI + `Apply + Reload WL`

### MDX Coordinate System (LH→RH)
- **Symptom**: Standalone MDX models appear inside-out/mirrored
- **Fix**: Apply `MirrorX = Matrix4x4.CreateScale(-1, 1, 1)` in `Render()` model matrix only
- **Note**: WorldScene uses `RenderWithTransform()` directly — no mirror needed

### MDX Animation — PIVT Chunk Order
- **Symptom**: Horror-movie deformation — bones rotate around world origin instead of joints
- **Cause**: PIVT chunk comes AFTER BONE in MDX files. Inline pivot assignment during `ReadBone()` found 0 pivots.
- **Fix**: Deferred pivot assignment in `MdxFile.Load()` after all chunks are parsed.

### MDX Animation — UpdateAnimation() Call Site
- **Symptom**: Bones never move despite correct parsing
- **Cause**: `ViewerApp` calls `RenderWithTransform()` directly, bypassing `Render()` which was the only place `_animator.Update()` was called
- **Fix**: Extracted `UpdateAnimation()` as public method on `MdxRenderer`. Called from `ViewerApp` (standalone) and `WorldScene` (terrain doodads).

### MDX KGRT Compressed Quaternions
- **Symptom**: Rotation data parsed incorrectly, wrong animation poses
- **Cause**: KGRT keys use `C4QuaternionCompressed` (8 bytes packed), not float4 (16 bytes)
- **Fix**: Added `C4QuaternionCompressed` struct with Ghidra-verified decompression (21-bit signed components, W from unit norm)

### MDX Bounding Box Pivot Offset
- **Symptom**: MDX doodads appear displaced from their MDDF placement positions
- **Cause**: MDX geometry is offset from origin (0,0,0). The MODL bounding box center is the effective pivot.
- **Fix**: Pre-translate geometry by `-boundsCenter` before scale/rotation/translation
- **Transform chain**: `pivotCorrection * mirrorX * scale * rotX * rotY * rotZ * translation`
- **Code**: `WorldAssetManager.TryGetMdxPivotOffset()` returns `(BoundsMin + BoundsMax) * 0.5f`
- **Note**: WMO models do NOT need pivot correction. PIVT chunk is for per-bone skeletal pivots, not placement.

### MCSH Shadow Blending
- **Symptom**: Shadow maps (MCSH) are washed out / invisible where overlay texture layers cover the base
- **Cause**: Shadow was only applied on base layer (`uIsBaseLayer == 1`). Overlay layers drawn on top with alpha blending cover the darkened base.
- **Fix**: Apply shadow to ALL layers, not just base. Remove `isBaseLayer` guard from both C# shadow texture binding and GLSL shader.
- **C#**: `bool hasShadow = chunk.ShadowTexture != 0` (removed `isBaseLayer &&`)
- **GLSL**: `if (uShowShadowMap == 1 && uHasShadowMap == 1)` (removed `&& uIsBaseLayer == 1`)

### BIDX Chunk Parsing
- **Symptom**: Geoset recovery failures, corrupted MaterialId/textures
- **Fix**: BIDX = 1 byte per vertex (like GNDX), NOT 4 bytes

### Terrain Alpha Map Seams
- **Symptom**: Grid pattern visible at chunk boundaries on layers 1-3
- **Fix**: Noggit edge fix — duplicate last row/column in 64×64 alpha data before upload

### Thread Safety for Async Tile Loading
- **Symptom**: Race conditions when background threads parse tiles concurrently
- **Fix**: 
  - `TileTextures` → `ConcurrentDictionary` in both `AlphaTerrainAdapter` and `VlmProjectLoader`
  - `_placementLock` object protects `_seenMddfIds`, `_seenModfIds`, and placement lists
  - `TerrainRenderer.AddChunks()` parameter: `IDictionary` (accepts both Dictionary and ConcurrentDictionary)

### VLM Coordinate Conversion
- **Symptom**: VLM terrain chunks don't render (invisible)
- **Cause**: WorldPosition calculation was wrong — posX/posY were not swapped, and MapOrigin was being subtracted
- **Fix**: In `VlmProjectLoader.cs`, use `worldX = posY, worldY = posX` (direct swap, no MapOrigin subtraction)

## Build & Run

```powershell
cd src/MdxViewer
dotnet build --no-restore
dotnet run
```

## Asset Catalog System

### Files
| File | Purpose |
|------|---------|
| `Catalog/AlphaCoreDbReader.cs` | Parses alpha-core SQL dump files directly (no MySQL needed) |
| `Catalog/AssetCatalogEntry.cs` | Unified data model for NPCs and GameObjects |
| `Catalog/AssetExporter.cs` | JSON metadata + GLB model + screenshot export (single + batch) |
| `Catalog/AssetCatalogView.cs` | ImGui browse/search/filter panel with export controls |
| `Catalog/ScreenshotRenderer.cs` | Offscreen FBO rendering + nameplate overlay + PNG save |

### Data Chain
- **Creatures**: creature_template.display_id1 → CreatureDisplayInfo.ModelID → mdx_models_data.ModelName
- **GameObjects**: gameobject_template.displayId → GameObjectDisplayInfo.ModelName
- **SQL dumps**: `{alphaCoreRoot}/etc/databases/world/world.sql` + `dbc/dbc.sql`

### Output Structure (planned per-object folders)
```
asset_catalog_output/
  creatures/
    {entryId}_{name}/
      metadata.json
      model.glb
      front.png
      back.png
      left.png
      right.png
      top.png
      three_quarter.png
  gameobjects/
    {entryId}_{name}/
      ...same structure...
```

## TODO

See `renderer_plan.md` for the full itemized 40-task implementation plan across 8 phases.

**Summary of phases:**
- [x] MDX model loading + rendering
- [x] WMO v14 loading + rendering
- [x] BLP2 texture loading
- [x] Phase 0: Foundation
- [x] Phase 3: Terrain (WDT/ADT loading, mesh gen, texture layering, lighting, shadow fix, alpha debug, async streaming)
- [x] Phase 4: World Scene (WMOs ✅, MDX pivot ✅, MDX textures ✅ except particles)
- [x] VLM: Dataset loading, minimap, generator UI, async streaming
- [x] Asset Catalog: SQL dump reader, browse/filter UI, JSON+GLB+screenshot export
- [x] Phase 1: MDX Animation (compressed quats, GPU skinning, standalone + terrain doodads)
- [ ] Asset Catalog: Per-object folders + multi-angle screenshots
- [ ] Lighting improvements (DBC light data, per-vertex, ambient)
- [ ] Phase 2: Particle System (emitters, physics, billboard rendering)
- [ ] Phase 5: Liquid Rendering — lava type mapping still broken (green)
- [ ] Phase 6: Detail Doodads (per-chunk grass/foliage)
- [ ] Phase 7: Polish (instancing, LOD, debug overlays)
