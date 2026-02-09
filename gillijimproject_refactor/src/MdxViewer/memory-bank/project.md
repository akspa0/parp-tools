# MdxViewer - Memory Bank

## Project Overview

**Purpose**: World of Warcraft Alpha 0.5.3 model viewer application.

**Key Technologies**:
- .NET 9.0 Windows Forms
- Silk.NET for OpenGL rendering
- SixLabors.ImageSharp for image processing
- DBCD for DBC database access

## Architecture

### Data Sources
- `MpqDataSource` - Main data source using MPQ archives
- Uses `NativeMpqService` for MPQ file reading
- Builds file list from MPQ internal listfiles
- Supports loose files on disk
- Custom scanning for nested WMO MPQ archives

### Rendering Pipeline
- OpenGL via Silk.NET
- Model rendering with vertex buffers
- Texture support for BLP format
- Animation system for models

## Key File Formats

### MDX (Alpha Model Format)
- Binary 3D model format used in Alpha 0.5.3
- Contains vertices, faces, textures, animations
- Uses C3Color type for color values
- Supports geoset animations

### WMO (World Map Object)
- Indoor/outdoor building models
- Supports BSP collision
- Uses portal system for visibility

### BLP (Blizzard Texture)
- Custom texture format with mipmaps
- JPEG or DXT compression
- Palette-based for older versions

### ADT/WDT (Terrain)
- Alpha client terrain format
- 16x16 chunks (MCNK)
- Supports alpha maps, heightmaps

## Key Classes

| Class | Purpose |
|-------|---------|
| `ViewerApp` | Main application entry point |
| `MpqDataSource` | MPQ file access and file list building |
| `FileBrowser` | File navigation UI |
| `ModelRenderer` | 3D model rendering |

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

### MDX Coordinate System (LH→RH)
- **Symptom**: Standalone MDX models appear inside-out/mirrored
- **Fix**: Apply `MirrorX = Matrix4x4.CreateScale(-1, 1, 1)` in `Render()` model matrix only
- **Note**: WorldScene uses `RenderWithTransform()` directly — no mirror needed

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

## TODO

See `renderer_plan.md` for the full itemized 40-task implementation plan across 8 phases.

**Summary of phases:**
- [x] MDX model loading + rendering
- [x] WMO v14 loading + rendering
- [x] BLP2 texture loading
- [x] Phase 0: Foundation
- [x] Phase 3: Terrain (WDT/ADT loading, mesh gen, texture layering, lighting, shadow fix, alpha debug, async streaming)
- [x] Phase 4: World Scene (WMOs ✅, MDX pivot ✅, MDX textures ❌)
- [x] VLM: Dataset loading, minimap, generator UI, async streaming
- [ ] Phase 1: MDX Animation (keyframes, bones, geoset animation, playback UI)
- [ ] Phase 2: Particle System (emitters, physics, billboard rendering)
- [ ] Phase 5: Liquid Rendering (water, magma, slime surfaces)
- [ ] Phase 6: Detail Doodads (per-chunk grass/foliage)
- [ ] Phase 7: Polish (instancing, LOD, debug overlays)
