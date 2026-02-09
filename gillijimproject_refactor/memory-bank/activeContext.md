# Active Context

## Current Focus: MdxViewer — 3D World Viewer (Feb 9, 2026)

### Recently Completed

- **WMO/MDX Geometry Mirroring Fix**: ✅ RESOLVED
  - Root cause: WoW/D3D uses CW triangle winding; OpenGL uses CCW
  - Fix: Reverse index winding at upload (swap v1↔v2 per triangle) in both `WmoRenderer.cs` and `ModelRenderer.cs`
  - Compensating 180° Z rotation in placement transforms (`WorldScene.cs`)
  - Vertices pass through raw — no coordinate conversion at vertex level
- **MDX GEOS Parsing Fix**: ✅ RESOLVED
  - Added `BWGT` (bone weights) handler with peek-ahead validation
  - Fixed `BIDX` (bone indices) stride detection (1-byte vs 4-byte)
  - Eliminates "unknown tag '?'" spam and viewer crashes

### Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| Alpha WDT terrain | ✅ | Monolithic format, 256 MCNK chunks per tile |
| Standard WDT+ADT (3.3.5) | ✅ | Split ADT files from MPQ/IDataSource |
| WMO v14 rendering | ✅ | Correct orientation, winding, placement |
| MDX rendering | ✅ | Geometry + textures, correct orientation |
| MCSH shadow maps | ✅ | 64×64 bitmask, all layers |
| MCLQ ocean liquid | ✅ | Inline liquid from MCNK flags |
| Async tile streaming | ✅ | AOI-based lazy loading |
| Frustum culling | ✅ | View-frustum + bounding box |
| Minimap overlay | ✅ | From minimap tiles |

### Next Steps (Priority Order)

1. **Liquid rendering** — rivers, lakes, magma, slime (currently only ocean renders)
2. **WMO interior liquid** — MLIQ chunk rendering
3. **MDX animations/bones** — skeletal animation system
4. **MDX texture improvements** — team colors, replaceable textures
5. **Lighting system** — ambient, directional, point lights from DBC data
6. **Skybox from game data** — Light.dbc, LightSkybox.dbc
7. **M2 model reader** — for 3.3.5 format support
8. **WMO v17 reader** — for 3.3.5 format support

---

## Key Architecture Decisions

### Coordinate System (Confirmed via Ghidra)
- WoW uses **right-handed** coordinates: X=North, Y=West, Z=Up
- WoW renders through **Direct3D** (CW front faces)
- OpenGL uses **CCW front faces**
- Fix: Reverse triangle winding at GPU upload + 180° Z rotation in placement
- Terrain positions: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`
- Model vertices: raw pass-through (no axis swap at vertex level)

### Key Files

| File | Purpose |
|------|---------|
| `WorldScene.cs` | Placement transforms, instance management |
| `WmoRenderer.cs` | WMO v14 GPU rendering, winding fix |
| `ModelRenderer.cs` | MDX GPU rendering, winding fix |
| `AlphaTerrainAdapter.cs` | Alpha WDT terrain loading |
| `StandardTerrainAdapter.cs` | 3.3.5 WDT+ADT terrain loading |
| `TerrainManager.cs` | AOI-based tile streaming |
| `MdxFile.cs` | MDX parser (GEOS/BIDX/BWGT fix) |
