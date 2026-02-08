# Progress — MdxViewer Renderer Reimplementation

## Status: Phase 4 Mostly Complete — MDX WorldScene Textures Next

## What Works Today

| Feature | Status |
|---------|--------|
| Terrain rendering + AOI lazy loading | ✅ |
| Terrain alpha map debug view | ✅ Show Alpha Masks toggle, Noggit edge fix |
| Standalone MDX rendering | ✅ MirrorX for LH→RH, front-facing, textured |
| MDX blend modes + depth mask | ✅ Transparent layers don't write depth |
| MDX doodads in WorldScene | ⚠️ Position OK, textures broken (magenta) |
| WMO v14 loading + rendering | ✅ Groups, BLP textures per-batch |
| WMO doodad sets | ✅ Loaded and rendered with WMO modelMatrix |
| WMO rotation/facing in WorldScene | ✅ Fixed — `-rz` negation for handedness |
| MDDF/MODF placements | ✅ Position correct |
| Bounding boxes | ✅ Actual MODF extents with correct min/max swap |
| BLP2 texture loading | ✅ DXT1/3/5, palette, JPEG |
| MPQ data source | ✅ Listfile, nested WMO archives |
| DBC integration | ✅ DBCD, CreatureModelData, CreatureDisplayInfo |
| Camera | ✅ Free-fly WASD + mouse look |
| ImGui UI | ✅ File browser, model info, visibility toggles |
| Live minimap + click-to-teleport | ✅ |
| AreaPOI system | ✅ DBC loading, 3D markers, minimap markers, UI list |
| Object picking/selection | ✅ |
| GLB export | ✅ MDX + WMO, Z-up → Y-up conversion |

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Foundation | ✅ Complete |
| 3 | Terrain | ✅ Complete (alpha seam fix applied) |
| 4 | World Scene | ⚠️ WMOs ✅, MDX textures ❌ |
| 1 | MDX Animation | ⏳ Not started |
| 2 | Particles | ⏳ Not started |
| 5-7 | Liquids, Detail Doodads, Polish | ⏳ Not started |

## Next Priority: MDX Doodad Textures in WorldScene

MDX doodads render in correct positions but have no textures (magenta). This is a pre-existing issue, not a regression from this session. Likely a texture path resolution issue when loading MDX models via WorldAssetManager.

## Recent Changes

- 2026-02-06: Phase 0 COMPLETE — 6 foundation files
- 2026-02-06: Phase 3 COMPLETE — 7 terrain files + ViewerApp integration
- 2026-02-07: Phase 4 — World Scene: lazy tile loading, MDDF/MODF placements, object rendering
- 2026-02-07: Fixed MDX holes — disabled backface culling
- 2026-02-07: Fixed WMO textures — BLP loaded per-batch from materials
- 2026-02-07: Fixed MODF bounding boxes — actual extents with min/max swap
- 2026-02-07: Added live minimap with click-to-teleport
- 2026-02-07: Added AreaPOI system (DBC loading, 3D/minimap markers, UI list)
- 2026-02-07: Fixed MDX blend modes — depth mask off for transparent layers, alpha discard 0.1
- 2026-02-07: WMO doodads now rendered with WMO's modelMatrix
- 2026-02-07: WMO rotation — BLOCKED after many attempts
- 2026-02-08: Fixed standalone MDX rendering — MirrorX model matrix for LH→RH conversion
- 2026-02-08: Fixed BIDX parsing — 1 byte per vertex (not 4)
- 2026-02-08: Reverted MTLS dual-format heuristic — back to stable count-header format
- 2026-02-08: Fixed WMO WorldScene regression — reverted to stable commit a1b0b41
- 2026-02-08: Fixed GLB export — Z-up → Y-up conversion for MDX and WMO
- 2026-02-08: Added terrain alpha mask debug view (Show Alpha Masks toggle)
- 2026-02-08: Applied Noggit edge fix for terrain alpha map seams
