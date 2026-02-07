# Progress — MdxViewer Renderer Reimplementation

## Status: Phase 4 Mostly Complete — WMO Rotation BLOCKED

## What Works Today

| Feature | Status |
|---------|--------|
| Terrain rendering + AOI lazy loading | ✅ |
| MDX model loading + rendering | ✅ Per-geoset, multi-layer materials, textured, no backface culling |
| MDX blend modes + depth mask | ✅ Transparent layers don't write depth |
| WMO v14 loading + rendering | ✅ Groups, BLP textures per-batch |
| WMO doodad sets | ✅ Loaded and rendered with WMO modelMatrix |
| WMO rotation/facing | ❌ BLOCKED — models in BB but face wrong direction |
| MDDF/MODF placements | ✅ Position correct |
| Bounding boxes | ✅ Actual MODF extents with correct min/max swap |
| BLP2 texture loading | ✅ DXT1/3/5, palette, JPEG |
| MPQ data source | ✅ Listfile, nested WMO archives |
| DBC integration | ✅ DBCD, replaceable texture resolution |
| Camera | ✅ Free-fly WASD + mouse look |
| ImGui UI | ✅ File browser, model info, visibility toggles |
| Live minimap + click-to-teleport | ✅ |
| AreaPOI system | ✅ DBC loading, 3D markers, minimap markers, UI list |
| Object picking/selection | ✅ |
| GLB export | ✅ MDX + WMO |
| Standalone WMO/MDX viewer | ❌ Black screen |

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Foundation | ✅ Complete |
| 3 | Terrain | ✅ Complete |
| 4 | World Scene | ⚠️ Mostly complete — WMO rotation blocked |
| 1 | MDX Animation | ⏳ Not started |
| 2 | Particles | ⏳ Not started |
| 5-7 | Liquids, Detail Doodads, Polish | ⏳ Not started |

## Key Blocker: WMO Rotation

See `activeContext.md` for full details. Models sit in bounding boxes but face ~180° wrong. Multiple vertex and transform approaches tried and failed. Need fresh approach — possibly study noggit's full WMO vertex pipeline or wow.export end-to-end.

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
- 2026-02-07: WMO rotation — BLOCKED after many attempts. Reverted to known-good baseline (raw vertices, simple rotation formula). Models in BBs but facing wrong.
