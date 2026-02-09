# Active Context

## Current Focus: MdxViewer — MDX Rendering Quality (Feb 9, 2026)

### Recently Completed (This Session)

- **WMO Liquid Rendering**: ✅ COMPLETE
  - Liquid type derived from `groupLiquid` + tile flags (noggit logic)
  - Positioning fixed: Z-up axis + 90° CCW rotation fix `(-j, +i)` on XY plane
  - Tile visibility: bit 0x08 hide flag from noggit SMOLTile
- **MDX Z-Buffer Fix**: ✅ Two-pass render (opaque then transparent)
- **WMO Transparent Textures**: ✅ 4-pass render: opaque → doodads → liquids → transparent
  - Fragment shader uses `texColor.a` + `uAlphaTest` uniform for per-BlendMode discard
- **WMO Doodad Loading**: ✅ Fixed 72% failure rate (4076/5614)
  - Added `FindInFileSet` case-insensitive MPQ lookup
  - Added `.mdx/.mdl` extension swap fallback
  - Result: 5614/5614 loaded (0 failures)

### Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| Alpha WDT terrain | ✅ | Monolithic format, 256 MCNK chunks per tile |
| Standard WDT+ADT (3.3.5) | ✅ | Split ADT files from MPQ/IDataSource |
| WMO v14 rendering | ✅ | 4-pass: opaque/doodads/liquids/transparent |
| WMO liquid (MLIQ) | ✅ | Type detection, positioning, tile visibility |
| WMO doodad loading | ✅ | Case-insensitive MPQ + mdx/mdl swap |
| MDX rendering | ✅ | Two-pass opaque/transparent, blend modes 0-6 |
| MCSH shadow maps | ✅ | 64×64 bitmask, all layers |
| MCLQ ocean liquid | ✅ | Inline liquid from MCNK flags |
| Async tile streaming | ✅ | AOI-based lazy loading |
| Frustum culling | ✅ | View-frustum + bounding box |
| Minimap overlay | ✅ | From minimap tiles |

### Next Steps (Priority Order)

1. **MDX rendering quality** — Fix alpha discard thresholds, per-geoset color/alpha ← SEE `memory-bank/planning/MDX_RENDERING_PLAN.md`
2. **WMO lighting** — v14-16 grayscale lightmap, v17 vertex color (MOCV)
3. **MDX particles** — Separate shader, billboard quads, blend modes
4. **GLB export test** — End-to-end on v16 WMO
5. **0.6.0 ADT+WDT loading** — New format support
6. **LK StandardTerrainAdapter fix** — 0 chunks loading

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
| `WmoRenderer.cs` | WMO v14 GPU rendering, 4-pass transparency, liquid |
| `ModelRenderer.cs` | MDX GPU rendering, two-pass, blend modes |
| `WmoV14ToV17Converter.cs` | WMO parser: materials, groups, doodads, MLIQ |
| `AlphaTerrainAdapter.cs` | Alpha WDT terrain loading |
| `StandardTerrainAdapter.cs` | 3.3.5 WDT+ADT terrain loading |
| `TerrainManager.cs` | AOI-based tile streaming |
| `MdxFile.cs` | MDX parser (GEOS/BIDX/BWGT fix) |
