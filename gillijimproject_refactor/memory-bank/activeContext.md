# Active Context

## Current Focus: MdxViewer — Performance & Rendering Quality (Feb 10, 2026)

### Recently Completed (This Session — Feb 9-10)

- **WMO Doodad Culling**: ✅ Distance cull (500u), nearest-first sort, max 64/frame, fog passthrough
  - Major performance win — WMO doodads were the primary bottleneck
- **WMO Liquid Type Fix**: ✅ `GroupLiquid=15` → magma (was incorrectly sampling tile flags → slime)
- **GEOS Footer Parsing**: ✅ `IsValidGeosetTag()` peek-ahead prevents footer misread as chunk tags
- **Alpha Cutout for Trees**: ✅ Layer 0 + BlendMode=Transparent → hard discard in opaque pass (no blend)
- **MDX Fog Skip**: ✅ Untextured (magenta) MDX fragments skip fog blending
- **AreaID Fix**: ✅ Extract low 16 bits from `Unknown3`, fallback to low byte for MapID mismatch
- **MDX Cull Relaxation**: ✅ DoodadCullDistance 1200→1500, SmallThreshold 20→10
- **Directional Tile Loading**: ✅ Camera heading tracking, forward lookahead tiles, priority-sorted load queue
- **DBC Lighting**: ✅ LightService loads Light.dbc + LightData.dbc, zone-based ambient/fog/sky colors
- **ReplaceableTexture Fix**: ✅ DBC resolver now validates all CDI variants against MPQ, picks correct skin
- **Rotation Revert**: Reverted X↔Y rotation swap — original `rx=Rotation.X, ry=Rotation.Y` is correct

### Reverted Changes (Caused Regressions)
- ❌ WMO fog skip for untextured fragments — broke WMO rendering entirely
- ❌ MDX rotation axis swap (X↔Y) — caused fence tilt issues
- ❌ MDX rotation negation — caused tree geometry to mirror/stretch into sky

### Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| Alpha WDT terrain | ✅ | Monolithic format, 256 MCNK chunks per tile |
| WMO v14 rendering | ✅ | 4-pass: opaque/doodads/liquids/transparent |
| WMO doodad culling | ✅ | Distance + cap + sort + fog passthrough |
| WMO liquid (MLIQ) | ✅ | GroupLiquid=15 → magma, type detection |
| MDX rendering | ✅ | Two-pass, alpha cutout for trees, fog skip for untextured |
| GEOS parser | ✅ | Tag validation + footer parsing |
| Async tile streaming | ✅ | Directional loading with heading-based priority |
| Frustum culling | ✅ | View-frustum + distance + fade |
| AreaID lookup | ✅ | Low 16-bit extraction + low byte fallback |
| DBC Lighting | ✅ | Light.dbc + LightData.dbc zone-based colors |
| Replaceable Textures | ✅ | DBC CDI variant validation + model dir scan fallback |
| Minimap overlay | ✅ | From minimap tiles |

### Known Issues / Next Steps

1. **Terrain liquid type** — Lava still green, diagnostic logging added, needs flag analysis
3. **MDX textures magenta** — ROOT CAUSE UNKNOWN, needs diagnostic logging session
4. **Water plane MDX rotation** — Flat water MDX models tilted wrong

---

## Key Architecture Decisions

### Coordinate System (Confirmed via Ghidra)
- WoW: right-handed, X=North, Y=West, Z=Up, Direct3D CW front faces
- OpenGL: CCW front faces
- Fix: Reverse winding at GPU upload + 180° Z rotation in placement
- Terrain: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`
- MDX rotations: `rx = Rotation.X`, `ry = Rotation.Y` (NO swap — swap was wrong)

### Performance Constants

| Constant | Value | Location |
|----------|-------|----------|
| DoodadCullDistance (world) | 1500f | WorldScene.cs |
| DoodadSmallThreshold | 10f | WorldScene.cs |
| WmoCullDistance | 2000f | WorldScene.cs |
| NoCullRadius | 150f | WorldScene.cs |
| WMO DoodadCullDistance | 500f | WmoRenderer.cs |
| WMO DoodadMaxRenderCount | 64 | WmoRenderer.cs |
| AoiRadius | 2 (5×5) | TerrainManager.cs |
| AoiForwardExtra | 1 | TerrainManager.cs |

### Key Files

| File | Purpose |
|------|---------|
| `WorldScene.cs` | Placement transforms, instance management, culling |
| `WmoRenderer.cs` | WMO v14 GPU rendering, doodad culling, liquid |
| `ModelRenderer.cs` | MDX GPU rendering, alpha cutout, fog skip |
| `AlphaTerrainAdapter.cs` | Alpha WDT terrain + AreaID + liquid type |
| `TerrainManager.cs` | Directional AOI tile streaming |
| `AreaTableService.cs` | AreaID → name with MapID filtering |
| `LightService.cs` | DBC Light/LightData zone-based lighting |
| `ReplaceableTextureResolver.cs` | DBC-based replaceable texture resolution with MPQ validation |
| `MdxFile.cs` | MDX parser (GEOS tag validation) |
