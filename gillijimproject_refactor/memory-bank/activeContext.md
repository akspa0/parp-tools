# Active Context

## Current Focus: MDX Compatibility Port + Rendering Parity (Feb 14, 2026)

MdxViewer is the **primary project** in the tooling suite. It is a high-performance 3D world viewer supporting WoW Alpha 0.5.3, 0.6.0, and LK 3.3.5 game data.

### Recently Completed (Feb 14)

- **GEOS Port (wow-mdx-viewer parity)**: ✅ `MdxFile.ReadGeosets` now routes by version with strict paths for v1300/v1400 and v1500, with guarded fallback.
- **SEQS Name Recovery**: ✅ Counted 0x8C named-record detection broadened so playable models no longer fall into `Seq_{animId}` fallback names in many cases.
- **PRE2 Parser Expansion**: ✅ Particle emitter v2 parser now reads full scalar payload layout, spline block, and skips known anim-vector tails safely for alignment.
- **RIBB Parser Expansion**: ✅ Ribbon parser now processes known tail anim-vector chunks safely for alignment.
- **Specular/Env Orientation Fix (shader)**: ✅ MDX fragment shader now flips normals/view-normals on backfaces before sphere-env UV and lighting/specular, targeting inside-out dome reflections.

### Previously Completed (Feb 11-12)

- **Full-Load Mode**: ✅ `--full-load` (default) / `--partial-load` CLI flags — loads all tiles at startup
- **Specular Highlights**: ✅ Blinn-Phong specular in ModelRenderer fragment shader (shininess=32, intensity=0.3)
- **Sphere Environment Map**: ✅ `SphereEnvMap` flag (0x2) generates UVs from view-space normals for reflective surfaces
- **MDX Bone Parser**: ✅ BONE/HELP/PIVT chunks parsed with KGTR/KGRT/KGSC keyframe tracks + tangent data
- **MDX Animation Engine**: ✅ `MdxAnimator` — hierarchy traversal, keyframe interpolation (linear/hermite/bezier/slerp)
- **Animation Integration**: ✅ Per-frame bone matrix update in MdxRenderer.Render()
- **WoWDBDefs Bundling**: ✅ `.dbd` definitions copied to output via csproj Content items
- **Release Build**: ✅ `dotnet publish -c Release -r win-x64 --self-contained` verified working (1315 .dbd files bundled)
- **GitHub Actions**: ✅ `.github/workflows/release-mdxviewer.yml` — tag-triggered + manual dispatch, creates ZIP + GitHub Release
- **No StormLib**: ✅ Pure C# `NativeMpqService` handles all MPQ access — no native DLL dependency

### Previously Completed (Feb 9-10)

- WMO doodad culling (distance + cap + sort + fog passthrough)
- GEOS footer parsing (tag validation)
- Alpha cutout for trees, MDX fog skip for untextured
- AreaID fix (low 16-bit extraction + fallback)
- Directional tile loading with heading-based priority
- DBC lighting (Light.dbc + LightData.dbc)
- Replaceable texture DBC resolution with MPQ validation

### Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| Alpha 0.5.3 WDT terrain | ✅ | Monolithic format, 256 MCNK chunks per tile |
| 0.6.0 split ADT terrain | ✅ | StandardTerrainAdapter, MCNK with header offsets |
| 0.6.0 WMO-only maps | ✅ | MWMO+MODF parsed from WDT |
| 3.3.5 split ADT terrain | ⚠️ | Loading freeze — needs investigation |
| WMO v14 rendering | ✅ | 4-pass: opaque/doodads/liquids/transparent |
| WMO liquid (MLIQ) | ✅ | matId-based type detection, correct positioning |
| Terrain liquid (MCLQ) | ✅ | Per-vertex sloped heights, absolute world Z |
| MDX rendering | ✅ | Two-pass, alpha cutout, blend modes 0-6 |
| Async tile streaming | ✅ | 9×9 AOI, directional lookahead, persistent cache |
| Frustum culling | ✅ | View-frustum + distance + fade |
| DBC Lighting | ✅ | Zone-based ambient/fog/sky colors |
| Minimap overlay | ✅ | BLP tiles, zoom, click-to-teleport |

### Known Issues / Next Steps

1. **Runtime validation pending (critical handoff item)** — verify PRE2/RIBB-heavy models visually after parser expansion.
2. **Specular/env dome check pending** — confirm Dalaran dome-like materials now reflect outward after backface normal correction.
3. **Residual SEQS/material parity work** — continue porting edge-case behavior from `lib/wow-mdx-viewer` if specific models still diverge.
4. **WMO semi-transparent window materials** — Stormwind glass still maps to wrong geometry (root cause unknown).
5. **MDX cylindrical texture stretching** — barrels/tree trunks still show stretched planks on some assets.
6. **3.3.5 ADT loading freeze** — needs investigation.
7. **WMO culling too aggressive** — objects outside WMO not visible from inside.

---

## Key Architecture Decisions

### Coordinate System (Confirmed via Ghidra)
- WoW: right-handed, X=North, Y=West, Z=Up, Direct3D CW front faces
- OpenGL: CCW front faces
- Fix: Reverse winding at GPU upload + 180° Z rotation in placement
- Terrain: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`
- WMO-only maps: raw WoW world coords (no MapOrigin conversion)

### Performance Constants

| Constant | Value | Location |
|----------|-------|----------|
| DoodadCullDistance (world) | 1500f | WorldScene.cs |
| DoodadSmallThreshold | 10f | WorldScene.cs |
| WmoCullDistance | 2000f | WorldScene.cs |
| NoCullRadius | 150f | WorldScene.cs |
| WMO DoodadCullDistance | 500f | WmoRenderer.cs |
| WMO DoodadMaxRenderCount | 64 | WmoRenderer.cs |
| AoiRadius | 4 (9×9) | TerrainManager.cs |
| AoiForwardExtra | 3 | TerrainManager.cs |
| MaxGpuUploadsPerFrame | 8 | TerrainManager.cs |
| MaxConcurrentMpqReads | 4 | TerrainManager.cs |

### Key Files

| File | Purpose |
|------|---------|
| `WorldScene.cs` | Placement transforms, instance management, culling |
| `WmoRenderer.cs` | WMO v14 GPU rendering, doodad culling, liquid |
| `ModelRenderer.cs` | MDX GPU rendering, alpha cutout, fog skip |
| `AlphaTerrainAdapter.cs` | Alpha 0.5.3 WDT terrain + AreaID + liquid type |
| `StandardTerrainAdapter.cs` | 0.6.0 / 3.3.5 split ADT terrain + MCLQ + WMO-only maps |
| `TerrainManager.cs` | AOI streaming, persistent cache, MPQ throttling |
| `LiquidRenderer.cs` | MCLQ/MLIQ liquid mesh rendering |
| `AreaTableService.cs` | AreaID → name with MapID filtering |
| `LightService.cs` | DBC Light/LightData zone-based lighting |
| `ReplaceableTextureResolver.cs` | DBC-based replaceable texture resolution |
| `MdxFile.cs` | MDX parser (GEOS, BONE, PIVT, HELP with KGTR/KGRT/KGSC tracks) |
| `MdxAnimator.cs` | Skeletal animation engine (hierarchy, interpolation, bone matrices) |
| `MdxViewer.csproj` | Project file with WoWDBDefs bundling |
| `.github/workflows/release-mdxviewer.yml` | CI/CD release workflow |
