# Active Context

## Current Focus: Terrain Regression Recovery (Mar 17, 2026)

MdxViewer — high-performance 3D WoW world viewer (0.5.3, 0.6.0, 3.3.5). Primary project.

### 3.x Terrain Status: BROKEN

Runtime later-client terrain texturing is still visibly wrong. Recent MCAL/MCCV fixes are **not sufficient closure**.

**Key fixes applied but insufficient:**
- Batched alpha/shadow slice no longer remaps row/col 63→62
- Alpha decode is now version-gated (0.x legacy vs 3.x strict Mcal)
- 3.x path uses MCAL layer span when WDT big-alpha bit absent
- 3.x trusts `MCLY.UseAlpha` first, relaxed span-decode only as fallback
- MCCV decoded as BGRA, `0..2` modulation range (WotLK semantics)
- Missing overlay BLPs invalidated in tile-array renderer
- `_tex0.adt` sourcing disabled for 3.x (root ADT only)
- Atlas roundtrip + packing parity tests pass

**Active blockers:**
- Alpha debug mode hides chunk/tile overlays (shader exits early) — blocks visual diagnosis
- `AlphaMapService` still separate from active viewer path
- WoWMuseum 3.3.5 sample missing overlay BLPs (separate from decode issues)

### Recovery Execution (Mar 17)

- **Baseline**: `343dadfa27df08d384614737b6c5921efe6409c8`
- **Recovery branch**: `recovery/terrain-surgical-343dadf` in `_recovery_343dadf` worktree
- **Safe replay**: `c1e0d29` — managers/models from `177f961` without fused terrain topology
- **Wave 0 evidence**: `177f961` = first fused alpha+shadow tile-pass merge
- **Version policy**: MPQ load requires explicit client version-family selection (no heuristic guessing)
- **NEXT STEP**: Wave 1 topology rollback in `TerrainRenderer`, `TerrainTileMeshBuilder`, `TerrainTileMesh`
- **Rule**: do NOT replay `177f961`, `d50cfe7`, `39799bf` wholesale; split-based replay with runtime gates only

### Other Recent Fixes (Mar 16-17)

- WMO/MDX cached-null reload bug fixed (needs runtime verify)
- WDL preview spawn confirmed working for Alpha 0.5.3
- Later-client maps bypass WDL preview, go direct WDT load
- World lifecycle cleanup on scene switch
- MH2O parsing improved (code-level, not runtime-verified)
- Skybox M2s routed to backdrop pass; fog M2 depth fixed
| Minimap overlay | ✅ | BLP tiles, zoom, click-to-teleport |

### Known Issues / Next Steps

1. **Runtime validation pending (critical handoff item)** — verify PRE2/RIBB-heavy models visually after parser expansion.
2. **Specular/env dome check pending** — confirm Dalaran dome-like materials now reflect outward after backface normal correction.
3. **Residual SEQS/material parity work** — continue porting edge-case behavior from `lib/wow-mdx-viewer` if specific models still diverge.
4. **WMO semi-transparent window materials** — Stormwind glass still maps to wrong geometry (root cause unknown).
5. **MDX cylindrical texture stretching** — barrels/tree trunks still show stretched planks on some assets.
6. **3.3.5 ADT loading freeze** — needs investigation.
7. **Terrain alpha regressions after baseline** — 3.x terrain texturing is still visibly broken at runtime, and the alpha debug UI currently blocks chunk/tile overlays when enabled, which slows diagnosis.
8. **WMO culling too aggressive** — objects outside WMO not visible from inside.

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
