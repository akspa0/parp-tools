# Active Context

## Current Focus: Terrain Alpha Regression Audit + Runtime Handoff (Mar 17, 2026)

MdxViewer is the **primary project** in the tooling suite. It is a high-performance 3D world viewer supporting WoW Alpha 0.5.3, 0.6.0, and LK 3.3.5 game data.

### Current Terrain Alpha Findings (Mar 16)

- **Validation scope correction (Mar 17)**: do not treat repo `test_data/development` or `test_data/WoWMuseum/*` samples as the active signoff target for 3.x terrain. Current runtime validation must follow the user's official 3.0.1-era client data, with repo samples kept only as archival parser references.
- **MPQ load path now requires explicit version-family choice**: `ViewerApp` no longer infers build/profile from folder names or MPQ heuristics during `Open Game Folder`. Users must select a client version family explicitly before DBC/profile routing is enabled.

- **Batched terrain upload was mutating decoded alpha/shadow data**: `TerrainTileMeshBuilder.FillAlphaShadowSlice` was remapping texel row/column 63 back to 62 for every slice, even though `Mcal` and the per-chunk upload path already treat edge-fix as decode-time behavior.
- **Real-data validation confirmed the bug on both Alpha and LK tiles**: a focused audit on `test_data/0.5.3/alphawdt/World/Maps/Azeroth/Azeroth.wdt` tile `(0,0)` and `test_data/WoWMuseum/335-dev/World/Maps/development` tile `(0,0)` found explicit alpha/shadow bytes that differed at the packed tile-array edge. The old remap would have changed 17,367 bytes on the Alpha tile and 5,166 bytes on the LK tile.
- **Atlas export/import parity fix is now in place**: `TerrainImageIo.BuildAlphaAtlasFromChunks` and `DecodeAlphaShadowArrayFromAtlas` now preserve decoded 64x64 alpha/shadow data without reapplying the old row/column 63→62 duplication. The same real-data audit now reports zero atlas roundtrip diffs on those Alpha and LK tiles.
- **Alpha-era terrain decode had also drifted from the baseline**: `AlphaTerrainAdapter.ExtractAlphaMaps` had been changed to use LK compressed-alpha helpers and to duplicate the last texel in each 4-bit row. That broke Alpha texture blending. The active path is now back to Alpha semantics: raw 4096-byte wide payloads stay raw, and 2048-byte packed rows expand directly without per-row edge duplication.
- **Viewer split-ADT alpha path is now version-gated instead of heuristic-driven**: `StandardTerrainAdapter.ExtractAlphaMaps` no longer tries to infer LK-vs-legacy decode from overlapping `MCLY` flags or per-chunk MCAL size. 0.x split ADT profiles stay on the legacy sequential 4-bit path; 3.x profiles use strict LK `Mcal.GetAlphaMapForLayer()` decode.
- **3.x strict alpha decode no longer blindly trusts the WDT big-alpha bit**: in the LK path, uncompressed layers now use the actual MCAL layer span to distinguish 4096-byte 8-bit alpha from legacy 2048-byte 4-bit alpha. This closes the case where a 3.x tile omitted the global big-alpha indicator and the viewer fell back to 4-bit decode even though the layer payload itself was 8-bit.
- **3.x strict alpha decode now trusts later-client layer flags first**: the viewer uses `Mcal.GetAlphaMapForLayer()` when `MCLY.UseAlpha` is present and only falls back to relaxed span-based decode when a layer is missing that flag but still has a nonzero alpha offset. This narrows the missing-metadata recovery path instead of making relaxed decode the default for 3.x tiles.
- **Batched terrain rendering no longer blends missing overlay textures**: `TerrainRenderer.CreateDiffuseArrayTexture` now invalidates per-vertex diffuse indices for slices whose BLP failed to load, so the tile-array path matches the per-chunk renderer instead of blending those overlays against a white fallback slice.
- **3.x later-client terrain no longer consults stray `_tex0.adt` files for texture and alpha sourcing**: split texture ADTs are now profile-gated back to the early split-ADT path. The active 3.x viewer path stays on the root ADT for MTEX, MCLY, and MCAL instead of opportunistically preferring `_tex0` when one happens to exist.
- **MCCV terrain tint now follows documented WotLK semantics**: the viewer no longer reads raw MCCV bytes as `RGBA` and multiplies terrain by them directly. `TerrainMeshBuilder` and `TerrainTileMeshBuilder` now decode MCCV as `BGRA`, ignore the alpha byte, and pre-scale the vertex color into the client-style `0..2` modulation range before the terrain shaders apply it. This removes the old whole-terrain darkening caused by treating neutral `0x7F` MCCV bytes like `0.5` diffuse multipliers.
- **`AlphaMapService` is still not the active viewer source of truth**: the viewer now uses explicit profile gating in `StandardTerrainAdapter`, and `AlphaMapService` remains separate until its semantics are brought fully in line with the active path.
- **Active terrain alpha path now has minimal first-party regression coverage**: `src/MdxViewer.Tests/Terrain/TerrainAlphaParityTests.cs` covers batched tile-array packing parity, `TerrainImageIo` atlas roundtrip parity, explicit legacy-vs-LK decode gating, and a fixed-path WoWMuseum 3.3.5 tile load smoke test. Broader decode parity across more LK samples is still missing.
- **Requested WoWMuseum 3.3.5 sample still has incomplete loose diffuse assets**: `test_data/WoWMuseum/335-dev/World/Maps/development/development_0_0.adt` preserves A1/A2/A3 on disk and through `StandardTerrainAdapter`, but the loose root does not contain the referenced overlay BLPs. Visual validation on that sample must account for missing texture assets separately from alpha decode.
- **3.x terrain texturing is still not resolved**: runtime validation is still showing broken later-client layer decode/blending in the viewer. Treat the current 3.x terrain path as unresolved even after the recent MCAL/MCCV changes.
- **Alpha debug UX regressed terrain diagnosis**: enabling the alpha-mask view currently prevents chunk/tile overlays from showing through in the same view. The active terrain shader returns early in alpha-debug mode before the chunk/tile boundary overlays are applied, which makes later-client layer debugging materially harder.

### Recovery Execution Checkpoint (Mar 17)

- **Surgical recovery branch is active from baseline**: `recovery/terrain-surgical-343dadf` in worktree `_recovery_343dadf` is anchored to baseline `343dadfa27df08d384614737b6c5921efe6409c8`.
- **Wave 0 evidence is locked**: commit `177f961` remains the first confirmed fused alpha+shadow tile-pass merge point (`uAlphaShadowArray`, `AlphaShadowArrayTexture`, `uShadowSampler`).
- **Safe replay slice already committed on baseline branch**: `c1e0d29` replays low-risk manager/model changes from `177f961` (`TerrainManager`, `VlmTerrainManager`, `ModelRenderer`) without reintroducing fused terrain topology.
- **Version policy hardening is implemented on active branch**: `ViewerApp` MPQ load now requires explicit client version-family selection; path and MPQ heuristic build guessing is disabled in the active flow.
- **Next locked engineering step**: execute Wave 1 topology rollback in `TerrainRenderer`, `TerrainTileMeshBuilder`, and `TerrainTileMesh` before replaying deferred terrain refactors.

### Current Handoff (Mar 17)

- **Current terrain handoff is back to active 3.x decode investigation**: runtime later-client terrain still looks wrong, so the active seam remains layer decode/sourcing/blending, not just validation.
- **Do not treat the recent 3.x MCAL heuristics as signed off**: runtime evidence now says later-client terrain texturing is still broken, so the recent big-alpha/compressed-alpha/MCCV fixes are not sufficient closure.
- **The alpha-mask debug path is currently insufficient for terrain investigation**: until the viewer can render alpha debug and chunk/tile overlays together, the debug UI itself is a blocker for diagnosing later-client terrain issues efficiently.
- **Recovery sequencing remains strict**: do not replay mixed high-risk terrain commits (`177f961`, `d50cfe7`, `39799bf`) wholesale; keep split-based replay with runtime gates only.

### Current WMO Streaming Findings (Mar 16)

- **Failed WMO loads were sticky across stream-out/stream-in**: `WorldAssetManager.EnsureWmoLoaded` and `EnsureMdxLoaded` treated cache-key presence as success even when the cached renderer was `null` from an earlier failed load.
- **The stream path now retries failed loads on re-entry**: if a cached WMO or MDX renderer is `null`, the next `Ensure*Loaded` call reloads it instead of treating the stale failure as a valid resident asset.

### WDL Preview Runtime Validation (Mar 16)

- **The WDL preview map-load path is confirmed working for Alpha 0.5.3**: the preview image orientation, clicked tile selection, and resulting world spawn line up in runtime testing instead of landing in empty sky or zero-tile areas.
- **Root cause was two-part**: the preview spawn path had regressed to `TileSize` scale, and the preview image/click path was still transposed relative to the terrain tile grid.
- **Scope correction**: that preview path is not generic across later clients. The current `WdlParser` is still Alpha-0.5.3-specific, so later builds must bypass preview and load WDTs directly.
- **Active behavior**: preview tiles render in normal screen order for supported Alpha WDLs, clicks are converted back into terrain tile coordinates before spawn calculation, WDL hide/show indexing now matches the WDL load path, and non-0.5.x clients no longer route normal map loads through the unsupported preview path.

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
