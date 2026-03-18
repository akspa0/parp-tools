# Active Context

## Current Focus: v0.4.0 Recovery Branch (Mar 17, 2026)

Working branch is now reset in the main tree, not only in side worktrees.

- Branch: recovery/v0.4.0-surgical-main-tree
- Baseline tag/commit: v0.4.0 / 343dadf
- .github metadata restored from main and committed: 845748b
- .github restore was pushed to origin/recovery/v0.4.0-surgical-main-tree

### Recovery Work Completed On This Branch

- Re-established v0.4.0 baseline in the primary tree and validated build.
- Restored the project instruction stack from main:
	- copilot-instructions
	- instructions
	- prompts
	- terrain-alpha-regression skill files
- Applied profile-driven terrain alpha decode routing in viewer terrain path:
	- Added TerrainAlphaDecodeMode to AdtProfile in FormatProfileRegistry
	- 3.x profiles route to LichKingStrict
	- 0.x profiles route to LegacySequential
	- StandardTerrainAdapter alpha extraction now routes by profile mode
	- Strict path includes UseAlpha-first decode plus offset/span fallback for mis-set flags
	- Legacy path remains sequential 4-bit nibble expansion

### Validation Status

- Build: dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug PASSED
- Runtime real-data spot-check: PENDING (Alpha-era + LK 3.3.5)
- No claim of full terrain regression safety without runtime real-data validation.

### Next Integration Queue (Ordered)

1. Commit and push the current profile/decode code slice if not already committed.
2. Runtime-check alpha decode behavior on both fixed data families.
3. Continue commit-by-commit intake from v0.4.0..main with strict triage:
	 - SAFE first
	 - MIXED only with dependencies proven and build gates
	 - RISKY terrain renderer/decode rewrites skipped unless explicitly approved
4. Keep UI changes incremental; avoid broad layout rewrites.
5. Pull selected import/export functionality in small batches after profile/decode stabilization.

### Surgical Intake Triage (Mar 17)

- Commit triage against `v0.4.0..main` is now documented for the current queue:
	- `177f961`: RISKY, skip entire commit (terrain renderer + tile mesh + alpha decode rewrite)
	- `37f669c`: RISKY, skip entire commit (relaxed alpha heuristics + MPQ decompression changes)
	- `d50cfe7`, `326e6f8`, `4e2f681`, `39799bf`, `62ecf64`: MIXED, only extract isolated safe slices
- First SAFE batch selected:
	- take only the corrected `TerrainImageIo` alpha-atlas helper from `62ecf64`
	- do not take the earlier `d50cfe7` version because it reintroduced atlas import/export edge remapping
	- do not take ViewerApp, TerrainRenderer, WorldScene, test-project, or alpha-decode hunks in the first batch
- Required gate after the first SAFE batch: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime terrain validation remains required after any terrain-adjacent batch; build success is not proof of terrain correctness.
- First SAFE batch status:
	- corrected `TerrainImageIo` helper has been applied in the recovery branch
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the change
	- runtime real-data validation is still pending

### Rendering Fix Batch (Mar 18)

- Applied the main-branch `WorldAssetManager` residency fix in the recovery branch:
	- MDX/WMO renderer residency now defaults to unlimited
	- only the raw file-data cache remains bounded
	- cached failed model loads are retried instead of becoming permanent null entries
	- lazy `GetMdx` / `GetWmo` lookups can now load on demand
- Applied the minimal main-branch skybox backdrop path without broad ViewerApp/UI churn:
	- skybox-like MDX/M2 placements are routed into a separate skybox instance list
	- nearest skybox placement renders as a camera-anchored backdrop before terrain
	- `ModelRenderer` now has a backdrop path that keeps depth test/write disabled for all layers
- Current branch already had the reflective M2 depth-flag fix and the guarded env-map backface handling, so those were not re-applied.
- Build gate passed again after this rendering batch: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime validation is still required for:
	- WMO/MDX disappearance when moving away and back
	- skybox model classification and backdrop behavior
	- MH2O liquid rendering on LK data

### MCCV + MPQ Recovery Batch (Mar 18)

- Restored active-branch MCCV terrain support in the chunk renderer path:
	- `StandardTerrainAdapter` now carries MCNK MCCV data into `TerrainChunkData`
	- `TerrainMeshBuilder` uploads per-vertex RGBA alongside position/normal/UV
	- `TerrainRenderer` consumes MCCV in the shader
- Follow-up correction after runtime feedback:
	- MCCV is now treated as BGRA, matching the repo's own `MinimapService.GenerateMccvData` documentation
	- neutral/no-tint MCCV is treated as mid-gray (`127`) rather than white
	- shader tinting now maps mid-gray to neutral and no longer relies on MCCV alpha as terrain tint strength
- Applied the isolated `NativeMpqService` slice from the mixed MPQ recovery commits:
	- broader patch archive priority ordering, including locale/custom patch variants
	- encrypted-file key derivation now tries the full normalized path first, then basename fallback
	- per-sector MPQ decompression now handles bitmask combinations instead of only single-byte cases
	- BZip2 sector decompression added via SharpZipLib
- Follow-up patch-chain fix:
	- `NativeMpqService.LoadArchives(...)` now discovers MPQs recursively instead of only scanning a few top-level directories
	- Alpha-style single-asset wrapper archives (`.wmo.mpq`, `.wdt.mpq`, `.wdl.mpq`) are still excluded from this generic path because `MpqDataSource` handles them separately
- Build gates passed after this batch:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime real-data validation is still required for:
	- 1.x+ patch-chain reads on patched client data
	- later-version encrypted MPQ entries
	- 3.x MCCV highlight/tint behavior on real LK terrain after the BGRA + mid-gray semantic correction

### Commit 39799bf Model Slice (Mar 18)

- The M2 load-failure fix associated with `39799bf` was the `NativeMpqService` encrypted-read compatibility slice, which is now already applied.
- The only additional model-renderer change from that commit was also applied:
	- `ModelRenderer` no longer renders particles on the world-scene batched instance path
	- standalone model viewing still renders particles
- Rationale:
	- world-scene batch instancing does not yet propagate per-instance transforms into particle simulation/rendering
	- leaving particles enabled there can produce camera-locked billboard artifacts on placed models
- Build gate passed again after applying this renderer hunk: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`

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
