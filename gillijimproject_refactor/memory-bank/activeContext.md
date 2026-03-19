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
- Runtime real-data spot-check: PARTIAL PASS
	- user confirmed Alpha-era 0.5.3 terrain renders correctly again after the alpha-edge-fix restoration
	- user confirmed a 3.0.1 alpha build now renders correctly on the current profile-driven 3.x path
	- earlier 3.3.5 spot-check also looked correct, but broader cross-map signoff is still pending
- Do not claim full terrain regression safety beyond the validated samples above.

### Next Integration Queue (Ordered)

1. Commit and push the current profile/decode code slice if not already committed.
2. Broaden runtime-check alpha decode behavior beyond the currently validated 0.5.3 and 3.0.1 samples.
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

### WDL + Model Compatibility Follow-up (Mar 18)

- Follow-up after runtime feedback on the newly ported WDL preview cache:
	- the main WDL failure on 1.x/3.x was not only path lookup; `WoWMapConverter.Core.VLM.WdlParser` hard-rejected every non-`0x12` WDL
	- parser is now version-tolerant and scans for `MAOF`/`MARE` instead of requiring Alpha-only layout assumptions
	- parser also tolerates MAOF offsets that point either at a `MARE` chunk header or directly at the height payload
- Viewer-side WDL read paths are now unified through `WdlDataSourceResolver`:
	- both preview warmup and 3D WDL terrain now try `.wdl` and `.wdl.mpq`
	- MPQ-backed loads also use `MpqDataSource.FindInFileSet(...)` so listfile/casing recovery works consistently
- Remaining 3.x doodad extension parity gap closed in `WmoRenderer`:
	- canonical doodad resolution now tries `.m2` in addition to `.mdx`/`.mdl`
- Semi-translucent model follow-up in `ModelRenderer`:
	- shared texture cache entries now carry a simple alpha classification (`Opaque`, `Binary`, `Translucent`)
	- classic non-M2 layer-0 `Transparent` now stays on the hard alpha-cutout path only when the loaded texture alpha is binary
	- textures with intermediate alpha values now render through the blended path instead of the old foliage-style cutout heuristic
- Build gate passed after this batch:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation is still required before claiming:
	- non-Alpha WDL previews / WDL 3D terrain actually load on the user's real 1.x/3.x map set
	- 3.x `.mdx` WMO doodads now resolve correctly as M2-family assets on real data
	- the semi-translucent material heuristic fixes the reported visuals without regressing classic cutout foliage

### M2 Empty-Fallback Guardrail (Mar 18)

- Follow-up after the standalone 3.x model-load freeze fix: some M2-family assets could still appear to load while producing a blank viewport and model info with zero geometry.
- Current conclusion is narrow:
	- this is at least partly a false-positive success path, not necessarily a valid render of an odd pre-release asset
	- raw `MD20` fallback conversion can yield an `MDX` shell that parses but has no renderable geosets
- Recovery change applied:
	- shared geometry validation added for converted M2 fallback results
	- standalone `ViewerApp`, world `WorldAssetManager`, and WMO doodad `WmoRenderer` now reject empty converted fallback models and keep the real failure path visible in logs
- Validation status:
	- alternate-OutDir build passed: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
	- no runtime real-data validation yet
	- do not over-claim this as a full M2 render fix for pre-release `3.0.1`; it is a guardrail that removes a misleading blank-success outcome

### Pre-release 3.0.1 M2 + Shared Transparency Follow-up (Mar 18)

- User runtime verification now narrows the remaining model issue further:
	- most unresolved M2 failures are specific to the pre-release `3.0.1` model family, not the later `3.3.5` layout
	- the active working assumption is that this pre-release family may be a hybrid or transitional `MDX` + `M2` variant rather than a clean later-WotLK `M2`
- Treat this as a separate compatibility track:
	- do not assume later `3.3.5` `MD20` / `.skin` semantics are sufficient for pre-release `3.0.1`
	- keep profile/version-aware model parsing on the roadmap instead of broadening generic fallback heuristics
	- the empty-fallback guardrail remains useful, but it is only a diagnostics fix
- Separate rendering issue still confirmed by runtime evidence:
	- neon-pink transparent surfaces still reproduce on both classic `MDX` and M2-family assets
	- that means the pink/transparency bug is not only an M2 parser problem; it is likely in shared material, texture binding, blend, or shader behavior
- Practical next investigation split:
	1. pre-release `3.0.1` model-structure compatibility in `WarcraftNetM2Adapter` / profile routing
	2. shared transparent-material shader parity across `ModelRenderer` and any M2-converted runtime path

### Pre-release 3.0.1 wow.exe Guide Handoff (Mar 19)

- Latest Ghidra pass mapped the common model load chain in `wow.exe` build `3.0.1.8303`:
	- `FUN_0077e2c0` -> `FUN_0077d3c0` -> `FUN_0079bc70` -> `FUN_0079bc50` -> `FUN_0079bb30` -> `FUN_0079a8c0`
- High-confidence parser contract now documented in `documentation/pre-release-3.0.1-m2-wow-exe-guide.md`:
	- root must be `MD20`
	- accepted version range is `0x104..0x108`
	- parser layout splits at `0x108`
	- shared typed span validators use strides `1`, `2`, `4`, `8`, `0x0C`, `0x30`, and `0x44`
	- confirmed nested record families include `0x70`, `0x2C`, `0x38`, `0xD4`, and `0x7C`
	- legacy side uses `0xDC` + `0x1F8`; later side uses `0xE0` + `0x234`
- Fresh-chat prompts now exist for implementation, deeper Ghidra follow-up, and runtime triage:
	- `.github/prompts/pre-release-3-0-1-m2-implementation-plan.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-ghidra-followup.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-runtime-triage.prompt.md`
- Do not treat the guide as proof that viewer support is implemented yet:
	- no new runtime validation happened in this documentation pass
	- Track B pink transparency remains separate

### Pre-release 3.0.1 Profile Routing Broadening (Mar 19)

- Active profile resolution is no longer restricted to exact build `3.0.1.8303`.
- `FormatProfileRegistry` now maps any parsed `3.0.1.x` build to the existing pre-release `3.0.1` ADT, WMO, and M2 profiles.
- Keep the scope narrow:
	- this is profile routing, not full parser completion for every remaining pre-release `3.0.1` model difference
	- other `3.0.x` builds still use the generic `3.0.x` fallback profile unless new binary evidence justifies a tighter mapping
- Validation status:
	- code change applied
	- build/runtime validation still pending for this exact routing update

### Pre-release 3.0.1 Parser + Fallback Alignment (Mar 19)

- `WarcraftNetM2Adapter` now has a dedicated pre-release `MD20` parse path based on the wow.exe contract instead of routing those files through Warcraft.NET's later-layout `MD21` parser.
- Current scope of the fix:
	- standalone model load
	- world doodad load
	- WMO doodad load
	- shared `M2ToMdxConverter` fallback for those entry points
- Important implementation boundary:
	- the prior profile-specific `.skin` parser path was disabled because its `0x70` / `0x2C` record-size assumptions were lifted from model-family validation, not proven `.skin` layout evidence
	- converter fallback now keeps pre-release handling geometry-focused by skipping later-layout animation / bone parsing and by not forcing optional fixed-stride `.skin` submesh / texture-unit parsing
- Current residual risk:
	- runtime validation on real `3.0.1` assets is still outstanding
	- active MPQ build selection still relies on path/build inference unless a more explicit selector is ported later

### 3.x Alpha Follow-up (Mar 18)

- The LK offset-0 fallback experiment in `StandardTerrainAdapter.ExtractAlphaMaps(...)` was reverted after runtime validation showed it was wrong for the active 3.x terrain path.
- Current conclusion:
	- the recent attempt to treat `AlphaMapOffset == 0` as a valid relaxed-LK fallback case was not the correct fix
	- keep that path reverted and continue investigating 3.x alpha sourcing/decode without broadening fallback heuristics blindly
- Alternate-output build validation passed after reverting the tweak because a live `MdxViewer` process still had the normal `bin/Debug` outputs locked.

### 3.x Profile-Driven Alpha Recovery (Mar 18)

- Follow-up investigation confirmed the active recovery branch was still missing two important 3.x inputs that existed in rollback code:
	- WDT/MPHD big-alpha detection should treat `0x4 | 0x80` as the effective big-alpha mask for 3.x profiles
	- 3.x layer/alpha/shadow sourcing may need to come from split `*_tex0.adt` MCNK data rather than the root ADT alone
- Recovery changes now applied:
	- `AdtProfile` carries `BigAlphaFlagsMask` and `PreferTex0ForTextureData`
	- 3.0.1 / 3.3.5 profiles use `0x4 | 0x80` and prefer `*_tex0.adt`
	- `StandardTerrainAdapter` can build a `*_tex0.adt` MCNK index map and source MTEX/layers/MCAL/MCSH from that file when the profile says to
	- `StandardTerrainAdapter` now passes the MCNK `0x8000` do-not-fix-alpha bit into MCAL decode and uses chunk-level big-alpha inference instead of the reverted offset-0 fallback
	- `WoWMapConverter.Core.Formats.LichKing.Mcal` now has the stronger compressed / big-alpha / 4-bit decode split with proper edge-fix suppression for big-alpha and do-not-fix chunks
- Build gates passed after this batch:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build "I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="I:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime signoff is still pending:
	- confirm real 3.x tiles stop falling back to obvious 4-bit Alpha-style layer-1-only behavior
	- confirm split `*_tex0.adt` sourcing is actually the missing piece on the user’s 3.x client data

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

### Mar 19, 2026 - PM4 Coordinate Validation Slice

- Active core PM4 support now has one explicit coordinate-validation path built around `MPRL` refs already stored in ADT placement order.
- New active-core pieces:
	- `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateService.cs` defines the authoritative PM4 placement helpers for this first validation pass.
	- `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateValidator.cs` validates transformed `MPRL` refs against real `_obj0.adt` placements from the fixed development dataset.
	- `WoWMapConverter.Cli` now exposes `pm4-validate-coords`.
- Real-data validation status for this slice:
	- `wowmapconverter pm4-validate-coords --tile-limit 100` validated 100 PM4 tiles with placements from the fixed development dataset
	- 38,133 `MPRL` refs landed in expected tile bounds (100.0%)
	- 36,070 refs landed within a 32-unit nearest-placement threshold (94.6%)
	- average nearest-placement distance was 10.86 units
- Scope boundary:
	- this validates the `MPRL` anchor path only
	- cross-tile CK24 aggregation is still pending
	- MSCN/world-space semantics are still not the validated contract for active core code
- Do not claim PM4 world placement is fully solved beyond this `MPRL` path until CK24 aggregation and MSCN semantics are also validated.

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
