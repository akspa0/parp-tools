# Progress

## Mar 17, 2026 - Recovery Branch Checkpoint (v0.4.0 base)

- Active branch reset in main tree: recovery/v0.4.0-surgical-main-tree (base 343dadf).
- Restored .github customization stack from main and committed as 845748b.
- Build from this branch passes in primary tree environment.
- Terrain alpha decode profile routing is now staged in code:
	- TerrainAlphaDecodeMode in AdtProfile
	- LichKingStrict for 3.x profiles
	- LegacySequential for 0.x profiles
	- StandardTerrainAdapter alpha extraction routes by profile mode

### Critical Pending Validation

- Runtime terrain checks still required on both families:
	- Alpha-era terrain
	- LK 3.3.5 terrain
- Do not mark terrain safety complete until these real-data checks are done.

### Immediate Next Work

1. Finalize commit state for the profile/decode changes (if still local).
2. Run manual runtime spot-checks for alpha decode output.
3. Resume surgical commit intake from v0.4.0..main in SAFE-first order.

### Mar 17, 2026 - Intake Triage Update

- Reviewed queued commits `177f961`, `d50cfe7`, `326e6f8`, `4e2f681`, `37f669c`, `39799bf`, and `62ecf64` against the recovery branch and terrain-alpha guardrails.
- Marked `177f961` and `37f669c` as RISKY and out of scope for safe-first intake.
- Marked `d50cfe7`, `326e6f8`, `4e2f681`, `39799bf`, and `62ecf64` as MIXED; only isolated helper/tooling slices are candidates.
- Selected first SAFE extraction: corrected `TerrainImageIo` alpha-atlas helper from `62ecf64` only.
- Explicitly rejected the earlier `d50cfe7` `TerrainImageIo` version because it hardcoded atlas edge remapping that the recovery notes already identified as changing shipped data.
- No claim of terrain safety from this triage alone; runtime real-data validation is still required.

### Mar 17, 2026 - First SAFE Batch Applied

- Added `src/MdxViewer/Export/TerrainImageIo.cs` from the corrected `62ecf64` implementation only.
- Kept ViewerApp, TerrainRenderer, WorldScene, test-project, and terrain decode heuristic changes out of this batch.
- Build gate passed: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime terrain validation remains pending; build-only status is not sufficient for terrain signoff.

### Mar 18, 2026 - Rendering Recovery Batch

- Applied the `WorldAssetManager` renderer-residency fix from main so placed MDX/WMO renderers are no longer evicted out from under live world instances.
- `GetMdx` / `GetWmo` now lazy-load missing models and cached failed loads can be retried.
- Added the minimal skybox backdrop path from main:
	- route skybox-like MDX/M2 placements into a dedicated list
	- render the nearest skybox as a camera-anchored backdrop before terrain
	- added `ModelRenderer.RenderBackdrop(...)` with forced no-depth state for all layers
- Verified that the recovery branch already contained the reflective M2 depth-flag fix and env-map backface guard, so those regressions were not reintroduced here.
- Build gate passed again: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime validation still required; build success does not prove:
	- doodad/WMO reload correctness after moving away and back
	- correct skybox classification on real map data
	- MH2O liquid correctness on LK 3.3.5 tiles

### Mar 18, 2026 - MCCV + MPQ Recovery Batch

- Restored MCCV terrain color support on the active chunk-based terrain path.
- `TerrainChunkData` now carries MCCV bytes, `StandardTerrainAdapter` populates them, `TerrainMeshBuilder` uploads them, and `TerrainRenderer` applies them in shader.
- Fixed the known MCCV black-tint regression by treating MCCV alpha as tint strength so no-tint/transparent areas remain neutral instead of darkening to black.
- Applied the isolated `NativeMpqService` recovery slice from the mixed MPQ commits:
	- expanded patch archive ordering for locale/custom patch names
	- full normalized path encrypted-key derivation with basename fallback
	- compression bitmask handling for MPQ sectors
	- BZip2 support via SharpZipLib
- Build gates passed:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation still required; build success does not prove:
	- patched 1.x+ MPQ read correctness on real patch chains
	- encrypted later-version MPQ entry reads on real data
	- MCCV highlight/tint correctness on real 3.x terrain

## ✅ Working

### MdxViewer (3D World Viewer) — Primary Project
- **Alpha 0.5.3 WDT terrain**: ✅ Monolithic format, 256 MCNK per tile, async streaming
- **0.6.0 split ADT terrain**: ✅ StandardTerrainAdapter, MCNK with header offsets (Feb 11)
- **0.6.0 WMO-only maps**: ✅ MWMO+MODF parsed from WDT (Feb 11)
- **Terrain liquid (MCLQ)**: ✅ Per-vertex sloped heights, absolute world Z, waterfall support (Feb 11)
- **WMO v14 rendering**: ✅ 4-pass: opaque → doodads → liquids → transparent
- **WMO liquid (MLIQ)**: ✅ matId-based type detection, correct positioning (Feb 11)
- **WMO doodad culling**: ✅ Distance (500u) + cap (64) + nearest-first sort + fog passthrough
- **WMO doodad loading**: ✅ FindInFileSet case-insensitive + mdx/mdl swap → 100% load rate
- **MDX rendering**: ✅ Two-pass opaque/transparent, alpha cutout, specular highlights, sphere env map
- **MDX GEOS version compatibility**: ✅ Ported version-routed GEOS parser behavior from `wow-mdx-viewer` (v1300/v1400 strict path + v1500 strict path + guarded fallback)
- **MDX SEQS name compatibility**: ✅ Counted 0x8C named-record detection broadened to reduce fallback `Seq_{animId}` names on playable models
- **MDX PRE2/RIBB parsing parity**: ✅ Expanded parser coverage for PRE2 and RIBB payload/tail animation chunks (runtime visual verification pending)
- **MDX animation engine**: ✅ BONE/PIVT/HELP parsing, keyframe interpolation, bone hierarchy (Feb 12)
- **Full-load mode**: ✅ `--full-load` (default) loads all tiles at startup with progress (Feb 11)
- **MCSH shadow maps**: ✅ 64×64 bitmask applied to all terrain layers
- **AOI streaming**: ✅ 9×9 tiles, directional lookahead, persistent tile cache, MPQ throttling (Feb 11)
- **Frustum culling**: ✅ View-frustum + distance + fade
- **AreaID lookup**: ✅ Low 16-bit extraction + low byte fallback for MapID mismatch
- **DBC Lighting**: ✅ LightService loads Light.dbc + LightData.dbc, zone-based ambient/fog/sky colors
- **Replaceable Textures**: ✅ DBC CDI variant validation against MPQ + model dir scan fallback
- **Minimap overlay**: ✅ From minimap tile images

### Model Parsers & Tools
- **MDX-L_Tool**: ✅ Core parsing and Archaeology logic complete.
- **GEOS Chunk (Alpha)**: ✅ Robust scanner for Version 1300 validated.
- **Texture Export**: ✅ DBC-driven `ReplaceableId` resolution working.
- **OBJ Splitter**: ✅ Geoset-keyed export verified on complex creatures.
- **0.5.3 Alpha WDT/ADT**: ✅ Monolithic format, sequential MCNK.
- **WMO v14/v17 converter**: ✅ Both directions implemented.
- **BLP**: ✅ BlpResizer complete.

### Data Generation
- **VLM Datasets (Alpha)**: ✅ Azeroth v10 (685 tiles).

## ⚠️ Partial / In Progress

### MdxViewer — Rendering Quality & Performance
- **3.3.5 ADT loading freeze**: Needs investigation
- **WMO culling too aggressive**: Objects outside WMO not visible from inside
- **MDX GPU skinning**: Bone matrices computed per-frame but not yet applied in vertex shader (needs BIDX/BWGT vertex attributes)
- **MDX animation UI**: Sequence selection combo box in ImGui panel not yet wired
- **MDX per-geoset color/alpha**: Only static alpha used; animated GeosetAnims not wired
- **MDX particles/ribbons**: Parser coverage expanded; runtime behavior verification still pending on effect-heavy assets
- **MDX texture UV animation**: Not implemented
- **MDX billboard bones**: Not implemented
- **WMO lighting**: v14-16 grayscale lightmap + v17 MOCV vertex colors not implemented
- **Vulkan RenderManager**: Research phase — `IRenderBackend` abstraction for Silk.NET Vulkan

### Build & Release Infrastructure
- **GitHub Actions**: ✅ `.github/workflows/release-mdxviewer.yml` — tag push or manual dispatch
- **WoWDBDefs bundling**: ✅ 1315 `.dbd` files copied to output via csproj Content items
- **Self-contained publish**: ✅ `dotnet publish -c Release -r win-x64 --self-contained` verified

### MDX-L_Tool Enhancements
- **M2 Export (v264)**: 🔧 Implementing binary writer.

## ❌ Known Issues

### MdxViewer Rendering Bugs (Feb 12, 2026)

#### MDX Sphere Env / Specular Orientation (Feb 14, 2026)
- **Symptom**: Reflective/specular surfaces (e.g., dome-like geometry) appeared inward-facing on some two-sided materials.
- **Fix Applied**: Fragment shader now flips normals/view-space normals on backfaces before env UV generation and lighting/specular.
- **Status**: 🔧 Patched in code, pending visual confirmation on Dalaran dome repro.

#### WMO Semi-Transparent Window Materials
- **Symptom**: Stormwind WMO maps blue/gold stained glass textures to white marble columns instead of window frames
- **Hypothesis 1**: Secondary MOTV chunk not skipped → MOBA batch parsing misalignment
- **Fix Attempt 1**: Added `reader.BaseStream.Position += chunkSize;` when secondary MOTV encountered in `WmoV14ToV17Converter.ParseMogp` (line 922)
- **Result**: ❌ FAILED — window materials still map to wrong geometry
- **Status**: Root cause still unknown. May not be MOTV-related. Need to check console logs to verify if secondary MOTV is even present in Stormwind groups.

#### MDX Cylindrical Texture Stretching
- **Symptom**: Barrels, tree trunks show single wood plank stretched around entire circumference instead of tiled texture
- **Hypothesis 1**: Texture wrap mode incorrectly clamping both S and T axes when only one should clamp
- **Fix Attempt 1**: Changed `ModelRenderer.LoadTextures` to use per-axis clamp flags (clampS/clampT) based on `tex.Flags & 0x1` and `tex.Flags & 0x2` (lines 778-779)
- **Result**: ❌ FAILED — textures still stretched on cylindrical objects
- **Status**: Root cause still unknown. May not be wrap mode related. Need to check console logs to verify texture flags and investigate UV coordinates.

### AdtModfInjector
- **Problem**: Appends MWMO/MODF chunks to end of file; result is Noggit-incompatible.

## Key Technical Insights

### MCLQ Liquid Heights (Feb 11, 2026)
- MCLQ per-vertex heights (81 entries × 8 bytes) are absolute world Z values
- Heights can slope for waterfalls — adjacent water planes at different Z levels
- MH2O (3.3.5) was overwriting valid MCLQ data with garbage on 0.6.0 ADTs
- Fix: Skip MH2O when MCLQ liquid already found; never overwrite existing MCLQ
- WMO MLIQ liquid type: use `matId & 0x03` from MLIQ header, NOT tile flag bits

### Performance Tuning (Feb 11, 2026)
- AOI: 9×9 tiles (radius 4), forward lookahead 3, GPU uploads 8/frame
- MPQ read throttling: `SemaphoreSlim(4)` prevents I/O saturation
- Persistent tile cache: `TileLoadResult` stays in memory, re-entry is instant
- Dedup sets removed: objects always reload correctly after tile unload/reload

### WMO/MDX Coordinate System (Feb 9, 2026)
- WoW: right-handed (X=North, Y=West, Z=Up), Direct3D CW winding
- OpenGL: CCW winding for front faces
- **Fix**: Reverse winding at GPU upload + 180° Z rotation in placement
- MDX rotations: `rx = Rotation.X`, `ry = Rotation.Y` — NO axis swap
- WMO-only maps: raw WoW world coords (no MapOrigin conversion)

### WMO MLIQ Liquid Positioning (Feb 9, 2026)
- MLIQ data has inherent 90° CW misrotation (wowdev wiki)
- Fix: `axis0 = cornerX - j * tileSize`, `axis1 = cornerY + i * tileSize`
- Tile visibility: bit 3 (0x08) = hidden
- GroupLiquid=15 always → magma (old WMO "green lava" type)

### Replaceable Texture Resolution (Feb 10, 2026)
- Try ALL CDI variants, validate each resolved texture exists in MPQ
- If no DBC variant validates, fall through to model directory scan
