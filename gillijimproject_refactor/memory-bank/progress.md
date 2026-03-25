# Progress

### Mar 25, 2026 - Enhanced Terrain Shader / Lighting Planning Prompt Captured

- Added planning prompt file:
	- `plans/enhanced_terrain_shader_lighting_prompt_2026-03-25.md`
- Purpose:
	- capture the current direction for an enhanced-quality terrain renderer path, shader-family reconstruction strategy, and lighting-model expansion without collapsing the active historical renderer into a speculative rewrite.
- Prompt requirements captured:
	- explicit `Historical` vs `Enhanced` render-mode architecture
	- terrain-only first vertical slice
	- render-quality UI/settings expansion
	- `LightService` expansion as a separate required seam from shader work
	- shader-family translation strategy for terrain, WMO/map-object, Model2, liquid, and particles
	- terrain decode/shading guardrails and real-data validation requirements
- Validation limits:
	- documentation/planning only
	- no code changes to the active renderer from this prompt file itself
	- no automated tests or builds were run for this planning-only slice

### Mar 25, 2026 - Enhanced Renderer Prompt Set Added

- Added focused companion planning prompts:
	- `plans/enhanced_renderer_plan_set_2026-03-25.md`
	- `plans/enhanced_renderer_architecture_prompt_2026-03-25.md`
	- `plans/enhanced_terrain_first_slice_prompt_2026-03-25.md`
	- `plans/shader_family_and_lighting_roadmap_prompt_2026-03-25.md`
- Purpose:
	- give Copilot narrower planning entry points instead of forcing every session through one umbrella renderer prompt.
- Split of responsibilities:
	- plan-set index selects the right prompt
	- architecture prompt covers runtime boundaries and mode ownership
	- first-slice prompt covers the first landable enhanced terrain implementation slice
	- roadmap prompt covers post-slice lighting and shader-family rollout
- Validation limits:
	- planning/documentation only
	- no renderer behavior changed by this prompt set

### Mar 24, 2026 - Fullscreen Minimap Tile-Scale Regression Fix

- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- floating and fullscreen minimap camera center math now uses `WoWConstants.TileSize` instead of `ChunkSize`.
	- minimap teleport now converts clicked tile coordinates back to world space with `TileSize`, so the camera lands inside the intended 64x64 map tile grid.
- `src/MdxViewer/MinimapHelpers.cs`
	- POI markers, taxi route polylines, and taxi node markers now project world positions onto the minimap in tile space instead of chunk space.
	- shared minimap click-to-world conversion now also uses `TileSize`.
- `src/MdxViewer/ViewerApp.cs`
	- the legacy `DrawMinimap_OLD()` path was updated to the same tile-scale camera math so the regression does not survive in fallback code.
- Root cause:
	- the minimap renders a `64x64` tile grid, but the camera/overlay math had drifted onto `ChunkSize`, inflating positions by `16x` and pushing the fullscreen camera marker outside the valid map area.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-fix/"` passed on Mar 24, 2026.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on fullscreen minimap behavior, marker placement, or minimap teleport feel.

### Mar 25, 2026 - Fullscreen Minimap Still Treated As Open Release Blocker

- Runtime user feedback after the earlier tile-scale patch says the fullscreen minimap is still broken.
- Current interpretation:
	- the Mar 24 tile-scale correction should now be treated as a partial fix attempt or narrowed hypothesis, not as a closed bug.
	- the fullscreen minimap remains an open `v0.4.5` release blocker until runtime validation proves otherwise.
- Current likely investigation seams for the follow-up:
	- camera world-to-tile axis mapping
	- tile row/column ordering across `ViewerApp_MinimapAndStatus`, `MinimapHelpers`, and `MinimapRenderer`
	- fullscreen interaction parity with the docked minimap path
	- click/teleport mapping versus displayed tile imagery
- Planning follow-up captured in:
	- `plans/mdxviewer_release_plan_set_v0_4_5_v0_5_0_2026-03-25.md`
	- `plans/v0_4_5_release_stabilization_prompt_2026-03-25.md`
	- `plans/fullscreen_minimap_repair_prompt_2026-03-25.md`
	- `plans/v0_5_0_goal_stack_prompt_2026-03-25.md`
- Validation limits:
	- runtime report only for the minimap still-broken state
	- no new code changes, tests, or builds were performed in this planning slice

### Mar 25, 2026 - Taxi Route Actor Prototype + Node Inspector Controls

- `src/MdxViewer/Terrain/TaxiPathLoader.cs`
	- taxi node loading now resolves mount metadata through the historical DBC chain:
		- `TaxiNodes.MountCreatureID[2]`
		- `Creature.DisplayID[4]`
		- `CreatureDisplayInfo.ModelID` + `CreatureModelScale`
		- `CreatureModelData.ModelName`
	- `TaxiNode` now exposes resolved mount creature IDs, display ID, scale, and model path for viewer-side taxi actor rendering.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- added animated taxi actor runtime support for selected taxi nodes/routes.
	- taxi actors now sample route waypoints, advance over time, and render through the existing MDX world-render path.
	- added viewer controls/state for `ShowTaxiActors` and `TaxiActorSpeedMultiplier`.
- `src/MdxViewer/ViewerApp.cs`
	- taxi list selection now routes through shared selection helpers instead of setting raw IDs directly.
	- viewport clicking can now pick visible taxi node indicators and sync them into the selected-object inspector state.
	- taxi selection now clears conflicting world/PM4 selections and populates selected-object info for node/route inspection.
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- the inspector now shows taxi-route controls when a taxi node or route is selected.
	- added the requested `Taxi Speed` slider plus a `Show Animated Taxi Actor` toggle.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-taxi/"` passed on Mar 25, 2026.
	- a normal build to the default output path was blocked by the running `ParpToolsWoWViewer` process holding file locks.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on taxi mount resolution, taxi node viewport picking, or in-scene route animation.

### Mar 24, 2026 - WMO Vertex-Light Prototype

- `src/MdxViewer/Rendering/WmoRenderer.cs`
	- WMO vertex buffers now include a baked vertex-light attribute alongside position, normal, and diffuse UV.
	- the renderer now consumes parsed `MOCV` colors when present and usable.
	- if parsed `MOCV` is missing but raw v14 lightmap payloads are present, it now samples preserved `MOLV` / `MOLD` / `MOLM` data into per-vertex baked-light colors during buffer build.
	- the WMO shader now multiplies the existing textured/diffuse lighting path by that baked-light color, which gives the active viewer a first object-light prototype without inventing a fake per-batch lightmap texture system.
- Scope limits:
	- not full client-faithful object-lightmap parity yet.
	- no dedicated batch-local lightmap texture binding path yet.
	- no runtime real-data signoff yet.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed.
	- no automated tests were added or run.
	- no real-data runtime validation was performed in this slice.

### Mar 24, 2026 - 0.5.3 Terrain/Object Render Fast-Path And Viewer Perf Gap

- Reverse-engineering only against the symbolized `0.5.3` client plus viewer code audit; no repo code changes landed in this slice.
- durable report extended at `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- confirmed `0.5.3` terrain-side render behavior relevant to performance/parity:
	- `CreateRenderLists` (`0x00698230`) is a real precompute/batch-build step for terrain texcoords/render lists
	- `RenderLayers` (`0x006a5d00`) and `RenderLayersDyn` (`0x006a64b0`) use locked GX buffers plus prepared chunk batches instead of a generic frame-time rebuild path
	- `0.5.3` terrain already has shader-assisted paths via `CMap::psTerrain` / `CMap::psSpecTerrain` plus `shaderGxTexture`
	- terrain draw cost is reduced by distance through runtime layer-count collapse (`textureLodDist`)
	- moving terrain layer behavior is now directly supported in the terrain path: runtime layer flag `0x40` triggers an extra texture transform indexed into the time-varying world transform tables
	- terrain shadows are drawn as a separate modulation pass
- confirmed `0.5.3` object/light behavior relevant to parity:
	- `RenderMapObjDefGroups` (`0x0066e030`) walks visible `CMapObjDefGroup` lists and dispatches group renders rather than using one generic world-object loop
	- `CreateLightmaps` (`0x006adba0`) allocates per-group lightmap textures and registers update callbacks
	- `RenderGroupLightmap(...)` and `RenderGroupLightmapTex(...)` tighten the lightmap conclusion further: the client has a dedicated group-lightmap render path with its own vertex stream and combine pass structure
	- `UpdateLightmapTex(...)` exposes CPU lightmap memory plus stride on `GxTex_Latch`, which supports a longer-lived lightmap texture path rather than ad hoc per-draw shading
	- `CalcLightColors` (`0x006c4da0`) computes substantially richer lighting state than the active viewer currently models (direct, ambient, multiple sky/cloud/water channels, fog, storm blending)
- viewer-side gap captured from the same slice:
	- `StandardTerrainAdapter` still actively uses `MPHD` only for big-alpha/profile handling and still flattens `MAIN` entries to tile presence
	- `TerrainRenderer` is still a generic base+overlay loop with only `MCLY 0x100` interpretation
	- `LightService` remains a simplified DBC interpolator
	- `WmoRenderer` / `MdxRenderer` still flatten renderer specialization heavily
	- `WorldScene` hot-path render work plus uncapped PM4 forensic budgets remain a practical perf risk when enabled, and the existing `RenderQueue` abstraction is not yet the active submission path for world rendering
- validation limits:
	- no automated tests were added or run
	- no viewer build or runtime real-data signoff was performed in this RE-only slice

### Mar 24, 2026 - WoW 2.0.0 Beta Ghidra Recon For M2 / Light / Particle Risk

- Reverse-engineering only against a loaded beta `2.0.0` client binary in Ghidra; no repo code changes landed in this slice.
- durable report added at `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- Confirmed engine-side anchors relevant to safe `2.x` support planning:
	- `FUN_00717b00` loads `shaders\vertex\Model2.bls` and `shaders\pixel\Model2.bls` for `Model2`.
	- `FUN_006b3b20` preloads map-object pixel BLS variants including translucent diffuse/specular programs.
	- terrain follow-up clarified:
		- the terrain shader split is now tighter than earlier notes:
			- `FUN_006a2360` loads `terrain1..4` into `DAT_00caf304..310` and `terrain1_s..4_s` into `DAT_00caf548..554`
			- `FUN_006cee30` uses those two contiguous tables as one-pass cached programs indexed by chunk layer count
			- `terrainp` / `terrainp_s` are the slower manual terrain fallback path inside `FUN_006cee30`
			- `terrainp_u` / `terrainp_us` are currently only confirmed in startup/shutdown, not yet in an active draw path
		- `XTextures\slime\slime.%d.blp` now traces into an animated `WCHUNKLIQUID` texture-family path through `FUN_0069b310` and its caller cluster
		- `WCHUNKLIQUID` rendering is not one single effect path: `FUN_006c65b0` dispatches modes `0/4/8` to animated texture-family renderers and modes `2/3/6/7` to a direct-coordinate path; `FUN_0069e200` builds cell strips for mode values `1/4/6`
		- `FUN_006c65b0` passes the raw mode nibble into `FUN_0069b310`, so liquid mode doubles as animated family index
		- currently recovered family table entries are `0=lake_a`, `1=ocean_h`, `2=lava`, `3=slime`, `4=lake_a` again; higher traced slots remain unresolved in this pass
		- novelty/dead-code candidates now include unresolved family slot `6`, unused `XTextures\river\fast_a.%d.blp`, and terrain-side `terrainp_u` / `terrainp_us` that still only show up in startup/shutdown
	- `FUN_0072d1a0` plus `FUN_0072cc60` / `FUN_0072cc90` / `FUN_0072cdc0` show `M2Light` objects being spatially bucketed / relinked at runtime instead of handled as a static flat light list.
	- `FUN_007c26c0`, `FUN_007ca9d0`, `FUN_007c3180`, and `FUN_007c79d0` show `ParticleSystem2` bootstrapping and runtime `CParticle2` / `CParticle2_Model` object storage, which keeps the smoke issue open on the renderer/runtime side.
	- `LightFloatBand.dbc`, `LightIntBand.dbc`, `LightParams.dbc`, `Light.dbc`, and `LightSkybox.dbc` all use strict `WDBC` loaders with schema checks and ID->row pointer tables.
- Current conclusion:
	- later `2.x` profile routing is a reasonable structural start, but real parity risk sits in shader/material selection and light/particle runtime interpretation, not in raw table loading.
- Validation limits:
	- no automated tests were added or run.
	- no viewer build or runtime signoff was performed as part of this RE-only slice.

### Mar 24, 2026 - Later 2.x M2-Family Profile Routing Enablement

- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
	- added `M2Profile_20x_Unknown` for later `2.x` / TBC-era model routing.
	- active `2.x` window is now `MD20` with versions `0x104..0x107` and the existing parser split threshold remains `0x108`.
- `src/MdxViewer/ViewerApp.cs`
	- fallback build options now include `2.4.3.8606`, so the viewer can select a later `2.x` model profile even without `Map.dbd` build metadata.
- `src/MdxViewer/Rendering/ReplaceableTextureResolver.cs`
	- added short-build alias support for `2.4.3 -> 2.4.3.8606`.
- `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
	- neutralized the profiled legacy `MD20` trace wording so TBC routing no longer logs as if it were only the pre-release `3.0.1` path.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on an actual later `2.x` client dataset.

### Mar 24, 2026 - 0.12 Standalone Model Browser Recovery

- `src/MdxViewer/DataSources/MpqDataSource.cs`
	- loose-file indexing now includes `.mdl`
	- Alpha nested wrapper scan now includes `.mdx.MPQ`, `.mdl.MPQ`, and `.m2.MPQ`
	- nested model wrappers now register alternate model-extension aliases into the file set / Alpha wrapper cache so the same wrapped asset can resolve through `.mdx`, `.mdl`, or `.m2`
- `src/MdxViewer/ViewerApp.cs`
	- the browser-side `.mdx` file bucket now includes early `.mdl` assets as part of the same standalone model family
	- standalone disk loads now accept `.mdl`
	- unsupported standalone M2-family loads now fail early with a clear error when the active build has no resolved `M2Profile`
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- the file-browser type selector now labels the early-model bucket as `.mdx/.mdl`
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed with existing warnings
	- no automated tests were added or run
	- no runtime real-data signoff yet on a real `0.12` client dataset

### Mar 24, 2026 - 0.6.0 Through 2.x Terrain Alpha Grid Regression Fix

- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
	- legacy standard-ADT terrain alpha decode no longer uses the naive sequential nibble expansion for the entire `0.6.0` through `2.x` band.
	- `TerrainAlphaDecodeMode.LegacySequential` now prefers relaxed MCAL per-layer decode with inferred layer spans and preserved `DoNotFixAlphaMap` handling.
	- fallback legacy 4-bit decode now goes through the existing row-aware unpack + legacy edge-fix helpers, which is the actual seam tied to the chunk-grid artifact.
- Build follow-up required by the same slice:
	- the earlier in-progress minimap candidate-path patch still had compile errors in `src/MdxViewer/Rendering/MinimapRenderer.cs`; those were corrected so the terrain change could be validated with a real solution build.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed.
	- no automated tests were added or run.
	- no runtime real-data validation yet on the affected legacy terrain clients.

### Mar 24, 2026 - v0.4.5 Branding + MH2O LiquidType Classification Fix

- `src/MdxViewer/ViewerApp.cs`
	- viewer window titles now use `parp-tools WoW Viewer`
	- Help -> About now opens a modal with version, author, and credits instead of only setting a status line
	- standard terrain world loads now pass the active DBC provider/build metadata into `StandardTerrainAdapter`
- `src/MdxViewer/MdxViewer.csproj` and `src/MdxViewer/MdxViewer.CrossPlatform.csproj`
	- version metadata now targets `0.4.5`
	- the emitted assembly/executable name is now `ParpToolsWoWViewer`
- `.github/workflows/release-mdxviewer.yml`
	- release workflow is now branded for `parp-tools WoW Viewer`
	- workflow dispatch example now points at `v0.4.5`
	- build environment now uses .NET 10 and publishes `parp-tools-wow-viewer-<version>-win-x64.zip`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
	- `MH2O` liquid family selection now prefers `LiquidType.dbc.Type` through DBCD instead of treating `LiquidTypeId` as a direct render class
	- fallback behavior now still handles later 3.3.5 / 4.0 liquid IDs when DBC metadata is unavailable
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/Liquids/LiquidConverter.cs`
	- shared fallback `LiquidTypeId -> MCLQ family` mapping now recognizes `13`, `14`, `17`, `19`, and `20`
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed
	- no automated tests were added or run
	- no runtime real-data signoff yet on the corrected 3.3.5 / 4.0 liquid-family rendering

### Mar 24, 2026 - PM4 Color Legend + Selected-Object Graph UI

- `src/MdxViewer/Terrain/WorldScene.cs`
	- now exposes viewer-derived PM4 legend data for the active color mode instead of forcing users to reverse-map swatches by eye
	- now exposes a selected-object PM4 hierarchy summary built from the current overlay assembly using:
		- CK24 root group
		- MSLK-linked subgroup
		- optional MDOS subgroup
		- connectivity/object-part leaf nodes
- `src/MdxViewer/ViewerApp.cs`
	- world-objects PM4 controls now show a `PM4 Color Legend` block under the color-mode selector so `MSUR Attr Mask` and other categorical modes are directly identifiable by value/count
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- selected PM4 objects now show a `PM4 Graph` tree in the inspector so users can inspect link-group / MDOS / part structure for the clicked object
- Mar 24 follow-up on the same PM4 UI slice:
	- PM4 graph leaf rows are now actionable: clicking a part reselects that exact PM4 part and `Frame` moves the camera to it
	- the selected PM4 graph can now be exported as JSON for later PM4 research/planning work
- Planning follow-up:
	- `plans/unified_format_io_overhaul_prompt_2026-03-23.md` now records the current pragmatic PM4 hierarchy contract and the rule that centroids are derived anchors, not raw PM4 graph nodes
- Validation limits:
	- no automated tests were added or run
	- runtime real-data signoff is still required before claiming the new graph view fully matches raw PM4 ownership semantics

### Mar 23, 2026 - Viewer Tool Dialogs Now Reuse Active Client / Loose Overlay Paths

- The viewer already retained enough session state to stop forcing repeated path browsing across several tools:
	- active MPQ base client path via `MpqDataSource.GamePath`
	- attached loose overlay roots via `MpqDataSource.OverlayRoots`
	- current loaded map name via `TerrainManager.MapName` / `VlmTerrainManager.MapName`
- `src/MdxViewer/ViewerApp.cs`
	- tool menu actions now prepare dialog inputs before opening:
		- `Generate VLM Dataset`
		- `Terrain Texture Transfer`
		- `Map Converter`
		- `WMO Converter`
	- added helper methods that resolve the current session’s base client, loose overlay, map directory, and WDT path from the already loaded viewer state instead of making the user browse for them again.
- Important current behavior:
	- VLM export prefers the active MPQ base path and current map name.
	- terrain transfer prefers loose overlay map dir as source and base-client map dir as target when both exist.
	- map converter seeds from the current map WDT/map dir when a usable on-disk path exists.
	- standalone WMO conversion still auto-seeds from the currently loaded WMO file.
- Validation limits:
	- file diagnostics on `src/MdxViewer/ViewerApp.cs` were clean after the change.
	- no automated tests were added or run.
	- no new full viewer build or runtime real-data signoff was recorded for this exact slice.

### Mar 23, 2026 - Unified Format I/O Overhaul Proposal Captured

- The user wants the current scattered terrain/model/WMO knowledge moved into one shared read/write library used by all tooling.
- Explicit proposal direction now captured for follow-up planning:
	- one shared format I/O library for Alpha, LK 3.3.5, and relevant 4.x read/write paths
	- terrain + placement + model + WMO conversion under one orchestration surface
	- retire the split where `MdxViewer` has newer runtime-read knowledge while `WoWMapConverter.Core` still carries older write/conversion assumptions
	- do not over-claim Alpha placement downconversion until MODF/MDDF write support is actually implemented and validated
- Planning prompt added at `plans/unified_format_io_overhaul_prompt_2026-03-23.md`.

### Mar 23, 2026 - Viewer Docs Refresh + Render Quality Follow-Up

- Initial doc refresh was followed by a user rewrite of `src/MdxViewer/README.md` to remove bad assumptions and make the support/workflow description more grounded.
- Preserve that correction for future handoff:
	- do not overstate platform restrictions
	- do not overstate supported versions beyond the user-corrected README
	- do not write branch-local language into docs intended for eventual `main`
	- keep the render-quality statement narrow: texture filtering is the landed win; MSAA availability is context-dependent and not required for this slice
- Validation limits:
	- the active viewer solution had already built successfully on Mar 23 before the doc refresh
	- no automated tests were added or run for the documentation update

### Mar 22, 2026 - Viewer Debug Workflow Follow-Up: PM4 OBJ Export + Minimap Guardrails + Terrain Hole Override

- Added a viewer-side offline PM4 OBJ export path so PM4 inspection no longer depends only on the live overlay's currently loaded subset.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- now exports per-tile OBJ, per-object OBJ, and `pm4_obj_manifest.json` from direct PM4 file scans against the active data source
- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- now exposes `Export PM4 OBJ Set` in the PM4 utilities UI
- `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`, `src/MdxViewer/Rendering/MinimapRenderer.cs`
	- minimap teleport now requires triple-clicking the same tile within the confirmation window
	- minimap drag-vs-click discrimination now uses full drag-origin distance
	- minimap zoom/pan/window visibility now persist in viewer settings
	- decoded minimap tiles now cache on disk under `output/cache/minimap/<cache-segment>`
- `src/MdxViewer/Terrain/TerrainMeshBuilder.cs`, `src/MdxViewer/Terrain/TerrainManager.cs`, `src/MdxViewer/Terrain/VlmTerrainManager.cs`, `src/MdxViewer/ViewerApp_Sidebars.cs`
	- added viewer-side terrain hole override controls
	- loaded terrain tiles can now be rebuilt with `HoleMask` ignored either globally or for the current camera tile
	- this does not edit ADT data on disk; it is a mesh rebuild / debug visibility feature only
- Validation limits:
	- file diagnostics were clean on the edited viewer files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- no automated tests were added or run
	- no runtime real-data signoff yet on PM4 OBJ parity, minimap UX/cache behavior, or terrain-hole rebuild behavior

### Mar 21, 2026 - Standalone PM4 Research Library

- Added `src/Pm4Research.Core` as a fresh-start PM4 reading library independent from the active viewer reconstruction path.
- Current implementation:
	- walks PM4 files chunk-by-chunk and preserves raw payloads with offsets/sizes
	- independently decodes the currently understood chunk layouts: `MVER`, `MSHD`, `MSLK`, `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR`, `MSCN`, `MPRL`, `MPRR`, `MDBH`, `MDBI`, `MDBF`, `MDOS`, `MDSF`
	- exposes `Pm4ResearchFile` and `Pm4ExplorationSnapshot` so future rediscovery work can compare raw chunk evidence without going through `WorldScene`
	- exposes decode-audit reports so future rediscovery work can measure chunk-size consistency and cross-chunk reference validity before inferring object semantics
- Why this was added:
	- current PM4 viewer work has hit repeated transform-contract ambiguity
	- the user requested a fresh perspective instead of continuing to layer fixes onto the existing PM4 read/reconstruct path
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Core/Pm4Research.Core.csproj -c Debug` passed.
	- no automated tests added or run.
	- no runtime real-data validation yet; this is groundwork for fresh PM4 format exploration.

### Mar 21, 2026 - PM4 Decode-Confidence Audit Pass

- Added raw decode-audit commands to `src/Pm4Research.Cli`:
	- `inspect-audit`
	- `scan-audit`
- The audit path measures:
	- chunk presence versus populated-data presence
	- stride/size consistency for the typed chunk layouts
	- cross-chunk reference validity for `MSVI -> MSVT`, `MSPI -> MSPV`, `MSUR -> MSVI`, `MSLK -> MSUR`, `MSLK -> MSPI`, `MDSF -> MSUR`, `MDSF -> MDOS`, and `MDOS -> MDBH`
- Real-data findings from the full development corpus (`616` PM4 files):
	- zero file-walk overrun or trailing-byte diagnostics
	- zero unknown chunk signatures after adding typed support for the documented destructible-building chunks
	- recurring decode structure is carried by the `MS*` / `MPR*` families, not by Wintergrasp destructible chunks
	- `MDBI` and `MDBF` are one-tile only in this corpus, and `MDBH` / `MDOS` / `MDSF` only carry populated destructible-building payload on the trusted `development_00_00.pm4` reference tile; their wider corpus presence is mostly placeholder/empty chunk stubs
	- meaningful open seam surfaced by the audit: aggregate `MSLK.RefIndex -> MSUR` mismatches still exist in the corpus, so the current standalone `MSLK.RefIndex == MSUR index` assumption is not fully closed
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- inspect-audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - Targeted `MSLK.RefIndex` Mismatch Audit

- Added dedicated commands to `src/Pm4Research.Cli`:
	- `inspect-mslk-refindex`
	- `scan-mslk-refindex`
- The new audit reports invalid `MSLK.RefIndex >= MSUR.Count` cases by tile and records which other PM4 index domains those values still fit on the same file.
- Real-data findings from the full development corpus:
	- `150` files contain `4553` total `MSLK.RefIndex -> MSUR` mismatches
	- the trusted `development_00_00.pm4` tile has zero such mismatches, so it is not the main linkage-problem reference tile
	- bad `RefIndex` values almost never fit `MPRL` counts, which weakens the idea that the unresolved `RefIndex` population is simply pointing into `MPRL`
	- many bad values still fit within `MSLK`, `MSPI`, `MSVI`, and `MSCN` counts on the affected files, making those domains stronger next-step candidates
	- some mismatch-heavy tiles show repeated `LinkId` clusters, which may help isolate families of alternate `RefIndex` semantics in a follow-up pass
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- inspect-mslk-refindex --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - Comprehensive PM4 Unknowns Map

- Added `scan-unknowns` to `src/Pm4Research.Cli` and `Pm4ResearchUnknownsAnalyzer` to `src/Pm4Research.Core`.
- Purpose:
	- stop spreading PM4 unknowns across isolated ad-hoc commands and stale notes
	- produce one corpus-scale report that states which raw PM4 relationships are verified, which are partial, and which fields remain open
- The report now covers:
	- chunk population vs populated payload counts
	- verified raw edges such as `MSUR -> MSVI`, `MSVI -> MSVT`, `MSLK -> MSPI`, `MSPI -> MSPV`, and `MDSF -> {MSUR, MDOS}`
	- partial/open edges such as `MSLK.RefIndex`, `MSLK.GroupObjectId -> MPRL.Unk04`, `MPRR.Value1`, and `MDOS.buildingIndex`
	- field distributions for `MSHD`, `MSLK`, `MSUR`, `MPRL`, and `MPRR`
	- `LinkId` pattern summary and `MSLK.MspiIndexCount` ambiguity buckets
	- a generated unknown list with evidence and next proof tasks
- Key real-data findings from the current corpus run:
	- `LinkId` is uniformly `0xFFFFYYXX` in the current dataset
	- `MPRL.Unk02` is always `-1` and `Unk06` is always `0x8000`
	- `MPRL.Unk14` ranges `-1..15` and still looks floor-like; `Unk16` collapses to two values (`0x0000`, `0x3FFF`)
	- `MSLK.MspiIndexCount` has no triangles-only evidence in the current corpus, but still has a large overlap bucket where both interpretations fit
	- `MPRR` remains a mixed/open field family; current counts do not justify naming it as purely `MPRL` or purely geometry-facing
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - MSCN Relationship Report

- Added `scan-mscn` to `src/Pm4Research.Cli` and `Pm4ResearchMscnAnalyzer` to `src/Pm4Research.Core`.
- Purpose:
	- stop treating MSCN as a vague side channel or re-importing old rollback claims without revalidating them in the standalone raw path
	- measure MSCN directly against CK24, `MSUR.MdosIndex`, mesh-side geometry, and `MSLK.GroupObjectId`
- The report now covers:
	- `MSUR.MdosIndex -> MSCN` validity across the full corpus
	- CK24-group MSCN coverage and mesh+MSCN coexistence
	- raw-vs-swapped MSCN bounds overlap against CK24 mesh bounds
	- low16 / low24 `MSLK.GroupObjectId` fits against CK24 identity layers
	- MSCN coordinate-space buckets against file tile coordinates
- Key real-data findings from the current corpus run:
	- `MSUR.MdosIndex -> MSCN` is strong but not closed (`511891` fits, `6201` misses)
	- `1886 / 1895` CK24 groups carry valid MSCN-backed node coverage
	- raw MSCN bounds overlap CK24 mesh bounds far more often than swapped-XY MSCN bounds (`1162` vs `10` fits)
	- current standalone corpus evidence does not support the older blanket claim that MSCN is simply world-space plus XY swap
	- `MSLK.GroupObjectId` is not a direct full CK24 key (`0 / 1272796` low24 fits) and only weakly overlaps CK24 low16 object ids (`399 / 1272397`)
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - PM4 Linkage / CK24 ObjectId Report

- Added `scan-linkage` to `src/Pm4Research.Cli` and `Pm4ResearchLinkageAnalyzer` to `src/Pm4Research.Core`.
- Purpose:
	- stop treating the UI `Ck24ObjectId` label as if it were a separately proven PM4 identity field
	- join `MSLK.RefIndex` mismatches, `MSLK.GroupObjectId`, CK24 identity layers, and bad `MSUR.MdosIndex` clusters in one corpus report
- The report now covers:
	- low16 and low24 `MSLK.GroupObjectId` fits against CK24 identity layers on mismatch entries
	- file-local reuse of non-zero `Ck24ObjectId` across multiple full CK24 values and type bytes
	- top mismatch families grouped by `LinkId + TypeFlags + Subtype`
	- top bad-`MdosIndex` CK24 clusters
- Key real-data findings from the current corpus run:
	- the UI `Ck24ObjectId` is just the low 16 bits of `MSUR.PackedParams -> CK24`
	- that low16 layer is usually a near one-to-one slice of full CK24 in-file, not a broadly reused hierarchy id (`2` reuse cases out of `1601` analyzed non-zero object-id groups)
	- both reuse cases occur on tile `36_24`, where one low16 object id survives across two full CK24 values and two type bytes
	- `MSLK.GroupObjectId` remains weak as the missing identity/hierarchy answer for unresolved `RefIndex` mismatches (`16` low16 matches, `15` low24 matches across `4553` mismatches)
	- `58` files carry bad `MSUR.MdosIndex` references, including several large non-zero CK24 families, not only `CK24=0` aggregates
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - PM4 Structure-Confidence / Decode-Trust Report

- Added `scan-structure-confidence` to `src/Pm4Research.Cli` and `Pm4ResearchStructureConfidenceAnalyzer` to `src/Pm4Research.Core`.
- Purpose:
	- stop collapsing two separate questions into one:
		- "is this chunk layout byte-level real?"
		- "are these field names and meanings actually proven?"
	- give the standalone PM4 path an explicit guardrail against inherited GPT-era structure lore and stale rollback assumptions
- The report now covers:
	- chunk-level layout confidence vs semantic confidence vs hallucination risk
	- field-level classification (`verified-reference`, `derived-bit-slice`, `named-guess`, `conflicted-reference`, `sparse-reference`, etc.)
	- explicit source-conflict inventory where older notes or field names overstate certainty
	- one summary that counts how much of the current standalone decoder is truly byte-closed versus only semantically guessed
- Key real-data findings from the current corpus run:
	- `13` tracked chunk families currently land in `high` layout confidence on the fixed development corpus
	- semantic confidence is much weaker:
		- `1` field `high`
		- `4` fields `medium`
		- `10` fields `low`
		- `4` fields `very-low`
	- strongest current byte+semantic anchors are `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR -> MSVI`, and `MDSF -> {MSUR, MDOS}`
	- highest current hallucination-risk zones are `MSLK.RefIndex`, `MSUR` bytes `4..19`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible fields such as `MDOS.buildingIndex`
	- the new conflict inventory now records concrete overstated legacy claims around `MSLK.LinkId`, `MSLK.RefIndex`, `MSUR.MdosIndex`, `MSUR.Normal + Height`, MSCN coordinate frame, and `MPRR.Value1`
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-structure-confidence --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_structure_confidence_report.json` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - MSUR Geometry Audit + RefIndex Family Classifier + Placement-Truth Refresh

- Added two new standalone PM4 commands in `src/Pm4Research.Cli`:
	- `scan-msur-geometry`
	- `scan-mslk-refindex-classifier`
- Added two new standalone analyzers in `src/Pm4Research.Core`:
	- `Pm4ResearchMsurGeometryAnalyzer`
	- `Pm4ResearchMslkRefIndexClassifier`
- Purpose:
	- close the specific decoder-trust seam around whether `MSUR` bytes `4..19` are real geometric fields or inherited naming fiction
	- replace the undifferentiated `4553`-row `MSLK.RefIndex` mismatch blob with likely target-domain family buckets
	- advance step 3 pragmatically by reusing the existing active `pm4-validate-coords` path for real `_obj0.adt` placement truth instead of pretending standalone PM4 can prove that by itself
- Key real-data findings from `scan-msur-geometry`:
	- analyzed surfaces: `518092`
	- degenerate surfaces: `0`
	- unit-length stored normals: `518092 / 518092`
	- strong positive stored-vs-geometry normal alignment: `518092 / 518092`
	- the trailing float currently named `Height` behaves like the negative plane-distance term along the stored normal, with best candidate `storedPlane.-` mean absolute error `0.00367829`
	- practical correction: `MSUR` bytes `4..19` are no longer a top decoder hallucination-risk seam, but the final float is semantically better described as a signed plane term than as generic height
- Key real-data findings from `scan-mslk-refindex-classifier`:
	- files with mismatches: `150`
	- total mismatch rows: `4553`
	- resolved/classified families: `505`
	- ambiguous families still remaining: `344`
	- resolved rows covered by classified families: `2651`
	- the classifier uses lift above corpus baseline so domains like `MPRR` do not win just by size (`98.4%` raw fit baseline)
	- largest current resolved family population: `probable-MSVT` (`293` families), with smaller but real `MSPI` / `MSPV` / `MSVI` / `MSCN` / `MPRL` slices
- Key real-data findings from the placement-truth refresh using existing active `pm4-validate-coords`:
	- tiles scanned: `616`
	- tiles validated against `_obj0.adt` placements: `206`
	- `MPRL` refs inside expected tile bounds: `114301 / 114301` (`100.0%`)
	- `MPRL` refs within `32` units of a nearest placement: `107907 / 114301` (`94.4%`)
	- average nearest placement distance: `10.98`
	- this materially strengthens `MPRL.Position` against real object-placement truth; it does **not** close `MPRR`
- Follow-up correction to decode-trust state:
	- refreshed `scan-structure-confidence` after these audits
	- semantic confidence counts moved from `1/4/10/4` (`high/medium/low/very-low`) to `2/4/9/4`
	- highest current hallucination-risk zones are now `MSLK.RefIndex`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible fields such as `MDOS.buildingIndex`
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-msur-geometry --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_msur_geometry_report.json` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex-classifier --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_mslk_refindex_classifier_report.json` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- pm4-validate-coords --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --json i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_pm4_coordinate_validation_report.json` passed, but only after a broader active-tree build that emitted many unrelated existing warnings.
	- no automated tests were added or run.

### Mar 21, 2026 - PM4 Research Workflow Validation On Trusted Tile `development_00_00.pm4`

- Validated the standalone PM4 research workflow on `test_data/development/World/Maps/development/development_00_00.pm4`, which is now the preferred reference tile for PM4 rediscovery work.
- Real-data findings from the standalone CLI on that tile:
	- `54` chunks total
	- `MSPV=8778`, `MSVT=6318`, `MSCN=9990`, `MPRL=2493`
	- top CK24 groups include large multi-surface families such as `0x40AA0A`, `0x418D9F`, and `0x421809`
	- `MPRL.Unk04` still spans only about `0.01°..22.30°` on this tile in the standalone read path, consistent with earlier viewer forensics that it is not a simple absolute building-yaw field here
- Dataset note:
	- the matching original `00_00` ADT triplet is not present anywhere in the workspace, so repository-side validation is currently limited to PM4 analysis and not full in-repo PM4-vs-ADT visual signoff
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Core/Pm4Research.Core.csproj -c Debug` passed.
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- no automated tests were added or run.
	- real-data PM4 analysis was performed on the file above, but no in-repo ADT-backed runtime signoff was possible because the companion ADTs are not in the workspace.

### Mar 21, 2026 - MPRL Semantic Correction From User Domain Knowledge

- User clarified the intended semantics of `MPRL` explicitly:
	- `MPRL` points are literal terrain/object collision-footprint intersections.
	- they mark the `XYZ` positions where ADT terrain is pierced by the object for collision stitching.
	- this makes terrain and object part of the same collision mesh at those points.
- Consequence for active PM4 work:
	- reject the old whole-object `MPRL` center/bounds translation idea, but do not reduce `MPRL` to vague anchor noise.
	- treat `MPRL` as collision-footprint reference data when scoring or comparing PM4 object hypotheses.
- Research-tooling follow-up:
	- `Pm4Research.Core` object hypotheses now include `MPRL` footprint counts against raw PM4 bounds so corpus analysis can see which candidate objects actually capture linked/tile-level `MPRL` seam points.

### Mar 21, 2026 - PM4 Link-Decode Sanity Fix + Linked-MPRL Summary Instrumentation

### Mar 21, 2026 - PM4 Object-Local Base Frame Layer

### Mar 21, 2026 - PM4 MPRL Axis Contract Correction

- Follow-up after direct comparison with older PM4 R&D exports and `WoWRollback/Pm4Reader` forensic notes.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- viewer-side `MPRL` position handling no longer assumes ADT-style planar `X/Z`, vertical `Y`.
	- the common `XY+Zup` mesh path now restores the older fixed `MSVT` viewer/world basis `(Y, X, Z)` instead of treating raw `(X, Y, Z)` as already canonical.
	- PM4 axis convention is now detected once per file and reused across CK24 groups instead of being redetected per CK24.
	- `BuildMprlPlanarPoints(...)`, `NearestPositionRefDistanceSquared(...)`, and `BuildPm4PositionRefMarkers(...)` now all convert `MPRL.Position` to world as `(PositionX, PositionZ, PositionY)` to match that restored `MSVT` basis.
- Why this was needed:
	- older PM4 forensics matched `MPRL` fields against raw `MSVT` axes, but the active viewer also needs to fold in the older successful `MSVT -> (Y, X, Z)` world basis from the R&D exporter.
	- without that fixed `MSVT` basis, the viewer was trying to approximate the right layout with per-object swap/invert heuristics, which can push PM4 into mirrored or polar-opposite fits against real WMO/M2 placements.
	- keeping axis convention per CK24 could still let neighboring wall/object fragments choose different mesh bases, which matched the remaining “random offset / mirrored” runtime symptom.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings.
	- no automated tests added or run.
	- no runtime real-data signoff yet that this restores PM4 placements on the affected tiles.

- Follow-up after the user chose the structural PM4 path instead of another heuristic yaw tweak.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- PM4 overlay objects now retain placed-space bounds/center, but their line and triangle geometry is localized around a preserved linked-group placement anchor instead of each fragment center.
	- each PM4 overlay object now carries a baked base placement transform that restores that anchored local geometry into the solved placed frame during rendering/export.
	- PM4 batched overlay rendering now applies `base placement -> overlay/object transforms` instead of assuming PM4 geometry is already final placed geometry.
	- PM4 JSON export re-applies the baked base transform so exported geometry remains in placed space.
- Why this was needed:
	- the viewer previously flattened PM4 geometry directly into placed space too early, which made “object inside container” reasoning and future placement-frame work harder.
	- the earlier experiment to move placement ownership down to linked subgroups regressed coherence and was reverted; this local-frame layer is structural groundwork without changing the CK24 solve boundary.
	- follow-up runtime diagnosis showed that rebasing split objects to per-fragment centers also discarded their original linked-group placement offsets, so split parts now preserve the pre-split anchor.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings.
	- no automated tests added or run.
	- no runtime signoff yet that this closes the remaining PM4 natural-rotation mismatch.

- Runtime forensics on `development_00_00.pm4` found that the active viewer path was consulting legacy `MSLK` fields that were never actually populated by `WoWMapConverter.Core`:
	- `MslkEntry.MsurIndex` defaulted to `0`
	- `WorldScene` was still checking that field when splitting surface groups and collecting linked `MPRL` refs
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/PM4/Pm4File.cs`
	- legacy `MSLK` entries now initialize unsupported fields to sentinels instead of zero (`MsurIndex = uint.MaxValue`, `MsviFirstIndex = -1`, `MsviIndexCount = 0`)
	- this prevents fake `surface 0` associations from leaking into PM4 viewer grouping/link logic
- `src/MdxViewer/Terrain/WorldScene.cs`
	- PM4 overlay objects now carry a linked-`MPRL` summary payload: total refs, normal refs, terminators, floor min/max, heading min/max/mean
	- PM4 JSON interchange export now includes the same linked-`MPRL` summary per object
- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- selected-object PM4 debug UI now shows linked-`MPRL` heading/floor summary directly in the alignment panel
- Tile-specific forensic result recorded from the raw dump of `development_00_00.pm4`:
	- `MPRL.Unk04` only spans about `0.01° .. 22.3°` across the tile, so it is not behaving like a simple absolute building-yaw field here
	- `Unk06` is constant `0x8000`
	- `Unk16` splits normal entries from terminators
	- `Unk14` still looks floor-like
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- no automated tests added or run
	- no runtime signoff yet on final PM4 object orientation

### Mar 21, 2026 - PM4 Documentation Refresh

- Refreshed `documentation/pm4-current-decoding-logic-2026-03-20.md` into a current viewer-side PM4 reconstruction contract instead of leaving PM4 behavior spread across many memory-bank entries and stale experiments.
- The updated document now records:
	- the three distinct PM4 reading layers: raw file data, linked object assembly, and final viewer render derivation
	- the active CK24 reconstruction pipeline in `WorldScene.BuildPm4TileObjects(...)`
	- the current `MPRL` contract as anchor/scoring input, not linked-center translation ownership
	- the stronger negative result from runtime evidence: current PM4 viewer behavior does not support an `MPRL` bounding-box/container paradigm
	- the split planar-candidate policy for tile-local versus world-space PM4
	- the `12°` coarse-only yaw correction guardrail
	- the list of rejected experiments that should not be reintroduced casually
- Updated memory-bank active-context files to point future PM4 work at that document first.
- Validation limits:
	- documentation and memory-bank update only; no code changes were made in this slice.
	- no automated tests were added or run.

### Mar 21, 2026 - Dockspace UI Recovery + PM4 Translation Rollback

### Mar 21, 2026 - Viewer PM4/WMO Correlation Export

- User asked to stop treating PM4, ADT placements, and WMO mesh data as separate lanes and to wire the correlation path into `MdxViewer` itself.
- `src/MdxViewer/Terrain/WorldAssetManager.cs`
	- added `WmoMeshSummary` and a new `TryGetWmoMeshSummary(...)` path.
	- factored WMO parsing so the existing v14/v17 read path can be reused for correlation output without depending on a live `WmoRenderer`.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- added `BuildPm4WmoPlacementCorrelationJson(...)`.
	- the export walks currently loaded tile WMO placements, derives WMO mesh summaries/local bounds, and ranks nearby PM4 overlay objects by tile-neighborhood plus bounds-gap / overlap heuristics.
	- output includes ADT placement identity, WMO mesh counts/bounds, and PM4 object metadata such as `CK24`, object part, linked-ref counts, and dominant `MSUR` fields.
- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- added `Dump PM4/WMO Correlation JSON` to the existing `PM4 Alignment` window.
	- save flow matches the existing PM4 object JSON export path.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings.
	- no automated tests were added or run.
	- no runtime real-data validation was performed for the new export workflow.

### Mar 21, 2026 - Viewer PM4/WMO Correlation Panel + Footprint Scoring Follow-Up

- User chose the next PM4/WMO steps explicitly:
	- add a live in-viewer correlation panel instead of staying export-only
	- strengthen ranking with actual transformed geometry / footprint comparison instead of only AABB heuristics
- `src/MdxViewer/Terrain/WorldAssetManager.cs`
	- extended `WmoMeshSummary` with sampled WMO geometry points so footprint comparison can reuse cached parse output instead of reopening meshes during report generation.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- added `BuildPm4WmoPlacementCorrelationReport(...)` as a typed viewer-side report model behind the JSON export.
	- correlation ranking now incorporates footprint overlap, footprint area similarity, and symmetric hull-distance metrics derived from transformed WMO sample geometry and PM4 object footprint hulls.
	- added `SelectPm4Object(...)` so a reported candidate can be promoted directly into the live PM4 selection state.
- `src/MdxViewer/ViewerApp.cs`
	- added persistent window/filter state for a new `PM4/WMO Correlation` tool window.
	- added a `View` menu toggle and a `PM4/WMO` launch button beside the existing PM4 controls.
- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- added a real `PM4/WMO Correlation` window with refresh, near-only filtering, model/path filtering, placement browsing, candidate drill-down, PM4 selection, and camera framing actions.
	- `PM4 Alignment` now links directly into that panel instead of only offering JSON export.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings after the follow-up.
	- no automated tests were added or run.
	- no runtime real-data validation has been performed yet for the new panel interactions or the footprint-based ranking quality.

- Latest user report after the ViewerApp partial split and earlier PM4 MPRL-frame experiment:
	- `World Maps` starting collapsed was wrong.
	- PM4 alignment had gotten worse.
	- the viewer still needed a real dock-panel UI.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- removed the linked-`MPRL` bounds-center translation path from `BuildPm4TileObjects(...)`.
	- removed the viewer-only `TryResolveMprlAuthoritativeAdjustment(...)` translation stage and the extra `worldTranslation` plumbing through PM4 line/triangle conversion.
	- kept the earlier geometry-pivot path plus coarse yaw-correction logic instead of forcing linked CK24 groups into one translated `MPRL` center.
- `src/MdxViewer/ViewerApp.cs`
	- enabled ImGui docking in source.
	- added a dockspace host between the menu/toolbar region and the status bar.
	- added a `View -> Dock Panels` toggle.
	- scene viewport math no longer subtracts fixed sidebar widths.
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- restored `World Maps` to default-open on first draw.
	- left/right shell panels can now run as normal titled dockable windows (`Navigator`, `Inspector`) when dock panels are enabled, while preserving the older fixed-sidebar mode as fallback.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings.
	- no automated tests added or run.
	- no runtime real-data signoff yet on PM4 alignment recovery or on the dock-panel workflow.

### Mar 21, 2026 - ViewerApp Partial-Class Refactor

- User asked for the oversized viewer shell file to be broken up instead of continuing to grow `src/MdxViewer/ViewerApp.cs` as one 6000+ line class.
- `src/MdxViewer/ViewerApp.cs`
	- removed the moved client-dialog, PM4 utility, minimap/status, and sidebar-heavy UI method bodies from the main file.
	- kept the remaining world-objects implementation in-place as `DrawWorldObjectsContentCore()` so the split stayed low-risk and behavior-preserving.
- Added new partials:
	- `src/MdxViewer/ViewerApp_ClientDialogs.cs`
	- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- `src/MdxViewer/ViewerApp_Sidebars.cs`
- Why this shape:
	- the repo already used partial-class decomposition for `ViewerApp`, so continuing that pattern was safer than a broad UI architecture rewrite.
	- this is a maintainability slice, not a user-facing viewer redesign.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings after fixing missing imports in the new partial files.
	- no automated tests added or run.

### Mar 21, 2026 - PM4 MPRL-Authoritative CK24 Frame Follow-Up

- This earlier viewer-side linked-`MPRL` translation experiment is no longer active.
- Runtime user validation reported that it made PM4 alignment worse, and the translation path was later removed in the `Dockspace UI Recovery + PM4 Translation Rollback` follow-up above.
- Keep this in mind when continuing PM4 work:
	- `MPRL` ownership as a semantic hypothesis is still open.
	- the specific implementation that translated CK24 groups into linked `MPRL` bounds centers should be treated as a regressed experiment, not current behavior.

### Mar 21, 2026 - PM4 Per-Object Bounds Overlay

### Mar 21, 2026 - PM4 Small-Yaw Correction Clamp

### Mar 21, 2026 - Viewer UI / Perf Slice: Hideable Chrome + Clipped Lists

- User priority shifted to viewer-shell usability and UI render cost because the fixed sidebar layout was getting in the way of PM4 debugging itself.
- `src/MdxViewer/ViewerApp.cs`
	- added `Tab`-driven hide-chrome mode so the menu bar, toolbar, sidebars, status bar, and floating utility windows can be suppressed quickly during scene inspection.
	- reduced default shell noise by no longer forcing every major sidebar section open on first draw.
	- clipped the obvious large per-frame UI lists instead of rendering every row:
		- file browser
		- discovered world maps
		- renderer subobject visibility toggles
		- WMO / MDX placement lists
		- POI / taxi node / taxi route lists
- Why this is scoped this way:
	- it attacks the two immediately visible pain points without doing a high-risk full UI rewrite: constant shell clutter and unbounded list drawing.
	- it does not attempt to restore the old dockspace/panel architecture yet.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing solution warnings
	- no automated tests added or run
	- no runtime signoff yet on actual frame-time improvement or on whether the new defaults are the right interaction model for PM4-heavy sessions

- Follow-up after user runtime report that PM4 objects were now almost correct but still carried a coherent `5..10` degree vertical-axis offset.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- `TryComputeWorldYawCorrectionRadians(...)` no longer applies tiny geometry-derived residual yaw corrections.
	- the CK24 world-yaw correction threshold moved from `2°` to `12°` so principal-axis noise does not override near-correct MPRL rotation.
- Why this is narrower than a raw-angle constant rewrite:
	- the PM4 repo tooling and format notes still treat MPRL `Unk04` / low-16 rotation as a standard `360 * value / 65536` angle.
	- the likely over-correction seam was the viewer-only continuous principal-axis fit, not the packed-angle scale itself.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- no automated tests added or run
	- no runtime signoff yet on the affected PM4 objects after this clamp

### Mar 21, 2026 - Archive I/O Performance Slice

- Reduced confirmed archive/path-resolution waste on the active MPQ-era viewer path without changing renderer or terrain behavior.
- `src/MdxViewer/DataSources/MpqDataSource.cs`
	- added `MpqDataSourceStats` instrumentation for `FileExists`, `ReadFile`, raw-byte cache behavior, and prefetch queue / worker timing
	- preserved the separate read-only prefetch MPQ workers, but now measures queue wait and worker read time explicitly
	- removed redundant normalized-vs-original duplicate MPQ existence probes in the MPQ-backed path
- `src/MdxViewer/Terrain/WorldAssetManager.cs`
	- added `WorldAssetReadStats` plus a resolved-read-path cache for world asset loads
	- `ReadFileData(...)` now caches the winning fallback path and no longer retries duplicate lowercase or `.mpq` forms that `MpqDataSource` already resolves internally
	- prefetch now warms the canonical model path and strongest `.skin` candidate first instead of broadly spraying alias permutations on every queued asset
- `src/MdxViewer/ViewerApp.cs`
	- world stats panel now shows asset-read probe counters plus MPQ read/prefetch counters so runtime profiling has exact signal instead of guesswork
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing solution warnings
	- no automated tests added or run
	- no runtime real-data validation yet on fixed MPQ-era datasets, so streaming-latency benefit is still instrumentation-backed but unproven in a live scene

- Added a PM4-specific bounds overlay in `src/MdxViewer/Terrain/WorldScene.cs` so PM4 object-part AABBs are visible in the main scene instead of only existing implicitly for picking/culling/debug text.
- Added a matching `PM4 Bounds` toggle in `src/MdxViewer/ViewerApp.cs` beside the existing PM4 MPRL and centroid toggles.
- Current behavior:
	- PM4 bounds draw through the existing `BoundingBoxRenderer` pass.
	- selected PM4 groups are highlighted, and the exact selected PM4 object is drawn white.
	- PM4 bounds rendering respects existing PM4 tile/object visibility checks and per-object transforms.
- Important limit:
	- the current PM4 object bounds still come from the rendered PM4 object geometry path, not from `MSCN` directly.
	- this is a debugging/visibility slice to validate extent mismatch hypotheses, not a solved MSCN container correction.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing solution warnings only
	- no automated tests added or run
	- no runtime signoff yet on the reported PM4 extent mismatch case

### Mar 21, 2026 - PM4 World-Space Orientation Solver Fix

### Mar 21, 2026 - PM4 Tile-Local Orientation Guardrail

- Follow-up after runtime report that PM4 tiles other than `0_0` and `0_1` were coherently rotated about `90°` counter-clockwise.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- `EnumeratePlanarTransforms(...)` now keeps tile-local PM4 on the established non-swapped south-west tile basis and only tests non-swapped mirror variants there.
	- `ConvertPm4VertexToWorld(...)` now assembles tile-local viewer-world positions with the correct WoW tile convention: file `tileY` advances world `X`, and file `tileX` advances world `Y`.
	- quarter-turn `swap` candidates remain available only for world-space PM4, where the earlier handedness fix was actually needed.
- Why this is narrower than reverting the whole solver expansion:
	- the quarter-turn solve is still needed for world-space PM4 cases.
	- the regression came from applying that same basis search to tile-local PM4, which already has a stable tile-frame mapping.
	- origin tiles masked a second seam: unswapped file tile indices can still place tile-local PM4 onto the wrong non-origin grid cell even when the planar basis is otherwise correct.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- no automated tests added or run
	- no runtime signoff yet on the reported non-origin tile placement/orientation case

- Fixed one concrete PM4 handedness bug in `src/MdxViewer/Terrain/WorldScene.cs` after runtime evidence showed mirrored solutions like `swap=True` / `windingFlip=True` on structures that should only need a quarter-turn basis correction.
- Root cause:
	- `ResolvePlanarTransform(...)` only tested `identity` and `swap` for world-space PM4 data
	- this forced some world-space objects into mirrored fits because the rigid `+/-90` degree candidates were never evaluated
- Current behavior:
	- world-space PM4 now evaluates the rigid planar transforms first (`identity`, `180`, `+90`, `-90`)
	- mirrored candidates are now removed from the active PM4 planar solver so winding parity stays rigid-only instead of drifting into reversed/opposite-facing fits
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests added or run
	- no runtime signoff yet on the guardtower staircase case

### Mar 21, 2026 - PM4 Picking Arbitration Fix

- Fixed a viewer interaction bug where visible PM4 overlay objects could not be selected because `ViewerApp.PickObjectAtMouse(...)` returned on WMO/MDX selection before PM4 picking ran.
- Current behavior:
	- `WorldScene` now provides nearest-hit helpers for regular scene objects and PM4 overlay objects.
	- `ViewerApp` compares both hit distances from the same mouse ray and selects the closer target instead of hard-prioritizing WMO/MDX.
- Why this matters:
	- PM4 alignment tooling depends on left-click selection, and PM4 geometry commonly overlaps the same world objects whose WMO/MDX AABBs were previously swallowing the click.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests added or run
	- no runtime click-validation yet on a live PM4 overlay session

### Mar 21, 2026 - 4.0.0.11927 M2 Wrap + Blend Correction

- Follow-up after the first M2 parity slice focused on the remaining Cataclysm-era runtime symptoms the user reported: texture clamping/stretching and incorrect blend family selection on `4.0.0.11927` assets.
- Root gaps corrected in the active viewer path:
	- `src/MdxViewer/Rendering/ModelRenderer.cs` now treats `WrapWidth` / `WrapHeight` as repeat flags for all M2-adapted models, while classic MDX keeps the legacy clamp-flag interpretation.
	- `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` no longer shifts M2 blend ids after mode `2`; ids `3`..`7` now map deliberately into the closest local renderer families.
- Current mapping details:
	- `0=Load`, `1=Transparent`, `2=Blend`, `3=Add` (`NoAlphaAdd`), `4=Add`, `5=Modulate`, `6=Modulate2X`, `7=AddAlpha` (`BlendAdd`)
	- `NoAlphaAdd` and `BlendAdd` are still approximations because the local MDX renderer has no separate states for them yet
- Validation limits for this checkpoint:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing solution warnings only
	- no automated tests added or run
	- no runtime real-data validation yet on the affected Cataclysm-era M2 assets

### Mar 21, 2026 - M2 Material Parity Slice: Explicit Env-Map + UV Selector Recovery

- Landed the first non-heuristic M2 material-parity implementation slice in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` and `src/MdxViewer/Rendering/ModelRenderer.cs`.
- Confirmed root gap before editing:
	- `ModelRenderer` already had separate alpha-cutout / blended / additive / env-map shader-state paths
	- `WarcraftNetM2Adapter` was still flattening M2 batch intent by hardcoding `CoordId = 0`, dropping raw `.skin` texture-coordinate lookup metadata, and only preserving the first UV set from vertex data
- Current code change:
	- merges raw `.skin` `textureCoordComboIndex` metadata back into the Warcraft.NET skin path
	- preserves both M2 UV sets from raw `MD20` vertex data
	- reads raw `textureCoordCombos` so `-1` now drives reflective `SphereEnvMap` and `1` can route to UV1
	- adds focused renderer trace output showing pass + resolved material family for M2 batches under debug focus
- Current scope/limits:
	- improved: reflective / env-mapped family selection and UV-set fidelity where metadata exists
	- still open: texture transform animation, transparency/color track parity, broader shader-combo parity, and real-data visible validation on heavy reflection/transparency assets
- Validation limits at this checkpoint:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests added or run
	- no runtime validation yet on real assets

### Mar 21, 2026 - PM4 Decode Triage Framed + Renderer Parity Planned

- PM4 overlay debugging is now in a more precise phase than the earlier loose-overlay indexing/precedence work:
	- user runtime symptom: `PM4: 2674 files found, none decoded into overlay data`
	- status interpretation: files are being discovered, but none survive into renderable overlay objects
	- `WorldScene` now has failure-bucket instrumentation for tile parse, tile range, read, decode, and parsed-but-zero-object cases
- Current root-cause direction for the `4.0` base-client failure versus `3.3.5`:
	- PM4 parser/object builder does not appear to key directly on `_dbcBuild`
	- map discovery / WDT resolution / active candidate set still does
	- the `2674` PM4 candidate count should be investigated against the fixed development dataset expectation of `616` PM4 files
- Formalized the rendering recovery program needed for PM4 object-variant matching:
	1. M2 material, transparency, and reflective parity
	2. lighting DBC expansion
	3. skybox / environment parity
- Added dedicated prompt plans for each implementation slice:
	- `.github/prompts/m2-material-parity-implementation-plan.prompt.md`
	- `.github/prompts/lighting-dbc-expansion-implementation-plan.prompt.md`
	- `.github/prompts/sky-environment-parity-implementation-plan.prompt.md`
- Validation limits for this update:
	- no new implementation code landed for the three rendering tracks yet
	- no automated tests were added or run in this planning pass
	- no runtime signoff yet for PM4 failure-bucket output or the rendering program

### Mar 21, 2026 - WMO Blend-Mode Correction + Loose PM4 Overlay Precedence

- Corrected one concrete WMO material/rendering mismatch in `src/MdxViewer/Rendering/WmoRenderer.cs`:
	- raw WMO material `BlendMode` is now mapped to `EGxBlend`
	- opaque pass handles `Opaque` and `AlphaKey`
	- transparent pass now handles only `Blend` and `Add` with matching blend funcs
- Fixed loose overlay precedence in `src/MdxViewer/DataSources/MpqDataSource.cs`:
	- loose-file resolution now searches `_looseRoots` newest-first so the most recently attached overlay overrides earlier roots
	- PM4 loose-path failures now emit the same trace help that previously existed only for WMO failures
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Validation limits:
	- no automated tests were added or run
	- no runtime real-data validation yet for the WMO sheen symptom
	- no runtime real-data validation yet for base+loose-overlay PM4 loading

### Mar 21, 2026 - Explicit Base-Build Selection Restored For Viewer MPQ Loads

- Restored explicit client build selection in `MdxViewer` instead of relying only on path-based build inference:
	- added `src/MdxViewer/Terrain/BuildVersionCatalog.cs`
	- `Open Game Folder (MPQ)...` now routes through a build-selection dialog before calling `LoadMpqDataSource(...)`
	- build options are loaded from `WoWDBDefs/definitions/Map.dbd` when available, with a fallback list that includes `4.0.0.11927` and `4.0.1.12304`
- Persisted base-build identity for saved clients:
	- `KnownGoodClientPath` now stores `BuildVersion`
	- viewer settings now also store `LastSelectedBuildVersion`
	- reopening a saved base or loading a loose map folder against a saved base now reuses the saved explicit build when present
- Added a runtime hint for PM4-era dataset mismatches:
	- loose overlay attach inspects the first PM4 version marker it finds
	- known markers currently map `11927 -> 4.0.0.11927` and `12304 -> 4.0.1.12304`
	- viewer logs a warning when PM4 overlay hint and active base-client build disagree
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Validation limits:
	- no automated tests were added or run
	- no runtime real-data validation yet with the development PM4 overlay against a verified `4.0.1.12304` base client

### Mar 21, 2026 - 4.0.0.11927 Terrain Blend Documentation + First Runtime Recovery Slice

- Closed the stale documentation gap around 4.0 terrain texturing by recording the wow.exe-backed runtime model instead of repeating the older "same as 3.3.5" shorthand.
- Reverse-engineered/runtime-documented behavior now preserved in repo docs and prompts:
	- chunk alpha assembly is neighbor-aware, not chunk-local only
	- neighbor layers are matched by texture id
	- 8-bit layers without direct alpha payload can be synthesized as residual coverage
	- runtime blend textures are created through the `TerrainBlend` path
- Documentation and prompt updates landed in:
	- `documentation/wow-400-terrain-blend-wow-exe-guide.md`
	- `docs/archive/WoW_400_ADT_Analysis.md`
	- `docs/archive/WoW_400_DeepDive_Analysis.md`
	- `docs/archive/WoW_301_DeepDive_Analysis.md`
	- `docs/ADT_WDT_Format_Specification.md`
	- `specifications/ghidra/prompt-400.md`
	- `.github/prompts/wow-400-terrain-blend-recovery.prompt.md`
- Active viewer implementation now includes the first 4.0 recovery slice in `StandardTerrainAdapter` / `TerrainChunkData`:
	- dedicated `Cataclysm400` alpha-decode mode stays separate from `LichKingStrict`
	- preserves per-layer `AlphaSourceFlags`
	- synthesizes missing residual 8-bit alpha when a layer lacks direct payload
	- stitches same-tile chunk-edge alpha texels by matching neighbor layer texture ids
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Validation limits:
	- no new automated tests were added or run for this slice
	- no real-data runtime verification yet on `test_data/development/World/Maps/development`
	- this is the first runtime-backed recovery slice, not full `TerrainBlend` parity closure

### Mar 20, 2026 - PM4 Tile Mapping Normalization + Reboot Handoff

- Applied PM4 viewer tile mapping guardrail in `WorldScene`:
	- map PM4 filename `x_y` into terrain tile keys as `(tileX=x, tileY=y)`
	- remove MPRL-centroid tile reassignment from PM4 overlay load path
	- merge duplicate PM4 tile payloads instead of overwriting (objects/stats/position refs)
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Runtime handoff:
	- user will restart machine before runtime checks
	- next required validation is the reported tile adjacency case (`00_00`, `01_00`, `01_01`) to confirm no PM4 tile drift/data loss

### Mar 20, 2026 - WDL Spawn Chooser Cross-Version Regression Handoff

- User runtime report: WDL heightmap spawn chooser currently does not function on tested versions.
- Status correction:
	- treat prior notes that implied chooser readiness/fallback behavior was sufficient as unverified for the active branch state
	- keep this issue open until runtime behavior is re-proven end-to-end
- Investigation target area for the next pass:
	- map-row spawn action gating (`WdlPreviewWarmState` and readiness transitions)
	- preview cache warmup readiness state propagation into chooser UI
	- chooser commit path versus fallback load path when preview prep fails
- Validation requirement for closure:
	- real-data runtime proof on at least one Alpha-era map and one 3.x map
	- explicit evidence that user can pick a spawn point and spawn is applied
- Validation limits in this handoff-only update:
	- no code changes made
	- no automated tests added or run

### Mar 19, 2026 - Terrain Texture Transfer Command (Backend Slice)

- Added first backend/library + CLI slice for mapped terrain texture transfer:
	- command: `terrain-texture-transfer`
	- payload scope: `MTEX`, `MCLY`, `MCAL`, `MCSH`, and MCNK holes
	- mapping modes: explicit `--pair` and auto `--global-delta`
	- supports `dry-run` manifests and `apply` output ADT writing
- Added split-ADT resilience for the active development dataset:
	- if `SplitAdtMerger` serialization fails, command now composes transferable texture payload from root + `_tex0.adt`
	- MCNK subchunk parsing now tolerates headerless tex0 MCNK payloads
	- top-level chunk walk/rebuild now handles odd-size boundary variance seen in split files
	- merge path now skips `obj0`-only sidecars (without `_tex0`) and uses root bytes directly for terrain-texture transfer
- Real-data validation performed (fixed path):
	- source/target: `test_data/development/World/Maps/development`
	- dry-run sample: `development_0_0 -> development_0_0` (chunk pairs=256, copied flags true for MTEX/MCLY/MCAL/MCSH/holes)
	- apply sample: same pair wrote output ADT + summary/tile manifests
	- non-identity sample: `development_0_0 -> development_1_0` succeeded in both dry-run and apply with full payload transfer and no manual-review flags
	- small global-delta batch (`--global-delta 1,0 --tile-limit 3`) completed; 2 tiles clean, 1 tile (`development_0_1 -> development_1_1`) still flagged manual-review due one target MCNK with no parseable subchunks
- Validation limits:
	- no viewer runtime visual signoff yet for transferred outputs in this pass
	- no new automated tests added in this pass

### Mar 19, 2026 - MdxViewer Thin UI Hook For Terrain Texture Transfer

- Added a thin UI entry in `MdxViewer` (`ViewerApp`) for the backend terrain texture transfer flow:
	- File menu item: `Terrain Texture Transfer...`
	- dialog supports source/target/output folders, dry-run/apply toggle, explicit-pair or global-delta mapping, chunk offsets, payload toggles, and optional manifest path
	- execution runs asynchronously via the existing app-thread pattern and surfaces summary + warnings in an in-dialog log panel
- Build validation passed for the viewer after wiring:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Validation limits:
	- no new runtime visual validation in the viewer yet for this dialog path
	- this UI slice does not resolve the known `development_0_1 -> development_1_1` target MCNK parse edge case from backend validation

### Mar 19, 2026 - Canonical Fresh-Output Pass For 3.3.5 Development Map

- Executed a full-map identity transfer pass to materialize a fresh canonical output folder for viewer use:
	- command: `terrain-texture-transfer --source-dir ...335-dev... --target-dir ...335-dev... --global-delta 0,0 --mode apply`
	- output root: `output/development-335-canonical-texture-transfer`
- Real-data result summary:
	- tiles planned/processed/written: 2303 / 2303 / 2303
	- manual review: 0
	- chunk pairs applied: 589,568
	- missing source/out-of-range chunk remaps: 0 / 0
	- summary manifest: `output/development-335-canonical-texture-transfer/manifests/summary.json`
	- companion `development.wdt` and `development.wdl` copied into output root
- Operational guidance:
	- this is now a viable "open the generated folder in MdxViewer" workflow for the tested 3.3.5 development dataset
	- this does not replace targeted non-identity remap validation when using non-zero global deltas or explicit cross-tile mappings

### Mar 19, 2026 - Development Repair WL Attribution + Texture Payload Manifests

- Reworked `DevelopmentRepairService` WL ingestion so repair no longer assumes tile-named `*.wl*` files.
	- new behavior pre-indexes all map-level WL files (`.wlw/.wlm/.wlq/.wll`) once, converts to MH2O by world position, and applies per-tile liquids from that coordinate-attributed index
	- tile manifests now record the actual WL source file paths used (for example `Clayton Test.wlw`) instead of synthetic `tileName.wlw` expectations
- Expanded per-tile JSON payload (`TextureData`) with terrain texturing data modeled after the VLM chunk-layer shape:
	- includes MTEX texture list
	- includes per-chunk layers with texture id/path, flags, alpha offset, effect id, plus optional base64 alpha bytes and byte count
	- extractor now chooses the richest source among output ADT, `_tex0.adt`, and root ADT so split-source tiles can still emit texture payload data
- Real-data validation performed on fixed paths:
	- command: `development-repair --mode repair --input-dir test_data/development/World/Maps/development --tile-limit 50`
	- observed manifests with `WlLiquidsConverted=true` and map-level WL source filenames attached to those tiles
	- reference check only: `development-repair --mode repair --input-dir test_data/WoWMuseum/335-dev/World/Maps/development --tile-limit 1` (used only to inspect payload shape, not as canonical pipeline input)
	- policy now enforced in code: `development-repair` rejects WoWMuseum `335-dev` input and requires building clean outputs from `test_data/development/World/Maps/development` constituent parts
- Validation limits:
	- this pass did not include viewer-side visual validation of generated MH2O/texturing results
	- no new automated regression tests were added in this pass

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
- Initial MCCV fix improved output but did not fully match runtime behavior.
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

### Mar 18, 2026 - MCCV + Patch-Letter Follow-up

- Reworked MCCV semantics after user runtime feedback showed the first shader heuristic was still wrong.
- Current interpretation now matches the repo's own MCCV writer comments:
	- bytes are treated as BGRA, not RGBA
	- neutral/no-tint values are mid-gray (`127`) rather than white
	- terrain tint uses RGB remapped around mid-gray, not MCCV alpha strength
- Extended `NativeMpqService.LoadArchives(...)` to discover MPQs recursively so nested/custom `patch-[A-Z].mpq` archives are included in the patch chain.
- Kept Alpha single-asset wrapper archives (`.wmo.mpq`, `.wdt.mpq`, `.wdl.mpq`) out of the generic recursive scan because they are handled separately by the viewer data source.
- Build gates passed again after this follow-up:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation is still the blocker:
	- confirm 3.x MCCV transparent/neutral regions no longer darken to black
	- confirm maps stored inside `patch-[A-Z].mpq` are now discovered and load through normal WDT/ADT lookup paths

### Mar 18, 2026 - 3.x Alpha Offset-0 Experiment Reverted

- The recent LK offset-0 fallback change in `StandardTerrainAdapter.ExtractAlphaMaps(...)` was reverted after runtime validation showed it was wrong.
- Updated conclusion:
	- treating `AlphaMapOffset == 0` as a valid relaxed fallback for the active 3.x terrain path is not the correct fix
	- keep the revert and continue investigating the real 3.x alpha decode/sourcing failure separately
- Validation status:
	- normal `dotnet build .../MdxViewer.sln -c Debug` still conflicts with the running viewer process locking `bin/Debug`
	- use the alternate-output build for compile validation while the viewer stays open

### Mar 18, 2026 - 3.x Profile-Driven Alpha Recovery

- Investigated the remaining 3.x terrain failure after the offset-0 revert.
- Confirmed the active recovery branch was still missing rollback-era handling for:
	- MPHD/WDT big-alpha mask `0x4 | 0x80`
	- split `*_tex0.adt` sourcing for textures/layers/alpha/shadow data
	- stronger MCAL decode semantics for compressed alpha, big alpha, and do-not-fix chunks
- Applied the recovery batch:
	- `FormatProfileRegistry`: added `BigAlphaFlagsMask` and `PreferTex0ForTextureData`; 3.0.1 and 3.3.5 profiles now use `0x4 | 0x80` and prefer `*_tex0.adt`
	- `StandardTerrainAdapter`: can read MTEX + MCNK data from `*_tex0.adt`, route layer/alpha/shadow sourcing through that file, pass the MCNK `0x8000` do-not-fix flag into alpha decode, and infer big-alpha per chunk
	- `WoWMapConverter.Core/Formats/LichKing/Mcal.cs`: replaced the broken/simple decoder with the stronger compressed / big-alpha / 4-bit implementation
- Build validation passed:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build "I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="I:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime validation is still the blocker:
	- no claim yet that 3.x alpha blending is correct on real user data
	- next check is whether the failing 3.x sample now uses more than one alpha layer and stops looking like 4-bit Alpha-era decode

### Mar 18, 2026 - Terrain Runtime Validation Update

- User runtime validation now confirms the current terrain alpha recovery on two real data families:
	- Alpha 0.5.3 terrain renders correctly again after restoring the alpha-era edge fix in `AlphaTerrainAdapter`
	- 3.0.1 alpha-build terrain renders correctly on the profile-driven strict 3.x path
- Earlier runtime feedback also reported the 3.3.5 sample looked correct before the 0.5.3 regression was fixed.
- Status change:
	- terrain validation is no longer build-only for the tested 0.5.3 and 3.0.1 samples
	- broader signoff across more 3.x maps is still pending, so do not generalize this to all LK-era terrain yet

### Mar 18, 2026 - Remaining ModelRenderer Slice From 39799bf

- Applied the last model-side hunk from `39799bf` after the MPQ reader work was already in place.
- `ModelRenderer` now skips particle rendering on the world-scene batched render path only.
- Standalone model viewing still renders particles as before.
- Reason: per-instance transforms are not yet propagated into particle simulation for placed models, and leaving them enabled there can produce visibly wrong camera-locked effects.
- Build gate passed: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.

### Mar 18, 2026 - WDL Preview Warmup + Texture Reuse Batch

- Ported the missing `main` WDL preview cache support into the recovery branch:
	- added `WdlPreviewCacheService`
	- `ViewerApp` now warms discovered WDL previews in the background and opens the preview dialog through the cache-aware path
	- `ViewerApp_WdlPreview` now shows warmup/error state instead of only a synchronous failure dialog
- Added a targeted model-load performance slice in `ModelRenderer`:
	- per-model texture diagnostic logs are now opt-in via `PARP_MDX_TEXTURE_DIAG`
	- BLP/PNG textures now use a shared refcounted GL texture cache so repeated world doodads do not decode/upload the same texture once per instance
- Build validation passed: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime validation is still required before claiming:
	- WDL preview warmup/cache behavior is correct on the user's real map set
	- M2 load time is materially improved in the real world scene

### Mar 18, 2026 - WDL Parser Recovery + Transparency Heuristic Follow-up

- Addressed the newly reported WDL read failure after the preview-cache port:
	- `WoWMapConverter.Core/VLM/WdlParser.cs` no longer rejects all non-`0x12` WDL versions up front
	- parser now scans the WDL chunk stream for `MAOF` and accepts MAOF offsets that reference either `MARE` headers or direct height payloads
- Unified active viewer WDL reads through `src/MdxViewer/Terrain/WdlDataSourceResolver.cs` so both preview warmup and `WdlTerrainRenderer` use the same `.wdl` / `.wdl.mpq` + file-set lookup path.
- Closed a remaining 3.x model-path gap in `WmoRenderer` by extending doodad extension fallback from only `.mdx`/`.mdl` to also include `.m2`.
- Adjusted `ModelRenderer` transparency routing:
	- shared texture cache entries now retain simple alpha-shape metadata
	- classic non-M2 `Transparent` layer-0 materials only use hard cutout when the texture alpha is binary
	- textures with intermediate alpha now stay on the blended path
- Build validation passed: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- No automated tests were added or run.
- No runtime real-data validation has been performed yet for this batch.

### Mar 18, 2026 - Standalone 3.x Model Load Freeze Follow-up

- Addressed the reported freeze / non-load behavior when opening individual 3.x `.mdx` files in the viewer.
- Root cause on the active standalone path was different from the world/WMO loaders:
	- standalone container probing only recognized `MD20`, not `MD21`
	- standalone M2 adaptation eagerly scanned the full `.skin` file list on the UI thread before trying the obvious same-basename candidates
	- standalone file loads also lacked the world path's canonical model-path recovery and MD20 converter fallback
- Current fix in `src/MdxViewer/ViewerApp.cs`:
	- standalone probe now routes both `MD20` and `MD21` through the M2-family path even when the file extension is `.mdx`
	- standalone M2 loads now resolve a canonical model path through MPQ file-set indexes before skin lookup
	- predictable `.skin` candidates are tried first, and the broader `.skin` file-list search is only used as a fallback with a per-session cache
	- standalone MD20 loads now also have the same M2->MDX converter fallback used elsewhere when direct adaptation cannot complete
	- standalone skin-path cache is cleared when a new MPQ data source is loaded
- Build validation passed: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- No automated tests were added or run.
- No runtime real-data validation has been performed yet for this batch.

### Mar 18, 2026 - M2 Empty-Fallback Guardrail

- Follow-up after runtime feedback that some M2-family models still "load" into an empty viewport with `0` geosets / vertices.
- Current conclusion:
	- at least some failures are not clean adapter failures; the raw `MD20` converter fallback can produce an `MDX` shell that parses but has no renderable geometry
	- that state is misleading in the UI because it looks like a loaded model rather than an unsupported / failed conversion
- Current fix:
	- `WarcraftNetM2Adapter` now exposes shared renderable-geometry checks
	- standalone `ViewerApp`, world `WorldAssetManager`, and WMO doodad `WmoRenderer` now reject converted fallback models unless they contain at least one renderable geoset
	- rejected fallback loads now preserve/log the underlying failure instead of silently treating an empty converted model as success
- Build validation passed: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`.
- No automated tests were added or run.
- No runtime real-data validation has been performed yet for this batch.
- This is a diagnostics/correctness guardrail, not proof that pre-release `3.0.1` M2 layouts are fully supported.

### Mar 18, 2026 - Pre-release 3.0.1 M2 Scope Clarified

- User runtime verification after the guardrail patch indicates most remaining M2 problems are specific to the pre-release `3.0.1` model family rather than the later `3.3.5` family.
- Current working assumption:
	- pre-release `3.0.1` model files may be a transitional or hybrid `MDX` + `M2` variant
	- later-WotLK assumptions should not be silently reused for that path
- Separate runtime issue remains open across both model families:
	- neon-pink transparent surfaces still appear on both `MDX` and M2-family assets
	- treat that as a shared renderer/material/shader problem, not proof of a model-parser-only defect
- Resulting investigation split for the next pass:
	1. add true version/profile-aware handling for pre-release `3.0.1` model structure
	2. audit shared transparent-surface handling, texture resolution, and blend/shader parity independently of format parsing
- No new code changes were made in this note-only follow-up.
- Runtime evidence came from the user's real data, not fixtures.

### Mar 19, 2026 - Pre-release 3.0.1 Model Profile Guardrail

- Live `wow.exe` decompilation for build `3.0.1.8303` confirmed the client-side model gate is stricter than the active generic adapter path:
	- required root magic is `MD20`
	- accepted version range is `0x104..0x108`
	- parser behavior splits structurally at `0x108`
- Active viewer code now routes that profile knowledge into all three shared M2-family entry points:
	- standalone `ViewerApp.LoadM2FromBytes(...)`
	- world `WorldAssetManager.LoadMdxModel(...)`
	- WMO doodad `WmoRenderer.LoadM2DoodadRenderer(...)`
- `WorldScene` / `WorldAssetManager` now receive the build string at construction time so constructor-time manifest loads use the same profile guard instead of waiting for later `SetDbcCredentials(...)`.
- `WarcraftNetM2Adapter` now fails fast on build/profile mismatches before `.skin` search or fallback conversion:
	- `3.0.1.8303` and unknown `3.0.x` profiles reject `MD21` roots and out-of-range MD20 versions
	- `3.3.5.12340` currently keeps `MD21` container allowance to avoid broad later-branch regression while the parser path remains shared
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no new runtime real-data validation yet for the guarded `3.0.1` model path
	- do not claim this as a full pre-release `3.0.1` render fix; it is a profile-routing/compatibility guardrail
- Separate shared renderer issue is still open:
	- neon-pink transparent surfaces remain a Track B problem across classic `MDX` and M2-family assets

### Mar 19, 2026 - Standalone Data-Source M2 Read-Path Fix

- The new user-visible `Failed to read: ...` symptom on standalone/browser-loaded M2-family assets was not a parser error.
- Root cause:
	- `ViewerApp.LoadFileFromDataSource(...)` still did an exact `_dataSource.ReadFile(virtualPath)` and returned early
	- M2-family assets in the file browser can appear under alias paths that need the same canonical resolution logic already used later in the standalone M2 path
- Current fix:
	- data-source loads for `.mdx` / `.mdl` / `.m2` now resolve through `ResolveStandaloneCanonicalModelPath(...)`
	- browser-side model reads now use `ReadStandaloneFileData(...)` before giving up
	- successful reads now carry the resolved virtual path into the later container-probe path
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- runtime retry on the user's actual 3.0.1 data is still required to confirm the failure moved from read-time to the next real blocker

### Mar 19, 2026 - Pre-release 3.0.1 wow.exe Documentation Pass

- Shifted from speculative code changes to binary-backed documentation after the user reported that models still do not load.
- New documented `wow.exe` facts for build `3.0.1.8303`:
	- common loader chain is `FUN_0077e2c0` -> `FUN_0077d3c0` -> `FUN_0079bc70` -> `FUN_0079bc50` -> `FUN_0079bb30` -> `FUN_0079a8c0`
	- accepted model-family extensions are normalized to `.m2` before parse/bootstrap continues
	- high-level failure falls back to `Spells\\ErrorCube.mdx`
	- root parser is `MD20`-only with version range `0x104..0x108`
	- parser layout splits at `0x108`
	- confirmed validator families now include shared span strides `1`, `2`, `4`, `8`, `0x0C`, `0x30`, `0x44` and nested record families `0x70`, `0x2C`, `0x38`, `0xD4`, `0x7C`
	- version split families are legacy `0xDC` + `0x1F8` versus later `0xE0` + `0x234`
- New artifacts created for fresh chats:
	- `documentation/pre-release-3.0.1-m2-wow-exe-guide.md`
	- `.github/prompts/pre-release-3-0-1-m2-implementation-plan.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-ghidra-followup.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-runtime-triage.prompt.md`
- Validation status for this pass:
	- no automated tests were added or run
	- no new build was needed because this pass only added documentation and prompts
	- no runtime real-data validation was performed

### Mar 19, 2026 - 3.0.1 Pre-release Profile Routing Broadening

- Follow-up after the wow.exe-backed profile guardrail: the active registry no longer binds the pre-release `3.0.1` profile only to exact build `3.0.1.8303`.
- Current behavior:
	- any parsed `3.0.1.x` build now resolves to the same pre-release `3.0.1` ADT, WMO, and M2 profiles
	- other `3.0.x` builds still fall back to the generic unknown `3.0.x` profile until there is binary evidence for a narrower mapping
- Why this matters:
	- standalone model loads, world doodads, WMO doodads, and terrain/WMO profile routing now stay on the pre-release path for the whole `3.0.1` family instead of silently downgrading non-`8303` builds to the generic `3.0.x` profile
- Validation status:
	- build validation pending for this specific routing change
	- no automated tests were added or run
	- no runtime real-data validation was performed

### Mar 19, 2026 - 3.0.1 Pre-release M2 Parser + Fallback Alignment

- Follow-up after the routing-only fix was not enough: active model loading now includes a dedicated pre-release `MD20` parse path in `WarcraftNetM2Adapter` instead of sending raw `3.0.1` files through Warcraft.NET's later-layout `MD21` assumptions.
- Current viewer-side behavior:
	- standalone, world, and WMO doodad adapter loads normalize pre-release `MD20` data through a local parsed-model abstraction
	- the old forced profile-specific `.skin` parser path was disabled because the wow.exe-derived `0x70` / `0x2C` family sizes were not proven `.skin` submesh / batch strides
	- converter fallback now receives the active build version and avoids hard-parsing later-layout animation / bone tables for pre-release `3.0.1`
	- converter skin fallback keeps only the index / triangle tables required for geometry conversion instead of forcing nonessential fixed-stride submesh / texture-unit tables
- Why this matters:
	- the primary runtime path and the fallback conversion path no longer disagree about pre-release `3.0.1` model-family assumptions
	- non-`8303` `3.0.1.x` builds now reach both the right profile and a compatible loader path
- Validation status for this pass:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests were added or run
	- no runtime real-data validation was performed

### Mar 19, 2026 - Standalone Alias Recovery + Unsuffixed Skin Candidates

- Follow-up after fresh runtime errors still showed two model-load gaps:
	- standalone/browser `DataSourceRead` failures could still stop at the unresolved `.mdx` alias path even when the world-model path already had broader file-set heuristics
	- companion skin discovery only tried `00`-`03` suffixed names, not the unsuffixed `.skin` form some transitional assets may use
- Current fix:
	- `ViewerApp` standalone canonical resolution and data-source reads now reuse the broader candidate set already proven useful on the world path: exact path, extension aliases, bare filename aliases, and `Creature\Name\Name.{mdx|m2|mdl}` guesses
	- standalone resolution now also probes guessed candidates through `FileExists` / `ReadFile` instead of depending only on the prebuilt file index
	- shared `WarcraftNetM2Adapter.BuildSkinCandidates(...)` now includes unsuffixed `.skin` candidates before the numbered `00`-`03` forms
- Why this matters:
	- user-visible `Failed to read requested='...mdx'` errors can now recover through the same alias breadth the world loader already had
	- `Missing companion .skin for M2` can now recover when the sidecar is present under the base `.skin` name instead of only numbered variants
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests were added or run
	- runtime validation on the specific failing assets is still pending

### Mar 19, 2026 - Cocoon Optional-Span Parser Follow-up

- Fresh runtime log from `Creature\Cocoon\Cocoon.mdx` showed the profiled pre-release parser was now reached, but it still failed before geometry extraction because an unresolved optional table span (`colors`, stride `0x2C`) was treated as fatal.
- Current fix:
	- `WarcraftNetM2Adapter.ParseProfiledMd20Model(...)` now hard-validates only the spans the runtime model builder actually dereferences for viewer geometry

### Mar 19, 2026 - MCNK Index Repair Hook For Development ADT Export

- Added a rollback-CLI `repair-mcnk-indices` command that audits or rewrites root ADT `MCNK` header `IndexX` / `IndexY` values.
- `development-repair` now runs the same fixup in-memory on exported root ADTs by default; disable with `--repair-mcnk-indices false` if raw output is needed.
- Repair logic prefers `MCIN` order when present and otherwise falls back to top-level `MCNK` scan order.
- Real-data audit on the loose source folder `test_data/development/World/Maps/development` found:
	- 466 root ADT filenames
	- 114 zero-byte placeholders
	- 352 non-empty roots with chunk data
	- 0 detected `MCNK` index mismatches under scan-order validation on those raw loose roots
- Validation limits:
	- this does not prove generated WDL-derived / repaired export sets are clean because the referenced `PM4ADTs/*` outputs are not present in this workspace
	- `dotnet run/build` for `WoWRollback.Cli` is still blocked here by pre-existing missing `WoWFormatLib` / `CascLib` references under `WoWRollback.AnalysisModule`, so end-to-end CLI execution was not revalidated in this environment
	- optional / unresolved table families now use a nonfatal validator that logs and skips invalid spans instead of rejecting the entire model
	- per-texture filename spans are also treated as optional so a bad embedded name table does not abort the whole model
- Why this matters:
	- `Cocoon.mdx` was failing in the parser before any real geometry read was attempted
	- this keeps the wow.exe-backed strictness for required geometry tables while avoiding false rejects from still-unmapped optional families on `0x104..0x107` models
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests were added or run
	- no runtime real-data validation was performed after the fix

### Mar 19, 2026 - Classic 0.5.3 MDX Regression Closed; 3.0.1 Still Open

- User runtime validation now confirms the classic Alpha `0.5.3` MDX rendering regression is fixed.
- Confirmed repair stack in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- direct-path replaceable fallback is restricted to `_isM2AdapterModel`
	- wrap/clamp interpretation is split between classic MDX and M2-adapted models
	- classic `Layer 0 + Transparent` once again always uses alpha-cutout
- A new direct-asset diagnostic path was added in `src/MdxViewer/AssetProbe.cs` and wired through `src/MdxViewer/Program.cs`:
	- `--probe-mdx` loads an asset from a real client path, prints parsed materials, and reports decoded BLP alpha statistics
	- this was used on `DuskwoodTree07.mdx` to prove the remaining canopy failure was in renderer behavior after decode, not in TEXS parsing or BLP decode
- Current status change:
	- classic `0.5.3` MDX should be treated as restored for the tested runtime sample
	- pre-release `3.0.1` rendering is still buggy and remains the active unresolved model-family track

### Mar 19, 2026 - PM4 Coordinate Validation Command

- Added `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateService.cs` as the first authoritative PM4 placement helper set in active core code.
- Added `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateValidator.cs` to validate transformed `MPRL` refs against real `_obj0.adt` placements from the fixed development dataset.
- Added CLI command: `wowmapconverter pm4-validate-coords [--input-dir <dir>] [--tile-limit <n>] [--threshold <units>] [--json <path>]`.
- Important scope limit:
	- this is a real-data validation path for `MPRL` only
	- it does not yet validate MSCN semantics
	- it does not yet build the cross-tile CK24 registry
- Validation status at this note:
	- initial real-data slice showed `MPRL` is already in ADT placement order, not tile-local
	- broadened sample run on 100 validated tiles reported 38,133 refs in expected tile bounds (100.0%) and 36,070 refs within 32 units of a nearest `_obj0.adt` placement (94.6%)
	- average nearest-placement distance on that sample was 10.86 units
	- broader work is still pending for CK24 aggregation and MSCN semantics

### Mar 20, 2026 - PM4 Viewer Overlay Diagnostics/Grouping/Winding Pass

- Added active PM4 overlay rendering + diagnostics in `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp.cs`.
- Added PM4 color modes for structural inspection (`CK24` type/object/key, tile, dominant group/attribute, height).
- Added optional PM4 3D markers (`MPRL` refs and object centroids).
- Added CK24 decomposition controls for disjoint geometry:
	- split by shared vertex connectivity
	- optional split by dominant `MSUR.MdosIndex` before connectivity
- Added per-object planar transform solve and winding parity correction:
	- candidate swap/invert U/V planar transforms scored against nearest `MPRL` anchors
	- mirrored parity now flips triangle winding order to avoid backward-wound faces
- Validation status:
	- repeated `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests were added or run
	- runtime real-data signoff still pending for merged/disjoint PM4 object cases
- Scope boundary:
	- this does not replace the pending map-level CK24 registry or finalize MSCN semantics
	- current PM4 reconstruction should be treated as viewer debug instrumentation + heuristics, not final export-grade identity mapping

## ✅ Working

### Mar 19, 2026 - 4.x Split ADT No-MCIN Fallback

- Real-data audit of the fixed `test_data/development/World/Maps/development` loose roots confirmed the current 4.x load failure is primarily a no-`MCIN` issue, not an `MCNK.IndexX/IndexY` issue:
	- 466 root ADT filenames
	- 114 zero-byte placeholders
	- 352 non-empty roots
	- 0 non-empty roots with `MCIN`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` now falls back to top-level `MCNK` scan order when a root ADT omits `MCIN`.
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/LkToAlphaConverter.cs` now uses the same root fallback so later split roots can flow into the existing Alpha conversion path instead of throwing immediately on missing `MCIN`.
- Scope limit:
	- this is a geometry/chunk-order recovery step first
	- full 4.x `_tex0.adt` texture-layer parity is still not claimed
	- the converter only consumes split texture companions when they expose LK-style `MCNK` payloads large enough for the current Alpha builder
- Validation status at this note:
	- code edits landed
	- build/runtime validation still pending after this patch

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
- **PM4 debug overlay (viewer-side)**: 🔧 In progress — color modes, 3D markers, CK24 split modes, and parity-aware winding fixes landed; runtime signoff still pending

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
