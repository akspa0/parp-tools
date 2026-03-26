# Active Context

## Mar 25, 2026 - wow-viewer Tool Inventory And Cutover Plan

- Added a concrete inventory and cutover document at `plans/wow_viewer_tool_inventory_and_cutover_plan_2026-03-25.md`.
- New planning decisions captured there:
	- first-class survivors are the main viewer shell, one converter CLI, one inspect CLI, one optional catalog CLI, and a real PM4 library plus workspace from day one.
	- do not port duplicate legacy executables as permanent apps; merge WoWMapConverter with still-useful WoWRollback or AlphaLkToAlpha conversion seams, merge the Alpha WDT inspectors, and keep DBCTool.V2 behavior only.
	- PM4 correction: current `MdxViewer` behavior is the de facto PM4 runtime reference implementation, and `Pm4Research` should be ported as the future `Core.PM4` library family because PM4 semantics are still under active research.
	- keep parpToolbox, PM4Tool, ADTPrefabTool, and the legacy WoWRollback GUI or viewer surfaces in `parp-tools` as archaeology or reference unless a specific algorithm is deliberately re-homed.
	- immediate follow-up planning docs now exist for bootstrap layout, CLI or GUI surfaces, and the PM4 library direction:
		- `plans/wow_viewer_bootstrap_layout_plan_2026-03-25.md`
		- `plans/wow_viewer_cli_gui_surface_plan_2026-03-25.md`
		- `plans/wow_viewer_pm4_library_plan_2026-03-25.md`
	- migration emphasis is now effectively `1, 3, 2`: bootstrap layout and project skeleton, then dual-surface tool design, then deeper PM4 library consolidation work.
- This plan refines `plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md` rather than replacing it.
- Validation status:
	- planning and documentation only
	- no viewer, converter, or renderer code changed in this slice

## Mar 25, 2026 - wow-viewer Initial Skeleton Created In Workspace

- A first-pass `wow-viewer/` scaffold now exists directly under the workspace root.
- Created projects:
	- `src/viewer/WowViewer.App`
	- `src/core/WowViewer.Core`
	- `src/core/WowViewer.Core.IO`
	- `src/core/WowViewer.Core.Runtime`
	- `src/core/WowViewer.Core.PM4`
	- `src/tools-shared/WowViewer.Tools.Shared`
	- `tools/converter/WowViewer.Tool.Converter`
	- `tools/inspect/WowViewer.Tool.Inspect`
- Added first-pass repo files:
	- `WowViewer.slnx`
	- `Directory.Build.props`
	- `Directory.Packages.props`
	- `eng/Version.props`
	- `scripts/bootstrap.ps1`
	- `scripts/bootstrap.sh`
	- `scripts/validate-real-data.ps1`
- PM4-specific rule carried into the scaffold:
	- `Core.PM4` exists from day one
	- the placeholder code explicitly treats `MdxViewer` as the PM4 runtime reference and `Pm4Research` as the PM4 library seed
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- this is only a structure lock and placeholder-code build, not a real code-port or runtime signoff

## Mar 25, 2026 - First PM4 Code-Port Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first real PM4 code port from `src/Pm4Research.Core`.
- Landed pieces:
	- typed chunk models for the trusted PM4 chunk set
	- `Pm4ResearchDocument`
	- `Pm4ResearchReader`
	- `Pm4ResearchSnapshotBuilder`
- Important boundary:
	- this is still a raw research-facing PM4 reader layer
	- current `MdxViewer` behavior remains the runtime PM4 reference implementation for reconstruction, grouping, transforms, and viewer-facing semantics
	- no viewer PM4 logic has been re-homed onto `Core.PM4` yet
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after the PM4 port
	- no runtime validation or app integration has happened yet

## Mar 25, 2026 - PM4 Inspect Verbs Now Work In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first single-file PM4 analyzer and report layer on top of the earlier reader port.
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` now has working PM4 commands:
	- `pm4 inspect --input <file.pm4>`
	- `pm4 export-json --input <file.pm4> [--output <report.json>]`
- Smoke-test result on the fixed reference tile:
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` succeeded
	- output included version `12304`, `54` chunks, `6318` `MSVT` vertices, `9990` `MSCN` points, and `2493` `MPRL` refs for `development_00_00.pm4`
- Important boundary:
	- this is still single-file research analysis, not viewer reconstruction or PM4 correctness closure
	- current `MdxViewer` behavior remains the runtime PM4 reference implementation

## Mar 25, 2026 - PM4 Audit And Placement Contracts Follow-Up

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first decode-audit path plus the first extracted MdxViewer-facing PM4 placement-contract seam.
- Landed pieces:
	- `Pm4ResearchAuditAnalyzer` with single-file and directory-level decode or corpus audit entry points
	- `WowViewer.Tool.Inspect` verbs for `pm4 audit --input <file.pm4>` and `pm4 audit-directory --input <directory>`
	- shared `Pm4AxisConvention`, `Pm4CoordinateMode`, `Pm4PlanarTransform`, `Pm4CoordinateService`, and `Pm4PlacementContract`
- New research note captured in the inspect layer:
	- CK24 low-16 object values, read as integers, appear to be plausible `UniqueID` candidates on the development map, but this remains a hypothesis until correlated against real placed-object data
- Important boundary:
	- this is still not the full MdxViewer PM4 reconstruction or transform solver port
	- current `MdxViewer` behavior remains the runtime reference implementation
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after this slice
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit-directory --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026 and scanned `616` PM4 files with no unknown chunks or diagnostics
	- early audit findings worth keeping visible:
		- `MDOS.buildingIndex->MDBH` shows real invalid references in the development corpus
		- `MSLK.RefIndex->MSUR` also shows corpus-level mismatches in nontrivial counts, which supports keeping linkage interpretation labeled as research

## Mar 25, 2026 - Post-v0.4.5 Branch And Roadmap Prompt Bundle

- Post-release planning is now intentionally split onto branch `feature/v0.4.6-v0.5.0-roadmap` so the next milestone work can stay isolated from `main` until the first real slices are ready.
- Detailed Copilot prompt assets for the `wow-viewer` tool-suite/library refactor now live under workspace `.github/prompts/`, not under `gillijimproject_refactor/plans`.
- For this tool-suite migration work, treat `gillijimproject_refactor/plans` as scratchpad/archeology notes and `.github/prompts/` as the canonical prompt surface.
- Current dedicated prompt set:
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-bootstrap-layout-plan.prompt.md`
	- `.github/prompts/wow-viewer-shared-io-library-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-inventory-cutover-plan.prompt.md`
	- `.github/prompts/wow-viewer-cli-gui-surface-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-migration-sequence-plan.prompt.md`
- New prompt bundle captured under `plans/` for the next branch of work:
	- `post_v0_4_5_plan_set_2026-03-25.md`
	- `v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md`
	- `wowrollback_uniqueid_timeline_prompt_2026-03-25.md`
	- `alpha_core_sql_scene_liveness_prompt_2026-03-25.md`
	- `viewer_performance_recovery_prompt_2026-03-25.md`
	- `v0_5_0_new_repo_library_migration_prompt_2026-03-25.md`
	- `v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`
- Current intended milestone split:
	- `v0.4.6` should carry the first visible WoWRollback / `UniqueID` timeline filter slice inside the active viewer, plus Alpha-Core SQL caching/fidelity follow-up and an initial performance recovery pass.
	- `v0.5.0` should move into `https://github.com/akspa0/wow-viewer` as the new production repo with one canonical shared library plus split viewer/tool consumers.
- Important boundaries for future sessions:
	- keep WoWRollback integration on the active viewer UI/data-loading path; do not drift back to the older separate web-viewer plan as the primary delivery target.
	- treat `parp-tools` as the R&D / archaeology repo and `wow-viewer` as the intended production home for the next major milestone.
	- external constructive guidance now explicitly supports a sane top-level `wow-viewer` layout: the main renderer app should have one obvious root, with libraries/dependencies/tools split into their own clear folders instead of repeating the current nested sprawl.
	- latest user constraint: fully refactor and re-own the first-party read/parse/write/convert stack, including current base libraries such as `gillijimproject-csharp`; keep upstream projects like `Warcraft.NET`, `DBCD`, `WoWDBDefs`, `Alpha-Core`, `WoWTools.Minimaps`, and `SereniaBLPLib` under `libs/` and track their original repos where practical.
	- repo bootstrap should automatically pull support repos like `wow-listfile` instead of relying on manual setup.
	- possible targeted integrations worth evaluating later include `MapUpconverter`, `ADTMeta`, `wow.export`, and `wow.tools.local`, but they should support the owned-library plan rather than replace it.
	- possible future upstream work on `Noggit` / `noggit-red` alpha-era support is interesting, but should stay an explicit stretch/outreach track rather than replacing the main `wow-viewer` migration target.
	- a concrete first-pass repo tree and migration order draft now exists in `plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`; future planning should refine that draft rather than re-deriving repo shape from scratch.
	- treat Alpha-Core SQL equipment correctness, animation-state handling, and pathing as separate seams.
	- do not assume SQL or PM4 already prove server-like NPC pathing; that remains a later research seam, not an implicit short-term deliverable.
	- performance recovery is now a first-class dependency, but the deeper overhaul should be planned against the new repo/library split instead of indefinite surgery inside the R&D tree.
- Documentation follow-up on the same slice:
	- root `README.md` was refreshed again to make the active support headline, conversion coverage, WMO `v14/v16/v17` handling, and built-in tooling more explicit.
	- screenshot reality remains unchanged: asset-catalog screenshot automation exists already, but a curated world/UI gallery is still future work.
- Validation status:
	- planning/documentation only
	- no viewer, converter, or renderer code changed in this slice

## Mar 24, 2026 - WMO Vertex-Light Prototype In Active Viewer

- First renderer-side object-lighting prototype is now in the active tree at `src/MdxViewer/Rendering/WmoRenderer.cs`.
- Scope of the implementation:
	- WMO group vertex buffers now carry a fourth attribute for baked vertex-light color.
	- `WmoRenderer` now prefers parsed `MOCV` vertex colors when they look usable.
	- if usable `MOCV` is missing but preserved v14 lightmap payloads exist (`MOLV` / `MOLD` / `MOLM`), the renderer now samples those on load into per-vertex baked-light modulation colors.
	- the fragment shader now modulates the existing diffuse/fog path by that baked-light color, so WMOs can show preserved object-light contribution instead of relying only on the generic ambient+directional path.
- Important limit:
	- this is not full `0.5.3` / early-client object-lightmap parity.
	- there is still no client-faithful group/batch lightmap texture pipeline, no recovered batch-to-lightmap index path, and no dedicated `RenderGroupLightmap` / `RenderGroupLightmapTex` analogue in the active renderer.
	- this is a first prototype using the data the active model already preserves.
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed on Mar 24, 2026 after the change.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on affected WMOs.

## Mar 24, 2026 - 0.5.3 Terrain/Object Render Fast-Path And Viewer Perf Gap

- Reverse-engineering follow-up against the symbolized `0.5.3` client materially tightened the current performance/parity story; no viewer code changed in this slice.
- durable write-up extended in `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- high-confidence `0.5.3` terrain findings from decompilation:
	- `CreateRenderLists` (`0x00698230`) is a real precompute step that builds terrain texcoord tables and batch/render-list data instead of leaving chunk draw setup entirely to the frame loop
	- `RenderLayers` (`0x006a5d00`) and `RenderLayersDyn` (`0x006a64b0`) use locked GX buffers plus prebuilt chunk batches, not a fully generic per-layer rebuild path
	- terrain already has shader-assisted paths in `0.5.3`: the chunk draw path binds `CMap::psTerrain` / `CMap::psSpecTerrain` plus `shaderGxTexture` when terrain/specular shader support is enabled
	- terrain layer count is reduced by distance (`textureLodDist` can clamp the runtime draw to one layer), and the dynamic path also fades diffuse alpha before collapse
	- per-layer moving-texture behavior is confirmed in the terrain path itself: when runtime layer flag `0x40` is set, `RenderLayers` / `RenderLayersDyn` apply an extra texture transform indexed by low flag bits into the time-varying world transform tables updated by `FUN_006804b0`
	- terrain shadows are drawn as a separate modulation pass rather than being flattened into one generic terrain blend loop
- high-confidence `0.5.3` object/light findings from decompilation:
	- `RenderMapObjDefGroups` (`0x0066e030`) walks visible `CMapObjDefGroup` lists, sets transforms once per group, and dispatches `CMapObj::RenderGroup(...)`; this is more structured than the active viewer's generic instance loops
	- `CreateLightmaps` (`0x006adba0`) allocates per-group lightmap textures (`256x256`) and registers `UpdateLightmapTex`, which strongly supports a dedicated object-lightmap path in the client
	- `RenderGroupLightmap(...)` uses dedicated group lightmap vertex streams and batch-local lightmap texture binding rather than one generic object UV/material path
	- `RenderGroupLightmapTex(...)` splits the lightmap composition work into dedicated subpasses with lighting forced off, and `UpdateLightmapTex(...)` exposes row-stride plus CPU memory on `GxTex_Latch`; taken together, the object lightmap path is a real rendering subsystem, not just a texture on the generic WMO path
	- `CalcLightColors` (`0x006c4da0`) computes a much richer lighting state than the active viewer currently models: direct, ambient, six sky channels, five cloud channels, four water channels, fog end, fog-start scalar, and storm blending
- viewer-side implication from the same slice:
	- the active viewer remains structurally flatter than the client in the exact places that matter for both performance and fidelity:
		- `StandardTerrainAdapter` still actively uses `MPHD` only for big-alpha/profile selection and still flattens `MAIN` entries to boolean tile existence
		- `TerrainRenderer` is still a generic base+overlay pass loop that only interprets `MCLY 0x100`; it has no terrain shader-family split, no per-layer motion support, no layer-count LOD collapse, and no specular terrain path
		- `LightService` remains a simplified nearest-zone DBC interpolator rather than a full terrain/object/sky/runtime-light system
		- `WmoRenderer` / `MdxRenderer` still rely on shared generic shader families instead of the client's stronger specialization
		- `WorldScene` hot paths remain heavy: MDX transparent items are re-collected/sorted every frame, optional PM4 forensic budgets are still `int.MaxValue`, and the current render-queue abstraction is not yet the active world submission path
- practical priority order now supported by evidence:
	1. preserve `MAIN` / `MPHD` / `MCLY` semantics as first-class runtime metadata
	2. split terrain renderer responsibilities into fallback vs client-faithful material/shader path
	3. treat object/lightmap parity as a separate seam from terrain lighting
	4. reduce generic hot-path state churn before layering on more fidelity features
	5. use the existing `WorldAssetManager` read/path-probe counters as the basis for an explicit scene residency/prefetch policy
- validation status:
	- reverse engineering plus code audit only; no viewer build or runtime signoff was produced by this slice

## Mar 24, 2026 - WoW 2.0.0 Beta Ghidra Recon For M2 / Light / Particle Risk

- Static reverse-engineering pass only against a loaded beta `2.0.0` `WoW.exe` in Ghidra. No viewer/converter code changed in this slice.
- durable write-up: `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- High-confidence findings from decompilation:
	- `Model2` has an explicit BLS shader bootstrap in `FUN_00717b00` (`M2Cache.cpp` path string present) and loads both `shaders\vertex\Model2.bls` and `shaders\pixel\Model2.bls`.
	- map objects preload a dedicated bank of pixel BLS programs in `FUN_006b3b20`, including `MapObjOverbright`, `MapObjSpecular`, `MapObjMetal`, `MapObjEnv`, `MapObjEnvMetal`, `MapObjExtWater0`, `MapObjTransDiffuse`, and `MapObjTransSpecular`.
	- `M2Light.cpp`-anchored logic in `FUN_0072d1a0` does not treat model lights as a flat passive list: lights are inserted either into a spatial bucket structure or a general linked list depending on runtime mode/type, and companion mutators (`FUN_0072cc60`, `FUN_0072cc90`, `FUN_0072cdc0`) relink them when state/position changes.
	- particle runtime is a real engine-side system, not just file payload playback: `FUN_007c26c0` bootstraps `CParticleEmitter2_idx` and global pools, while `FUN_007ca9d0` / related constructors copy emitter payload regions into runtime `CParticle2` / `CParticle2_Model` objects.
	- the `Light*.dbc` family is loaded through strict `WDBC` schema-checked table loaders with ID-index maps, not ad-hoc parsing. Confirmed table shapes:
		- `LightFloatBand.dbc` and `LightIntBand.dbc`: `0x22` columns, `0x88` row size, two `0x40`-byte band payloads plus two leading scalars.
		- `LightParams.dbc`: `9` columns, `0x24` row size.
		- `Light.dbc`: `0xc` columns, `0x30` row size with a trailing `0x14`-byte block.
		- `LightSkybox.dbc`: `2` columns, `8` byte rows with string-table resolution.
- Practical viewer risk guidance from this RE pass:
	- do not collapse early/later `2.x` materials into one generic shader path if the goal is parity; the client uses distinct BLS programs for `Model2` and multiple map-object material families.
	- do not expect smoke / particle projection issues to close from parser tweaks alone; the particle and light systems are runtime-managed and likely need render-path/state investigation in addition to format parsing.
	- terrain follow-up is now split into two separate engine tracks:
		- cached per-layer terrain programs are now pinned down more precisely:
			- `terrain1..4` at `DAT_00caf304..310` are the one-pass layer-count table used when `DAT_00cb3594 == 0` and `DAT_00ca31b8 != 0`
			- `terrain1_s..4_s` at `DAT_00caf548..554` are the alternate one-pass layer-count table used when `DAT_00cb3594 != 0`
			- `terrainp` / `terrainp_s` belong to the slower manual terrain fallback path in `FUN_006cee30`, not the cached layer-count table
			- `terrainp_u` / `terrainp_us` are loaded at startup but are still untraced in an active draw branch
			- terrain also has a separate time-varying layer-transform path: `FUN_006c00f0` copies a source layer flag field into each runtime layer object, `FUN_006cee30` / `FUN_006cf590` apply an extra transform when bit `0x40` is present, and `FUN_006804b0` updates the transform tables every world tick
		- `XTextures\slime\slime.%d.blp` resolves into an animated `WCHUNKLIQUID` surface path, not yet proven to be a terrain diffuse-layer effect
		- latest `WCHUNKLIQUID` pass shows a real mode dispatcher: `FUN_006c65b0` splits modes `0/4/8` into animated texture-family rendering and modes `2/3/6/7` into a direct-coordinate/UV-style path
		- `FUN_006c65b0` passes the raw mode nibble into `FUN_0069b310`, so the liquid mode is also the animated family index
		- currently recovered family table entries:
			- `0 -> lake_a`
			- `1 -> ocean_h`
			- `2 -> lava`
			- `3 -> slime`
			- `4 -> lake_a` again
		- novelty/dead-content candidates:
			- `FUN_0069e690(2)` currently reaches `FUN_0069b310(6)`, but the family slot is still unresolved via data xrefs
			- `XTextures\river\fast_a.%d.blp` exists in strings but is not in the traced active family table
	- viewer-side audit against the active tree shows terrain flag under-parsing is real:
		- `StandardTerrainAdapter` currently uses `MPHD` only for big-alpha selection
		- `ReadMainChunk(...)` treats any non-zero `MAIN` entry as generic tile presence instead of keeping entry semantics like `has ADT` vs `all water`
		- raw `MCLY` flags are preserved into `TerrainLayer.Flags`, but `TerrainRenderer` only interprets `0x100` as the implicit-alpha hint
	- the dangerous seam for `2.x` support is downstream interpretation of light/material/particle IDs and runtime state, not raw DBC ingestion.
- Validation status:
	- reverse engineering only; no automated tests, no solution build, and no runtime real-data signoff were performed in this slice.

## Mar 24, 2026 - 0.12 Standalone Model Browser Recovery

- The latest standalone-model regression for the `0.12` client split into two separate seams in the active viewer:
	- `MpqDataSource` was no longer indexing Alpha-style nested model wrappers at all (`.mdx.MPQ`, `.mdl.MPQ`, `.m2.MPQ`), and it also skipped loose `.mdl` files entirely.
	- standalone `MD20` / `MD21` routing in `ViewerApp.LoadM2FromBytes(...)` still allowed an unsupported build with no resolved `M2Profile` to continue into the M2-family adapter path instead of failing cleanly.
- Root cause now fixed in the active tree:
	- `src/MdxViewer/DataSources/MpqDataSource.cs`
		- loose-file indexing now includes `.mdl`
		- Alpha nested wrapper scan now includes model wrappers (`.mdx.MPQ`, `.mdl.MPQ`, `.m2.MPQ`)
		- model wrappers now register extension aliases into the file set / Alpha wrapper cache so the browser and path resolver can find the same wrapped asset through `.mdx`, `.mdl`, or `.m2`
	- `src/MdxViewer/ViewerApp.cs`
		- the standalone browser's `.mdx` filter now aggregates early model files from both `.mdx` and `.mdl`
		- disk loads now accept `.mdl` through the same container-probe path already used by the data-source loader
		- `LoadM2FromBytes(...)` now hard-fails with a clear unsupported-build error when no `M2Profile` resolves for the active client build instead of continuing into an unsafe best-effort adapter path
	- `src/MdxViewer/ViewerApp_Sidebars.cs`
		- the file-browser type label now reflects that the early-model bucket is `.mdx/.mdl`
- Scope boundary:
	- this fix restores file discovery/indexing and turns the unsupported `.m2` route into a safe load failure for pre-M2 builds; it is not proof that standalone `0.12` runtime model rendering is fully signed off across a real client dataset.
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed on Mar 24, 2026 after this fix.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on actual `0.12` client browsing/loading because no fixed `0.12` data path is currently recorded in `memory-bank/data-paths.md`.

## Mar 24, 2026 - 0.6.0 Through 2.x Terrain Alpha Grid Regression Fix

- The terrain grid-pattern regression affecting standard ADT clients from `0.6.0` through the `2.x` era was not a newly proven shader/blend-style difference. The active viewer was still decoding that whole legacy band through a naive sequential 4-bit MCAL unpack path in `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`.
- Root cause now fixed in the active tree:
	- `StandardTerrainAdapter.ExtractAlphaMaps(...)` for `TerrainAlphaDecodeMode.LegacySequential` now prefers the relaxed MCAL path (`Mcal.GetAlphaMapForLayerRelaxed(...)`) and preserves `DoNotFixAlphaMap` behavior.
	- the old naive legacy fallback now routes through the existing row-aware 4-bit decode + legacy edge-fix helpers instead of writing raw nibble pairs straight into the `64x64` output.
- Scope boundary:
	- this change is limited to the standard-terrain legacy band (`0.6.0` through `2.x`) and does not change the separate `AlphaTerrainAdapter` path for `0.5.x` or the strict `3.x` / Cataclysm `4.0.0` decode branches.
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed on Mar 24, 2026 after this fix and after correcting unrelated compile breaks in the in-progress minimap candidate-path patch.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on affected `0.6.0` / `0.7.0` / `0.8.0` / `0.9.0` / `1.x` / `2.x` terrain tiles.

## Mar 24, 2026 - v0.4.5 Branding + MH2O LiquidType Classification Fix

- Active viewer branding/release metadata is now aligned toward `parp-tools WoW Viewer` version `0.4.5` without renaming the `MdxViewer` root namespace.
- Current user-facing changes in the active tree:
	- viewer window title now uses `parp-tools WoW Viewer`
	- Help -> About now opens a modal with author + credits instead of only writing a transient status line
	- project metadata now emits `ParpToolsWoWViewer` as the executable/assembly name
	- `.github/workflows/release-mdxviewer.yml` now packages/releases `parp-tools-wow-viewer-<version>-win-x64.zip` and uses the .NET 10 SDK required by the active project target
- MH2O follow-up on the same slice:
	- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` now classifies `MH2O` liquids from `LiquidType.dbc -> Type` when DBC metadata is available for the active client build
	- when DBC loading is unavailable or an ID is missing from the loaded table, the viewer now falls back to an expanded static family map that includes the real 3.3.5 / 4.0 IDs already used elsewhere in the repo (`13`, `14`, `17`, `19`, `20`)
	- `src/WoWMapConverter/WoWMapConverter.Core/Formats/Liquids/LiquidConverter.cs` now recognizes those late-style IDs in the shared `LiquidTypeId -> MCLQ family` fallback path as well
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed on Mar 24, 2026
	- no automated tests were added or run
	- no runtime real-data signoff yet on 3.3.5 / 4.0 liquid visual parity; the build only proves the implementation compiles

## Mar 25, 2026 - Fullscreen Minimap Release Blocker Closed For v0.4.5

- The fullscreen/docked minimap repair is now treated as closed for `v0.4.5` after the final transpose-only follow-up and runtime user confirmation on the fixed development minimap dataset.
- Final landed behavior in the active tree:
	- the bad `WoWConstants.TileSize` minimap hypothesis stays reverted; the active `64x64` minimap grid continues to use `WoWConstants.ChunkSize`
	- the broad world-axis swap attempted during the first Designer Island follow-up was backed out
	- the landed fix instead keeps the direct world/click mapping and only transposes the screen-space marker placement seam that had drifted away from the drawn tile grid
	- docked and fullscreen minimap now agree well enough for the user to describe the bug as fixed after runtime checking the top-right Designer Island scenario
- Practical release consequence:
	- the fullscreen minimap is no longer an open `v0.4.5` blocker
	- remaining minimap work should be treated as future polish or new regressions, not as justification to keep `v0.4.5` open
- Validation status:
	- build plus targeted runtime user signoff: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-transpose-repair/"` passed on Mar 25, 2026 after the final transpose-only repair
	- runtime user feedback then confirmed the repaired minimap behavior on the fixed development minimap dataset
	- no automated tests were added or run
	- this is not broad automated minimap coverage; it is targeted real-data runtime confirmation for the previously broken release-blocker scenario

## Current Focus: v0.4.0 Recovery Branch (Mar 17, 2026)

Working branch is now reset in the main tree, not only in side worktrees.

- Branch: recovery/v0.4.0-surgical-main-tree
- Baseline tag/commit: v0.4.0 / 343dadf
- .github metadata restored from main and committed: 845748b
- .github restore was pushed to origin/recovery/v0.4.0-surgical-main-tree

### Tooling Path Reuse + Unified Format I/O Proposal (Mar 23)

- Viewer tool dialogs should stop forcing repeated folder browsing when the session already knows the active base client and loose overlay roots.
- Current viewer-side behavior now seeds tool inputs from the active session where practical:
	- `Generate VLM Dataset` pulls the active MPQ base client path and current map name.
	- `Terrain Texture Transfer` prefers the attached loose-overlay map directory as source and the base-client map directory as target when those roots exist.
	- `Map Converter` now seeds WDT/map-directory inputs from the currently loaded local WDT when available, otherwise from the current map under the active loose/base roots.
	- `WMO Converter` still seeds from the currently loaded standalone WMO when applicable.
- Important scope limit:
	- this is UI/tool input seeding only, not proof that all downstream conversion paths are correct for Alpha, LK 3.3.5, or 4.x data.
	- after the Mar 23 seeding follow-up, edited-file diagnostics were clean on `src/MdxViewer/ViewerApp.cs`, but no new full viewer build or runtime signoff was recorded yet for this slice.
- Larger project direction requested by the user:
	- consolidate terrain, ADT/WDT, M2/MDX, and WMO read/write knowledge into one shared library used by viewer, converter, and tooling instead of continuing to split capabilities across `MdxViewer` and `WoWMapConverter.Core`.
	- do not assume the existing map converter is already closed for Alpha placement writing: MODF/MDDF downconversion for Alpha WDT remains an explicit open seam until reimplemented and validated.
	- planning prompt captured in `plans/unified_format_io_overhaul_prompt_2026-03-23.md`.
	- new PM4 planning guardrail from Mar 24 viewer forensics/UI work:
		- the practical viewer hierarchy is `CK24 -> MSLK-linked subgroup -> optional MDOS subgroup -> connectivity part`
		- PM4 centroids are useful derived display anchors for those nodes, not proven raw PM4 node records
		- `MSUR.AttributeMask` colors should be surfaced as explicit value legends, but their semantics remain open and must not be hardcoded into format contracts prematurely

### Documentation Refresh + Render Quality Follow-Up (Mar 23)

- Repo-level docs were refreshed, but the first pass still contained bad assumptions.
- The user then rewrote `src/MdxViewer/README.md` to be more grounded and truthful.
- Current documentation/handoff rule:
	- treat the user-corrected viewer README as the authoritative public summary for support and usage claims
	- do not reintroduce speculative platform restrictions or inflated support statements without direct evidence
	- do not write branch-local language into README text intended for eventual `main`
- Important current README claims to preserve in future sessions:
	- support headline: `0.5.3` through `4.0.0.11927`
	- later `4.0.x` ADT support exists
	- later split-ADT support through `4.3.4` exists but remains explicitly untested
	- Alpha-Core SQL world NPC/gameobject support is relevant to the README and should not be dropped casually
	- asset-catalog screenshot automation exists already; broader UI/menu showcase capture is still future work
- Validation status:
	- docs were updated after the Mar 23 viewer build had already passed
	- the documentation update itself adds no runtime validation and should not be read as new visual signoff

### Viewer Debug/Workflow Follow-Up (Mar 22)

- Latest viewer-side work moved away from treating PM4 runtime streaming as the only inspection path.
- Current additions in the active tree:
	- PM4 offline OBJ export from `src/MdxViewer/Terrain/WorldScene.cs`, surfaced through `ViewerApp_Pm4Utilities.cs`, so per-tile/per-object PM4 geometry can be compared outside the live overlay window.
	- minimap interaction/caching follow-up in `ViewerApp_MinimapAndStatus.cs`, `ViewerApp.cs`, and `Rendering/MinimapRenderer.cs`:
		- teleport now requires triple-clicking the same tile instead of a single short click
		- minimap zoom/pan/window state now persist in viewer settings
		- decoded minimap tiles now cache on disk under `output/cache/minimap/<cache-segment>`
	- terrain-hole debug override in `TerrainMeshBuilder`, `TerrainManager`, `VlmTerrainManager`, and `ViewerApp_Sidebars.cs`:
		- viewer can ignore terrain hole masks globally or on the current camera tile by rebuilding loaded chunk meshes only
		- source ADT hole flags are unchanged; this is viewer-side inspection only
- Validation status:
	- file diagnostics were clean on the edited viewer files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 22, 2026 after these viewer-side follow-ups were in the active tree
	- no automated tests were added or run
	- no runtime real-data signoff yet on PM4 OBJ correctness, minimap feel/cache benefit, or terrain-hole rebuild behavior while streaming

### Standalone PM4 Research Library (Mar 21)

- Added a new isolated project at `src/Pm4Research.Core` for fresh PM4 format work outside the current viewer/converter reconstruction path.
- Current scope of that library:
	- raw chunk walking with preserved signatures, offsets, sizes, and payload bytes
	- standalone typed decoding for `MVER`, `MSHD`, `MSLK`, `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR`, `MSCN`, `MPRL`, `MPRR`, `MDBH`, `MDBI`, `MDBF`, `MDOS`, and `MDSF`
	- lightweight exploration snapshot generation for counts and chunk bounds
	- raw decode-audit reporting for per-file and corpus-wide chunk consistency and cross-chunk reference checks
- Important boundary:
	- no viewer/world transform policy
	- no CK24 object reconstruction
	- no dependency on `MdxViewer` PM4 solver code or the current `WoWMapConverter.Core` PM4 models
- Preferred real-data reference tile for PM4 rediscovery:
	- use `test_data/development/World/Maps/development/development_00_00.pm4` first when checking raw chunk assumptions or viewer-forensics hypotheses
	- Mar 21 standalone analysis on that tile showed it is a dense PM4 file, not a degenerate edge case: `54` chunks, `MSPV=8778`, `MSVT=6318`, `MSCN=9990`, `MPRL=2493`
	- new Mar 21 audit result: `00_00` is also the only currently populated destructible-building payload tile in the in-repo development PM4 corpus; `MDBI` and `MDBF` are one-tile only, while `MDBH` / `MDOS` / `MDSF` mostly appear as empty or placeholder stubs elsewhere
	- the matching original ADTs are not present in this repo, so in-repo validation is currently PM4-side only; external visual cross-checks should still prefer this tile because the user has the trusted ADT placements for it
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Core/Pm4Research.Core.csproj -c Debug` PASSED on Mar 21, 2026.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and found zero file-walk/stride diagnostics across the 616-file corpus, but did surface `MSLK.RefIndex -> MSUR` mismatches in aggregate and the Wintergrasp-only destructible payload split described above.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and narrowed that open seam further: `150` files carry `4553` mismatches, `development_00_00.pm4` carries zero mismatches, and the bad values almost never fit `MPRL` counts but often still fit `MSLK`, `MSPI`, `MSVI`, and `MSCN` counts on the affected tiles.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and materially tightened the identity/hierarchy seam: the UI `Ck24ObjectId` is just the low 16 bits of `MSUR.PackedParams -> CK24`, it is almost always one-to-one with a full CK24 within a file (`2` reuse cases out of `1601` analyzed non-zero object-id groups), and `MSLK.GroupObjectId` remains very weak as the missing hierarchy/ownership key for the unresolved `RefIndex` population (`16` low16 matches and `15` low24 matches across `4553` mismatches).
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and materially tightened the MSCN seam: `MSUR.MdosIndex -> MSCN` is strong (`511891` fits, `6201` misses), `1886 / 1895` CK24 groups carry MSCN coverage, and in the standalone raw path raw MSCN bounds overlap CK24 mesh bounds far more often than swapped-XY MSCN bounds (`1162` vs `10` fits). Current standalone corpus evidence does not support the older blanket claim that MSCN is simply world-space plus XY swap.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-msur-geometry --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_msur_geometry_report.json` PASSED on Mar 21, 2026 and materially tightened a major decoder-trust seam: all `518092` analyzed `MSUR` surfaces had unit-length stored normals with strong positive alignment to geometry-derived polygon normals, and the trailing float currently named `Height` behaves like the negative plane-distance term along that normal (best candidate mean absolute error `0.00367829`).
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex-classifier --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_mslk_refindex_classifier_report.json` PASSED on Mar 21, 2026 and replaced the old all-or-nothing mismatch story with family buckets: `505` mismatch families are now classified beyond pure ambiguity, covering `2651` of `4553` mismatch rows, with the largest resolved family population currently landing in `probable-MSVT` plus smaller `MSPI` / `MSPV` / `MSVI` / `MSCN` / `MPRL` slices.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-structure-confidence --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and is now the explicit decode-trust guardrail for the standalone PM4 path: `13` tracked chunk families currently land in `high` layout confidence, but field semantics are much weaker (`1` high, `4` medium, `10` low, `4` very-low). The main hallucination-risk zone is semantic over-closure, not raw stride parsing.
	- refreshed `scan-structure-confidence` result after the new audits: field semantics are still weaker than layout confidence, but the picture improved materially (`2` high, `4` medium, `9` low, `4` very-low). Current highest-risk zones are `MSLK.RefIndex`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible fields; `MSUR` bytes `4..19` are no longer in that top-risk bucket.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- pm4-validate-coords --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --json i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_pm4_coordinate_validation_report.json` PASSED on Mar 21, 2026 and materially strengthened `MPRL` against real placement truth on the fixed dataset: `206` tiles validated, `114301 / 114301` refs inside expected tile bounds (`100.0%`), `107907 / 114301` refs within `32` units of a nearest `_obj0.adt` placement (`94.4%`), average nearest placement distance `10.98`. This helps `MPRL`, not `MPRR`.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and now serves as the main corpus-scale PM4 unknowns map: it records verified raw edges, partial fits, field distributions, and open proof tasks in one place.
	- structure-confidence highlights to preserve for future PM4 work:
		- strongest byte+semantic anchors: `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR` plane fields, `MSUR -> MSVI`, and `MDSF -> {MSUR, MDOS}`
		- highest hallucination-risk fields: `MSLK.RefIndex`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible payload fields such as `MDOS.buildingIndex`
		- explicit conflict inventory now exists for overstated legacy claims around `MSLK.LinkId`, `MSLK.RefIndex`, `MSUR.MdosIndex`, `MSUR.Normal + Height`, MSCN coordinate frame, and `MPRR.Value1`
	- no automated tests were added or run.
	- no real-data runtime signoff exists yet because this is a standalone decode/exploration foundation, not an integrated viewer fix.

### M2 Material Parity Slice: Explicit Env-Map + UV Selector Recovery (Mar 21)

### Archive I/O Performance Slice: Read-Path Probe Reduction + Useful Prefetch Instrumentation (Mar 21)

### ViewerApp Partial-Class Refactor (Mar 21)

- `src/MdxViewer/ViewerApp.cs` was reduced by extracting cohesive UI domains into partial-class files instead of doing a behavior rewrite:
	- `src/MdxViewer/ViewerApp_ClientDialogs.cs`
	- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- `src/MdxViewer/ViewerApp_Sidebars.cs`
- The goal of this slice is maintainability only: keep existing viewer behavior while shrinking the single 6000+ line shell file and making future UI changes more localized.
- Current limit of the extraction:
	- the large world-objects body still lives behind `DrawWorldObjectsContentCore()` in `ViewerApp.cs`; the refactor did not attempt a full inspector redesign in this pass.
- Validation status for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 after the split.
	- no automated tests were added or run.
	- no runtime real-data validation was done because this change is structural, not a terrain/data-path behavior fix.

### Viewer UI / Perf Slice: Hideable Chrome + Clipped Long Lists (Mar 21)

### Viewer UI Follow-Up: Dockspace Host + Dockable Side Panels (Mar 21)

### Viewer PM4/WMO Correlation Export (Mar 21)

- `MdxViewer` now exposes a viewer-side PM4/WMO correlation export in the existing `PM4 Alignment` window.
- Current implementation:
	- `ViewerApp_Pm4Utilities.cs` adds `Dump PM4/WMO Correlation JSON` next to the existing PM4 object dump.
	- `WorldScene.BuildPm4WmoPlacementCorrelationJson(...)` exports loaded ADT WMO placements, parsed WMO mesh summaries, and top nearby PM4 overlay object candidates per placement.
	- `WorldAssetManager` now exposes `WmoMeshSummary`, reusing the existing WMO v14/v17 parsing path to capture local bounds plus group/vertex/index/triangle counts without depending on a renderer instance.
- Scope / limit:
	- this is a correlation/export utility, not closure on PM4-to-WMO semantic identity.
	- current matching is still heuristic, but it is no longer AABB-only: ranking now uses transformed WMO footprint samples versus PM4 footprint hulls in addition to bounds-gap / overlap metrics and PM4 object metadata.
- Follow-up now landed on top of the export path:
	- `ViewerApp_Pm4Utilities.cs` now adds a real `PM4/WMO Correlation` window with refresh/filter controls, placement browsing, candidate inspection, PM4 selection, and camera framing actions.
	- `WorldScene` now exposes a typed PM4/WMO correlation report for viewer use instead of forcing the UI to go through JSON only.
	- `WorldScene.SelectPm4Object(...)` lets the panel drive live PM4 selection from a reported candidate row.
	- `WorldAssetManager.WmoMeshSummary` now caches sampled WMO geometry points so the correlation path can compare transformed footprint shape instead of only transformed bounds.
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 after the interactive panel + footprint follow-up, with existing warnings.
	- no automated tests were added or run.
	- no runtime real-data signoff was performed yet for the new panel workflow or the footprint-based ranking changes.

- Latest user feedback after the clipped-list shell pass: `World Maps` starting collapsed was wrong, and the viewer still did not have a real dock-panel UI.
- Current correction in `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- ImGui docking is now explicitly enabled in source instead of relying on stale layout state in `imgui.ini`.
	- the viewer now creates a real central dockspace host between the menu/toolbar region and the status bar.
	- the old fixed left/right sidebars can now render as normal dockable windows (`Navigator` and `Inspector`) when dock panels are enabled from the `View` menu.
	- `World Maps` now defaults open again on first draw.
	- scene viewport math no longer subtracts fixed sidebar widths, which was incompatible with docked/floating panels.
- Validation status for this follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on the docking workflow or interaction feel; do not over-claim the UI recovery from build success alone.

- Latest user priority shifted from PM4 transform tuning to viewer usability and frame-time friction while debugging PM4.
- Current implementation in `src/MdxViewer/ViewerApp.cs` is intentionally incremental, not a dockspace/UI-shell rewrite:
	- `Tab` now toggles a hide-chrome mode for the menu bar, toolbar, sidebars, status bar, and floating utility windows while keeping modal dialogs available.
	- left/right sidebar sections no longer all default open on first draw; the shell now starts less expanded by default.
	- large UI lists now use clipped child-list rendering instead of drawing every row every frame:
		- file browser
		- discovered maps
		- subobject/group visibility toggles
		- WMO / MDX placement lists
		- POI / taxi node / taxi route lists
- Scope / limit of this slice:
	- this reduces known UI hot spots and improves focus-mode usability, but it is not a full restoration of the older dockable UI and not proof yet of runtime frame-time recovery on the fixed development dataset.
- Validation status for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on actual UI responsiveness or PM4-debugging flow; do not over-claim the perf impact from build success alone.

- Confirmed hot seam on the active viewer path:
	- `WorldAssetManager.ReadFileData(...)` was still issuing repeated alias/fallback `ReadFile(...)` probes on top of `MpqDataSource`, including duplicate lowercase and `.mpq` retries that the MPQ data source already handled internally through case-insensitive normalization and Alpha wrapper resolution.
	- `MpqDataSource` had a raw-byte cache and worker prefetch path already, but it did not expose exact counters for direct read cache behavior, resolution source, or prefetch queue latency.
- Current implementation change:
	- `MpqDataSource` now exposes precise archive-I/O counters through `MpqDataSourceStats`:
		- `FileExists` request/cache/source counters
		- `ReadFile` request/cache/source counters (`loose`, `alpha wrapper`, `MPQ`, `miss`)
		- average uncached read latency
		- prefetch enqueue/dedup/cache-skip/completion counters plus average queue-wait and worker-read latency
	- `WorldAssetManager` now exposes `WorldAssetReadStats` and caches the winning resolved asset path per requested model/WMO read so later retries can jump straight to the known-good candidate instead of replaying the whole fallback chain.
	- Redundant work removed from the active world-asset path:
		- removed duplicate lowercase retry in `WorldAssetManager.ReadFileData(...)`
		- removed duplicate `.mpq` retry there for Alpha wrapper reads because `MpqDataSource.ReadFile(...)` already resolves the wrapper path directly
		- deduped candidate enumeration before trying alternates / stripped-filename / prefixed fallbacks
	- Prefetch policy is now narrower and more scene-aligned:
		- prefetch uses the canonical resolved model path first
		- if that canonical path is known, it no longer fans out across all extension aliases
		- M2 prefetch now warms the best resolved `.skin` path first and only falls back to generic skin candidates when no indexed best match exists
	- Viewer terrain/world stats panel now surfaces both `WorldAssetManager` probe counters and `MpqDataSource` cache/prefetch counters for runtime measurement.
- Validation status for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice.
	- no runtime real-data validation has been run yet on fixed MPQ-era data; do not claim generalized scene-streaming improvement from build success alone.

- The active M2-family renderer gap was confirmed to be material-state flattening inside `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`, not missing shader hooks in `ModelRenderer` first.
- Current landed slice recovers one explicit source seam instead of adding new transparency heuristics:
	- M2 skin batch metadata now preserves `textureCoordComboIndex` from raw `.skin` data and merges it back into the Warcraft.NET-derived skin path.
	- raw `MD20` vertex decode now preserves both UV sets instead of dropping everything to the first texture coordinate pair.
	- `textureCoordCombos` lookup now drives `MdlTexLayer.CoordId`; lookup value `-1` now marks the layer as `SphereEnvMap`, and lookup value `1` can select UV1 where present.
	- `ModelRenderer` now emits focused debug traces showing pass + resolved material family for M2-adapted batches when MDX debug focus is enabled.
- Scope of this slice:
	- improved family: reflective / env-mapped M2 surfaces, plus UV1-routed layers that were previously flattened to UV0
	- unchanged gaps: texture transform animation, color/transparency tracks, broader per-batch shader/material combo parity, and any runtime sorting issues beyond the existing pass split
- Validation status for this exact slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice
	- no runtime real-data signoff yet on reflective/env-mapped assets; do not claim PM4 matching benefit from this change alone

### M2 Material Parity Follow-Up: 4.0.0.11927 Wrap + Blend Correction (Mar 21)

- Follow-up runtime triage on Cataclysm-era M2 assets found two concrete material-state mismatches after the env-map / UV recovery slice:
	- `ModelRenderer` was only treating `WrapWidth` / `WrapHeight` as M2 repeat flags for the pre-release `3.0.1` profile, leaving later M2 builds on the old classic-MDX clamp interpretation.
	- `WarcraftNetM2Adapter.MapBlendMode(...)` was shifted after mode `2`, so M2 blend ids `4`..`7` were routed into the wrong local material families.
- Current correction:
	- all M2-adapted models now interpret wrap X/Y as repeat flags; classic MDX keeps the legacy clamp-flag behavior.
	- M2 blend ids now map as `0=Load`, `1=Transparent`, `2=Blend`, `3=Add` (`NoAlphaAdd`), `4=Add`, `5=Modulate`, `6=Modulate2X`, `7=AddAlpha` (`BlendAdd`).
	- the local renderer still has no distinct `NoAlphaAdd` or `BlendAdd` states, so those cases are now collapsed intentionally into the nearest additive families instead of landing there because of an off-by-one bug.
- Validation status for this exact follow-up slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice
	- no runtime real-data signoff yet on `4.0.0.11927` M2 assets; do not claim visual parity from build success alone

### PM4 Orientation Follow-Up: World-Space Solver No Longer Forces Mirrored Swap Fits (Mar 21)

### PM4 Link-Decode Follow-Up: Legacy `MSLK` Surface Index Defaults No Longer Leak As Real Data (Mar 21)

### PM4 MPRL Axis Contract Correction (Mar 21)

- Follow-up after comparing the active viewer path against older PM4 R&D exports and `WoWRollback/Pm4Reader` forensic notes.
- Current correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the active viewer restores the older fixed `MSVT` viewer/world basis `(Y, X, Z)` for the common `XY+Zup` path instead of trying to recover that basis later with per-object planar heuristics.
	- axis convention is now held file-level again across CK24 groups instead of being redetected per CK24; this avoids neighboring PM4 pieces drifting into different mesh bases.
	- viewer-side `MPRL` positions are now converted to world as `(PositionX, PositionZ, PositionY)` so they line up with that restored `MSVT` basis during planar scoring, nearest-anchor comparisons, and PM4 position-ref marker rendering.
	- the previous viewer assumption that `MPRL` could be treated as ADT-style planar `X/Z`, vertical `Y` or as raw `Z/X/Y` world output is no longer the active contract.
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet that this closes the reported PM4 placement failure.

### PM4 Render-Derivation Follow-Up: Overlay Objects Now Keep An Explicit Local Frame (Mar 21)

- Follow-up after runtime evidence that PM4 mesh pieces were effectively being treated as if they were already in final placed space, which makes it too easy to conflate object-local shape with world placement.
- Current correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- `Pm4OverlayObject` now localizes its line/triangle geometry around a preserved pre-split linked-group placement anchor instead of storing only fully placed geometry.
	- each PM4 overlay object now carries a baked base placement transform that restores that anchored object-local geometry into the solved placed frame.
	- when one CK24 is split into linked-group / MDOS / connectivity-derived parts, those parts keep the original linked-group placement anchor instead of rebasing to per-fragment centers.
	- overlay rendering now applies that baked base transform first, then any global PM4 overlay transform and object-local alignment edits on top.
	- PM4 JSON export now rehydrates placed-space geometry from the baked base transform so the interchange dump still matches what the viewer is rendering.
- Scope / limit:
	- this is structural groundwork for the missing “mesh inside stable object frame” layer; it is not a claim that final PM4 natural-rotation decoding is solved.
	- the CK24 placement solve itself is unchanged in this slice.
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet that this resolves the remaining PM4 orientation mismatch.

- Runtime investigation on `test_data/development/World/Maps/development/development_00_00.pm4` found a concrete active-path bug during PM4 rotation forensics:
	- `WoWMapConverter.Core.Formats.PM4.MslkEntry` exposes `MsurIndex`, `MsviFirstIndex`, and `MsviIndexCount`
	- `WorldScene` consults `MsurIndex` when grouping/linking surfaces and linked `MPRL` refs
	- but `Pm4File.PopulateLegacyView(...)` was never populating those legacy fields, so `MsurIndex` defaulted to `0`
- Current correction:
	- legacy `MSLK` entries created from the canonical decoder now explicitly set sentinel values for the unsupported fields (`MsurIndex = uint.MaxValue`, `MsviFirstIndex = -1`, `MsviIndexCount = 0`) instead of leaking fake `0` values into the viewer
	- this keeps `WorldScene` on the existing `RefIndex` fallback path unless a real surface index is available in the future
- PM4 rotation-forensics result from `development_00_00.pm4`:
	- raw `MPRL.Unk04` values only span about `0.01° .. 22.3°` on this tile
	- treat that field as a narrow local heading/placement signal on this file, not as proven absolute object yaw for the whole placed building set
	- `Unk06` is constant `0x8000` on this tile, and `Unk16` still behaves like normal-vs-terminator entry typing
	- `Unk14` continues to look like floor/level bucketing, not pitch/roll
- Viewer debugging follow-up:
	- selected PM4-object debug info now shows linked `MPRL` normal/terminator counts, floor range, and heading min/max/mean so runtime object picks can be compared against raw PM4 placement stats directly
- Validation status for this follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026
	- no automated tests were added or run
	- no runtime signoff yet that the selected `CK24=0x421809` object now has the final correct orientation; this slice improves link-data integrity and observability first

- Use `documentation/pm4-current-decoding-logic-2026-03-20.md` as the authoritative viewer-side PM4 reconstruction contract for the active branch.
- That doc was refreshed on Mar 21, 2026 to capture the current CK24 pipeline, the tile-local versus world-space planar candidate split, and the rollback of the linked-`MPRL` center-translation experiment.

### PM4 Tile-Local Orientation Follow-Up: Quarter-Turn Swap Solve No Longer Rotates Non-Origin Tiles (Mar 21)

- Latest runtime PM4 report narrowed a second orientation seam after the world-space solver fix: tiles beyond `0_0` / `0_1` were coherently rotated about `90°` counter-clockwise while origin-adjacent tiles still aligned.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the quarter-turn planar transform expansion was also being offered to tile-local PM4
	- tile-local PM4 already has a fixed south-west tile basis, so per-tile `swap` solving could rotate whole non-origin tiles even when the underlying tile basis was correct
- Current correction:
	- tile-local PM4 now tests only non-swapped mirror candidates inside the established tile basis
	- tile-local PM4 world assembly now applies the file tile indices in viewer-world order (`tileY -> worldX`, `tileX -> worldY`) instead of the naive unswapped pairing that only looked right on origin tiles
	- quarter-turn `swap` candidates remain world-space only
- Validation status for this exact follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this follow-up.
	- no runtime real-data signoff yet on the non-origin tile placement/orientation case; do not claim PM4 tile closure from build success alone.

- Runtime PM4 alignment evidence showed some objects resolving to mirrored planar transforms like `swap=True, invertU=False, invertV=False`, which reverses handedness and makes stairs/ramps wind the wrong way around structures.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
	- world-space PM4 candidate enumeration only tested `identity` and `swap`
	- rigid quarter-turn candidates were never considered, so some world-space objects could only be approximated by mirrored solutions
- Current correction:
	- world-space PM4 now evaluates the rigid planar set first: identity, 180 degree, +90 degree, and -90 degree basis changes
	- mirrored candidates are no longer part of the active PM4 planar solver; the viewer now stays on rigid candidates only to avoid reversed winding and opposite-facing fits
- Validation status for this exact PM4 solver slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice
	- no runtime real-data signoff yet on the guardtower staircase case; do not claim closure from build success alone

### PM4 Bounds Overlay Follow-Up: Per-Object PM4 Bounds Are Now Visible In-Scene (Mar 21)

### PM4 MPRL Frame Follow-Up: Linked-Center Translation Experiment Reverted (Mar 21)

- The earlier linked-`MPRL` frame experiment turned out to regress PM4 placement badly in runtime user validation.
- Latest runtime evidence also argues against the broader `MPRL` bounding-box/container paradigm itself: PM4 geometry and PM4 bounds are not conforming to that model in the viewer.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the viewer-side reconstruction path was translating whole CK24 groups into the linked `MPRL` world-bounds center after geometry pivot/yaw solve.
	- that shared translation was too aggressive and made PM4 alignment worse instead of better.
- Current correction:
	- the linked-center translation path was removed from `BuildPm4TileObjects(...)`.
	- CK24 rendering is back to the prior geometry-pivot path with the existing coarse yaw-correction logic.
	- this keeps the earlier `12°` suppression of small principal-axis yaw deltas, but no longer forces linked PM4 groups into an MPRL-center translation frame.
- Current interpretation:
	- user/domain correction: `MPRL` points are terrain/object collision-footprint intersections where ADT terrain is pierced by object collision geometry.
	- keep rejecting the old `MPRL` center/bounds translation experiment.
	- do not assume PM4 objects should fit inside an `MPRL` bounding box or container frame; use `MPRL` as footprint/collision reference data instead.
- Validation status for this exact follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 with existing solution warnings only.
	- no automated tests were added or run for this follow-up.
	- no runtime real-data signoff yet on whether PM4 alignment is restored; do not claim placement closure from build success alone.

### PM4 Yaw Follow-Up: Small Principal-Axis Corrections No Longer Override Near-Correct MPRL Rotation (Mar 21)

- Latest runtime user feedback on PM4 overlay alignment: objects were no longer wildly mis-rotated, but many still looked consistently off by roughly `5..10` degrees around the vertical axis.
- Root cause narrowed in `src/MdxViewer/Terrain/WorldScene.cs`:
	- PM4 MPRL yaw decode was already being rebased and then compared against a geometry-derived principal-axis yaw.
	- the follow-up CK24 world-yaw correction stage was still applying small residual deltas (`>= 2°`), which is too aggressive for irregular object footprints and can turn "almost correct" PM4 orientation into a visible small bias.
- Current correction:
	- CK24 continuous yaw correction is now treated as a coarse recovery step only.
	- residual yaw deltas below `12°` are ignored, leaving MPRL-derived orientation authoritative for near-correct objects.
- Validation status for this exact follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this follow-up.
	- no runtime real-data signoff yet after the threshold change; do not claim PM4 rotation closure from build success alone.

- Latest PM4 alignment feedback showed MPRL anchors lining up while other PM4 object extents still felt offset or nested inside the wrong container, making click-and-compare work too opaque.
- Current correction in `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp.cs`:
	- PM4 per-object bounds that were already computed for picking/culling/debug info are now rendered directly in-scene through the existing `BoundingBoxRenderer` path.
	- the PM4 alignment controls now expose a dedicated `PM4 Bounds` toggle beside `PM4 MPRL Refs` and `PM4 Centroids`.
	- selected PM4 object groups get a highlighted bounds color, and the exact selected PM4 object gets a white bounds box.
- Important scope note:
	- these bounds are currently built from the rendered PM4 object geometry (`MSVT`/`MSVI`/`MSUR` path), not from `MSCN`.
	- treat this as a visibility/debugging aid for the current PM4 reconstruction path, not proof yet that the active PM4 extents are sourced from the final correct container.
- Validation status for this exact PM4 bounds slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 with existing solution warnings only.
	- no automated tests were added or run for this slice.
	- no runtime real-data signoff yet on PM4 bounds usefulness or on the MSCN-versus-MSVT extent question.

### MPQ Base Build Selection Recovery (Mar 21)

- The active viewer no longer relies only on `InferBuildFromPath(...)` for new MPQ loads.
- `ViewerApp` now restores explicit build selection before loading a game folder:
	- MPQ open flow now pauses on a build-selection dialog.
	- build choices come from `Terrain/BuildVersionCatalog.cs` using `WoWDBDefs/definitions/Map.dbd` when available, with a built-in fallback list that includes `4.0.0.11927` and `4.0.1.12304`.
	- path/build tokens are now treated as preselection hints, not authoritative routing.
- Known-good base-client entries now persist `BuildVersion` in viewer settings and reuse it when reopening a saved base or attaching a loose overlay against that base.
- Loose overlay attach now emits a PM4 build hint when the overlay contains PM4 files with known version markers:
	- `12304` => `4.0.1.12304`
	- `11927` => `4.0.0.11927`
	- if that hint disagrees with the active base build, the viewer logs a warning instead of silently continuing with no build-era signal.
- Validation status for this build-routing slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- No automated tests were added or run for this slice.
	- No runtime real-data signoff yet on PM4/world-object matching with a `4.0.1.12304` base client.

### 4.0.0.11927 Terrain Blend Recovery (Mar 21)

- The earlier working assumption that 4.0 terrain texturing was effectively "3.3.5 MCAL decode with split files" is now documented as incomplete.
- Latest wow.exe RE confirms the missing behavior is runtime blend assembly, not only local MCAL byte decode:
	- `CMapChunk_UnpackChunkAlphaSet` stitches the current chunk with three linked neighbor chunks.
	- Neighbor alpha is matched by texture id, not only by local overlay slot index.
	- In 8-bit mode, layers without direct alpha payload can be synthesized as residual coverage `255 - other layer alphas`.
	- Blend textures are rebuilt through the `TerrainBlend` resource path (`CMapChunk_BuildSingleLayerBlendTexture`, `CMapChunk_BuildChunkBlendTextureSet`, `CMapChunk_RefreshBlendTextures`).
- Active viewer implementation now reflects the first verified slice of that model:
	- `FormatProfileRegistry.AdtProfile40xUnknown` routes to `TerrainAlphaDecodeMode.Cataclysm400`.
	- `StandardTerrainAdapter` captures per-layer source flags, synthesizes residual 8-bit alpha for missing direct payloads, and stitches same-tile chunk edges by matching neighbor layer texture ids.
	- `TerrainChunkData` now preserves `AlphaSourceFlags` for runtime post-processing.
- Documentation/handoff files updated for this recovery line:
	- `documentation/wow-400-terrain-blend-wow-exe-guide.md`
	- `docs/archive/WoW_400_ADT_Analysis.md`
	- `docs/archive/WoW_400_DeepDive_Analysis.md`
	- `docs/archive/WoW_301_DeepDive_Analysis.md`
	- `docs/ADT_WDT_Format_Specification.md`
	- `specifications/ghidra/prompt-400.md`
	- `.github/prompts/wow-400-terrain-blend-recovery.prompt.md`
- Validation status for this exact 4.0 recovery slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- No real-data runtime signoff yet on the fixed development terrain after residual synthesis + edge stitching.
	- Do not claim 4.0 terrain correctness from build success or diagnostics alone.

### WMO Blend + Loose PM4 Overlay Follow-Up (Mar 21)

- WMO distant "foggy sheen" triage found one concrete renderer mismatch in `src/MdxViewer/Rendering/WmoRenderer.cs`:
	- the active branch had flattened WMO material blend handling into opaque vs generic transparent
	- current code now maps raw WMO `BlendMode` to `EGxBlend` semantics (`Opaque`, `Blend`, `Add`, `AlphaKey`)
	- opaque pass now keeps `AlphaKey` with alpha-test, while transparent pass only handles `Blend` / `Add`
- Loose overlay PM4 resolution now gives precedence to the most recently attached overlay root in `src/MdxViewer/DataSources/MpqDataSource.cs`.
	- this matters when a base path and a later loose overlay both expose the same PM4 virtual path
	- older behavior searched loose roots in insertion order, so base loose files could shadow the attached overlay
	- current resolver searches newest overlay first and now traces PM4 loose-path misses like WMO misses
- Validation status for these viewer fixes:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- No automated tests were added or run for these fixes.
	- No runtime real-data signoff yet for the WMO sheen symptom or the loose-overlay PM4 workflow.

### PM4 Decode Triage + Rendering Parity Program (Mar 21)

- Current PM4 overlay failure state has moved past indexing/attach into decode-or-reconstruction triage:
	- runtime symptom seen by the user: `PM4: 2674 files found, none decoded into overlay data.`
	- this means PM4 candidates are being found, but none produced renderable overlay objects
	- latest `WorldScene.LazyLoadPm4Overlay()` instrumentation now buckets that failure into:
		- tile-parse rejection
		- tile-range rejection
		- read failure
		- decode failure
		- parsed-but-zero-object files
- Working hypothesis for the `4.0` versus `3.3.5` split:
	- PM4 parsing/object assembly itself appears build-agnostic
	- the likely seam is build-dependent map discovery / WDT resolution / candidate-set selection through `_dbcBuild`
	- the observed `2674` candidate count is suspicious versus the fixed development dataset note in `memory-bank/data-paths.md` (`616 PM4 files`) and should be treated as a clue, not normal noise
- Rendering work is now explicitly grouped as one coordinated program because PM4 object-variant matching depends on visually trustworthy output, not only PM4 geometry placement.
- The ordered rendering program is now:
	1. M2 material, transparency, and reflective-surface parity
	2. lighting DBC expansion beyond the current `Light` + `LightData` subset
	3. skybox / environment parity so backdrop and lighting context stop misleading object matching
- Planning artifacts created for this program live under `.github/prompts/`:
	- `m2-material-parity-implementation-plan.prompt.md`
	- `lighting-dbc-expansion-implementation-plan.prompt.md`
	- `sky-environment-parity-implementation-plan.prompt.md`
- Validation status for this planning slice:
	- no rendering code changes landed yet from this program
	- no automated tests were added or run for the planning-only pass
	- no runtime real-data validation yet on the new PM4 failure-bucket diagnostics

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

### WDL Spawn Chooser Regression Handoff (Mar 20)

- Latest runtime report from the active branch: WDL heightmap spawn chooser is currently non-functional across tested versions.
- Treat earlier notes that framed the spawn chooser path as working as stale until revalidated.
- Scope this as a viewer flow regression, not a parser-complete claim:
	- likely touchpoints are spawn action enablement (`WdlPreviewWarmState` gating), preview readiness transitions, and preview dialog/open fallback routing
	- this may involve both UI state and async warmup timing, not just WDL decode
- Do not close this issue on build success or file-level diagnostics alone.
- Required signoff for closure:
	- real-data runtime verification on at least one Alpha-era map and one 3.x map
	- explicit proof that spawn chooser opens/commits a spawn point and that fallback load behavior still works when preview prep fails

### PM4 Tile Mapping Runtime Handoff (Mar 20)

- PM4 viewer tile assignment now follows direct filename indices (`map_x_y.pm4` maps to `(tileX=x, tileY=y)`).
- The old MPRL-based tile reassignment heuristic has been removed from the PM4 overlay load path.
- Duplicate PM4 files mapping to one tile now merge object payloads/stats/refs instead of replacing prior data.
- Immediate next step after restart is runtime validation on the reported adjacency mismatch (`00_00`, `01_00`, and `01_01`) before further PM4 transform work.
- Do not claim this fixed from build-only validation; runtime signoff is still pending.

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

### Mar 20, 2026 - PM4 Viewer Overlay Diagnostics + Grouping/Winding Pass

- PM4 support advanced from coordinate-validation-only into active viewer diagnostics in `src/MdxViewer/Terrain/WorldScene.cs` + `src/MdxViewer/ViewerApp.cs`.
- New viewer PM4 overlay capabilities now include:
	- multi-mode color classification (`CK24` type/object/key, tile, dominant group/attribute, height)
	- optional `MPRL` reference pins and PM4-object centroid pins
	- selected-object PM4 metadata readout (dominant group key, attribute mask, `MdosIndex`, planar transform flags, winding parity)
	- CK24 disjoint-geometry splitting toggles: connectivity split and optional `MdosIndex` pre-split
- Orientation correction changed from translation-first nudging to per-object planar transform solving with parity-aware triangle winding correction.
- Scope boundary for this pass:
	- still a viewer-side PM4 debug/reconstruction layer, not final cross-tile object identity
	- map-wide CK24 registry + MSCN semantics remain pending
- Validation status:
	- repeated `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests were added or run
	- runtime real-data visual signoff remains pending for merged/disjoint object edge cases

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
