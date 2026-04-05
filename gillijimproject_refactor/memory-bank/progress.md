# Progress

### Apr 05, 2026 - Switched the active viewer back toward near-field ADT residency with WDL fallback and tighter object streaming

- followed the user's explicit requirement that only about `3-4` detailed ADTs should stay loaded while WDL covers distance terrain, because the prior retest was still around `18 FPS` with too many visible objects and too much terrain detail resident
- landed the streaming-policy follow-up:
	- `Camera.cs` now exposes a reusable forward vector and `ViewerApp.cs` passes it into `TerrainManager.UpdateAOI(...)`
	- `TerrainManager.cs` now uses a much smaller near-field AOI plus forward-biased tile lookahead instead of the old broad square radius
	- `WorldScene.cs` no longer globally hides WDL for ADT-backed tiles at startup and now restores WDL visibility when an ADT tile unloads
	- object streaming now defaults to `0.50x` and can be lowered to `0.25x` in both the active viewer and the shared `wow-viewer` visibility collector
	- `ViewerApp_Investigation.cs` now exposes the lower object-stream floor in the live UI
	- `WdlTerrainRenderer.cs` now fades WDL tiles in and out over a short blend window instead of hard-popping fallback terrain on ADT load/unload
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started without immediate startup errors before the process was stopped
- proof boundary:
	- this proves the active viewer compiles and still starts after the AOI/WDL/object-range changes
	- it also proves the WDL fade transition compiles and starts cleanly
	- no automated tests were added or run for this slice
	- no live FPS or visible-count retest has been captured yet, so the performance outcome is still unproven

### Apr 05, 2026 - Added chunk-bucket broad-phase culling for streamed MDX/WMO objects and trimmed redundant WMO doodad transparent work

- followed a new live retest screenshot where the viewer had improved to around `14 FPS` but still showed object-heavy steady-state costs (`WMO vis/draw ~17 ms`, `MDX vis ~17 ms`, `MDX opaque ~18 ms`)
- landed the active viewer follow-up:
	- `Terrain/WorldScene.cs` now tracks aggregate bounds per streamed chunk bucket for MDX and WMO instances
	- per-frame visibility now checks those chunk buckets first and only runs the existing per-instance collectors inside buckets that survive the coarse frustum/cone/range gate
	- `Rendering/WmoRenderer.cs` now reuses a scratch list for visible doodads and skips the transparent doodad replay when a doodad renderer has no transparent world pass
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this proves the new object broad-phase and WMO doodad trim compile in the active viewer
	- no automated tests were added or run for this slice
	- no live runtime retest has been captured yet, so no FPS claim is made yet

### Apr 05, 2026 - Restored tile-batched terrain submission in the active viewer after the user clarified the slowdown scales with loaded map tiles, not just object-heavy scenes

- followed the user's direct correction that all normal continent-sized maps are still around `5 FPS`, while only tiny maps with about `12` tiles normalize toward `60 FPS`
- landed the active terrain batching restore:
	- added `TerrainTileMesh` and `TerrainTileMeshBuilder`
	- switched `TerrainManager` from per-chunk terrain uploads to one batched terrain mesh per loaded tile
	- replaced the active `TerrainRenderer` with the tile-batching-capable path while preserving current public hooks such as MCCV toggles, runtime alpha/shadow replacement, and render-quality resampling
	- exposed terrain draw/uniform/texture-bind counters in the live renderer stats so the next retest has concrete evidence for whether terrain submission dropped
- terrain-alpha guardrail step completed:
	- compared the touched terrain batching file set against baseline commit `343dadfa27df08d384614737b6c5921efe6409c8`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this proves the active viewer compiles with the restored tile-batched terrain path
	- no automated tests were added or run for this slice
	- no live runtime or real-data terrain validation has been captured yet, so no FPS or alpha-blend safety claim is made yet

### Apr 05, 2026 - MDX object-pass route planning moved into wow-viewer runtime so WorldScene no longer decides batching and transparent order inline

- followed the user's post-fix retest that said the UI was less laggy but the scene was still only around `2-5 FPS`, then switched from host-side symptom work to the requested `WorldScene`-thinning path in `wow-viewer`
- landed the slice:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassFrame.cs` now stores planned opaque and transparent MDX routes plus the first batched opaque visible index
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs` now plans opaque/transparent MDX routes and executes the planned route lists
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldVisibleMdxPassRoute.cs` is the new shared route contract
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now consumes that route-planning seam and only does renderer lookup plus actual draw submission for the planned entries
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldObjectPassCoordinatorTests|WorldFramePassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this proves the extracted routing contract and the active compatibility build
	- it does not yet prove FPS recovery or full runtime closure on the live scene

### Apr 05, 2026 - UI multi-click lag traced to low-FPS mouse-event loss in the active ImGui backend path

- followed live feedback that the active UI itself was often taking `3+` clicks to register button presses, which was too specific to explain away as generic scene sluggishness alone
- confirmed the current root issue in `ViewerApp.cs`:
	- the active Silk.NET OpenGL ImGui backend samples mouse buttons once per frame with `CaptureState()`
	- at very low frame rates, short click down/up transitions can happen entirely between frames and never reach ImGui, making buttons appear randomly dead until one click happens to overlap a frame
- landed the fix:
	- `ViewerApp.cs` now queues raw Silk mouse down/up transitions and flushes them into ImGui as explicit mouse-button events right after `_imGui.Update(...)`
	- this is a host-side mitigation for low-FPS input loss; it does not claim the broader scene-performance collapse is solved
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this is build validation only
	- no live runtime signoff has been captured yet for actual button responsiveness improvement

### Apr 05, 2026 - Liquid pass now frustum- and fog-culls loaded meshes instead of drawing every loaded chunk

- followed a fresh runtime screenshot showing `World CPU` still near `89 ms` with the terrain liquid stage alone around `16.69 ms`
- found a direct hot-path issue in the active viewer: `Terrain/LiquidRenderer.cs` was iterating and drawing all loaded liquid meshes with no frustum test and no fog-range distance cull
- landed the fix:
	- terrain and WL liquid meshes now carry bounds and are culled against the current frustum plus a fog-range distance threshold before draw submission
	- `ViewerApp_Sidebars.cs` now reports `Liquid visible: visible/total` for terrain and WL liquid meshes so the next live screenshot can confirm whether the pass is still oversubmitting
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this is build validation only
	- no live runtime signoff has been captured yet for actual liquid-stage or FPS improvement

### Apr 05, 2026 - LIT base sampling bug fixed, and scene doodad visibility now suppresses WMO-internal doodads too

- followed new live evidence that the active LIT result still looked implausible and that scene-level doodad control might not actually be gating all doodad work
- landed two narrow fixes in the active `MdxViewer` path:
	- `Terrain/LitLoader.cs` now uses only an actual default light as the global/base LIT sample instead of accidentally treating the first light with any groups as the base, which was letting a local light tint the whole scene when file ordering was unfavorable
	- `Rendering/WmoRenderer.cs` plus `Terrain/WorldScene.cs` now apply the world scene's doodad visibility to WMO-internal doodad rendering, so `Show Doodads` can cut that hidden render path instead of leaving WMO doodads active behind the scene-level toggle
	- `ViewerApp_Investigation.cs` now says more honestly that LIT table selection is inspection-only while runtime sampling remains camera-driven and group-0-only
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this is build validation only
	- no automated tests were added or run for this slice
	- no live viewer signoff has been captured yet for either corrected LIT output or WMO-related FPS improvement

### Apr 04, 2026 - FOV-aware object visibility profiles and viewer-side object-family controls landed on the active renderer path

- responded to new live feedback that the terrain-world viewer was still only reaching roughly `5 FPS` and needed explicit efficiency layers instead of only passive range throttles
- landed a new runtime-owned visibility policy slice in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility`:
	- `WorldObjectVisibilityProfile` with `Quality`, `Balanced`, and `Performance`
	- `WorldObjectVisibilityContext` now carries vertical FOV and the chosen visibility profile
	- `WorldObjectVisibilityCollector` now performs projected-size culling and skips queueing tiny low-value missing assets, while preserving near/front-view candidates
- active `MdxViewer` integration now consumes that policy:
	- `Terrain/WorldScene.cs` derives the live vertical FOV from the projection matrix and forwards it to the shared collector
	- the active viewer exposes `Show Scene Objects`, `Show WMOs`, `Show Doodads`, and `Object Detail` controls in the terrain/investigation surfaces
	- renderer stats now show the selected object-detail profile alongside the existing stream-range/readout data
- focused proof landed:
	- added new collector tests for performance-profile projected-size culling and missing-asset load gating in `wow-viewer/tests/WowViewer.Core.Tests/WorldObjectVisibilityCollectorTests.cs`
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldObjectVisibilityCollectorTests|WorldObjectPassCoordinatorTests|WorldFramePassCoordinatorTests"`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- no live same-scene retest has been captured yet, so no FPS claim is made from this slice alone
	- if standstill FPS is still unacceptable after retest, the next likely hotspot remains WMO draw/runtime decomposition rather than object-load admission alone

### Apr 04, 2026 - Integrated wow-viewer reference-renderer performance plan landed as the new parent renderer program

- added `gillijimproject_refactor/plans/wow_viewer_reference_renderer_performance_plan_2026-04-04.md` as the new high-level renderer program for `wow-viewer`
- this new plan:
	- treats `wow-viewer` as the canonical cross-version C# renderer target, not only a parser/tool repo
	- unifies world-runtime extraction, M2 runtime completion, real batching/submission work, spatial indexing/residency, shared lighting, and consumer cutover under one staged effort
	- keeps Alpha-era and 3.x-era ownership under one profile-driven engine instead of separate renderer designs
	- makes real-data proof harness work a first-class phase so later performance claims have fixed-scene evidence
- current recommended next implementation slices from the plan:
	- `wow-viewer` visible-set runtime extraction
	- `wow-viewer` M2 scene submission and batching design
	- fixed Alpha/3.3.5 performance proof harness
- proof boundary:
	- this is planning/continuity work only
	- no renderer code or runtime validation landed in this slice

### Apr 04, 2026 - World runtime slice 02 is now specified as a real build slice

- refined `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md` so the `visible-set extraction` item now names:
	- the exact first extraction seam
	- the exact runtime files to add under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility`
	- the exact `WorldScene` responsibilities that remain host-only afterward
	- the exact validation floor for the first PR-sized slice
- normalized the implementation queue item for world runtime slice 02 to match that narrower execution boundary instead of the older generic wording
- proof boundary:
	- this is planning/continuity refinement only
	- no runtime extraction code, build, or real-data validation landed in this documentation pass

### Apr 04, 2026 - First visible-set extraction bridge landed in wow-viewer runtime

- landed the first code slice of world runtime slice 02:
	- shared `WorldObjectInstance` plus runtime-owned visibility contracts and collector in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility`
	- active `MdxViewer` now consumes that seam for WMO/MDX/taxi visibility admission and visible-bucket ownership
- current host/runtime split after this slice:
	- `wow-viewer` owns pure visibility admission and visible-bucket scratch
	- `MdxViewer.WorldScene` still owns asset-ready lookup, pending-load queueing, animation advance, transparent sort, and submission
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldObjectVisibilityCollectorTests`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- no real-data runtime capture or performance proof was done in this slice
	- pass extraction, host thinning, and `WowViewer.App` consumer work remain open

### Apr 04, 2026 - First object-pass coordinator slice landed on top of visible-set extraction

- landed a first slice of world runtime slice 03:
	- runtime-owned object-pass scratch and pass helpers in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes`
	- `WorldScene` now routes WMO opaque iteration, MDX animation dedup, MDX opaque iteration, transparent MDX sorting, and transparent MDX iteration through that runtime coordinator layer
- current host/runtime split after this slice:
	- `wow-viewer` owns object-pass sequencing scratch and iteration order for the visible object families
	- `MdxViewer.WorldScene` still owns GL state, renderer lookup, batch begin timing, and all non-object passes
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldObjectPassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- capture-proof boundary clarified during the same pass:
	- current `MdxViewer` capture automation can execute queued captures automatically, but queue creation is still UI-only and startup args do not yet provide a non-interactive capture path for the split development-map workflow
- proof boundary:
	- no real-data capture or performance signoff was completed in this slice
	- terrain/WDL/liquid/sky/overlay pass services and broader host thinning remain open

### Apr 04, 2026 - Startup capture hook landed, and slice 03 now owns the frame-order seam in wow-viewer

- landed a narrow non-interactive startup path in `gillijimproject_refactor/src/MdxViewer/ViewerApp_StartupAutomation.cs`:
	- direct base-client load with `--game-path` and `--build`
	- loose-overlay attach with `--loose-map-overlay`
	- world/asset load with `--world`
	- queued saved-shot capture with `--capture-shot`, optional `--capture-output`, optional `--capture-with-ui`, and optional `--exit-after-capture`
- widened world runtime slice 03 with a new runtime-owned frame-order seam:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs`
	- `wow-viewer/tests/WowViewer.Core.Tests/WorldFramePassCoordinatorTests.cs`
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now routes the current lighting/sky/skybox/WDL/terrain and object-tail pass order through that coordinator while keeping host-side callbacks for the concrete renderer work
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldFramePassCoordinatorTests|WorldObjectPassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- no real-data startup capture was run in this session
	- no automated tests were added for the `MdxViewer` startup-automation path itself
	- no runtime/performance signoff is claimed yet

### Apr 04, 2026 - Scene picking now uses the same docked viewport projection as rendering, and WDL tile lookup is X-major again

- fixed the active `MdxViewer` selection-offset regression in `src/MdxViewer/ViewerApp.cs`:
	- the live picker and hover tooltip were already normalizing mouse coordinates against the docked scene viewport rect
	- the 3D scene itself was still being rendered with full-window viewport sizing and projection aspect, which let the mouse ray, tooltip, bounding boxes, and visible objects drift apart when docked side panels changed the usable scene width
	- the render path now applies the same docked scene viewport rectangle to OpenGL viewport setup and projection aspect, then restores the full framebuffer viewport before UI rendering
- fixed WDL tile lookup drift across active consumers:
	- `src/MdxViewer/Terrain/WdlTerrainRenderer.cs` now reads and hides WDL tiles with X-major indexing (`tileX * 64 + tileY`) instead of Y-major indexing
	- `WoWRollback/WoWRollback.PM4Module/Services/WdlService.cs` and `WoWRollback/WoWRollback.PM4Module/WdlToAdtProgram.cs` now use the same X-major MAOF lookup, which matches the rest of the map tile codepath and avoids swapped-tile WDL terrain generation
- validation completed:
	- file diagnostics were clean for the touched files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run for this slice
	- no live runtime signoff was captured yet for the corrected click-selection alignment or the WDL terrain/generation output

### Apr 04, 2026 - Transparent MDX geosets now sort by camera depth within each priority plane

- fixed the active `MdxViewer` MDX material-order issue in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- transparent geosets were only being ordered by material priority plane and static geoset index during the transparent pass
	- that left some translucent MDX surfaces rendering as if they were behind their own model or nearby objects when multiple transparent geosets shared the same priority plane
	- the renderer now caches a model-space bounds center per geoset and sorts transparent geosets back-to-front by world-space camera distance within each priority plane instead of falling back to raw geoset index
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/Rendering/ModelRenderer.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run for this slice
	- no live runtime validation was captured yet for the affected MDX materials, so viewer signoff is still pending

### Apr 04, 2026 - Map GLB tile export now converts placement transforms into glTF Y-up space correctly

- fixed the active `MdxViewer` terrain-plus-objects GLB export mismatch in `src/MdxViewer/Export/MapGlbExporter.cs`:
	- the exporter was conjugating object placement transforms with the Z-up to Y-up basis in the wrong order for `System.Numerics` row-vector semantics
	- terrain vertices were already converted correctly, but placement matrices were landing in the wrong Y-up space, which made exported objects appear rotated or mirrored relative to the tile terrain
	- `ConvertTransformZupToYup(...)` now uses `C^{-1} * T_zup * C`, matching the direct `(X,Y,Z) -> (X,Z,-Y)` position conversion already used by the mesh builders
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/Export/MapGlbExporter.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run for this slice
	- no real-data viewer export re-run was performed in this session, so runtime/export signoff is still pending

### Apr 04, 2026 - MDX clicked-object selection now follows hovered identity, and terrain worlds prewarm streamed object assets

- fixed the remaining clicked-object mismatch in the live viewer path for dense MDX scenes:
	- `HoveredAssetInfo` now carries scene-object identity for MDX/WMO hits instead of only PM4 identity plus display text
	- `ViewerApp` now selects the hovered scene instance directly before falling back to the generic scene ray pick, so the clicked MDX should line up with the hovered tooltip target instead of a competing overlapping AABB
- replaced the old terrain-world object-load bottleneck with a scene load policy in `WorldScene`:
	- streamed terrain tiles now queue their tile-local MDX/WMO assets immediately on `OnTileLoaded(...)`
	- terrain maps now use a higher deferred-load throughput path with queue-pressure scaling instead of the old fixed `ProcessPendingLoads()` defaults and the old `6/3` visible-load promotion cap
	- WMO-only maps keep their eager-manifest path; terrain-streamed worlds share the same warmup behavior across early Alpha and later roots instead of adding new exact-version branches
- validation completed:
	- file diagnostics were clean for the touched files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with warnings in this workspace
- proof boundary:
	- no automated tests were added or run for this slice
	- no live runtime validation was captured yet for the MDX selection fix or the terrain-world performance change

### Apr 04, 2026 - MdxViewer inspector build repaired; UniqueId range edits now activate hides directly

- repaired the current `ViewerApp_Sidebars.cs` compile break from the selected-object inspector regrouping slice:
	- `DrawModelInfoContent()` is restored to a valid standalone model-inspection block
	- the fixed right inspector now also exposes an `Inspector Width` slider so width can be changed without relying only on the custom splitter
- tightened the `UniqueId Archaeology` UI in `ViewerApp.cs`:
	- hide-range slider changes now set `UniqueIdFilterEnabled = true` immediately
	- the detected-layer table is compressed so the action buttons fit more reliably in the inspector
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run for this slice
	- no live runtime validation was captured yet for the inspector recovery, UniqueId hide behavior, or fixed-sidebar width control

### Apr 04, 2026 - MdxViewer now keeps a grouped dirty-source queue for staged placement moves

- extended the first selected-placement save consumer into a multi-change dirty-map slice in `gillijimproject_refactor/src/MdxViewer`:
	- staged translation-only MDDF and MODF moves now persist across selection changes instead of being reset to the active selection only
	- pending moves are grouped by source ADT and can be written with `Save Current Source` or `Save All Pending`
	- the `Publish` workspace now exposes the same pending dirty-source queue so save packaging is visible outside the object inspector
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- proof boundary:
	- this is still translation-only object persistence for existing ADT placements
	- no automated tests were added or run for this slice
	- no real-data interactive workflow validation was captured yet for the grouped save queue

### Apr 04, 2026 - MdxViewer now consumes the shared selected-placement save seam

- landed the first end-to-end UI wiring on top of the existing `wow-viewer` placement writer:
	- selected existing ADT MDDF/MODF placements can now be translated from the `Objects` workspace in `MdxViewer`
	- live preview updates propagate through `WorldScene`, tile-instance caches, and adapter placement lists instead of staying as status text only
	- save-target plumbing now supports either a resolved writable loose source path or an explicit user-chosen `.adt` output path
- supporting runtime/data-source work landed in the active viewer path:
	- writable loose-path resolution on `IDataSource`
	- placement-source and writable-path resolution on `ITerrainAdapter`
	- cached tile placement mutation in `TerrainManager`
	- tile-local placement entry tracking on `WorldScene.ObjectInstance`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- proof boundary:
	- this is not general map-save closure; it only covers translation-only saves for selected existing ADT object placements
	- no add/remove placement support, aggregated dirty-map model, terrain persistence, or runtime signoff was completed in this slice

### Apr 03, 2026 - wow-viewer first save-capable ADT object move transaction landed

- landed the first shared editor-save seam in `wow-viewer` instead of keeping object edits as viewer-only state:
	- new shared placement move contracts in `wow-viewer/src/core/WowViewer.Core/Maps/AdtPlacementEditTransaction.cs`
	- new shared in-place ADT placement writer in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtPlacementWriter.cs`
	- translation-only persistence for existing `MDDF` and `MODF` entries
	- `MODF` bounds are shifted with the moved placement so shared readers see a coherent translation result after save
- validation completed:
	- focused synthetic roundtrip coverage for `MDDF` and `MODF`
	- real-data roundtrip coverage against `gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt`
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "AdtPlacementReaderTests|AdtPlacementWriterTests"` passed
- proof boundary:
	- this is a translation-only move seam for existing placements, not full editor save closure
	- no add/remove placement support, dirty-map pipeline, terrain write path, or save packaging landed in this slice

### Apr 03, 2026 - MdxViewer viewer/editor workspace shell landed in the live UI

- implemented the first actual editor-surface regrouping inside `gillijimproject_refactor/src/MdxViewer` instead of leaving the editor UI plan as prompt-only continuity:
	- new `Viewer` vs `Editor` workspace mode in the existing menu and toolbar
	- editor task routing for `Terrain`, `Objects`, `PM4 Evidence`, `Inspect`, and `Publish`
	- editor-mode navigator task rail on the left sidebar
	- editor-mode task inspector on the right sidebar
	- explicit status-bar affordances for workspace mode, active task, current target, and current save boundary
	- terrain task now hosts chunk clipboard inline in the inspector, while publish task makes export/capture-only status explicit
- proof boundary:
	- this is an MdxViewer UI-shell change only; it does not add map save, object persistence, or new format ownership
	- object task still reuses the existing mixed `DrawWorldObjectsContentCore()` surface as a first regrouping step, so follow-up extraction is still needed
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- validation not completed:
	- no automated tests were added or run for this slice
	- no live runtime validation of the new workspace/task flow was performed yet

### Apr 03, 2026 - Plan-state audit and implementation queue normalization

- audited active and prompt-era planning docs under `gillijimproject_refactor/plans` to separate landed work from still-open slices
- added `gillijimproject_refactor/plans/plan_audit_2026-04-03.md` as the continuity snapshot for:
	- implemented-but-still-worded-as-pending items
	- truly open implementation gaps
	- stale/superseded prompt-era docs
- updated active plan statuses to reduce queue confusion:
	- `wow_viewer_m2_runtime_plan_2026-03-31.md` now explicitly treats slice 01 as landed and slices 02-05 as open
	- `wow_viewer_world_runtime_service_plan_2026-03-31.md` now marks slice 01 as partial and slices 02-05 as open
	- `mdxviewer_renderer_performance_plan_2026-03-31.md` now includes an Apr 03 status snapshot and an updated next-slice focus on phase 3
	- `wow_viewer_format_parity_matrix_2026-03-28.md` now reflects M2 foundation ownership as `partial` instead of `none`
- added `gillijimproject_refactor/plans/implementation_queue_2026-04-03.md` as the numbered chat-by-chat execution queue for upcoming implementation sessions
- proof boundary:
	- this slice is documentation/continuity maintenance only
	- no new runtime or library behavior was implemented in this audit pass

### Apr 03, 2026 - wow-viewer editor-transition prompts and continuity plan landed

- added a dedicated planning surface for the user’s stated shift from viewer-first tooling toward a real viewer-editor:
	- `.github/prompts/wow-viewer-editor-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-map-editing-foundation-plan.prompt.md`
	- `.github/prompts/wow-viewer-editor-ui-surface-plan.prompt.md`
	- matching `.codex/prompts/` mirrors
	- `gillijimproject_refactor/plans/wow_viewer_editor_plan_2026-04-03.md`
- wired the new prompt family into the existing workflow discovery surfaces:
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `.codex/prompts/wow-viewer-tool-suite-plan-set.md`
	- `.github/copilot-instructions.md`
	- `AGENTS.md`
	- `wow-viewer/README.md`
- follow-up correction on Apr 03, 2026:
	- the editor plan set and map-editing foundation prompt are now explicitly worded to produce implementation-ready build plans with exact slice scope, validation commands, and immediate next actions rather than generic planning commentary
	- the editor UI surface prompt now follows the same rule, so workspace or panel planning should also return an implementation-ready UI slice with explicit dependencies and proof targets
	- the remaining companion prompts for CLI or GUI dual-surface planning and tool-migration sequencing now follow the same rule, so the full editor-transition prompt family behaves like a build queue rather than an architecture essay
- current proof boundary:
	- this slice is workflow-asset and continuity maintenance only
	- no editor runtime, map persistence, or UI-mode implementation has landed yet

### Apr 03, 2026 - MdxViewer adapted M2 material stacks and WDL far-terrain spacing were corrected in build-verified slices

- narrowed two live-runtime regressions in the active viewer path:
	- adapted M2 shiny or semi-transparent surfaces could collapse into incomplete translucent shells because `WarcraftNetM2Adapter.BuildMaterialsFromBatches(...)` still locked each skin section after its first texture unit
	- WDL far terrain was still being laid out on `WoWConstants.TileSize` instead of the viewer's 64x64 chunk grid, which stretched the low-detail mesh by `16x`
- landed focused fixes:
	- removed the per-section first-batch lock so the existing `MaterialLayer` grouping logic can build full adapted M2 material stacks again
	- switched `WdlTerrainRenderer` to `WoWConstants.ChunkSize` spacing for WDL cell placement
- current proof:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- proof boundary:
	- no automated tests were added in this slice
	- no runtime viewer signoff has been captured yet for the affected shiny M2 models or the corrected WDL far-terrain path

### Apr 03, 2026 - wow-viewer now emits per-build ADT UniqueId manifests for later timeline diffs

- landed `map uniqueid-report` in `WowViewer.Tool.Inspect` as the first build-manifest workflow over the shared `AdtPlacementReader` seam
- the command now persists raw `MDDF` and `MODF` `UniqueId` evidence with model paths, placement metadata, duplicate-id summaries, per-source counts, and explicit failure rows instead of only printing placement counts
- current proof:
	- focused validation passed with `ArchiveVirtualFileReaderTests`, `ArchiveCatalogBootstrapperTests`, and `AdtPlacementReaderTests`
	- real development-map output now exists at `wow-viewer/output/reports/map-uniqueids/development.json`
	- that report currently records `64435` placements, `62490` distinct `UniqueId` values, `1701` reused IDs, and `114` explicit `Unknown`-kind file failures
- proof boundary:
	- this is a per-build report artifact, not yet a cross-build added/removed-object timeline engine

### Apr 03, 2026 - wow-viewer now caches trusted MPQ-era known-file universes per client/build

- landed the first shared archive listfile-cache seam in `wow-viewer`:
	- `ArchiveListfileCache` / `ArchiveListfileCacheManifest`
	- direct `IArchiveCatalog.LoadListfileEntries(...)` seeding
	- `ArchiveCatalogBootstrapOptions`-driven cache load/persist behavior
	- `archive build-listfile-cache` in `WowViewer.Tool.Inspect`
- trust model now matches the current MPQ-era rule:
	- internal MPQ listfiles from the client are primary
	- the vendored/community listfile is supplemental only
- current proof:
	- focused archive bootstrap tests passed after the compatibility fix that restored external-listfile path forwarding
	- real `0.6.0` archive data produced `wow-viewer/output/cache/archive-listfiles/0.6.0.3592.json` with `56742` trusted internal entries, `1291033` supplemental entries, and `1347773` merged known files
- consumer boundary:
	- archive-backed `mdx chunk-carriers` now benefits from the merged bootstrap file universe, but this is still discovery infrastructure rather than deeper parser/runtime closure

### Apr 03, 2026 - wow-viewer WMO flag typing now names exterior and exterior-lighting, but not `0x2`

- broadened the real-data WMO audit from Castle into Alpha Ironforge with `wmo inspect --flag-correlation`
- result:
	- the larger corpus kept the already-typed chunk-gating reads stable for BSP, lights, doodads, and liquid
	- `0x00000008` and `0x00000040` are no longer left anonymous in the shared layer; they are now typed as exterior and exterior-lighting based on the repo-local WMO notes plus the in-repo `Warcraft.NET` names
	- `0x00000002` remains intentionally unnamed because the current corpus still does not separate it into a clean shared behavior signal
- proof boundary:
	- this is still inspect/shared-summary progress in `wow-viewer`, not runtime culling or lighting signoff

### Apr 03, 2026 - wow-viewer WMO summary now carries root skybox presence and a real-data flag-correlation report

- extended the shared `wow-viewer` WMO seam one step past raw `MOSB` and `MOGP` readers:
	- `WmoSummary` now exposes root skybox presence directly as `HasSkybox`
	- `wmo inspect` now supports `--flag-correlation` to correlate `MOGP` bits against actual group chunk signals within a real root WMO
- real-data validation on `castle01.wmo.MPQ` now gives an explicit per-file evidence readout instead of only raw flag words:
	- `0x00000001` cleanly aligns with BSP presence in both groups
	- `0x00000800` aligns with doodad refs on the flagged group
	- `0x00000002` remains intentionally unknown
- proof boundary:
	- this is shared summary/reporting progress in `wow-viewer`, not runtime collision closure

### Apr 03, 2026 - LK to Alpha converter recovered from false-success and chunk-walker regressions

- fixed the active `WoWMapConverter.Core` LK→Alpha path so it no longer reports success when zero tiles convert and no longer dies behind the recent Alpha write regressions
- concrete root causes fixed in the active converter path:
	- `AlphaMcnkBuilder` had an impossible header contract: `McnkHeaderSize` was `0x88` while the writer immediately required a 128-byte Alpha header layout
	- `LkToAlphaConverter` rejected tiles too early before MCIN/top-level MCNK fallback completed
	- MCIN offsets were trusted without validating that they actually point at `MCNK` chunks
	- the top-level ADT chunk walker hard-coded odd-size padding and drifted one byte after real chunks like `MTEX` size `187`, which broke later chunk discovery on tiles such as `development_0_0.adt`
	- `MMDX` / `MWMO` extraction trusted chunk bounds too aggressively and could surface `startIndex` range failures on malformed scans
- real-data validation completed against the fixed museum path:
	- command: `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- convert-lk-to-alpha i:/parp/parp-tools/gillijimproject_refactor/test_data/WoWMuseum/335-dev/World/Maps/development/development.wdt --map-dir i:/parp/parp-tools/gillijimproject_refactor/test_data/WoWMuseum/335-dev/World/Maps/development -o i:/parp/parp-tools/output/tmp/lk-to-alpha-dev/development.wdt --verbose`
	- result after the first repair pass: `1358/2303` tiles
	- result after the chunk-walker/MCIN repair: `2303/2303` tiles
- proof boundary:
	- this is real-data CLI conversion proof for the old compatibility path, not active viewer runtime signoff and not yet a `wow-viewer` library migration

### Apr 02, 2026 - Canonical M2 documentation set landed under wow-viewer/docs/architecture/m2

- consolidated the active M2 implementation surface into one canonical doc set:
	- `wow-viewer/docs/architecture/m2/README.md`
	- `wow-viewer/docs/architecture/m2/implementation-contract.md`
	- `wow-viewer/docs/architecture/m2/native-build-matrix.md`
	- `wow-viewer/docs/architecture/m2/consumer-cutover.md`
- the new set intentionally separates:
	- implementation contract
	- per-build proof matrix
	- wow-viewer versus MdxViewer cutover rules
	- raw evidence and historical plan sources
- updated the main M2 entrypoints so future sessions land on the consolidated docs first:
	- `wow-viewer/README.md`
	- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
	- `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`

### Apr 02, 2026 - Cataclysm next-build substitution narrowed to `4.0.0.11927`, with a hard binary-availability blocker

- advanced the cross-build investigation setup after the Wrath baseline by checking what later Win32 clients are actually reproducible in the current environment
- result:
	- the default ladder's next target `4.0.6a.13623` is not the build with existing repo-native evidence
	- the nearest documented Cataclysm-era native evidence in-repo is Win32 `4.0.0.11927`
	- a direct filesystem search under `I:\parp` found only local `WoW.exe` testdata for `0.5.5` and `0.6.0`, not Cataclysm/Mists/Warlords clients
- updated the canonical note and continuity to reflect the real proof boundary:
	- `4.0.0.11927` is the honest next Cataclysm substitution
	- current support for that slot in this session is static-only from older repo Ghidra notes
	- no fresh Cataclysm runtime attach or direct M2 choose/load/init/effect chain was possible from the currently visible files

### Apr 02, 2026 - Cataclysm `4.0.0.11927` first dedicated M2 static anchor map recovered

- used the live Ghidra-loaded `4.0.0.11927` binary to recover concrete Cataclysm M2 seams instead of relying only on older terrain/performance notes
- new confirmed static anchors recorded in the canonical note:
	- `FUN_007242d0` exact `%02d.skin` formatter
	- `FUN_00724270` exact `%04d-%02d.anim` formatter
	- `FUN_0072a740` choose-skin-profile seam
	- `FUN_0072a620` exact skin load + async callback setup
	- `FUN_0072a5f0` completion callback into `FUN_0072a4e0`
	- `FUN_0072a4e0` strict init + loaded-bit set + callback rebuild drain
	- `FUN_00725e00` active section/effect materialization from loaded skin data
	- `FUN_00724320` explicit `Diffuse_*` + `Combiners_*` effect builder with `Diffuse_T1Combiners_Opaque` fallback
	- `FUN_0072b3f0` external `%04d-%02d.anim` load path
	- `FUN_00402390` M2 runtime option registration with low bits matching Wrath but default mask `0x2008`
- runtime boundary after this slice:
	- x64dbg tools were available and attached, but the session dropped during the first rebasing attempt before a live Cataclysm breakpoint chain could be harvested
	- proof level for Cataclysm remains static-only until that runtime chain is recaptured

### Apr 02, 2026 - Win32 `0x20` flag now narrowed to a track-bearing shared-record class

- extended the canonical native note with a stronger conclusion for the long-running Wrath `0x20` question:
	- the repeated `0x20` checks in bootstrap relocation helpers now line up with exact wowdev record sizes for `M2Track<T>`, `M2Color`, `M2TextureTransform`, and `M2Light`
	- current best reading is that `0x20` marks a shared-record class with nested animated payloads that receives special relocation handling and is excluded from the compact runtime render list
- updated continuity to reflect the new proof boundary:
	- the remaining gap is the final user-facing label for that class, not whether `0x20` is real or whether it matters to bootstrap/runtime ownership

### Apr 02, 2026 - First Win32 world-path M2 choose-load capture recorded

- extended the Win32 `3.3.5.12340` native notes beyond UI-path traffic with a real in-world doodad load chain in:
	- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
- newly confirmed world-path evidence:
	- model path `world\expansion02\doodads\generic\barbershop\barbershop_mirror_01.m2`
	- exact numbered skin output `world\expansion02\doodads\generic\barbershop\barbershop_mirror_0100.skin`
	- post-load success at `0x0083cd32` with `EAX=1`
	- downstream callback rebuild hits at `0x00832ea0`
- proof boundary:
	- this is world-path choose/load proof, not full world-path choose/load/init/effect closure yet
	- explicit `0x00838490` init capture and world-path `0x00836600` combiner capture are still pending

### Apr 02, 2026 - Second world-path skin sample and explicit init reachability captured before debugger drop

- added a second confirmed world-path sample to the canonical Win32 notes:
	- `world\expansion02\doodads\generic\barbershop\barbershop_shavecup.m2`
	- exact numbered skin `world\expansion02\doodads\generic\barbershop\barbershop_shavecup00.skin`
- proved downstream init-path reachability in the same in-world session after removing noisy front-half breakpoints:
	- `0x00838490`
	- `0x00838561`
	- `0x00836600`
- proof boundary:
	- this still does not close a world-attributed init or world-attributed combiner sample because the isolated downstream samples stayed UI-heavy
	- the x64dbg MCP session timed out and disconnected before further targeted sampling could continue

### Apr 02, 2026 - Fresh reattach pass captured world-attributed combiner and init completion

- after x64dbg restart and reattach, the narrowed downstream-only breakpoint set produced a clean world-attributed combiner sample in:
	- `world\generic\human\passive doodads\beds\duskwoodbed.m2`
- concrete world-path effect-routing result recorded in the canonical native note:
	- `Diffuse_T2`
	- `Combiners_Mod2x`
- the same world object also surfaced at `0x00838561`, the loaded-state write inside the skin-init routine
- proof level change:
	- Win32 Wrath now has direct world-path runtime evidence for choose/load, init completion, and at least one concrete combiner-family output

### Apr 02, 2026 - Static Wrath M2 runtime contract consolidated from decompilation

- expanded the canonical native note with direct decompilation-backed behavior for:
	- `M2_ChooseAndLoadSkinProfile`
	- `FUN_0083cb40`
	- `FUN_0083cb10`
	- `M2_InitializeSkinProfileAndRebuildInstances`
	- `FUN_00837a40`
	- `FUN_00836980`
	- `FUN_00837680`
	- `M2_BuildCombinerEffectName`
	- `FUN_00836c90`
	- `M2_RegisterRuntimeFlags`
	- `M2_NormalizeModelPathAndProbeSkins`
- concrete new facts now recorded include:
	- skin choose threshold ladder `0x100`, `0x40`, `0x35`, `0x15`
	- the normal and special-case combiner-family decision trees
	- the exact startup and callback-owned runtime flag bits
	- the callback-drain loop after successful skin init
	- the batching relevance of fallback bit `0x40`

### Apr 02, 2026 - Strict extension gate and external anim naming added to the Wrath contract

- extended the canonical note with direct Win32 decompilation of:
	- `FUN_0081c390` strict cache-open and extension normalization
	- `M2_FormatAnimFilename_04d_02d`
	- `FUN_00837ee0` animation-track relocation during root-model bootstrap
- new concrete facts recorded:
	- `.mdl` and `.mdx` are normalized to `.m2` in the real Win32 loader path
	- unsupported extensions still hard-fail through the `Model2: Invalid file extension` path
	- external animation filenames are formatted as `%04d-%02d.anim`
	- animation relocation is part of the strict bootstrap path, not post-init glue code

### Apr 02, 2026 - Prompt location correction: use .github, not .copilot, in this repo

- corrected continuity guidance for workflow asset placement:
	- workspace prompt and agent workflow assets for this repo stay in `.github/` (with `.codex/` mirrors)
	- `.copilot/` is not the canonical location for these repo-scoped workflow assets here
- cross-build M2 investigation prompt remains correctly located at:
	- `.github/prompts/m2-cross-build-native-investigation.prompt.md`
	- `.codex/prompts/m2-cross-build-native-investigation.md`
- boundary:
	- this is a continuity correction only
	- no runtime, parser, or renderer code behavior changed

### Apr 02, 2026 - Cross-build M2 native investigation prompt added for 3.3.5 through 6.x

- added a dedicated cross-build workflow asset for native M2 behavior recovery across expansion branches where current library support is partial:
	- `.github/prompts/m2-cross-build-native-investigation.prompt.md`
	- `.codex/prompts/m2-cross-build-native-investigation.md`
- routing and discoverability updates landed so this prompt is reachable from existing wow-viewer prompt sets and instruction registries:
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`
	- `.codex/prompts/wow-viewer-tool-suite-plan-set.md`
	- `.codex/prompts/wow-viewer-m2-runtime-plan-set.md`
	- `.github/copilot-instructions.md`
	- `AGENTS.md`
	- `.codex/prompts/README.md`
	- `wow-viewer/README.md`
- continuity boundary:
	- this slice adds workflow and investigation guidance only
	- no parser, renderer, or runtime behavior change was implemented in code in this slice

### Apr 02, 2026 - Restarted x64dbg live-open sampling added to native notes

- after restarting x64dbg and reattaching to Win32 WoW, a targeted live sample at `FUN_004609b0` captured active open paths including:
	- `sound\\emitters\\Emitter_Stormwind_BehindtheGate_03.wav`
	- `Shaders\\Pixel\\ps_3_0\\Desaturate.bls`
- canonical and Session A docs now include a tighter LIT-status boundary that combines:
	- strict `Model2` extension-gate logic from `FUN_0081c390` / `M2_NormalizeModelPathAndProbeSkins`
	- restarted live open-traffic samples
- current blocker captured in continuity:
	- repeated `DebugRun` pauses in system DLL frames are still disrupting efficient world-path M2 chain harvest, so debugger run-state stabilization remains the immediate prerequisite for full live world-path closure
- validation boundary:
	- documentation/reverse-engineering continuity only
	- no renderer code changes landed in this slice

### Apr 02, 2026 - Win32 subsystem deep-dive notes added for shaders, liquids, particles, lighting, and LIT status

- expanded native Win32 documentation packet and canonical handoff with subsystem-specific anchors:
	- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/01-runtime-log.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/02-win32-m2-anchor-map.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/03-console-and-render-controls.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/04-next-steps.md`
- new evidence now explicitly maps:
	- rendering/effect ownership (`*.wfx` reload path, shader effect load/cache/bind seams)
	- liquid shader families and DBC-backed material/settings fallback behavior
	- particle dual-path submission (direct vs merged batch) and batch-compatibility constraints
	- world/map lighting seams (`Light*.dbc`, `WLIGHT`, `WCACHELIGHT`) plus debug script-light command path
- LIT status boundary in current Win32 pass:
	- no positive `.lit` or `.LIT` loader/path formatter anchor recovered
	- recovered `Unlit` labels are effect-mode names, not standalone file-family ownership
	- classification remains evidence-bounded and is recorded as unconfirmed/unsupported until positive anchors are found
- validation boundary:
	- documentation and reverse-engineering continuity updates only
	- no renderer code changes landed in this slice

### Apr 01, 2026 - Session A Deep-Capture and Hidden-Path Native Notes Landed

- expanded the Session A packet and canonical native research docs with second-pass Win32 evidence:
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/01-runtime-log.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/02-win32-m2-anchor-map.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/03-console-and-render-controls.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/04-next-steps.md`
	- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
- new confirmed runtime chain evidence now includes profile select, exact `%02d.skin` path output, init-state transition, callback rebuild hits, and combiner return-handle capture (all currently in UI-model context)
- new hidden-path notes now include:
	- shared Win32 M2 runtime flag-word helpers (`DAT_00d3fcf4` + getter/setter/OR helper)
	- callback-owned bit toggles for doodad batching, particle batching, and additive particle sorting
	- `M2Faster`/`M2FasterDebug` high-bit mode routing and parser caveats
	- likely-dead startup fallback branch (`0x40`) under the normal `M2_RegisterRuntimeFlags` init flow
	- repeated `M2_NormalizeModelPathAndProbeSkins` prewarm chain callsites
- validation boundary:
	- this was documentation and reverse-engineering continuity work only
	- no renderer code changes landed in this slice
	- x64dbg control session timed out and ended (`is_debugging=false`), so world-path runtime captures remain pending until reattach

### Apr 01, 2026 - Adapted M2 Skeletal Animation Re-enabled With Material-Track Guardrails

- landed a renderer-side animation recovery in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- adapted M2 models now create/use `MdxAnimator` again by default (gateable with `PARP_M2_ENABLE_ANIMATION=0`)
	- GPU bone upload now runs for adapted M2 when enabled
	- vertex shader skinning now clamps bone indices to `0..127` and normalizes weight sums before matrix blend
- intentionally kept high-risk animation channels suppressed for adapted M2 while visibility recovery continues:
	- material alpha/color tracks remain static for M2 path
	- geoset animation alpha overrides remain disabled for M2 path
	- UV animation transforms remain disabled for M2 path
- this keeps skeleton motion online without reopening the known transparency-driven invisibility seam.
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 01, 2026 with existing warnings only
- validation boundary:
	- no runtime manual signoff captured yet in this slice for world doodad animation parity
	- this is a staged motion-recovery pass, not full M2 animation/material parity

### Apr 01, 2026 - M2 Visibility Hotfix Targets Shader Alpha Path With Animator Disabled

- landed a narrow renderer change in `src/MdxViewer/Rendering/ModelRenderer.cs` focused on the active M2 invisible-geometry seam:
	- when `_isM2AdapterModel` is true and animator is disabled, `EvaluateLayerAlpha(...)` now uses `StaticAlpha` only and does not multiply by `StaticColorAlpha`
	- added `PARP_M2_FORCE_SOLID=1` diagnostic mode to force adapted M2 geosets through an untextured solid-color shader path (opaque pass) for hard geometry-vs-material isolation
- rationale:
	- adapted M2 layers can carry static color-alpha metadata that evaluates to zero at frame 0
	- with animator suppressed, that value was still reaching shader uniform `uColor.a`, making all submitted geometry fully transparent even when vertices/indices were valid
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 01, 2026 with existing workspace warnings only
- validation boundary:
	- no manual world-scene runtime signoff captured in this slice yet
	- this is a targeted visibility hotfix and not full M2 material/animation parity closure

### Apr 01, 2026 - Added Headless `--probe-m2-adapter` Triage Mode For Phase 1 M2 Visibility Diagnostics

- landed a new non-UI probe entrypoint in `src/MdxViewer/AssetProbe.cs`:
	- `--probe-m2-adapter <gamePath> <modelVirtualPath> [--build <version>] [--skin <virtualPath>] [--listfile <path>]`
	- alias: `--probe-m2`
- probe behavior now explicitly targets Phase 1 investigation evidence without requiring interactive viewer rendering:
	- loads model bytes from MPQ/loose game roots
	- validates build/profile compatibility through `FormatProfileRegistry` and `WarcraftNetM2Adapter.ValidateModelProfile(...)`
	- tries companion skin candidates (or forced `--skin`) through `WarcraftNetM2Adapter.BuildRuntimeModel(...)`
	- prints renderer-equivalent geoset outcomes as `[M2-DIAG-CPU]`: total geosets, valid, index-rejected, empty-skipped
	- preserves adapter per-geoset logs already emitted as `[M2-ADAPT]`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 01, 2026 with existing workspace warnings only
	- command wiring was exercised through `dotnet run ... -- --probe-m2-adapter ...`
- validation boundary:
	- no successful real 3.3.5 M2+skin probe was captured in this slice because the attempted in-repo `0.6.0` testdata run did not contain the chosen 3.3.5 UI-model path
	- this slice proves command availability and compile integration, not M2 visibility closure

### Apr 01, 2026 - Fresh A/B Session A Investigation Packet Started (Stormwind Runtime)

- created a new clean-room documentation packet for A/B analysis under:
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/README.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/01-runtime-log.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/02-win32-m2-anchor-map.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/03-console-and-render-controls.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/04-next-steps.md`
- this packet is intentionally framed as a fresh-session baseline for later A/B comparison and does not rely on prior-session conclusions as the primary evidence source
- static/decompilation outputs captured in this slice:
	- confirmed Win32 M2 anchors for skin/profile choose-load-init and combiner family selection
	- recovered world/terrain render command registration and high-value console toggles (`showCull`, `showLowDetail`, `showSimpleDoodads`, `detailDoodadAlpha`, `terrainAlphaBitDepth`, plus M2 runtime flags)
- runtime status in this slice:
	- x64dbg breakpoints were set on the M2 anchors
	- confirmed live anchor hits captured after restart:
		- `0x0083cc80` (`M2_ChooseAndLoadSkinProfile`)
		- `0x00835a80` (`M2_FormatSkinFilename_02d`)
		- `0x00838490` (`M2_InitializeSkinProfileAndRebuildInstances`)
		- `0x00836600` (`M2_BuildCombinerEffectName`)
	- captured model path in those first hits is a UI model (`interface\\glues\\models\\ui_mainmenu_northrend\\ui_mainmenu_northrend.m2`), so world-path capture is still pending
- validation boundary:
	- this slice delivered documentation and anchor setup only
	- no renderer or adapter parity fix is claimed yet

### Apr 01, 2026 - M2 Investigation Tooling Boundary Updated To Offline Ghidra + x64dbg-mcp

- updated `.github/prompts/m2-rendering-investigation.prompt.md` to remove the old live-Ghidra requirement
- Phase 2 now explicitly uses:
	- offline static analysis in Ghidra against `WoW.exe` (3.3.5.12340)
	- live runtime debugging in x64dbg through `x64dbg-mcp`
- the prompt now requires Ghidra-mapped Win32 targets to be validated dynamically with x64dbg breakpoints/watchpoints before claiming parity conclusions
- continuity intent:
	- keep native reverse-engineering evidence grounded in an executable workflow that is actually available in this environment
	- continue recording both static and runtime findings in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
- validation boundary:
	- this slice updates workflow guidance only
	- no new renderer/adapter runtime fix was claimed from this change

### Apr 01, 2026 - Win32 3.3.5.12340 M2 Runtime Anchors Mapped For x64dbg

- mapped and renamed concrete Win32 M2 functions in the loaded `WoW.exe` for immediate `x64dbg-mcp` breakpoint usage:
	- `0x0083cc80` `M2_ChooseAndLoadSkinProfile`
	- `0x00838490` `M2_InitializeSkinProfileAndRebuildInstances`
	- `0x00836600` `M2_BuildCombinerEffectName`
	- `0x00835a80` `M2_FormatSkinFilename_02d`
	- `0x00835a20` `M2_FormatAnimFilename_04d_02d`
	- `0x00402760` `M2_RegisterRuntimeFlags`
	- `0x0053c430` `M2_NormalizeModelPathAndProbeSkins`
- source evidence came from direct string-xref anchors in offline Ghidra (`%02d.skin`, `%04d-%02d.anim`, `Combiners_Opaque`, `Diffuse_T1`, `M2UseZFill`, `CM2Model`)
- recorded these anchors in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` under a dedicated Win32 breakpoint section
- validation boundary:
	- this slice establishes native-analysis anchors only
	- no runtime breakpoint-hit capture or renderer-fix claim yet

### Mar 31, 2026 - wow-viewer M2 Foundation Slice 01 Implemented

- implemented the first narrow `wow-viewer`-owned M2 seam rather than leaving slice 01 as planning-only work
- landed code:
	- `WowViewer.Core/M2` model identity, model document, skin document, submesh, batch, and profile-selection contracts
	- `WowViewer.Core.IO/M2` strict `MD20` and `SKIN` readers
	- `WowViewer.Core.Runtime/M2` choose/load/initialize skin-profile state
	- `WowViewer.Tool.Inspect` `m2 inspect` command for local-path or archive-backed model inspection
	- `WowViewer.Core.Tests/M2FoundationTests` coverage for identity normalization, strict root checks, strict skin parsing, and runtime-stage transitions
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- all `234` tests passed in the current wow-viewer solution run
- boundary:
	- this is not viewer-runtime parity or active `MdxViewer` signoff
	- no in-repo extracted real `.m2` / `.skin` asset was available, so this landing proves library/build/test behavior plus inspect ownership only

### Mar 31, 2026 - Ordered wow-viewer M2 Runtime Prompt Set Landed

- added the missing workflow surface for M2 runtime and renderer recovery so future chats stop mixing parser ownership, skin-state recovery, material routing, lighting, batching, and compatibility-only `MdxViewer` fixes in one prompt
- landed assets:
	- `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/01-md20-and-skin-runtime-foundation.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/02-section-classification-and-material-routing.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/03-animation-lighting-and-effect-runtime.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/04-scene-submission-and-batching.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/05-consumer-cutover-and-parity-harness.prompt.md`
	- matching Codex prompt mirrors plus `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
- routing effect:
	- M2 runtime ownership, exact `%02d.skin` behavior, section/material routing, animation/lighting state, and scene batching now have a dedicated staged prompt set in the same style as PM4/shared-I/O/world-runtime work
	- broader `WorldScene` extraction and repeated asset-miss suppression still belong to the separate world-runtime prompt family
- validation boundary:
	- this entry was workflow/continuity work only when it landed
	- slice 01 has since landed as a separate implementation step; keep using this entry for prompt-routing history, not current implementation status

### Mar 31, 2026 - Conservative Adapted-M2 Material Rollback Restored A Sane Giant-Root Payload

- followed stronger runtime evidence from the standalone viewer: `AzjolRoofGiant.m2` still loaded as a selectable adapted/runtime model but rendered no visible geometry even outside the world-scene path
- found a concrete regression seam in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`:
	- the current adapted M2 path was applying `ApplyLayerAnimationMetadata(...)` to each generated material layer
	- that is newer behavior than the old conservative path and can zero final layer alpha through transparency/color animation tracks even for static doodads
- landed a narrow rollback:
	- adapted M2 materials now stay on the conservative path again and do not graft raw layer transparency/color/UV animation metadata into runtime layers
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Mar 31, 2026 with existing warnings only
	- direct probe validation using the built viewer binary now reports `AzjolRoofGiant.m2` as a sane opaque adapted model with `574` vertices, `1063` triangles, `AZULBLANKROCK.BLP`, and `StaticAlpha=1.000`
- validation boundary:
	- no automated tests were added or run
	- live viewer runtime signoff is still pending until the rebuilt app is reopened on the standalone giant-root asset and the development world

### Mar 31, 2026 - Remaining Invisible World M2 Set Is Now Scoped To The Shaded Pass, Not Placement

- followed the first live viewer report after the build-mismatch correction: many MPQ-backed M2s are still missing, especially the giant root structures that should cover the development terrain
- the newest runtime evidence changed the problem statement again:
	- object tooltip selection still resolves those missing models
	- world bounding-box overlays still show their placements and extents in the expected locations
	- this means scene registration and bounds are alive, but does not yet prove the shaded geoset draw path is succeeding for those instances
- current best reading:
	- the stale-build fix remains valid because it explained and reproduced one real failure mode
	- the remaining blocker is now a narrower render-path problem in the active viewer for adapted M2 shading/submission/material state
- next investigation target is explicit:
	- verify whether those root models reach `ModelRenderer.RenderGeosets(...)` in opaque and transparent passes
	- if they do, compare their layer family / blend routing against visible adapted M2s
	- if they do not, add temporary runtime diagnostics or forced solid-color rendering so the viewer can separate geometry submission failure from texture/material invisibility
- validation boundary:
	- no new fix landed for this remaining seam yet
	- no runtime signoff should be claimed from the build-mismatch correction alone

### Mar 31, 2026 - M2 Loads Now Override Stale Build Selection With The Real Client Build

- followed the next concrete blocker after the shared renderer fixes: `AzjolRoofGiant.m2` still showed as invisible in the viewer, but a new headless probe proved the real issue was build mismatch rather than necessarily bad adapter extraction
- captured hard evidence with the same asset on the same 3.3.5 client root:
	- `--build 3.3.5.12340` produced sane adapted output (`574` verts / `1063` tris, valid bounds, resolved skin/texture)
	- `--build 3.0.1.8303` produced degenerate output (`1` vert / `1` tri, broken bounds)
- landed a narrow runtime correction in `src/MdxViewer/Terrain/BuildVersionCatalog.cs`, `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/Terrain/WorldAssetManager.cs`, and `src/MdxViewer/Rendering/WmoRenderer.cs`:
	- M2-family loads now prefer the build inferred from the actual game/client path over a stale selected build when those disagree
	- standalone M2 open, world M2 loading, and WMO doodad M2 loading all use that effective build for adapter and converter fallback paths
- validation completed:
	- `get_errors` returned clean for the edited files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed with existing warnings only
- validation boundary:
	- no real viewer runtime confirmation has been captured yet
	- the next check is to reopen the failing standalone asset and the development map under the saved 3.3.5 client and confirm the previously invisible MPQ-backed M2s now render

### Mar 31, 2026 - World M2 Doodads Now Use Per-Instance Rendering In WorldScene Again

- followed the next active runtime blocker after slice 01: user reported fewer repeated asset hiccups, but many world M2s were still invisible even though hover/picking showed the objects existed
- landed a narrow world render-path correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- M2-adapted world doodads now use `RenderWithTransform(...)` again in both opaque and transparent passes instead of the shared batched `RenderInstance(...)` path
	- classic MDX doodads still stay batched
	- batch initialization now comes from the first actually batched renderer rather than the first visible renderer overall
- why this matters:
	- the active batch/unbatch gate only covered particle/ribbon cases, so adapted world M2s were still falling onto the generic instanced path despite earlier continuity pointing at that path as the likely invisible-model seam
	- this keeps the fix narrow in the active viewer without reopening the broader renderer split yet
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime confirmation has been captured yet, so remaining WMO/M2 hiccups are still open until the viewer is exercised on the development map

### Mar 31, 2026 - Missing Base Textures No Longer Make Adapted M2s Fully Invisible

- followed the next user-visible M2 blocker after the world-scene submission fix: more objects now appear, but another set of MPQ-backed M2s still stays invisible in both world view and standalone model view
- landed a shared renderer-path correction in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- adapted M2s now treat a missing base-layer texture as a neutral fallback-texture case instead of letting the whole geoset render path disappear
	- the renderer also no longer suppresses the normal fallback-geoset draw merely because an adapted/pre-release layer missed texture resolution
- why this matters:
	- the shared `ModelRenderer` path is used by both world placements and standalone model viewing, so this fix targets the common “loaded but fully invisible” symptom directly
	- it keeps the next investigation honest: if some MPQ M2s are still malformed after this, the likely remaining bug is in adapter skin/submesh/material extraction rather than another world-scene visibility seam
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/ModelRenderer.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime confirmation has been captured yet, so any remaining partial or malformed MPQ M2s are still open

### Mar 31, 2026 - First Negative Asset Lookup Suppression Slice Landed In MdxViewer

- implemented the ordered world-runtime slice 01 in the active viewer path instead of starting the broader service extraction early
- landed behavior in `src/MdxViewer/Terrain/WorldAssetManager.cs`, `src/MdxViewer/Rendering/WmoRenderer.cs`, `src/MdxViewer/ViewerApp.cs`, and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- cached failed MDX world loads now stay failed for the current session rather than being retried through lazy load, queue, and deferred drain paths
	- known-missing external `.skin` results now suppress repeated world-path prefetch fanout and repeated companion-skin miss logs
	- standalone M2 open and WMO doodad M2 load paths now also log the same missing `.skin` problem once per resolved model path instead of flooding repeats
	- the terrain sidebar now reports suppressed failed-MDX retries plus known/suppressed skin-miss telemetry so the miss path is visible without reading raw logs
- why this matters:
	- this removes one concrete source of repeated asset-miss noise before later visibility/pass extraction slices
	- it keeps the slice narrow: no broad asset-service rewrite, no pass ownership move, and no new renderer abstraction was introduced here
- validation completed:
	- `get_errors` returned clean for the edited files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime capture or fixed-shot smoke was run in this slice, so measured log/frame improvement on the development map is still pending

### Mar 31, 2026 - Ordered wow-viewer World Runtime Prompt Set Landed

- user selected the staged world-runtime decomposition path instead of a one-off next extraction only
- added a dedicated Copilot workflow surface for fresh implementation chats:
	- `.github/prompts/wow-viewer-world-runtime-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/01-negative-asset-lookup-suppression.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/02-visible-set-runtime-extraction.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/03-world-pass-service-extraction.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/04-world-scene-host-thinning.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/05-wow-viewer-app-runtime-consumer.prompt.md`
	- `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
- why this matters:
	- the next chats now have one ordered path for the `WorldScene` split instead of rediscovering the sequence every time
	- the first slice explicitly targets repeated `.skin` miss churn and failed MDX retry noise before deeper pass extraction, which should improve measurement quality and reduce obvious hidden runtime waste
- validation boundary:
	- this step only created workflow and continuity assets
	- no code fix for the `.skin` retry issue landed yet in this step
### Mar 31, 2026 - First WorldScene Seam Extracted Into wow-viewer Core.Runtime

- followed the new architectural direction to split `WorldScene.cs` by moving the first stable slice into `wow-viewer` instead of performing another app-local refactor only
- landed the first shared runtime seam across `wow-viewer` and `MdxViewer`:
	- added `WowViewer.Core.Runtime.WorldRenderStageStats`, `WorldRenderFrameStats`, and `WorldRenderOptimizationAdvisor`
	- moved `WorldScene` to consume those runtime-owned contracts instead of keeping the public telemetry surface embedded in the app project
	- added xUnit coverage for empty-frame, MDX-dominant, and terrain-dominant optimization hints in `wow-viewer/tests/WowViewer.Core.Tests/WorldRenderOptimizationAdvisorTests.cs`
	- added `WowViewer.Core.Runtime` project references to the active `MdxViewer` consumers so the extracted seam is compile-proven in the legacy app
- why this matters:
	- this establishes `wow-viewer` as the canonical owner of the first reusable world-render contract rather than leaving the seam trapped in `WorldScene`
	- it creates a concrete next extraction path for visibility, pass ownership, and scene composition without overstating that the renderer itself has already moved
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with 226 tests succeeding
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings only
- validation boundary:
	- no runtime viewer signoff was performed
	- no render behavior changed intentionally in this slice beyond where the telemetry contract is owned

### Mar 31, 2026 - First Renderer-Stats Slice Landed In WorldScene And The Sidebar

- followed the renderer-first roadmap by implementing the first measurement slice instead of jumping straight into another broad renderer rewrite
- landed the change in `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- added a reusable per-frame render contract in `WorldScene` that now owns visible WMO/MDX scratch lists, transparent-sort scratch, stage timings, and MDX batched-vs-unbatched counts
	- the active world frame now records timings for deferred asset drain, taxi actor update, lighting, sky, skybox backdrop, WDL, terrain, WMO visibility, WMO submission, MDX animation, MDX visibility, MDX opaque submission, liquids, MDX transparent sort, MDX transparent submission, and the late overlay/debug block
	- the terrain sidebar now exposes a `Renderer Stats` tree showing the last captured world-frame CPU timings and a heuristic `next win` hint based on those numbers
- why this matters:
	- this is the first active `WorldScene` seam that turns the renderer-performance roadmap into runtime data instead of only continuity notes
	- it also establishes the smallest viable world render-frame contract needed for the next batching/culling slice without pretending the full render-layer refactor is already done
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime capture was performed yet in this chat, so the new stats panel and its next-win hint still need manual confirmation on the development map

### Mar 31, 2026 - Renderer-First Performance Roadmap Recorded For The Active MdxViewer World Path

- followed the direct reprioritization that camera-movement performance is now the biggest blocker, ahead of more shell tweaks or more isolated feature additions
- recorded the active renderer roadmap in `gillijimproject_refactor/plans/mdxviewer_renderer_performance_plan_2026-03-31.md`
- plan decisions locked for future slices:
	- work the active `src/MdxViewer/Terrain/WorldScene.cs` path first instead of treating dormant `RenderQueue.cs` as if it already owned the frame
	- start with per-frame instrumentation plus an explicit world render-frame contract
	- then reduce MDX submission churn and batching waste
	- then pull WMO shell/liquid/transparent ownership outward from renderer-local sequencing into clearer scene-level layers
	- keep PM4/debug/editor overlays as explicit late layers instead of letting them stay mixed into the main world submission cost
	- finish DBC lighting integration after render-layer ownership is explicit
	- add graveyards from `WorldSafeLocs.dbc` only after the renderer frame is stabilized, reusing the Area POI / taxi lazy-load overlay model
- validation boundary:
	- this slice is planning only
	- no automated tests were added or run
	- no runtime performance measurements were captured yet from the new plan itself

### Mar 31, 2026 - Fixed Sidebar Shell Now Uses Draggable Split Panels

- followed direct viewer-shell feedback that the current fixed sidebars were still not meaningfully resizable and felt like a broken layout mode
- landed shell changes in `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- fixed mode now renders explicit draggable left/right splitter bars instead of relying on hidden ImGui window-border resize behavior
	- left and right panels stay edge-anchored while splitter drag updates the stored sidebar widths directly
	- fixed panels now opt into `NoResize` because the supported resize path is the splitter itself, not window-border grabbing
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for the new fixed-panel splitter behavior

### Mar 31, 2026 - Mouse Camera Input Restored After Narrowing The Fixed-Sidebar Splitter Windows

- followed direct runtime feedback that mouse camera control stopped working after the fixed-sidebar splitter shell landed
- root cause was the splitter host itself in `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- the first splitter implementation used one transparent full-width window over the whole viewport height
	- that window could still cause ImGui mouse capture outside the actual splitter bars, which interfered with scene camera input
- landed fix:
	- replaced the full-width splitter host with narrow splitter-only windows for the left/right drag handles
	- only the actual splitter strips now capture mouse input, leaving the rest of the viewport available to the normal camera path again
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp_Sidebars.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for restored mouse-look behavior

### Mar 31, 2026 - Hover Tooltip Toggle And UniqueId Archaeology Filter Landed In MdxViewer

- followed the latest direct viewer workflow request to make scene exploration less noisy and more controllable:
	- hover cards needed an explicit disable path
	- object layers needed a `UniqueId`-based scrubber for digital archaeology across either the whole map or the current camera tile
- landed behavior in `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/Terrain/WorldScene.cs`:
	- added a `Hover Tooltips` checkbox that suppresses scene hover overlay rendering without removing the hover metadata pipeline itself
	- replaced the first cutoff-only archaeology filter with explicit min/max range semantics so the viewer can hide placements within a chosen `UniqueId` span
	- propagated tile coordinates onto flattened scene object instances so camera-tile filtering works for terrain-loaded and external spawn objects
	- applied the hide filter to render submission, hover hit testing, scene picking, and bounding-box debug drawing so hidden ranges behave consistently
	- added gap-based archaeology layer detection for the active scope plus a viewer table that lists detected layers with range, count, WMO/M2 breakdown, and one-click hide actions
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for the new archaeology controls on the development map

### Mar 31, 2026 - User Fog Range Control Restored Over Zone Lighting

- followed direct runtime feedback that fog could no longer be effectively removed and that the older farther-view behavior had regressed
- root cause was the Mar 30 shared-lighting change letting `LightService` overwrite `TerrainLighting.FogStart` and `FogEnd` every frame while zone lighting was active
- landed fix in `src/MdxViewer/Terrain/TerrainLighting.cs` and `src/MdxViewer/Terrain/WorldScene.cs`:
	- external lighting override now applies only directional light, ambient light, and fog color
	- live fog distance remains on the user-controlled `TerrainLighting` values, so terrain sidebar fog sliders and far visibility budget work again
- also fixed compile blockers still present in the current `WorldScene` hover helpers so the viewer solution could be revalidated cleanly
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/TerrainLighting.cs` and `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for the restored no-fog / farther-view behavior

### Mar 31, 2026 - VLM Dataset Reconstruction Planning Reset Landed

- followed a direct request to stop treating VLM work as a vague training exercise and define how a v7-like missing-layer reconstruction model should be grounded in real map data
- confirmed the current exporter boundary before planning:
	- `WoWMapConverter.Core.VLM.VlmDatasetExporter` already exports chunk heights, local/global heightmaps, normals, MCCV, raw shadow bits, derived shadow analysis, alpha masks, liquids, objects, WDL data, and binary tile output
	- the older docs under `docs/VLM_Training_Guide.md` and `docs/VLM_DATASET_EXPORTER.md` do not fully describe that active schema
- landed new planning surfaces:
	- `gillijimproject_refactor/plans/vlm_dataset_reconstruction_plan_2026-03-31.md`
	- `.github/prompts/vlm-dataset-reconstruction-plan.prompt.md`
	- updated `.github/prompts/development-repair-implementation-plan.prompt.md` so future chats route VLM dataset/model asks away from the repair-pipeline prompt
- planning decision recorded in continuity:
	- `development` is now explicitly the reconstruction target/evaluation corpus, not the only teacher corpus
	- the next dataset slice should be manifest/provenance/completeness classification and curation across real exported maps before additional model work
- validation boundary:
	- no exporter code behavior changed in this slice
	- no automated tests were added or run
	- no real-data export, curation, or training command was executed in this slice

### Mar 31, 2026 - Terrain WDT Global WMO Parsing Fixed For ADT Maps; M2 UV Regression First Mitigation Landed

- followed direct runtime bug reports after the corrupted-chat continuity recovery:
	- terrain maps were missing WDT-level global WMO placements that should render roof or shell geometry over ADT terrain
	- M2s still had active material regressions, including oversized leaf-like detail doodads and inconsistent transparency behavior
- landed terrain fix in `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`:
	- WDT global `MWMO` or `MODF` parsing now triggers for terrain maps when the file carries the historical `MPHD` global-map-object flag or plainly contains both chunks, not only for `IsWmoBased` maps
	- terrain-map WDT placements now convert into renderer coordinates like ADT `MODF` placements instead of staying in raw WMO-map coordinates
- landed first M2 mitigation in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`:
	- negative resolved texture coord ids now fall back to UV0
	- negative coord ids no longer imply `SphereEnvMap`
	- this intentionally moves current behavior back toward the older known-good adapter path while preserving explicit positive UV-set selection
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run for these viewer regressions
	- no development-map runtime signoff has been completed yet for either the restored WDT global WMO path or the current M2 adapter mitigation

### Mar 31, 2026 - Active Tree-Trunk M2 Regression Trimmed Back By Restoring The Conservative Per-Section Material Path

- followed direct runtime feedback that the remaining M2 regression still made some trees appear to be made of leaves with no trunks
- landed a narrower compatibility fix in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`:
	- restored the old stable runtime behavior of taking the first batch/material per section
	- forced the active runtime material path back to `UV0` for that conservative section material path
	- this intentionally backs away from the newer richer batch/layer interpretation until it can be proven against real viewer output
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for restored tree-trunk rendering on the development map

### Mar 30, 2026 - WorldScene WMO Submission Now Uses A Visible-Instance Bucket

- followed the renderer-performance continuity work after the shared LightService or TerrainLighting lighting fix
- landed a narrow structural slice in `src/MdxViewer/Terrain/WorldScene.cs`:
	- added a reusable visible-WMO scratch contract so world-scene WMO instances are culled once per frame before submission instead of recomputing cull decisions inline inside the WMO draw loop
	- this brings the WMO path closer to the existing visible-MDX path and creates a cleaner seam for a future split between opaque shell, liquids, doodad transparent, and transparent shell passes
	- current user-visible behavior should stay the same because `WmoRenderer.RenderWithTransform(...)` still owns the actual WMO-local pass order
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run for this renderer-structure slice
	- no development-map runtime signoff has been completed yet for the visible-WMO submission path

### Mar 30, 2026 - PM4 UI Glossary Clarified Viewer-Derived `part` Labels

- followed direct user feedback that the PM4 inspector had become too opaque to trust for day-to-day use
- landed clarification across `src/MdxViewer/ViewerApp_Pm4Utilities.cs`, `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/Terrain/WorldScene.cs`, and `src/MdxViewer/README.md`:
	- the PM4 workbench now includes a glossary/evidence block explaining which labels are raw PM4 chunk names, viewer aliases, or viewer-generated structure
	- `part` / `ObjectPartId` is now explicitly described as a viewer-generated split id from the current overlay build, not a raw PM4 field on disk
	- selected-object and graph text now repeat that distinction where the user actually sees `part`
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime validation was performed for this terminology-only clarification slice

### Mar 30, 2026 - DBC LightService Now Drives One Shared World Lighting State

- followed user direction to stop deferring renderer correctness work while the world path is still slow and visually inconsistent
- landed a first lighting-correctness slice in `src/MdxViewer/Terrain/TerrainLighting.cs` and `Terrain/WorldScene.cs`:
	- `TerrainLighting` can now take an external per-frame lighting override for direct light, ambient light, fog color, and fog range
	- when `LightService` has an active zone, `WorldScene.Render(...)` now maps `LightService.TimeOfDay` into the shared terrain lighting clock, applies the DBC-driven colors/fog override, and updates that shared state before rendering skybackdrops, WDL, terrain, liquids, WMOs, or MDXs
	- when no zone is active, rendering falls back to the old procedural `TerrainLighting` path cleanly
- why this matters:
	- before this slice, one frame could mix `Light.dbc` sky/fog with fallback terrain/object light colors, so lighting parity was already broken even before the larger render-layer work
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/TerrainLighting.cs`, `src/MdxViewer/Terrain/WorldScene.cs`, and the associated capture-fix file `src/MdxViewer/ViewerApp_CaptureAutomation.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run for this viewer-side lighting correction
	- no development-map runtime signoff has been completed yet for the shared LightService or TerrainLighting path

### Mar 30, 2026 - v0.4.6.1 Release Prep Added PM4 Tooltip-Focused Notes And Beginner UI Guidance

- followed release request for `v0.4.6.1` with emphasis on clearer PM4 WoW-styled tooltip display messaging and reduced first-run confusion
- landed behavior/docs/workflow updates:
	- bumped `src/MdxViewer/MdxViewer.csproj` version metadata to `0.4.6.1`
	- updated welcome/status UI wording so users are directed to open a base game path first (`File > Open Game Folder (MPQ)`) instead of defaulting to standalone file usage
	- refreshed `src/MdxViewer/README.md` and `gillijimproject_refactor/README.md` with `0.4.6.1` snapshot, explicit beginner flow, and conservative support-range language
	- added `src/MdxViewer/docs/ui-screenshot-guide.md` and `src/MdxViewer/docs/screenshots/README.md` to standardize screenshot capture/drop workflow for README/release image selection
	- updated release-note body and quick-start text in both release workflows so GitHub release output matches the new onboarding and PM4 tooltip emphasis
- validation boundary:
	- no automated tests were added or run
	- this slice is build/docs/workflow prep and still needs runtime screenshot curation for final README hero-image selection

### Mar 30, 2026 - PM4 Workbench Tab Flicker/Snapback Removed

- followed user runtime feedback that `Selection` and `Correlation` in the PM4 workbench were only visible for a split second and then dropped out
- landed behavior:
	- added `_pendingPm4WorkbenchTab` one-shot tab focus state in `src/MdxViewer/ViewerApp.cs`
	- `OpenPm4Workbench(...)` now sets one-shot pending tab focus instead of continuously forcing the live tab state
	- `src/MdxViewer/ViewerApp_Pm4Utilities.cs` now uses non-closable tab items and applies `SetSelected` only when there is a pending tab request
	- pending tab request is cleared after tab-bar draw, so manual tab clicks persist across frames
- documentation update landed in the same slice:
	- `src/MdxViewer/README.md` now explicitly documents fixed sidebars as startup default and records the missing screenshot-guide follow-up for key workflows
- runtime boundary:
	- no automated tests were added or run for this UI-state fix
	- no live viewer/runtime signoff has been completed yet for tab persistence in the PM4 workbench

### Mar 30, 2026 - PM4 Sidebar Tabs Restored And Hover Info Hit Testing Narrowed

- followed direct user runtime feedback that the PM4 sidebar workflow had regressed badly after the inspector-first shell change:
	- `Overlay`, `Selection`, and `Correlation` felt like dead tabs because the workbench section was effectively missing
	- sidebar match-save flows were blocked because PM4 selection could still yield to normal scene picks
	- `WL*` hover/info was too hard to reach near PM4 content and the hover radius felt too broad
- landed behavior:
	- `ViewerApp_Pm4Utilities.OpenPm4Workbench(...)` now forces the right inspector open for PM4 workbench requests
	- `ViewerApp_Sidebars.DrawRightSidebar()` now renders `PM4 Workbench` whenever a world scene exists instead of hiding it until overlay or selection state already exists
	- `ViewerApp.PickObjectAtMouse(...)` now prefers the hovered PM4 object before regular scene-hit arbitration and uses the same preference for `Shift + Left Click` collection adds
	- `WorldScene` now separates hover-info hit testing from the larger wireframe-reveal brush, using a tighter screen-space brush for `WMO`, `MDX`, `WL*`, and PM4 hover cards while preserving the broader reveal brush for overlay visibility
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/ViewerApp_Pm4Utilities.cs`, `src/MdxViewer/ViewerApp_Sidebars.cs`, and `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run for this viewer-side interaction fix
	- no live viewer/runtime signoff has been completed yet for restored PM4 sidebar behavior or WL hover reachability on the development map

### Mar 30, 2026 - PM4 Camera-Window Load Regression Reduced By Removing A Redundant Zero-CK24 Link Rescan

- followed the user runtime report that PM4 overlay loads were stalling around `1/12` or `1/15` camera-window files and effectively never finishing
- root cause found in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the Mar 30 zero-`CK24` regrouping follow-up added `SplitZeroCk24SeedGroup(...)`
	- that path did an extra `MSLK` grouping pass and then re-scanned each returned group again to decide whether to preserve it or connectivity-split it
	- on large zero-`CK24` seed groups this added avoidable whole-link rescans to an already expensive PM4 object assembly path
- landed fix:
	- added shared `TryPartitionSurfaceGroupByMslk(...)` so both `SplitSurfaceGroupByMslk(...)` and `SplitZeroCk24SeedGroup(...)` reuse one partition result
	- zero-`CK24` handling now keeps linked `MSLK` families intact and only connectivity-splits the true unlinked remainder, without rebuilding the same grouping state again
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run for this viewer-side PM4 load path
	- no live viewer/runtime signoff has been completed yet on the fixed development map workflow

### Mar 30, 2026 - PM4 Unknowns Family Expansion Landed For MSLK And MSUR Attribution

- extended `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4ResearchUnknownsAnalyzer.cs` and the shared unknowns report contracts so `pm4 unknowns` now emits family-level attribution summaries instead of only field distributions and edge-fit counts
- landed new report sections for:
	- dominant `MSLK` families grouped by `TypeFlags` or `Subtype` or `SystemFlag`
	- dominant `MSUR` families grouped by `AttributeMask` or `GroupKey` or `IndexCount`
	- per-family linkage signals against direct `MSUR` fits, `MPRL` fits, `LinkId` patterns, `GroupObjectId -> MPRL.Unk04`, `CK24`, `MDOS`, and incoming-link fanout
- fixed-corpus result on `gillijimproject_refactor/test_data/development/World/Maps/development` now gives sharper evidence for where to dig next:
	- dominant `MSLK` families are sentinel-tile-link heavy and concentrated in a small repeated set of `TypeFlags` or `Subtype` combinations
	- dominant `MSUR` families split between large zero-`CK24` umbrella families and non-zero-`CK24` object-facing families with much broader `CK24` and `MDOS` fanout
	- this strengthens the current interpretation that some `group=3` zero-`CK24` families are umbrella/root-style surfaces, while several `group=18` families are better candidates for object-facing attribution analysis
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 30, 2026
	- `dotnet i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll pm4 unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/wow-viewer/output/pm4_unknowns_development_report.json` passed on Mar 30, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter UnknownsDirectory_DevelopmentCorpus_ProducesExpectedHighLevelSignals` passed on Mar 30, 2026
- interpretation boundary:
	- this still does not close the actual names or bit-level semantics of `MSLK.TypeFlags` or `Subtype` or `MSUR.AttributeMask` or `GroupKey`
	- it does give a much better corpus-scale target list for the next outlier or family-specific investigation

### Mar 30, 2026 - PM4 MSHD Corpus Analyzer And Inspect Verb Landed

- extended `wow-viewer/src/core/WowViewer.Core.PM4` with a dedicated `Pm4ResearchMshdAnalyzer` plus reusable MSHD report models so `MSHD` can be measured against actual PM4 chunk-family and grouping metrics instead of guessed from one-off tiles
- added a new Tool.Inspect verb:
	- `pm4 mshd --input <directory> [--output <report.json>]`
- fixed-corpus result on `gillijimproject_refactor/test_data/development/World/Maps/development` currently weakens the specific theory that `MSHD` directly stores active root-group or type-bucket counts:
	- `616` files scanned, `502` with `MSHD`
	- `Field0C..Field1C` are zero in all `502` sampled headers
	- `Field00 == Field08` in only `233/502` files
	- no measured field produced the kind of strong exact-match or high-correlation signal that would directly tie it to current `MSLK` or `MSUR` or `MPRL` grouping counts in this corpus slice
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 30, 2026
	- `dotnet i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll pm4 mshd --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/wow-viewer/output/pm4_mshd_development_report.json` passed on Mar 30, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter MshdDirectory_DevelopmentCorpus_PreservesCurrentTrailingZeroAndWeakBucketSignals` passed on Mar 30, 2026
- interpretation boundary:
	- `MSHD` is still unresolved semantically
	- this slice rules against one specific bucket-count interpretation for the current development corpus; it does not prove the surviving fields are inert or final placeholders in every PM4 family

### Mar 30, 2026 - PM4 Workbench Moved Into The Inspector And Startup PM4 Noise Reduced

- followed the user's shell-overhaul request instead of adding another isolated PM4 panel:
	- PM4 bounds now start disabled
	- PM4 x-ray now starts disabled
	- fixed sidebars are now the default startup shell mode so the inspector no longer drifts by default
	- `World Objects` now keeps only a light PM4 summary plus a `PM4 Workbench` entry point
	- the right sidebar now owns the main PM4 workflow through one consolidated workbench surface
	- the hover card now stays shorter and more selection-oriented, pushing the heavy detail into click-time inspection instead of mouse-over spam
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing project warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet on the development map for the new inspector-first PM4 shell

### Mar 30, 2026 - PM4 Hover Overlay Now Uses A Tooltip-Style Card And Hover-Time Match Preview

- followed the user's request to make the new hover overlay look more like a WoW item tooltip and to expose PM4 potential matches directly from mouse-over
- landed behavior:
	- `WorldScene.UpdateHoveredAssetInfo(...)` now recognizes PM4 overlay objects and returns a PM4 object key for the hovered part when the PM4 overlay is active
	- `ViewerApp.DrawSceneHoverAssetOverlay()` now renders a darker gold-bordered tooltip-style card with brighter title text and stronger path or detail styling
	- hovered PM4 parts now show a compact top-candidate list sourced from a separate hovered-object PM4 match cache, so the overlay can preview likely `WMO` or `M2` matches without changing selection
	- PM4 derived-report invalidation now also clears the hovered-object match cache so tooltip suggestions do not go stale after PM4 reloads or regrouping changes
- continuity updates landed alongside the code change:
	- `wow-viewer/README.md` now records the current `CK24` low-16 versus `CK24=0x000000` research framing
	- `plans/wow_viewer_pm4_library_plan_2026-03-25.md` now records the same note and keeps the hover or graph ranking surfaces labeled as research instrumentation only
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet for PM4 hover-time tooltip behavior or candidate quality

### Mar 30, 2026 - PM4 Shift-Click Collection Now Treats PM4 As The Primary Target

- followed direct runtime feedback that the first PM4 multi-select slice was still failing in practice:
	- `Shift + Left Click` could silently do nothing because normal scene-object hit priority still beat PM4 picking
	- per-item collection removal could leave stale PM4 highlight state behind
- landed behavior:
	- `ViewerApp.PickObjectAtMouse(...)` now handles `addPm4ToCollection` as a PM4-first branch, directly selecting and toggling the PM4 part when any PM4 hit exists under the cursor instead of comparing against regular scene hits first
	- failed additive clicks now report a clear status message telling the user no PM4 hit was found and to use graph `Collect` buttons for dense overlaps
	- per-item `Remove` in `ViewerApp_Pm4Utilities.DrawPm4ObjectCollectionSummary(...)` now resyncs PM4 collection highlighting immediately
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet for the corrected Shift-click PM4 collection path

### Mar 30, 2026 - PM4 Collection JSON Export Added For Multi-Part Comparison

- added a viewer-side PM4 collection workflow to help inspect whether several PM4 parts are one object family or repeated overlapping copies
- landed behavior:
	- graph-driven collection now provides direct `Collect` buttons on MSLK link groups, MDOS groups, and individual parts, so the workflow does not depend on unreliable viewport PM4 picking
	- collected PM4 parts now show a distinct in-scene highlight color in the PM4 overlay and bounds path
	- export JSON now includes per-part debug info, merged-group ownership, signature buckets, same-signature center-overlap clusters, and `likelyDuplicateScore` metrics for quick duplicate inspection
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/Terrain/WorldScene.cs`, `src/MdxViewer/ViewerApp.cs`, and `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet for the new PM4 collection workflow

### Mar 30, 2026 - PM4 Selection Family Split Rolled Back To Stop Half-Object Selection

- user runtime feedback after the Mar 30 selection-family experiments showed a hard regression:
	- the separate family-selection path could first pull in unrelated nearby PM4 pieces
	- after the same-tile helper rollback, it could also under-select and visibly split one object into smaller fragments
- landed rollback in `src/MdxViewer/Terrain/WorldScene.cs`:
	- removed `_pm4SelectedObjectFamilyGroupKeys`
	- removed `_selectedPm4ObjectFamilyGroupKey`
	- returned PM4 selection, highlight, and selected-object graph grouping to `_pm4MergedObjectGroupKeys`
	- kept the selected-only PM4 match cache path and selected-object match builder introduced in `ViewerApp` and `ViewerApp_Pm4Utilities`
- validation completed:
	- file diagnostics for `src/MdxViewer/Terrain/WorldScene.cs` were clean after the rollback
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing solution warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet after this rollback

### Mar 30, 2026 - Selected PM4 Match Lookup Stops Rebuilding The Global Report And Zero-Root Ranking Stops Forcing WMO First

- followed new user feedback that clicking one zero-`CK24` part stalled before the sidebar reacted and then surfaced obviously wrong `WMO` suggestions for what looked like `M2`-family data
- root causes confirmed in the active viewer path:
	- `ViewerApp_Pm4Utilities.TryGetSelectedPm4ObjectMatch(...)` was forcing the full global `BuildPm4ObjectMatchReport(...)` path even when the UI only needed the currently selected PM4 object
	- `WorldScene.BuildPm4ObjectMatchReport(...)` hard-ranked all candidates with `WMO` mesh evidence first, which skewed zero/root-family selections toward `WMO` even before local anchor or overlap evidence could compete
- landed behavior:
	- the selected-object sidebar/window path now uses a lightweight selected-object PM4 match builder plus a small cache keyed by the selected PM4 object and match-count setting instead of rebuilding the global object-match report on click
	- `WorldScene` now reuses a shared object-match evaluation helper for both the full report and the selected-object path
	- zero/root PM4 objects (`CK24 == 0` or root-like link ownership) no longer get blanket `WMO`-first ranking; when linked refs exist they now prefer `M2` candidates before the normal same-tile, anchor-gap, planar, overlap, and footprint checks
	- non-zero PM4 families keep the old `WMO`-mesh-first ranking path
- validation completed:
	- editor diagnostics were clean for `src/MdxViewer/Terrain/WorldScene.cs`, `src/MdxViewer/ViewerApp.cs`, and `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026
- important boundary:
	- this is still build validation only in this session
	- no live viewer/runtime signoff has been completed yet to prove the click latency is now acceptable or that the zero/root top matches are materially better on the development map

### Mar 30, 2026 - Zero-CK24 Same-Tile Merge Gap Closed In WorldScene

- traced the remaining zero/root PM4 selection fragmentation in `src/MdxViewer/Terrain/WorldScene.cs` past the seed split and into the later merged-group map that drives selected-family ownership
- root cause confirmed in this slice:
	- zero-`CK24` parts already use synthetic per-part keys by default
	- `_pm4MergedObjectGroupKeys` was the only later regrouping seam
	- shared `Core.PM4` merged-group math intentionally skips same-tile merges, so same-tile zero-`CK24` families could never be recombined there
- landed behavior:
	- preserved shared cross-tile connector merging
	- added a local same-tile merge pass for synthetic zero-`CK24` keys using connector overlap plus local frame evidence from bounds, placement anchors, linked `MPRL` floors, and linked-heading summaries
- supporting real-data evidence recorded during the slice:
	- zero-`CK24` forensic export on `development_23_18.pm4` reported `1150` surfaces across `204` distinct link groups, with `203` non-zero `MSLK.GroupObjectId` values
	- this supported the conclusion that the missing family was not simply one blob with no link ownership
- validation completed:
	- editor diagnostics for `src/MdxViewer/Terrain/WorldScene.cs` were clean after the final fix
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no live viewer/manual signoff yet for the new same-tile zero-`CK24` regrouping behavior on the development map

### Mar 30, 2026 - Zero-CK24 Regrouping Stops Skipping MSLK Ownership In WorldScene

- narrowed the remaining zero/root PM4 regrouping problem in `src/MdxViewer/Terrain/WorldScene.cs` to the first split step after `(GroupKey, AttributeMask)` seed buckets were formed
- previous behavior:
	- zero-`CK24` seed buckets always skipped `SplitSurfaceGroupByMslk(...)`
	- they went directly into connectivity splitting, so disconnected but still `MSLK.GroupObjectId`-related pieces were split apart before placement or matching logic could treat them as one family
- landed behavior:
	- zero/root seed buckets now run through `SplitZeroCk24SeedGroup(...)`
	- groups with a non-zero dominant `MSLK.GroupObjectId` stay intact
	- only the remaining groups with no `MSLK` ownership evidence fall back to connectivity splitting
	- linked `MPRL` collection now also follows `MSLK.GroupObjectId` families first when one of those groups is already established for the current PM4 subgroup
- important semantic boundary:
	- this does not restore the old fake `MSLK.MsurIndex` path; current repo evidence still supports the 20-byte active `MSLK` layout and keeps `RefIndex` semantics partially open
- validation completed:
	- editor/file diagnostics for `src/MdxViewer/Terrain/WorldScene.cs` were clean after the change
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` was attempted on Mar 30, 2026 but did not complete because `ParpToolsWoWViewer (16096)` held output DLL locks (`MSB3021` / `MSB3027` copy failures)
	- no live viewer/runtime signoff was completed yet on the development map

### Mar 29, 2026 - v0.4.6 Release Prep Aligns PM4 Wins With The Next Renderer Seam

- user runtime feedback after the latest PM4 runtime fixes is now strongly positive: PM4 objects are described as almost `100%` correct on the development map
- release target is now being moved from `0.4.5` to `0.4.6`
- release-facing notes that need to stay grouped for this build:
	- PM4 overlay decoding and placement improved through the recent camera-window, tile-remap, empty-carrier, and linked-group placement fixes
	- first rendering-performance slices landed by removing repeated MDX visibility work and deferring WMO doodad expansion
- next renderer priority recorded for continuity:
	- add real render layers / submission buckets instead of keeping all world-scene submission embedded directly in `WorldScene.Render(...)`
	- focus the next performance slice on draw-call/state churn and layer ownership, not only on another isolated culling micro-fix
- validation/build boundary for this continuity update:
	- versioning and release-note surfaces were aligned in this pass
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Release` passed on Mar 29, 2026 with existing warnings only
	- local self-contained publish for `0.4.6` completed successfully after updating the release workflow publish step to tolerate duplicate dependency-side publish outputs from `WoWRollback.PM4Module`
	- local release archive `parp-tools-wow-viewer-v0.4.6-win-x64.zip` was produced on Mar 29, 2026 and the publish output still bundled `1315` WoWDBDefs `.dbd` files
	- the first cloud `v0.4.6` Actions run failed for two real release-workflow reasons, not because of local environment dirt:
		- the root release workflow, not the `gillijimproject_refactor` copy, was the workflow GitHub actually executed
		- cross-platform publish was still assuming a bundled `WowViewer.Core.IO` `area_crosswalk.csv` resource that is not tracked and should not be shipped
	- follow-up fix landed on Mar 29, 2026:
		- `WowViewer.Core.IO` now treats the embedded area crosswalk as optional instead of required, keeping runtime mapping on archive-backed or explicit user data paths
		- `MdxViewer.CrossPlatform.csproj` now carries the same `WowViewer.Core.IO` and `WowViewer.Core.PM4` references as the Windows project so Linux publish no longer fails on missing PM4 namespaces
		- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter AreaIdMapperTests` passed on Mar 29, 2026
		- `dotnet publish i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.CrossPlatform.csproj -c Release -f net10.0 -p:TargetFramework=net10.0 -r linux-x64 --self-contained true -p:PublishSingleFile=false -p:IncludeNativeLibrariesForSelfExtract=true -p:ErrorOnDuplicatePublishOutputFiles=false -p:TreatWarningsAsErrors=false -o i:/parp/parp-tools/output/MdxViewer-linux-x64-smoke` passed on Mar 29, 2026

### Mar 29, 2026 - WMO Doodads Stop Eagerly Expanding On The Render Thread And Object Fog Defaults Off

- followed the first `WorldScene` render-pass optimization with a second narrower slice aimed at the remaining reported symptoms:
	- multi-second hitches while tiles or data stream in
	- world objects appearing inside unwanted fog
- root cause for the new hitch slice:
	- `src/MdxViewer/Rendering/WmoRenderer.cs` constructor work still eagerly loaded the active doodad set, which recursively constructs doodad `MdxRenderer`s and can stall badly when new WMOs enter view
- landed behavior:
	- added deferred initial doodad loading state to `WmoRenderer` so world-scene WMO shells can appear first and doodad model loads are then drained incrementally under a small per-frame budget
	- `src/MdxViewer/Terrain/WorldAssetManager.cs` now creates world-scene `WmoRenderer` instances with `deferInitialDoodadLoads: true`
	- `src/MdxViewer/Terrain/WorldScene.cs` now reduces main-thread deferred asset processing to `6` loads with a `4 ms` budget per frame
	- `WorldScene` now uses a dedicated object-fog policy that defaults off, while WMO cull distance still uses terrain fog end instead of the disabled object-fog range
	- `src/MdxViewer/ViewerApp.cs` now exposes `Fog Objects` in the world-objects UI so the fogged-object path can be toggled back on when needed
- validation completed:
	- `get_errors` returned clean for the touched viewer files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings only
- runtime boundary:
	- this is compile-validated only in this session
	- no manual viewer traversal or measured hitch reduction was captured yet after the deferred doodad change

### Mar 29, 2026 - WorldScene MDX Render Passes Stop Repeating The Same Visibility Work

- started the user-requested performance pivot away from PM4-first work by targeting the hottest obvious CPU path in `src/MdxViewer/Terrain/WorldScene.cs`
- root cause for this first slice:
	- the scene was re-walking `_mdxInstances` and `_taxiActorInstances` across separate animation/update, opaque, and transparent passes
	- opaque and transparent passes were repeating frustum checks, AABB distance checks, and `TryGetQueuedMdx(...)` lookups for the same instances in the same frame
- landed behavior:
	- added a reusable visible-instance scratch list for loaded MDX/taxi instances
	- `CollectVisibleMdxInstances(...)` now performs the cull and renderer-resolution pass once, computing reusable opaque/transparent fade factors at the same time
	- opaque rendering now iterates the preclassified visible list
	- transparent rendering now sorts only the already-visible list instead of re-culling the world again
- validation completed:
	- `get_errors` on `WorldScene.cs` returned clean
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings only
- runtime boundary:
	- this is compile-validated only in this session
	- no measured frame-time comparison or manual viewer FPS confirmation has been captured yet

### Mar 29, 2026 - PM4 Terminology Guardrail Synced Across wow-viewer Continuity Files

- reconciled current PM4 field names against wowdev `PM4` and `PD4` documentation plus the standalone corpus-backed confidence reports
- recorded the hard boundary that several active names are local research aliases rather than documentation-backed field names:
	- `MSUR.GroupKey`
	- `MSUR.AttributeMask`
	- `MSUR.MdosIndex`
	- `MSUR.PackedParams`
	- derived `CK24`, `Ck24Type`, `Ck24ObjectId`
	- `MSLK.GroupObjectId`
- also recorded the stronger current corrections that should survive even when names change:
	- `MSUR` bytes `0x04..0x0f` are geometry-validated normals
	- the current `MSUR.Height` name is misleading because the float behaves like a signed plane-distance term
	- `MSLK.RefIndex` is not closed as a universal `MSUR` index across the corpus
- continuity surfaces updated:
	- `gillijimproject_refactor/src/Pm4Research.Core/README.md`
	- `gillijimproject_refactor/memory-bank/activeContext.md`
	- `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`
	- `wow-viewer/README.md`
	- `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
- validation boundary:
	- this was a terminology and continuity correction only; no new PM4 code or runtime behavior changed in this pass

### Mar 29, 2026 - Shared PM4 Hierarchy Research Slice Landed And `MdxViewer` Cut Over

- moved the active viewer PM4 research path off legacy `src/Pm4Research.Core` and into shared `wow-viewer/src/core/WowViewer.Core.PM4`
- landed `Research/Pm4ResearchHierarchyAnalyzer.cs` in `Core.PM4`, porting the old split-family object-hypothesis research and extending each candidate with:
	- dominant `MSLK.GroupObjectId`
	- shared placement comparison (`CoordinateMode`, planar transform, world pivot, frame yaw, heading delta)
	- the existing MPRL footprint evidence
- added `WowViewer.Tool.Inspect pm4 hierarchy --input <file.pm4> [--output <report.json>]`
- rewired `src/MdxViewer/Terrain/WorldScene.cs` so selected-object PM4 research now uses shared snapshot, shared decode audit, and shared hierarchy analysis from `Core.PM4`
- updated the `PM4 Research` viewer panel to show the new shared hierarchy and placement signals for top hypothesis matches
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 29, 2026 with existing environment warnings
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter "Hierarchy_DevelopmentTile_ExposesSharedPlacementAndLinkGroupEvidence|Pm4ResearchIntegrationTests"` passed on Mar 29, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 hierarchy --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` produced a real hierarchy report on Mar 29, 2026
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this lands more grounded PM4 scene-graph evidence in shared code and in the viewer UI, but it does not claim the CK24 placement regression is solved yet

### Mar 29, 2026 - PM4 Incremental Loads Stop Clearing Prior Residency

- fixed a real PM4 runtime residency bug in `src/MdxViewer/Terrain/WorldScene.cs` that could make PM4 objects disappear permanently while moving around the viewer, especially when crossing into a new PM4 camera-window load
- root cause:
	- `BeginPm4OverlayLoad(...)` was resetting `_pm4LoadedCameraWindow` to `null` before the async load finished
	- `TryFinalizePm4OverlayLoad()` uses `_pm4LoadedCameraWindow.HasValue` to decide whether a load should merge into existing PM4 state or replace it
	- because the window had already been nulled, every incremental load finalized as a full replacement and cleared earlier PM4 tiles instead of preserving them
- landed behavior:
	- normal background PM4 loads now keep the previously loaded camera window intact until finalize decides whether to merge
	- manual `Reload PM4` now explicitly clears PM4 runtime state first and then starts a fresh cache-bypassing load, so reload behaves like a real full reload instead of a partial merge with stale bookkeeping
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is build-validated only; the exact user repro of “PM4 disappears as I approach and reload does not bring it back” still needs manual viewer confirmation after this fix

### Mar 29, 2026 - PM4 Same-Tile Candidate Collisions Now Keep One Canonical File

- narrowed the PM4 file-selection path in `src/MdxViewer/Terrain/WorldScene.cs` so both runtime loading and offline PM4 OBJ export now keep only one preferred `.pm4` candidate per effective tile instead of blindly merging every file that parses to the same tile coordinate
- reason: the latest non-zero `CK24` graph exports still showed exact paired duplicate parts like `part=0` and `part=495` even with `Split CK24 by MdosIndex` and `Split CK24 by Connectivity` disabled, which strongly fits same-tile candidate collisions rather than a pure transform-math bug
- current selection policy prefers the most canonical `.../World/Maps/<map>/<map>_<x>_<y>.pm4` style path and logs dropped same-tile candidates for follow-up diagnosis
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is still build-validated only; manual viewer confirmation is still required to prove the paired duplicate-part pattern and opposite-corner PM4 placements actually disappear on the development map

### Mar 29, 2026 - Global PM4 Y Mirror No Longer Applies To Zero-CK24 Root Buckets

- narrowed the PM4 object transform path in `src/MdxViewer/Terrain/WorldScene.cs` so the global `Mirror PM4 N/S` flip no longer applies to `CK24=0x000000` objects
- reason: current live viewer evidence showed the bad zero/root PM4 overlays snapping back into the correct placed-object location only after pressing `Wind Obj Y`, which is effectively cancelling the global Y mirror on that selected object
- preserved the global Y mirror for non-zero `CK24` groups, since those were the original reason the default north/south mirror was enabled in the first place
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is still build-validated only; manual viewer confirmation is still required to prove the zero/root PM4 objects now overlap the placed objects again without using `Wind Obj Y`

### Mar 29, 2026 - Zero-CK24 Seed Groups No Longer Re-Split After Seed Connectivity Split

- narrowed the PM4 zero/root-bucket runtime path in `src/MdxViewer/Terrain/WorldScene.cs` so seed groups that already require the mandatory connectivity split no longer also honor the later viewer toggle stages for `Split CK24 by MdosIndex` or `Split CK24 by Connectivity`
- reason: the current viewer evidence and graph exports point to zero/root `CK24` groups being fragmented twice, which can manufacture paired sub-parts after the first seed-level split and then make later frame or winding experiments look like rotation regressions instead of grouping regressions
- preserved the later MDOS/connectivity toggle behavior for non-zero `CK24` groups; this change is only for the zero/root seed path that already split once at the seed stage
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is build-validated only; no live viewer signoff was completed after the regrouping change, so the development map still needs manual confirmation that the artificial paired-part split is actually gone

### Mar 29, 2026 - Viewer Shell Resize/Input Sync Hardened Again

- fixed the recurring viewer-shell regression in `src/MdxViewer/ViewerApp.cs` where resize and mouse-hit behavior could break again even outside PM4-specific windows
- the old bridge only partially synchronized Silk window metrics into ImGui: it used the private `ImGuiController.WindowResized(...)` hook against logical size, but did not explicitly keep `ImGuiIO.DisplaySize` and `DisplayFramebufferScale` synchronized from both logical and framebuffer sizes
- `ViewerApp` now subscribes to both logical `Resize` and `FramebufferResize`, and `SyncImGuiWindowMetrics(...)` updates the private Silk hook plus explicit `ImGui` size/framebuffer-scale values before each `_imGui.Update(...)`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is build-validated only; no live shell resize or hit-testing signoff was completed after the patch

### Mar 29, 2026 - Zero-CK24 PM4 Mixed-Bucket Placement Fix Landed In MdxViewer

- fixed a PM4 runtime-consumer regression in `src/MdxViewer/Terrain/WorldScene.cs` where zero-`CK24` / root-style seed buckets were connectivity-split only after one shared placement basis had already been resolved for the whole mixed bucket
- root cause fit the latest manual symptom: some `M2`-aligned PM4 data drifted while nearby WMO-aligned PM4 remained mostly stable, because non-zero `CK24` WMO-style groups still had coherent per-`CK24` placement while zero/root buckets could mix unrelated parts before placement resolution
- the zero/root-style path now resolves coordinate mode or placement solution or connector keys per linked connectivity group instead of reusing one mixed-bucket planar transform or pivot or frame yaw across all zero-`CK24` parts in the seed bucket
- preserved the existing non-zero `CK24` behavior: shared per-`CK24` frame basis is still reused across split parts for the WMO-style path
- also recorded the user clarification that `CK24=0x000000` should not be treated as "just M2 data"; it is better treated as an unresolved root or umbrella bucket for now
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is compile-validated only; the development map still needs manual runtime confirmation to prove the M2/root-bucket drift is actually corrected

### Mar 29, 2026 - PM4 CK24 Alignment Controls Narrowed To Tile-Local Buckets

- replaced the earlier raw-`CK24` alignment state in `src/MdxViewer/Terrain/WorldScene.cs` so those transforms are now keyed by `(tileX, tileY, ck24)` instead of only `ck24`
- this stops exploratory fixes for `CK24=0x000000` from rotating or mirroring every matching raw bucket across the loaded PM4 overlay when the actual issue appears tile-local
- updated `src/MdxViewer/ViewerApp_Pm4Utilities.cs` so the PM4 alignment window now describes and edits tile-local `CK24` transforms, and added direct tile/object winding toggle buttons for faster handedness checks
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is build-validated only; no live viewer signoff was completed after the tile-local control change

### Mar 29, 2026 - Shared CK24 PM4 Forensic Export Slice Landed

- added `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4ForensicsModels.cs` and `Research/Pm4Ck24ForensicsAnalyzer.cs` so `Core.PM4` now owns a research-only CK24 forensic report instead of relying on viewer-only PM4 graph JSON
- the new report exposes component-level link groups, raw MSLK rows, raw linked MPRL rows, footprint counts, and placement-vs-heading comparison for a target `CK24`
- extended `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` so `pm4 export-json --ck24 <decimal|0xHEX>` emits the targeted CK24 forensic report while the original no-`--ck24` path still emits the coarse single-file PM4 analysis report
- fixed PM4 inspect JSON serialization for vector payloads by enabling field serialization, so the new shared forensic JSON shows real `Vector3` coordinates instead of empty objects
- added real-data PM4 regression coverage in `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4ResearchIntegrationTests.cs` for a dense linked CK24 case (`0x412CDC`) and a sparse no-linked-MPRL case (`0x41C0F5`) on `development_00_00.pm4`
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 29, 2026 after the slice landed
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter Pm4ResearchIntegrationTests` passed on Mar 29, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect -- pm4 export-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4 --ck24 0x412CDC --output i:/parp/parp-tools/wow-viewer/output/pm4_ck24_412CDC_forensics.json` wrote real-data shared forensic JSON
- boundary:
	- this slice proves shared export and regression behavior, not final PM4 runtime semantics or viewer signoff

### Mar 29, 2026 - PM4 CK24 Frame Yaw No Longer Rotates Visible Mesh Geometry

- changed the PM4 CK24 object-generation path in `src/MdxViewer/Terrain/WorldScene.cs` so `worldYawCorrection` stays on the object frame or anchor path instead of being baked directly into the generated mesh lines and triangles
- root cause was viewer evidence that CK24 objects were being visually rotated and displaced as though frame correction had been applied to the mesh itself, which inverted the intended ownership between visible geometry and the object frame basis
- `BuildCk24ObjectLines(...)` and `BuildCk24ObjectTriangles(...)` now convert PM4 mesh vertices without the CK24 frame yaw correction, while placement-anchor computation still retains the frame-yaw path
- validation completed:
	- editor/language-service checks passed for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this fix has not yet been re-checked in the live viewer during this session, so the exact effect on the opposite-side `CK24` placements still needs manual confirmation

### Mar 29, 2026 - PM4 Raw CK24 Layer Alignment Added In MdxViewer

- added a parallel PM4 raw-`CK24` layer transform path in `src/MdxViewer/Terrain/WorldScene.cs` so any selected PM4 object can now drive whole-layer translation or rotation or scale keyed by the original `ck24` value instead of only the resolved object-group key
- this specifically unblocks exploratory work on `CK24=0x000000`, which had been structurally split into synthetic per-part groups for object transforms and therefore could not previously be rotated as one layer
- `BuildPm4ObjectTransform(...)` now applies raw-`CK24` layer transform before the existing object-group transform, and the scene now tracks raw-layer pivots from combined bounds across all loaded tiles
- extended `src/MdxViewer/ViewerApp_Pm4Utilities.cs` with `CK24 Layer` move or rotate or scale controls, reset actions, and a print action while preserving the existing per-object-group controls beneath them
- extended PM4 interchange JSON reporting so each exported object also includes the raw-layer transform state currently affecting its `ck24`
- validation completed:
	- editor/language-service checks passed for `WorldScene.cs` and `ViewerApp_Pm4Utilities.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- the new `CK24 Layer` controls were not yet exercised in the live viewer during this session, so the `0x000000` per-tile or quadrant-orientation hypothesis remains unverified

### Mar 28, 2026 - PM4 Graph JSON Export No Longer Fails On Non-Finite Heading Values

- fixed the selected-object `PM4 Graph` JSON export in `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
- root cause was raw `System.Text.Json` serialization of `Pm4LinkedPositionRefSummary`, whose heading fields can be non-finite when a graph link group has no normal heading evidence
- replaced raw struct serialization with a JSON-safe projected payload and finite-or-null handling for linked-position-ref heading values so the export stays valid standard JSON instead of throwing in the status bar
- also repaired the remaining `Pm4OverlayCacheService` type reference so the earlier shared `Pm4PlanarTransform` cleanup still builds cleanly
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 28, 2026 after the fix
- runtime boundary:
	- the export button itself was not re-clicked in this session after the patch, so UI-level runtime confirmation is still pending

### Mar 28, 2026 - PM4 Overlay North/South Mirror Defaulted In MdxViewer

- removed the duplicate viewer-local `Pm4PlanarTransform` contract from `src/MdxViewer/Terrain/WorldScene.cs` so the active PM4 consumer now uses the shared `WowViewer.Core.PM4` placement contract directly
- updated the CK24 coordinate-mode consumer path to keep the shared typed `Pm4CoordinateModeResolution` result instead of collapsing it to a bool immediately
- switched viewer-side default world-space planar-transform fallback usage onto shared `Pm4PlacementContract`
- enabled the existing PM4 object-group mirror path by default through `_pm4FlipAllObjectsY = true`, because current manual runtime evidence showed PM4 overlay geometry landing on the wrong north/south side of nearby `M2` placements with handedness-related rotation mismatch
- kept the PM4 correlation-input builder viewer-local for now; the current correction is about overlay alignment, not correlation ownership
- updated the PM4 UI toggle label in `src/MdxViewer/ViewerApp.cs` from `Flip All Obj Y` to `Mirror PM4 N/S` so the control matches the behavior it actually applies
- validation so far is editor/language-service only; no build or new real-data runtime signoff was completed in this pass because the requested `MdxViewer` build was cancelled before completion

### Mar 28, 2026 - Post-MDX Default Continuation Reset

- recorded that the default next implementation area after pausing `MDX` is `wow-viewer` `Core.PM4` library completion
- preserved non-`MDX` shared-I/O as a secondary path only for narrow ADT/WDT/WMO slices with concrete proof targets

### Mar 28, 2026 - MDX Audit Reclassified Recent Shared Readers

- audited recent `wow-viewer` classic `MDX` work against active `MdxViewer` implementation instead of continuity claims alone
- confirmed that `GEOS` is the clearest real classic-parser parity seam among the recent payload slices
- reclassified `TXAN` payload ownership as a new shared-reader seam informed by legacy runtime concepts and M2-adapter usage, not a direct classic `MdxViewer` parser port
- reclassified `HTST` payload ownership as a new shared-reader seam with no current active classic `MdxViewer` parser/runtime parity
- reclassified `CLID` payload ownership as beyond active classic parser parity; active viewer currently uses only shared collision summary metadata in probe/model-info surfaces
- identified the hotter missed legacy parity seam: classic `ATSQ` geoset-animation and related material animation behavior already used by the active renderer

### Mar 28, 2026 - MDX Chunk Expansion Explicitly Paused

- recorded user direction that further speculative `MDX` chunk implementation should stop
- future continuation should not treat unresolved `MDX` families as the automatic next slice just because carrier discovery or inspect support exists
- only resume new `MDX` chunk work if the user explicitly asks for a named seam or if a concrete active consumer requirement makes it necessary

### Mar 29, 2026 - Shared Classic `MDX` `TXAN` Payload Slice Landed

- advanced the shared classic `MDX` migration from unresolved `TXAN` discovery into first typed texture-animation payload ownership so actual `KTAT` or `KTAR` or `KTAS` keyframes no longer remain only as a top-level chunk id in inspect output
- added shared `MdxTextureAnimationFile` and `MdxTextureAnimation` contracts for indexed classic texture-animation entries
- added `WowViewer.Core.IO.Mdx.MdxTextureAnimationReader` for classic `TXAN` payload reads in `v1300` and `v1400`, including counted section framing and actual translation or rotation or scaling keyframe payload parsing
- extracted the reusable vector3 and compressed-quaternion keyframe parsing into `WowViewer.Core.IO.Mdx.MdxTrackReader` and switched `MdxHitTestReader` to reuse it, so `HTST` and `TXAN` now share one track interpretation instead of drifting apart
- extended `WowViewer.Tool.Inspect mdx export-json` with `--include-texture-animations` so the new library-owned payload seam is immediately reusable on filesystem or archive-backed inputs without adding a tool-local parser path
- added `wow-viewer/tests/WowViewer.Core.Tests/MdxTextureAnimationReaderTests.cs` with a synthetic tracked texture-animation fixture, a real Alpha no-TXAN regression on `Wisp.mdx`, and a fixed real standard-era archive-backed AirElemental case
- this landing proves payload ownership and JSON export only; it does not claim runtime texture-transform evaluation or viewer cutover
- this should not be used as justification to continue automatic `MDX` chunk expansion work

### Mar 29, 2026 - Shared Classic `MDX` `HTST` Payload Slice Landed

- advanced the shared classic `MDX` migration from `HTST` summary-only ownership into first typed hit-test payload ownership so fixed shape payloads and actual `KGTR` or `KGRT` or `KGSC` keyframes no longer have to stay trapped behind summary-only metadata
- added shared `MdxHitTestFile` and `MdxHitTestShape` contracts plus reusable typed node-track payload contracts for vector3 and compressed-quaternion keyframes with interpolation metadata
- added `WowViewer.Core.IO.Mdx.MdxHitTestReader` for classic `HTST` payload reads in `v1300` and `v1400`, including fixed box or cylinder or sphere or plane payloads and actual transform keyframe payload parsing
- extended `WowViewer.Tool.Inspect mdx export-json` with `--include-hit-test` so the new library-owned payload seam is immediately reusable on filesystem or archive-backed inputs without adding a tool-local parser path
- added `wow-viewer/tests/WowViewer.Core.Tests/MdxHitTestReaderTests.cs` with a synthetic tracked hit-test fixture, a fixed real Alpha `Wisp.mdx` case, and a fixed real standard-era archive-backed `anubisath.mdx` case
- this landing proves payload ownership and JSON export only; it does not claim runtime hit detection or viewer cutover

### Mar 28, 2026 - Shared Classic `MDX` `CLID` Payload Slice Landed

- advanced the shared classic `MDX` migration from `CLID` summary-only ownership into first typed collision-mesh payload ownership so full `VRTX` or `TRI ` or `NRMS` data no longer has to stay trapped behind inspect-only summaries
- added shared `MdxCollisionFile` and `MdxCollisionMesh` contracts plus `WowViewer.Core.IO.Mdx.MdxCollisionReader` for classic `CLID` payload reads in `v1300` and `v1400`
- extracted the chunk-level payload logic into a shared `MdxCollisionChunkReader` helper and switched `MdxSummaryReader` to reuse it, so summary and payload coverage now share one `CLID` interpretation instead of drifting apart
- extended `WowViewer.Tool.Inspect mdx export-json` with `--include-collision` so the new library-owned payload seam is immediately reusable on filesystem or archive-backed inputs without adding a new tool-local parser path
- added `wow-viewer/tests/WowViewer.Core.Tests/MdxCollisionReaderTests.cs` with a synthetic collision fixture, a fixed real Alpha `Wisp.mdx` case, and a fixed real standard-era archive-backed dwarf-character case
- this landing proves payload ownership and JSON export only; it does not claim runtime collision behavior or viewer cutover

### Mar 28, 2026 - `mdx export-json` Inspect Slice Landed

- added `WowViewer.Tool.Inspect mdx export-json` as a thin JSON export surface over the shared `MdxSummaryReader`, with optional `--include-geometry` over the current shared `MdxGeometryReader` seam
- kept the slice library-first: the new command reuses the shared readers for both filesystem and archive-backed inputs instead of adding any second `MDX` parser in the inspect tool
- fixed the initial JSON serialization bug by enabling field serialization for `System.Numerics` payloads, so vectors and UV data now serialize as real coordinates instead of empty objects
- validated the slice with real data on both the fixed Alpha `Wisp.mdx` summary surface and the fixed standard-era `chest01.mdx` summary-plus-geometry surface
- this slice is still export of already-owned seams, not new chunk-family ownership or runtime `MDX` parity

### Mar 28, 2026 - `mdx chunk-carriers` Inspect Workflow Landed

- added `WowViewer.Tool.Inspect mdx chunk-carriers --chunks <FOURCC[,FOURCC...]>` as a thin carrier-discovery surface over shared `MdxSummaryReader`
- kept the slice tool-thin and library-first: the new command scans either a filesystem `MDX` file or directory or an archive-backed dataset via `MpqArchiveCatalog`, but it still uses the shared summary reader for chunk ownership instead of adding tool-local chunk parsing
- added practical scan controls with `--path-filter <text>` and `--limit <n>` so standard archive scans can stay targeted when future sessions are looking for the next fixed positive carrier
- validated the new workflow with a real positive archive-backed `LITE` scan over `braziers`, which found `dwarvenbrazier01.mdx` plus `3` more positive standard-era carriers, and with a real negative alpha-corpus scan proving the current unpacked `0.5.3` sample set still has no `TXAN`, `PREM`, or `CORN` carriers across `229` files
- this slice does not add new `MDX` chunk-summary ownership by itself; it adds the discovery workflow needed to choose the next real-data-backed seam without guessing

### Mar 28, 2026 - Viewer UI Resize And Input Regression Fixed

- fixed a real `MdxViewer` shell regression where panels drew at the wrong size and toolbar or sidebar buttons stopped responding after window resize or maximize
- updated `src/MdxViewer/ViewerApp.cs` to explicitly resync the packaged Silk `ImGuiController` logical window size through its private `WindowResized(Vector2D<int>)` hook, while keeping the OpenGL viewport bound to framebuffer resize
- validated the patch with `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` plus a short viewer startup smoke
- the user manually retested the resized UI on Mar 28, 2026 and reported that it now seems to be working
- this is still manual runtime validation only; there is no automated UI regression coverage for the resize or hit-testing path

### Mar 28, 2026 - ViewerApp Shared `MDX` Runtime Metadata Consumer Landed

- started the first non-probe runtime `MDX` consumer cutover in `MdxViewer` without changing the renderer ownership boundary
- updated `src/MdxViewer/ViewerApp.cs` so the real `MDLX` disk or data-source route now reads shared `MdxSummaryReader` plus `MdxGeometryReader` metadata before the legacy `MdxFile.Load(...)` render load
- switched standalone runtime model-info and load-status counts to prefer shared version or model-name or geoset or vertex or triangle or pivot or collision metadata, while keeping explicit legacy fallback when shared reads fail
- validated the slice with `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` plus a real startup smoke on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` that printed `[SharedMDX] Runtime metadata consumer: summary=yes geometry=yes file=Wisp.mdx`
- this is still a metadata-only runtime consumer cutover, not a renderer cutover or world-scene asset-loader cutover

### Mar 28, 2026 - AssetProbe Shared `PIVT` And `CLID` Signals Landed

- expanded the existing `AssetProbe` shared `MDX` compatibility surface past summary or `GEOS` counts into visible shared pivot-table and collision-mesh reporting
- updated `src/MdxViewer/AssetProbe.cs` so `SharedMDX` probes now emit `SharedPIVT` and `SharedCLID` lines when the shared summary exposes those chunks
- validated the output on archive-backed real assets: `chest01.mdx` now reports `SharedPIVT: count=6`, and `Creature/AncientOfWar/AncientofWar.mdx` now reports both `SharedPIVT: count=72` and `SharedCLID: vertices=8 triangles=12`
- this is still probe-only validation; it does not move collision or pivot handling into the runtime renderer path by itself

### Mar 28, 2026 - AssetProbe Shared `GEOS` Consumer Cutover Landed

- advanced the active `MdxViewer` compatibility surface from shared `MDX` summary-only probe validation into first shared `GEOS` payload consumer validation
- updated `src/MdxViewer/AssetProbe.cs` so probe-side geoset reporting now comes from `WowViewer.Core.IO.Mdx.MdxGeometryReader` instead of depending on `MdxFile.Load(...)` geoset objects
- kept the cutover narrow: the probe still uses legacy `MdxFile.Load(...)` for the rest of the model parse and texture or material reporting
- validated the change by building `MdxViewer` and running `--probe-mdx` on both archive-backed `chest01.mdx` and `Creature/AncientOfWar/AncientofWar.mdx`, with the latter confirming full shared reporting across `5` geosets
- this is still non-UI compatibility validation only, not a runtime model-loading cutover

### Mar 28, 2026 - Shared Classic `MDX` `GEOS` Payload Slice Landed

- advanced the shared classic `MDX` migration from `GEOS` summary-only ownership into first typed geoset payload ownership so render-facing mesh data no longer has to stay trapped behind `MdxFile.Load(...)`
- added shared `MdxGeometryFile` and `MdxGeosetGeometry` contracts for vertices, normals, UV sets, primitive types, face groups, indices, vertex groups, matrix tables, bone tables, and footer metadata
- added `WowViewer.Core.IO.Mdx.MdxGeometryReader` with classic counted `GEOS` payload support for `v1300` and `v1400`, including Alpha-style direct `UVAS` reads and optional explicit `UVBS` support
- added `wow-viewer/tests/WowViewer.Core.Tests/MdxGeometryReaderTests.cs` with a synthetic classic-`GEOS` payload fixture, a fixed real standard-era positive carrier, and a real on-disk alpha-era positive carrier from the existing `0.5.3` corpus
- validated the slice with focused shared-reader tests against both standard-era and alpha-era real data
- this landing now has the first shared classic `GEOS` payload seam in `wow-viewer`; it is still not runtime buffer assembly, skinning evaluation, or viewer render cutover

### Mar 28, 2026 - Shared Classic `MDX` `LITE` Summary Slice Landed

- advanced the shared classic `MDX` migration from `GLBS` into `LITE` so classic light metadata no longer remains only as a known-but-unparsed top-level chunk id in `wow-viewer`
- added shared `MdxLightType` and `MdxLightSummary` and extended `MdxSummary` with `Lights` plus `LightCount`
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `LITE` summary support for `v1300` and `v1400`, including inherited node metadata plus static attenuation or color or intensity fields and optional `KLAS`, `KLAE`, `KLAC`, `KLAI`, `KLBC`, `KLBI`, and `KVIS` metadata
- updated `WowViewer.Tool.Inspect mdx inspect` to report `lights=` and print `LITE[n]` lines
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`LITE` fixture and a fixed real archive-backed `0.6.0` `dwarvenbrazier01.mdx` light case
- added a real unpacked `0.5.3` alpha-corpus smoke over `229` MDX files to prove the new `LITE` path does not regress current alpha-era parsing even though the bundled `0.5.3` sample set contains no `LITE` chunks today
- validated the seam with focused shared-reader tests plus real inspect output on `world/generic/dwarf/passive doodads/braziers/dwarvenbrazier01.mdx`
- this landing now has strong synthetic coverage plus fixed real standard `0.6.0` `MDX` validation for classic `LITE`

### Mar 28, 2026 - Shared Classic `MDX` `GLBS` Summary Slice Landed

- advanced the shared classic `MDX` migration from `CLID` into `GLBS` so global sequence duration tables no longer remain only as known-but-unparsed top-level chunk ids in `wow-viewer`
- added shared `MdxGlobalSequenceSummary` and extended `MdxSummary` with `GlobalSequences` plus `GlobalSequenceCount`
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with strict `GLBS` summary support for counted `uint32` durations and invalid payload-size rejection
- updated `WowViewer.Tool.Inspect mdx inspect` to report `globalSequences=` and print `GLBS[n]` lines
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic `GLBS` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` global-sequence case
- validated the seam with focused shared-reader tests plus real inspect output on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx`
- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `GLBS`

### Mar 28, 2026 - Shared Classic `MDX` `CLID` Summary Slice Landed

- advanced the shared classic `MDX` migration from `HTST` into `CLID` so collision meshes no longer remain only as known-but-unparsed top-level chunk ids in `wow-viewer`
- added shared `MdxCollisionSummary` and extended `MdxSummary` with nullable collision ownership
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic `CLID` summary support for `v1300` and `v1400`, including ordered `VRTX` or `TRI ` or `NRMS` subchunk parsing, derived bounds, and max-index coverage
- updated `WowViewer.Tool.Inspect mdx inspect` to report `collisionVertices=` or `collisionTriangles=` and print a `CLID:` line
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`CLID` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` collision case
- validated the seam with focused shared-reader tests plus real inspect output on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx`
- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `CLID`

### Mar 28, 2026 - Shared Classic `MDX` `HTST` Summary Slice Landed

- advanced the shared classic `MDX` migration from `EVTS` into `HTST` so hit-test shapes no longer remain only as known-but-unparsed top-level chunk ids in `wow-viewer`
- added shared `MdxGeometryShapeType` and `MdxHitTestShapeSummary` contracts and extended `MdxSummary` with `HitTestShapes` plus `HitTestShapeCount`
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `HTST` summary support for `v1300` and `v1400`, including inherited node metadata plus fixed box or cylinder or sphere or plane payload fields
- updated `WowViewer.Tool.Inspect mdx inspect` to report `hitTestShapes=` and print `HTST[n]` lines
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`HTST` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` hit-test-shape case
- validated the seam with focused shared-reader tests plus real inspect output on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx`
- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `HTST`

### Mar 28, 2026 - Shared Classic `MDX` `EVTS` Summary Slice Landed

- advanced the shared classic `MDX` migration from `CAMS` into `EVTS` so event nodes no longer remain only as known-but-unparsed top-level chunk ids in `wow-viewer`
- added shared `MdxEventSummary` and `MdxEventTrackSummary` contracts and extended `MdxSummary` with `Events` plus `EventCount`
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `EVTS` summary support for `v1300` and `v1400`, including inherited node metadata and optional `KEVT` key-time metadata
- updated `WowViewer.Tool.Inspect mdx inspect` to report `events=` and print `EVTS[n]` lines
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`EVTS` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` event case
- validated the seam with focused shared-reader tests plus real inspect output on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx`
- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `EVTS`

### Mar 28, 2026 - Shared Classic `MDX` `CAMS` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `RIBB` summary ownership so `MdxSummaryReader` now also exposes classic camera summary coverage for fixed camera metadata and summary-only camera-track metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxCameraSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Cameras` and `CameraCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `CAMS` summary support for `v1300` and `v1400`, including per-camera section sizing, fixed payload fields, and optional `KCTR`, `KCRL`, `KVIS`, and `KTTR` metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `CAMS[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`CAMS` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` camera case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `CAMS`
	- this is still summary-only classic camera ownership, not camera playback, interpolation evaluation, viewer camera selection, or runtime portrait parity

### Mar 28, 2026 - Shared Classic `MDX` `PRE2` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `RIBB` summary ownership so `MdxSummaryReader` now also exposes classic particle-emitter-v2 summary coverage for `MDLGENOBJECT`-derived effect metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxParticleEmitter2Summary`
	- reused `WowViewer.Core.Mdx.MdxTrackSummary` for classic `PRE2` scalar track metadata
	- extended `WowViewer.Core.Mdx.MdxSummary` with `ParticleEmitters2` and `ParticleEmitter2Count`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `PRE2` summary support for `v1300` and `v1400`, including outer emitter sizing, inner node sizing, classic scalar payload fields, spline-count handling, and optional `KVIS` or `KP2V` plus `KP2S`, `KP2R`, `KP2L`, `KPLN`, `KP2G`, `KLIF`, `KP2E`, `KP2W`, `KP2N`, and `KP2Z` metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `PRE2[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`PRE2` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` particle-emitter case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `PRE2`
	- this is still summary-only classic particle-emitter ownership, not particle simulation, UV animation playback, spline playback, or runtime render parity

### Mar 28, 2026 - Shared Classic `MDX` `ATCH` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `HELP` summary ownership so `MdxSummaryReader` now also exposes classic attachment summary coverage for `MDLGENOBJECT`-derived attachment metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxAttachmentSummary`
	- added `WowViewer.Core.Mdx.MdxVisibilityTrackSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Attachments` and `AttachmentCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `ATCH` summary support for `v1300` and `v1400`, including outer section sizing, inner node sizing, `KGTR` or `KGRT` or `KGSC` transform metadata, and optional `KVIS` or `KATV` visibility metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `ATCH[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`ATCH` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` attachment case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `ATCH`
	- this is still summary-only classic attachment ownership, not visibility evaluation, asset resolution, or runtime attachment/render parity

### Mar 28, 2026 - Shared Classic `MDX` `HELP` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `BONE` summary ownership so `MdxSummaryReader` now also exposes classic helper-node summary coverage for `MDLGENOBJECT` metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxHelperSummary`
	- added `WowViewer.Core.Mdx.MdxNodeTrackSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Helpers` and `HelperCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `HELP` summary support for `v1300` and `v1400`, including `KGTR` or `KGRT` or `KGSC` track metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `HELP[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`HELP` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` helper case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `HELP`
	- this is still summary-only classic helper ownership, not transform evaluation, attachment behavior, billboard handling, or viewer playback parity

### Mar 28, 2026 - Shared Classic `MDX` `BONE` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `GEOA` summary ownership so `MdxSummaryReader` now also exposes classic bone summary coverage for render-facing skeleton metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxBoneSummary`
	- added `WowViewer.Core.Mdx.MdxNodeTrackSummary` as the shared classic node-track contract reused by `BONE` and `HELP`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Bones` and `BoneCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `BONE` summary support for `v1300` and `v1400`, including `KGTR` or `KGRT` or `KGSC` track metadata plus geoset-link fields
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `BONE[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`BONE` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` bone case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `BONE`
	- this is still summary-only classic bone ownership, not transform evaluation, runtime skeleton assembly, or viewer playback parity

### Mar 28, 2026 - Shared Classic `MDX` `GEOA` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `GEOS` structure ownership so `MdxSummaryReader` now also exposes classic geoset-animation summary coverage for render-facing metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxGeosetAnimationSummary`
	- added `WowViewer.Core.Mdx.MdxGeosetAnimationTrackSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `GeosetAnimations` and `GeosetAnimationCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `GEOA` summary support for `v1300` and `v1400`, including `KGAO` or `KGAC` track metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `GEOA[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`GEOA` fixture and an optional real archive-backed `GEOA` probe case
- Validation limits:
	- this landing now has strong synthetic coverage plus real Alpha `0.5.3` `MDX` validation for classic `GEOA`; the fixed `0.6.0` archive corpus still has no guaranteed positive `GEOA` asset identified
	- this is still summary-only classic geoset-animation ownership, not animation evaluation, viewer playback parity, or runtime geoset-state cutover

### Mar 28, 2026 - Shared Classic `MDX` `GEOS` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past `SEQS` and `PIVT` so `MdxSummaryReader` now also exposes classic geoset summary coverage for render-facing mesh structure.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxGeosetSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Geosets` and `GeosetCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted tagged `GEOS` summary support for `v1300` and `v1400`
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `GEOS[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with synthetic classic-geoset coverage and a real archive-backed `chest01.mdx` geoset case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `8` passing tests
	- real archive-backed `mdx inspect` on `world/generic/activedoodads/chest01/chest01.mdx` passed and reported `geosets=2`, `CHUNK[5]: id=GEOS`, and real `GEOS[0]` plus `GEOS[1]` lines
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `174` passing tests
	- this is still summary-only classic geoset ownership, not full mesh decode, geoset-animation ownership, skinning parity, or runtime render cutover

### Mar 28, 2026 - Shared `MDX` `PIVT` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past the new `SEQS` summary layer so `MdxSummaryReader` now also exposes `PIVT` pivot-table coverage.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxPivotPointSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `PivotPoints` and `PivotPointCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with strict `PIVT` `12`-byte-entry summary support and legacy-matching invalid-size failure behavior
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `PIVT[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with synthetic pivot fixtures and optional real pivot-positive archive coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `6` passing tests
	- real archive-backed `mdx inspect` on `world/generic/activedoodads/chest01/chest01.mdx` passed and reported `pivotPoints=6`, `CHUNK[8]: id=PIVT`, and real `PIVT[n]` lines
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `172` passing tests
	- this is still summary-only pivot ownership, not bone binding, helper or emitter placement parity, or runtime node transform ownership

### Mar 28, 2026 - Shared `MDX` `SEQS` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` beyond `TEXS` and `MTLS` so `MdxSummaryReader` now also exposes first `SEQS` summary coverage.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxSequenceSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Sequences` and `SequenceCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with counted legacy named `128/132/136/140`-byte `SEQS` summary support, counted named `0x8C` support, and the numeric-heavy `0x8C` `0.9.0` path as summary-only metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `SEQS[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with synthetic sequence fixtures and optional real animated-archive coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `4` passing tests
	- real archive-backed `mdx inspect` on `world/generic/passivedoodads/particleemitters/greengroundfog.mdx` passed and reported `sequences=1`, `CHUNK[2]: id=SEQS`, and `SEQS[0]: name=Stand ... blendTime=150`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `170` passing tests
	- this is still summary-only animation ownership, not animation-track parsing, bone/geoset ownership, or runtime viewer playback parity

### Mar 28, 2026 - Shared Root-ADT Plus `_tex0` Texture Reader And Broadened JSON Export Landed

- Generalized the earlier `_tex0`-only terrain-detail seam into a shared ADT texture reader for root `ADT` and `_tex0.adt` files.
- Landed pieces:
	- replaced `_tex0`-specific `AdtTex*` contracts with `AdtTextureChunkLayer`, `AdtTextureChunk`, and `AdtTextureFile`
	- added `WowViewer.Core.IO.Maps.AdtTextureReader` for shared root or `_tex0` per-chunk layer-table and decoded-alpha reads
	- updated `AdtMcalSummaryReader` to aggregate through the generalized shared reader
	- broadened `WowViewer.Tool.Converter export-tex-json` so it now accepts `file.adt` and `file_tex0.adt`
	- updated inspect `--dump-tex-chunks` to consume the generalized shared reader
	- replaced `_tex0`-only regression coverage with `AdtTextureReaderTests`, including synthetic root plus real development-root coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtTextureReaderTests|AdtMcalSummaryReaderTests|AdtMcalDecoderTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `37` passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `168` passing tests
	- real-data converter export now passes on both `development_0_0.adt` and `development_0_0_tex0.adt`, and root `--output` file-write also passed
	- the fixed development root dataset still does not positively prove real root-layer payload decode because its texture layering lives in `_tex0.adt`; that proof remains synthetic in this slice

### Mar 28, 2026 - Thin `_tex0` JSON Export Surface Landed In `WowViewer.Tool.Converter`

- Added the first real converter/export consumer for the new shared `_tex0` terrain seam.
- Landed pieces:
	- updated `WowViewer.Tool.Converter` with `export-tex-json --input <file_tex0.adt> [--output <report.json>]`
	- validated `_tex0` inputs through shared `WowFileDetector`
	- serialized shared `AdtTexReader` output directly to stdout or an output file instead of adding tool-local parsing
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `166` passing tests
	- real-data stdout export on `development_0_0_tex0.adt` passed and printed JSON with shared `SourcePath`, `TextureNames`, and `Chunks`
	- real-data file export on `development_0_0_tex0.adt` passed and wrote `wowviewer-development_0_0_tex0.json` under `%TEMP%`
	- this is still a thin JSON export over the shared read seam, not a broader terrain conversion workflow or a WoW terrain write path

### Mar 28, 2026 - Shared `_tex0` Per-Chunk Layer And Decoded Alpha Reader Landed

- Added the next terrain shared-I/O slice in `wow-viewer` after split-family routing plus aggregate `MCAL` summary.
- Landed pieces:
	- added `WowViewer.Core.Maps.AdtTexChunkLayer`
	- added `WowViewer.Core.Maps.AdtTexChunk`
	- added `WowViewer.Core.Maps.AdtTexFile`
	- added `WowViewer.Core.IO.Maps.AdtTexReader`
	- extended `MapSummaryReaderCommon` with shared `ReadStringEntries(...)`
	- updated `AdtMcalSummaryReader` so `_tex0` summary aggregation now consumes the shared reader instead of duplicating `_tex0` parsing logic
	- updated `WowViewer.Tool.Inspect map inspect` with `--dump-tex-chunks` so `_tex0` reports can print shared per-chunk `MCNK(tex)` and `LAYER` detail lines on demand
	- added synthetic plus real-data coverage in `AdtTexReaderTests`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtTexReaderTests|AdtMcalSummaryReaderTests|AdtMcalDecoderTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `35` passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `166` passing tests
	- real-data `map inspect --dump-tex-chunks` on `test_data/development/World/Maps/development/development_0_0_tex0.adt` passed and reported `ADT TEX detail: textures=5 chunks=256`, preserved aggregate `decodedLayers=519`, and printed real per-layer `Compressed` plus `BigAlpha` outputs
	- this is still not runtime `MdxViewer` terrain signoff and not a shared port of Cataclysm residual-alpha synthesis or chunk-edge stitching

### Mar 28, 2026 - Shared `ADT` Split-Family Routing And Direct `MCAL` Decode Summary Seams Landed

- Added the first terrain-focused shared ownership slice in `wow-viewer` under the broader full-format-ownership reset.
- Landed pieces:
	- added `WowViewer.Core.Maps.AdtTileFamily`
	- added `WowViewer.Core.Maps.AdtTextureLayerDescriptor`
	- added `WowViewer.Core.Maps.AdtMcalDecodeProfile`
	- added `WowViewer.Core.Maps.AdtMcalAlphaEncoding`
	- added `WowViewer.Core.Maps.AdtMcalDecodedLayer`
	- added `WowViewer.Core.Maps.AdtMcalSummary`
	- added `WowViewer.Core.IO.Maps.AdtTileFamilyResolver`
	- added `WowViewer.Core.IO.Maps.AdtMcalDecoder`
	- added `WowViewer.Core.IO.Maps.AdtMcalSummaryReader`
	- updated `WowViewer.Tool.Inspect map inspect` to print shared ADT family routing and `MCAL` decode summary lines
	- updated `MapFileKind` plus `MapFileSummaryReader` so `_lod.adt` is preserved as `AdtLod`
	- added focused synthetic and real-data coverage in:
		- `AdtTileFamilyResolverTests`
		- `AdtMcalDecoderTests`
		- `AdtMcalSummaryReaderTests`
		- plus adjacent map-summary and detector assertions
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtMcalDecoderTests|AdtMcalSummaryReaderTests|AdtTileFamilyResolverTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `35` passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `164` passing tests
	- real-data `map inspect` on `test_data/development/World/Maps/development/development_0_0_tex0.adt` passed and reported `overlayLayers=519`, `decodedLayers=519`, `missingPayloadLayers=0`, `compressed=515`, and `bigAlpha=4`
	- this is still not runtime `MdxViewer` terrain signoff and not a full shared port of Cataclysm residual-alpha synthesis or chunk-edge stitching

### Mar 28, 2026 - Full Format Ownership Program Reset Captured

- The migration target for `wow-viewer` was clarified beyond the earlier narrow summary-seam framing.
- New explicit rule:
	- `wow-viewer` must fully re-own every active `MdxViewer` format family as first-party library and tooling behavior.
	- current detector and summary slices are progress, but not closure.
- Added `gillijimproject_refactor/plans/wow_viewer_full_format_ownership_plan_2026-03-28.md` to lock the broader program target, ownership standard, format-family scope, workstreams, and execution order.
- Added `gillijimproject_refactor/plans/wow_viewer_format_parity_matrix_2026-03-28.md` to track the family-by-family gap between active `MdxViewer` behavior and `wow-viewer` ownership.
- Updated the shared-I/O plan and continuity docs so future sessions do not mistake current `BLP`, `MDX`, `WMO`, `ADT`, or `WDT` summary ownership for the final migration target.
- This was planning and continuity work only. No new implementation or validation was performed in this reset itself.

### Mar 28, 2026 - Shared `MDX` Top-Level Plus `TEXS` And `MTLS` Summary Seams And Consumer Validation Landed

- Added the first shared `MDX` model-family seam in `wow-viewer` and immediately validated it through the existing non-UI `MdxViewer` probe path.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxChunkIds`
	- added `WowViewer.Core.Mdx.MdxChunkSummary`
	- added `WowViewer.Core.Mdx.MdxTextureSummary`
	- added `WowViewer.Core.Mdx.MdxMaterialLayerSummary`
	- added `WowViewer.Core.Mdx.MdxMaterialSummary`
	- added `WowViewer.Core.Mdx.MdxSummary`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with shared `TEXS` texture-table summary support and narrow `MTLS` material-layer summary support
	- updated `WowViewer.Tool.Inspect` with `mdx inspect --input <file.mdx>` and `mdx inspect --archive-root <dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]`
	- added synthetic plus real standard-archive coverage in `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs`
	- extended `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` so the consumer probe now prints shared `MDX` summary signals for the probed model bytes, including `TEXS` texture-count and first-path signals plus compact `MTLS` material-layer summary signals
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "MdxSummaryReaderTests|WowFileDetectorTests"` passed on Mar 27, 2026 with `11` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --virtual-path world/generic/activedoodads/chest01/chest01.mdx --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed and reported `version=1300`, `model=Chest01`, `textures=2`, `materials=2`, and real `TEXS` plus `MTLS` layer lines on the archive-backed asset
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed and now prints `SharedMDX` plus real first-texture `TEXS` paths and first-layer `MTLS` signals alongside the earlier `SharedBLP` signals
	- this is still top-level `MDX` plus narrow `TEXS` and `MTLS` summary ownership and consumer validation only, not runtime viewer model-path signoff, deep material semantics, animation-track parity, or `M2` parity

### Mar 27, 2026 - `MdxViewer` Compatibility Validation Now Exercises Shared `BLP` Summary Reads

- Updated `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` so the active viewer consumer now uses the latest shared `wow-viewer` `BLP` seam during non-UI asset probing.
- Landed pieces:
	- added shared `WowFileDetector` output for probed model and texture bytes
	- added shared `BlpSummaryReader` output for resolved texture files classified as `Blp`
	- kept `SereniaBLPLib` decode in place for width and alpha inspection, so the probe now shows both shared-header signals and legacy decode signals together
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 27, 2026 with existing warnings
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 27, 2026 and printed real `SharedBLP` summary lines for both chest textures
	- this is not automated test coverage and not runtime viewer signoff

### Mar 27, 2026 - Shared `BLP` Header Summary Seam And Inspect Surface Landed

- Added the first real texture-family shared-I/O seam in `wow-viewer` instead of stopping at cross-family file detection.
- Landed pieces:
	- added `WowViewer.Core.Blp.BlpFormat`
	- added `WowViewer.Core.Blp.BlpCompressionType`
	- added `WowViewer.Core.Blp.BlpPixelFormat`
	- added `WowViewer.Core.Blp.BlpMipMapEntry`
	- added `WowViewer.Core.Blp.BlpSummary`
	- added `WowViewer.Core.IO.Blp.BlpSummaryReader`
	- updated `WowViewer.Tool.Inspect` with `blp inspect --input <file.blp>` and `blp inspect --archive-root <dir> --virtual-path <path/to/file.blp> [--listfile <listfile.txt>]`
	- added synthetic and archive-backed real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/BlpSummaryReaderTests.cs`
	- extended `wow-viewer/tests/WowViewer.Core.Tests/WowFileDetectorTests.cs` with a synthetic `BLP2` detector case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "BlpSummaryReaderTests|WowFileDetectorTests"` passed on Mar 27, 2026 with `11` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- blp inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path interface/minimap/minimaparrow.blp` now reports a real archive-backed `BLP2` summary and per-mip coverage through the shared reader
	- this is still header-summary ownership only, not full pixel decode, write support, or model-family parity

### Mar 27, 2026 - Shared `MOLT` Per-Light Detail Seam And Opt-In Inspect Dump Landed

- Added the next narrow WMO follow-up after the root-light summary fix: shared per-entry `MOLT` detail ownership plus an inspect flag that exposes those raw fields on demand.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoLightDetail`
	- added `WowViewer.Core.IO.Wmo.WmoLightReaderCommon`
	- added `WowViewer.Core.IO.Wmo.WmoLightDetailReader`
	- updated `WowViewer.Core.IO.Wmo.WmoLightSummaryReader` so summary aggregation reuses the shared detail decode path instead of duplicating `MOLT` layout logic
	- updated `WowViewer.Tool.Inspect wmo inspect` with `--dump-lights` so root WMO reports can print `MOLT[n]` lines for Alpha `32`-byte and standard `48`-byte entries without changing the default summary output
	- added synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoLightDetailReaderTests.cs`
	- extended `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` with real Alpha and standard per-light detail assertions on Ironforge
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|WmoLightDetailReaderTests|Read_IronforgeAlphaPerAssetMpq_ProducesExpectedRootLightSummary|Read_IronforgeAlphaPerAssetMpq_RootLightDetails_UseLegacyLayout|Read_IronforgeStandard060_RootLightSummary_UsesStandardTailAttenuationOffsets|Read_IronforgeStandard060_RootLightDetails_ExposeRawStandardLayoutFields"` passed on Mar 27, 2026 with `8` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path world/wmo/khazmodan/cities/ironforge/ironforge.wmo --dump-lights` now prints real standard per-light `MOLT[n]` lines with raw `headerFlagsWord` and quaternion rotation values
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ --dump-lights` now prints real Alpha per-light `MOLT[n]` lines with legacy `32`-byte entry sizing and no later-layout fields
	- this is still a shared detail-read and inspect-surface slice, not broader light-behavior interpretation or a write path

### Mar 27, 2026 - WMO Group Optional `MOLR`, `MOBN`, `MOBR`, And `MOBN->MOBR` Summary Slice Landed

- Added the next narrow shared WMO group slice in `wow-viewer` for the remaining low-risk optional group chunks plus one first linkage seam between BSP nodes and BSP face refs.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupLightRefSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupLightRefSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupBspNodeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupBspNodeSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupBspFaceSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupBspFaceSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupBspFaceRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupBspFaceRangeSummaryReader`
	- extended `WmoGroupSummary` and `WmoEmbeddedGroupSummary` so shared consumers can see `lightRefs`, `bspNodes`, and `bspFaceRefs` without their own chunk scans
	- updated `WowViewer.Tool.Inspect wmo inspect` so group files now print `MOLR`, `MOBN`, `MOBR`, and `MOBN->MOBR`, and Alpha monolithic root aggregate output now includes optional-chunk totals
	- added synthetic regression coverage for all four new seams and real-data `castle01.wmo.MPQ` coverage for embedded BSP totals and embedded-group reader replay
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoRealDataTests|WmoEmbeddedGroupSummaryReaderTests|WmoGroupSummaryReaderTests|WmoGroupLightRefSummaryReaderTests|WmoGroupBspNodeSummaryReaderTests|WmoGroupBspFaceSummaryReaderTests|WmoGroupBspFaceRangeSummaryReaderTests"` passed on Mar 27, 2026 with `9` passing tests
	- real `castle01.wmo.MPQ` inspect now reports `lightRefs=0`, `bspNodes=583`, and `bspFaceRefs=6716` across its embedded Alpha root groups
	- this is still summary and range coverage only, not full BSP traversal or consumer cutover

### Mar 27, 2026 - Alpha Root Per-Embedded-Group Inspect Routing Landed For `MOBN`, `MOBR`, And `MOBN->MOBR`

- Added the next narrow follow-up after the embedded-group aggregate: real per-group inspect routing for Alpha monolithic roots so the existing shared BSP summaries are visible on each embedded `MOGP` instead of only in totals.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoEmbeddedGroupDetail`
	- added `WowViewer.Core.IO.Wmo.WmoEmbeddedGroupDetailReader`
	- extended the group optional readers with internal `ReadMogpPayload(...)` entry points for embedded-root reuse
	- updated `WowViewer.Tool.Inspect wmo inspect` so Alpha roots now print `MOGP(root)[n]`, `MOBN(root)[n]`, `MOBR(root)[n]`, and `MOBN->MOBR(root)[n]`
	- added synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupDetailReaderTests.cs`
	- extended real-data `castle01.wmo.MPQ` coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupDetailReaderTests|WmoRealDataTests|WmoEmbeddedGroupSummaryReaderTests|WmoGroupBspNodeSummaryReaderTests|WmoGroupBspFaceSummaryReaderTests|WmoGroupBspFaceRangeSummaryReaderTests"` passed on Mar 27, 2026 with `8` passing tests
	- real `castle01.wmo.MPQ` inspect now prints per-group BSP lines for both embedded groups, including `127` or `456` `MOBN` nodes and `1145` or `5571` `MOBR` refs respectively
	- this is still shared inspect routing for the current BSP summaries, not full per-group routing for every embedded subchunk family

### Mar 27, 2026 - Alpha Root Per-Embedded-Group Inspect Routing Expanded To Existing Shared Group Summaries

- Expanded the earlier BSP-only root detail seam so Alpha monolithic roots now reuse the already-owned shared group readers for additional geometry and metadata lines instead of only printing BSP summaries per embedded group.
- Landed pieces:
	- extended `WowViewer.Core.Wmo.WmoEmbeddedGroupDetail` to carry `MLIQ`, `MOBA`, `MOPY`, `MOTV`, `MOCV`, `MODR`, `MOVI` or `MOIN`, `MOVT`, and `MONR` summaries
	- added internal `ReadMogpPayload(...)` entry points to the matching shared group readers
	- updated `WowViewer.Core.IO.Wmo.WmoEmbeddedGroupDetailReader` to populate those additional summaries from root-embedded `MOGP` payloads
	- updated `WowViewer.Tool.Inspect wmo inspect` so Alpha roots now print `MONR(root)[n]`, `MOVT(root)[n]`, `MOVI(root)[n]` or `MOIN(root)[n]`, `MODR(root)[n]`, `MOCV(root)[n]`, `MOTV(root)[n]`, `MOPY(root)[n]`, and `MOBA(root)[n]`, with `MLIQ(root)[n]` ready when present
	- extended synthetic and real-data regression coverage in `WmoEmbeddedGroupDetailReaderTests` and `WmoRealDataTests`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupDetailReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `4` passing tests
	- real `castle01.wmo.MPQ` inspect now prints positive per-group lines for `MONR`, `MOVT`, `MOIN`, `MOCV`, `MOTV`, `MOPY`, and `MOBA`, plus `MODR` on the embedded group that actually has doodad refs
	- `castle01.wmo.MPQ` still does not positively prove per-group `MOLR` or `MLIQ`, because those embedded groups remain zero-ref or liquid-free on this asset

### Mar 27, 2026 - `ironforge.wmo.MPQ` Added Positive Real Coverage For Per-Group `MOLR` And `MLIQ`

- Used real `ironforge.wmo.MPQ` as the next Alpha validation asset because it exercises the remaining per-group light-ref and liquid paths that `castle01.wmo.MPQ` does not.
- Landed pieces:
	- added `WmoRealDataTests.Read_IronforgeAlphaPerAssetMpq_EmbeddedGroupDetailsExposePositiveLightAndLiquidSignals`
	- updated `WowViewer.Tool.Inspect wmo inspect` so an invalid optional `MOLT` summary no longer aborts the whole report before later root or embedded-group lines print
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoRealDataTests"` passed on Mar 27, 2026 with `4` passing tests
	- real `ironforge.wmo.MPQ` inspect now reaches positive per-group `MOLR(root)[n]` lines such as groups `120`, `121`, `123`, `124`, and `125`, plus a positive `MLIQ(root)[127]` line with `liquidType=Magma`
	- this validates the per-group shared detail seam on a second real Alpha monolithic root, but it does not claim Ironforge `MOLT` root-light parsing is fully understood yet

### Mar 27, 2026 - Shared `MOLT` Root-Light Summary Now Reads Real Alpha `ironforge.wmo.MPQ`

- Fixed the narrow real-data gap exposed by Ironforge: the shared `MOLT` reader now handles legacy 32-byte Alpha root-light entries instead of assuming only the later 48-byte layout.
- Landed pieces:
	- updated `WowViewer.Core.IO.Wmo.WmoLightSummaryReader` with version-aware `MOLT` entry-size inference
	- extended `WmoLightSummary` and inspect output with `attenStartRange`, a raw later-layout `headerFlagsWord` summary from bytes `2..3`, and later-layout rotation metrics (`rotationEntries`, `nonIdentityRotations`, `rotationLenRange`)
	- extended `WmoLightSummaryReaderTests` with explicit synthetic `v14` and `v17` coverage
	- extended `WmoRealDataTests` with a direct Ironforge root-light assertion, including positive attenuation-start coverage
	- corrected the standard 48-byte layout after real `0.6.0` archive proof: bytes `2..3` now land as a raw `headerFlagsWord`, quaternion rotation reads from offsets `24..39`, and attenuation reads from offsets `40` and `44`
	- added shared `ArchiveVirtualFileReader` and updated `WowViewer.Tool.Inspect wmo inspect` with `--archive-root` plus `--virtual-path` so standard-archive root WMOs can be inspected without extracting them first
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `7` passing tests
	- real Ironforge inspect now prints `MOLT: payloadBytes=6976 entries=218 distinctTypes=1 attenuated=218 intensityRange=[0.120, 1.000] attenStartRange=[1.306, 8.333] maxAttenEnd=29.611 ...`
	- a real `0.6.0` standard-archive Ironforge regression now also proves `48`-byte `MOLT` uses a non-zero `headerFlagsWord` of `0x0101` at bytes `2..3`, quaternion rotation at offsets `24..39`, and attenuation at offsets `40` and `44`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path world/wmo/khazmodan/cities/ironforge/ironforge.wmo` now also reports the same standard root-light summary through the CLI, including `headerFlagsWordRange=[0x0101, 0x0101]`, `headerFlagsWordDistinct=1`, `headerFlagsWordNonZero=218`, `rotationEntries=218`, `nonIdentityRotations=218`, and `rotationLenRange=[1.118, 1.118]`
	- this is still a shared semantic-summary slice, not deeper light behavior or a write path
	- the per-light inspect dump has now landed, so the clean next step is to test more real standard roots for `headerFlagsWord` variability instead of re-opening the already-settled Ironforge attenuation and rotation offsets

### Mar 27, 2026 - Alpha `MOGI -> MOGP(root)` Linkage Summary Landed

- Added the next narrow Alpha follow-up by linking root `MOGI` entries to embedded top-level `MOGP` blocks by ordinal pairing.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoEmbeddedGroupLinkageSummary`
	- added `WowViewer.Core.IO.Wmo.WmoEmbeddedGroupLinkageSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so Alpha monolithic roots now print an `MOGI->MOGP(root)` linkage line
	- added synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupLinkageSummaryReaderTests.cs`
	- extended real-data coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `130` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupLinkageSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `2` targeted passing tests
	- real `castle01.wmo.MPQ` inspect now reports `flagMatches=0` and `boundsMatches=2` for the paired Alpha group-info vs embedded-group linkage surface
	- this is still linkage-summary ownership, not detailed per-group route selection or remediation logic

### Mar 27, 2026 - Alpha Monolithic Root Embedded-Group Aggregate Summary Landed

- Added the next narrow Alpha follow-up after `MOMO` root support by summarizing the embedded top-level `MOGP` group blocks that still live in monolithic 0.5.3 root files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoEmbeddedGroupSummary`
	- added `WowViewer.Core.IO.Wmo.WmoEmbeddedGroupSummaryReader`
	- reused `WmoGroupSummaryReader` logic through a shared internal `MOGP` payload helper instead of duplicating group-header interpretation
	- updated `WowViewer.Tool.Inspect wmo inspect` so Alpha monolithic roots now print an `MOGP(root)` aggregate line when embedded groups are present
	- added synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupSummaryReaderTests.cs`
	- extended real-data coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `129` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `2` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports embedded-group aggregate metrics
	- this is still aggregate ownership, not per-embedded-group detailed Alpha consumer routing

### Mar 27, 2026 - Alpha MOMO Root WMO Support And Real 0.5.3 `.wmo.MPQ` Validation Landed

- Added shared Alpha root-WMO support for the `MOMO` container so the existing root-summary stack can read real 0.5.3 monolithic WMO roots.
- Landed pieces:
	- added shared `MOMO` chunk id in `WowViewer.Core.Wmo.WmoChunkIds`
	- updated `WowViewer.Core.IO.Files.WowFileDetector` so `MVER` + `MOMO` is classified as `Wmo`
	- expanded `WowViewer.Core.IO.Wmo.WmoRootReaderCommon` to flatten Alpha `MOMO` subchunks into a root-chunk view reusable by later shared readers
	- moved the main root-summary readers onto `WmoRootReaderCommon`, including the semantic summary reader, group-info reader, material reader, texture-table reader, doodad-name reader, doodad-set reader, doodad-placement reader, group-name table reader, skybox reader, and the shared portal-root helper
	- loosened `WowViewer.Core.Wmo.WmoGroupInfoSummary` so negative `MOGI` name offsets from real Alpha data are treated as valid summary signals instead of rejected input
	- improved `WowViewer.Core.IO.Files.AlphaArchiveReader` internal-name candidate generation for non-map `World\...` paths and direct `.MPQ` inputs
	- updated `WowViewer.Tool.Inspect wmo inspect` to load `.wmo.MPQ` inputs through the shared Alpha archive fallback and then run the shared stream-based readers
	- added real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `128` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AlphaArchiveReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `7` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports real Alpha-era root-WMO semantic lines directly from the per-asset MPQ
	- this is still root-summary ownership; it is not yet full Alpha monolithic group-consumer ownership

### Mar 27, 2026 - Batched Root WMO Portal Linkage Summary Slices For MOPT->MOPV, MOPR->MOPT, And MOPR->MOGI Landed

- Added a portal-linkage focused batched root-WMO landing in `wow-viewer` after the earlier raw portal summary slice.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoPortalVertexRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalVertexRangeSummaryReader`
	- added `WowViewer.Core.Wmo.WmoPortalRefRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalRefRangeSummaryReader`
	- added `WowViewer.Core.Wmo.WmoPortalGroupRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalGroupRangeSummaryReader`
	- expanded `WmoRootReaderCommon` with optional chunk reads to avoid false-positive optional root-chunk lookups
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated portal-linkage lines for `MOPT->MOPV`, `MOPR->MOPT`, and `MOPR->MOGI`
	- added synthetic regression coverage for all three portal-linkage seams plus a missing-`MOVV` guard regression
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `125` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `94` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portal-linkage-batch-test.wmo` passed on Mar 27, 2026 for a synthetic root-portal-linkage smoke case
	- this is still summary work, not full portal topology validation or runtime culling ownership

### Mar 27, 2026 - Batched Root WMO Visibility Summary Slices For MOVV, MOVB, And MOVB->MOVV Landed

- Added another batched root-WMO landing in `wow-viewer` for visibility-owner chunks plus their first narrow linkage seam.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoVisibleVertexSummary`
	- added `WowViewer.Core.IO.Wmo.WmoVisibleVertexSummaryReader`
	- added `WowViewer.Core.Wmo.WmoVisibleBlockSummary`
	- added `WowViewer.Core.IO.Wmo.WmoVisibleBlockSummaryReader`
	- added `WowViewer.Core.Wmo.WmoVisibleBlockReferenceSummary`
	- added `WowViewer.Core.IO.Wmo.WmoVisibleBlockReferenceSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated `MOVV`, `MOVB`, and `MOVB->MOVV` semantic lines when those chunks are present
	- added synthetic regression coverage for all three seams
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `121` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `90` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-visibility-batch-test.wmo` passed on Mar 27, 2026 for a synthetic root-visibility smoke case
	- this is still summary work, not runtime visibility-volume ownership or write support

### Mar 27, 2026 - Batched Root WMO Linkage Summary Slices For MODD->MODN, MOGI->MOGN, And MODS->MODD Landed

- Added a linkage-focused batched root-WMO landing in `wow-viewer` instead of another raw-payload-only step.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoDoodadNameReferenceSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadNameReferenceSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupNameReferenceSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupNameReferenceSummaryReader`
	- added `WowViewer.Core.Wmo.WmoDoodadSetRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadSetRangeSummaryReader`
	- added shared `WowViewer.Core.IO.Wmo.WmoRootReaderCommon`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated linkage lines for `MODD->MODN`, `MOGI->MOGN`, and `MODS->MODD`
	- added synthetic regression coverage for all three linkage seams
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `118` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `87` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-linkage-batch-test.wmo` passed on Mar 27, 2026 for a synthetic root-linkage smoke case
	- this is still summary work, not full consumer cutover or write support

### Mar 27, 2026 - Batched Root WMO Metadata Slices For MOLT, MFOG, And MCVP Landed

- Added another batched root-WMO metadata landing in `wow-viewer` for lights, fog, and an opaque trailing chunk.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoLightSummary`
	- added `WowViewer.Core.IO.Wmo.WmoLightSummaryReader`
	- added `WowViewer.Core.Wmo.WmoFogSummary`
	- added `WowViewer.Core.IO.Wmo.WmoFogSummaryReader`
	- added `WowViewer.Core.Wmo.WmoOpaqueChunkSummary`
	- added `WowViewer.Core.IO.Wmo.WmoOpaqueChunkSummaryReader`
	- expanded shared `WmoChunkIds` with `MOLT`, `MFOG`, `MCVP`, `MOVV`, and `MOVB`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated `MOLT`, `MFOG`, and `MCVP` semantic lines when present
	- added synthetic regression coverage for all three seams
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `115` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `84` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-meta-batch-test.wmo` passed on Mar 27, 2026 for a synthetic root-metadata smoke case
	- this is still summary work, not deeper light/fog rendering semantics or opaque `MCVP` ownership

### Mar 27, 2026 - Batched Root WMO Portal Summary Slices For MOPV, MOPT, And MOPR Landed

- Added a second batched root-WMO landing in `wow-viewer` for portal-owner chunks.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoPortalVertexSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalVertexSummaryReader`
	- added `WowViewer.Core.Wmo.WmoPortalInfoSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalInfoSummaryReader`
	- added `WowViewer.Core.Wmo.WmoPortalRefSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalRefSummaryReader`
	- expanded shared `WmoChunkIds` with `MOPV`, `MOPT`, and `MOPR`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated `MOPV`, `MOPT`, and `MOPR` semantic lines when portal data is present
	- added synthetic regression coverage for all three portal seams
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `112` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `81` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portals-test.wmo` passed on Mar 27, 2026 for a synthetic root-portal smoke case
	- this is still summary work, not root-to-group portal routing ownership or write support

### Mar 27, 2026 - Batched Root WMO Summary Slices For MODD, MOGN, And MOSB Landed

- Added a batched three-slice root-WMO landing in `wow-viewer` instead of another single-slice step.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoDoodadPlacementSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadPlacementSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupNameTableSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupNameTableSummaryReader`
	- added `WowViewer.Core.Wmo.WmoSkyboxSummary`
	- added `WowViewer.Core.IO.Wmo.WmoSkyboxSummaryReader`
	- expanded shared `WmoChunkIds` with `MOGN` and `MOSB`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated `MODD`, `MOGN`, and `MOSB` semantic lines when present
	- added synthetic regression coverage for all three seams

- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `109` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `78` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-batch-test.wmo` passed on Mar 27, 2026 for a synthetic batched root-WMO smoke case
	- this is still summary work, not root-table linkage or write support

### Mar 27, 2026 - Shared WMO Root Doodad-Set Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MODS` doodad-set semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoDoodadSetSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadSetSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MODS` semantic line when doodad sets are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadSetSummaryReaderTests.cs` for synthetic empty and non-empty `MODS` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `106` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `75` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mods-test.wmo` passed on Mar 27, 2026 for a synthetic root doodad-set smoke case
	- this is still semantic summary work, not `MODD` linkage or write support

### Mar 27, 2026 - Shared WMO Root Doodad-Name Table Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MODN` doodad-name-table semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoDoodadNameTableSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadNameTableSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MODN` semantic line when doodad-name tables are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadNameTableSummaryReaderTests.cs` for synthetic mixed `.mdx` or `.m2` `MODN` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `105` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `74` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-modn-test.wmo` passed on Mar 27, 2026 for a synthetic root doodad-name smoke case
	- this is still semantic summary work, not `MODD` linkage or write support

### Mar 27, 2026 - Shared WMO Root Texture-Table Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MOTX` texture-table semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoTextureTableSummary`
	- added `WowViewer.Core.IO.Wmo.WmoTextureTableSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MOTX` semantic line when texture tables are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoTextureTableSummaryReaderTests.cs` for synthetic mixed-extension `MOTX` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `104` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `73` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-motx-test.wmo` passed on Mar 27, 2026 for a synthetic root texture-table smoke case
	- this is still semantic summary work, not `MOMT` offset resolution or write support

### Mar 27, 2026 - Shared WMO Root Material Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MOMT` material semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoMaterialSummary`
	- added `WowViewer.Core.IO.Wmo.WmoMaterialSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MOMT` semantic line when material entries are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoMaterialSummaryReaderTests.cs` for synthetic standard and legacy `MOMT` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `103` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `72` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-momt-test.wmo` passed on Mar 27, 2026 for a synthetic root-material smoke case
	- this is still semantic summary work, not `MOTX` resolution or write support

### Mar 27, 2026 - Shared WMO Root Group-Info Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MOGI` group-info semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupInfoSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupInfoSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MOGI` semantic line when group info is present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupInfoSummaryReaderTests.cs` for synthetic standard and legacy `MOGI` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `101` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `70` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mogi-test.wmo` passed on Mar 27, 2026 for a synthetic root-group-info smoke case
	- this is still semantic summary work, not `MOGN` name resolution or write support

### Mar 27, 2026 - Shared WMO Group Normal Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MONR` normal semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupNormalSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupNormalSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MONR` semantic line when normal payloads are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupNormalSummaryReaderTests.cs` for synthetic component-range and near-unit coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `99` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `68` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-normal-test.wmo` passed on Mar 27, 2026 for a synthetic normal smoke case
	- this is still semantic summary work, not tangent-space ownership or write support

### Mar 27, 2026 - Shared WMO Group Vertex Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOVT` vertex semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupVertexSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupVertexSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOVT` semantic line when vertex payloads are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupVertexSummaryReaderTests.cs` for synthetic vertex-bound coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `98` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `67` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-vertex-test.wmo` passed on Mar 27, 2026 for a synthetic vertex smoke case
	- this is still semantic summary work, not topology linkage or write support

### Mar 27, 2026 - Shared WMO Group Index Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOVI` or `MOIN` index semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupIndexSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupIndexSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOVI` or `MOIN` semantic line when index payloads are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupIndexSummaryReaderTests.cs` for synthetic `MOVI` and `MOIN` coverage including a degenerate-triangle case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `97` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `66` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-index-test.wmo` passed on Mar 27, 2026 for a synthetic index smoke case
	- this is still semantic summary work, not topology ownership or write support

### Mar 27, 2026 - Shared WMO Group Doodad-Ref Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MODR` doodad-ref semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupDoodadRefSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupDoodadRefSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MODR` semantic line when doodad refs are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupDoodadRefSummaryReaderTests.cs` for synthetic duplicate-ref coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `95` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `64` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-doodadref-test.wmo` passed on Mar 27, 2026 for a synthetic doodad-ref smoke case
	- this is still semantic summary work, not root-linkage ownership or write support

### Mar 27, 2026 - Shared WMO Group Vertex-Color Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOCV` vertex-color semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupVertexColorSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupVertexColorSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOCV` semantic line when vertex colors are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupVertexColorSummaryReaderTests.cs` for synthetic primary plus extra-set color coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `94` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `63` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-color-test.wmo` passed on Mar 27, 2026 for a synthetic vertex-color smoke case
	- this is still semantic summary work, not runtime lighting interpretation or write support

### Mar 27, 2026 - Shared WMO Group UV Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOTV` UV semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupUvSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupUvSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOTV` semantic line when UV data is present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupUvSummaryReaderTests.cs` for synthetic primary plus extra-set UV coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `93` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `62` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-uv-test.wmo` passed on Mar 27, 2026 for a synthetic UV smoke case
	- this is still semantic summary work, not runtime UV selection or write support

### Mar 27, 2026 - Shared WMO Group Face-Material Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOPY` face-material semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupFaceMaterialSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupFaceMaterialSummaryReader`
	- extended shared `WmoGroupReaderCommon` with shared `MOPY` entry-size inference
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOPY` semantic line when face-material entries are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupFaceMaterialSummaryReaderTests.cs` for synthetic v17-style and v16-style `MOPY` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `92` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `61` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-face-v17-test.wmo` passed on Mar 27, 2026 for a synthetic face-material smoke case
	- this is still semantic summary work, not face-to-batch reconstruction or write support

### Mar 27, 2026 - Shared WMO Group Batch Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOBA` batch semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupBatchSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupBatchSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOBA` semantic line when batches are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupBatchSummaryReaderTests.cs` for synthetic v17-style and v16-style batch coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `90` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `59` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-batch-test.wmo` passed on Mar 27, 2026 for a synthetic group-batch smoke case
	- this is still semantic summary work, not full batch reconstruction or write support

### Mar 27, 2026 - Shared WMO Group Liquid Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MLIQ` semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoLiquidBasicType`
	- added `WowViewer.Core.Wmo.WmoGroupLiquidSummary`
	- added shared `WowViewer.Core.IO.Wmo.WmoGroupReaderCommon` so WMO group readers share one `MOGP` payload and subchunk scan surface
	- added `WowViewer.Core.IO.Wmo.WmoGroupLiquidSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MLIQ` semantic line when liquid is present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupLiquidSummaryReaderTests.cs` for synthetic `MLIQ` height-range and ocean-inference coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `88` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `57` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-liquid-test.wmo` passed on Mar 27, 2026 for a synthetic group-liquid smoke case
	- this is still semantic summary work, not full WMO liquid mesh generation or write support

### Mar 27, 2026 - Shared WMO Group Semantic Summary Slice Landed

- Added the next narrow WMO follow-up seam in `wow-viewer`: shared `MOGP` group semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupSummaryReader`
	- expanded shared `WmoChunkIds` to cover group subchunk ids used by the summary seam
	- updated shared `WowFileDetector` so `MOGP`-first files classify as `WmoGroup`
	- updated `WowViewer.Tool.Inspect wmo inspect` so it prints either a root-WMO or group-WMO report through shared detection
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupSummaryReaderTests.cs` and an additional `WowFileDetectorTests` case for `MOGP`-first detection
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `87` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `56` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-summary-test.wmo` passed on Mar 27, 2026 for a synthetic group-file smoke case
	- this is still summary work, not deep WMO group parsing or write support

### Mar 27, 2026 - Shared ADT MCNK Semantic Summary And First WMO Root Summary Slices Landed

- Added the next narrow ADT chunk-internal semantic-summary layer in `wow-viewer` and the first shared WMO root semantic-summary seam.
- Landed ADT pieces:
	- added `WowViewer.Core.Maps.AdtChunkIds`
	- added `WowViewer.Core.Maps.AdtMcnkSummary`
	- added `WowViewer.Core.IO.Maps.AdtMcnkSummaryReader`
	- updated `WowViewer.Tool.Inspect map inspect` to print a shared `MCNK` semantic-summary line for ADT-family files
	- added `wow-viewer/tests/WowViewer.Core.Tests/AdtMcnkSummaryReaderTests.cs` for synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
- Landed WMO pieces:
	- added `WowViewer.Core.Wmo.WmoChunkIds`
	- added `WowViewer.Core.Wmo.WmoSummary`
	- added `WowViewer.Core.IO.Wmo.WmoSummaryReader`
	- added `wmo inspect --input <file.wmo>` to `WowViewer.Tool.Inspect`
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoSummaryReaderTests.cs` for a synthetic WMO root summary case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `84` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `53` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and now prints the shared ADT `MCNK` semantic summary on real split-texture data
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-summary-test.wmo` passed on Mar 27, 2026 for a synthetic root-WMO smoke case because no checked-in fixed real WMO file was available in this workspace snapshot
	- this is still semantic summary work, not deep MCNK parsing, group-file WMO parsing, or write support

### Mar 27, 2026 - Shared ADT Semantic Summary Slice Landed

- Added the first shared ADT semantic-summary layer in `wow-viewer` beyond raw chunk inventory.
- Landed pieces:
	- added `WowViewer.Core.Maps.AdtSummary`
	- added `WowViewer.Core.IO.Maps.AdtSummaryReader`
	- added shared `MapSummaryReaderCommon` helper coverage for top-level payload and string-block reads used by both WDT and ADT summary readers
	- expanded `MapChunkIds` with top-level `MAMP`
	- updated `WowViewer.Tool.Inspect map inspect` to print the shared ADT semantic summary for root, `_tex0.adt`, and `_obj0.adt` files
	- added `wow-viewer/tests/WowViewer.Core.Tests/AdtSummaryReaderTests.cs` for synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `77` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `46` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and now prints the shared ADT semantic summary on real texture-split data
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt` passed on Mar 27, 2026 and now prints the shared ADT semantic summary on real object-split data
	- this is still top-level semantic summary work, not deep ADT parsing or write support

### Mar 27, 2026 - Shared WDT Semantic Summary Slice Landed

- Added the first shared WDT semantic-summary layer in `wow-viewer` beyond raw chunk inventory.
- Landed pieces:
	- added `WowViewer.Core.Maps.WdtSummary`
	- added `WowViewer.Core.IO.Maps.WdtSummaryReader`
	- extended the shared WDT seam with standard `MAIN` flag summary metadata instead of flattening every non-zero standard entry to occupancy only
	- added `WowViewer.Core.Maps.WdtMainFlagsSummary` and `WdtMainFlagValueSummary` so standard `MAIN` readers can expose `hasAdt`, `allWater`, `loaded`, unknown-bit, async-id, and distinct-flag distribution signals without taking over tile discovery ownership
	- expanded `MapChunkIds` with Alpha-only `MDNM` and `MONM`
	- updated `WowViewer.Tool.Inspect map inspect` to print the shared WDT semantic summary plus a standard `MAIN` flag-distribution line when available
	- added `wow-viewer/tests/WowViewer.Core.Tests/WdtSummaryReaderTests.cs` coverage for synthetic standard WDT flags, synthetic Alpha WDT boundary behavior, and real-data `development.wdt`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `71` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WdtSummaryReaderTests` passed on Mar 31, 2026 with `3` passing focused WDT-summary tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development.wdt` passed on Mar 31, 2026 and now prints `WDT MAIN flags: any=1496 hasAdt=1496 allWater=0 loaded=0 unknown=0 asyncIds=0 distinct=0x1:1496`
	- this is still top-level semantic summary work, not deep WDT parsing, per-tile contract ownership, or write support

### Mar 27, 2026 - Shared AreaIdMapper Archive-Backed Loading Replaced Constructor-Time Extracted-Tree Probing

- Reworked the shared `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` seam so it can load `AreaTable` and `Map` directly from shared archive readers instead of assuming extracted `DBFilesClient` trees.
- Landed pieces:
	- added archive-backed `TryLoadFromArchives(...)` using `IArchiveReader`, `DbClientFileReader`, and an in-memory DBCD provider
	- normalized shorthand build strings like `0.5.3` and `3.3.5` to the full WoWDBDefs-compatible builds expected by DBCD
	- changed `WoWMapConverter.Core.Converters.AlphaToLkConverter` to lazy mapper initialization from explicit DBC paths or explicit `AlphaClientPath` and `LkClientPath` archive roots
	- added CLI options `--alpha-client` and `--lk-client` in `WoWMapConverter.Cli`
	- expanded `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` with synthetic archive-backed coverage and explicit archive-missing diagnostics
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `37` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 27, 2026 with the existing warning floor
	- no real MPQ-root conversion smoke test was run in this slice, so the active proof level is shared-library regression plus converter buildability, not end-to-end runtime signoff

### Mar 27, 2026 - Shared AreaIdMapper DBCD Wiring And Explicit Fallback Warning Landed

- Upgraded the shared `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` seam from raw-file-only loading to a real DBCD + WoWDBDefs-backed path for the active `AreaTable` and `Map` use case.
- Landed pieces:
	- `WowViewer.Core.IO.csproj` now uses the viewer-aligned vendored DBCD project from `gillijimproject_refactor/lib/wow.tools.local/DBCD` and bundles `gillijimproject_refactor/lib/WoWDBDefs/definitions` into output
	- `AreaIdMapper` now discovers bundled or vendored `WoWDBDefs` definitions and uses DBCD for known `0.5.3` and `3.3.5` extracted table trees when present, preferring `gillijimproject_refactor/test_data` roots first
	- `AreaIdMapper` still falls back to the existing narrow raw `DbcReader` when definitions or build inference are unavailable, preserving the old tests and compatibility path
	- `TryAutoLoadFromTestData()` now reports an explicit missing-tree diagnostic instead of silently returning false
	- `AlphaToLkConverter` now forwards that diagnostic as a visible runtime warning before falling back to crosswalk-only behavior
	- added shared-library tests for explicit missing-tree reporting and a synthetic DBCD-backed `AreaTable`/`Map` auto-load case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `66` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 27, 2026 with the existing warning floor
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- convert i:/parp/parp-tools/gillijimproject_refactor/test_data/0.5.3/alphawdt/World/Maps/PVPZone01/PVPZone01.wdt -o i:/parp/parp-tools/output/pvpzone01-alpha-to-lk-smoke-dbcd-check3 -v` passed on Mar 27, 2026 and confirmed the new explicit warning path now names the preferred `gillijimproject_refactor/test_data/*/tree/DBFilesClient` roots first when extracted DBC trees are missing
	- no runtime proof was added yet for the schema-backed path against real extracted `test_data/*/tree/DBFilesClient` tables because those files are still absent in this workspace

### Mar 26, 2026 - Shared AreaIdMapper And Crosswalk Ownership Landed

- Finished the remaining live old-repo DBC-backed area-mapping cutover onto shared `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs`.
- Moved the embedded `area_crosswalk.csv` resource into `wow-viewer` and wired `WowViewer.Core.IO.csproj` to embed it there.
- Retargeted `WoWMapConverter.Core.Converters.AlphaToLkConverter` to the shared mapper.
- Deleted the old-repo `Dbc/AreaIdMapper.cs`, the dead `Services/AreaIdCrosswalk.cs`, and the old `Resources/area_crosswalk.csv` copy from `WoWMapConverter.Core`.
- Added focused shared-library regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` for:
	- constructor-loaded embedded crosswalk defaults
	- matching-report CSV parsing through `LoadCrosswalkCsv(...)`
	- continent-hinted exact-name matching through `LoadDbcs(...)`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `64` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `53` warnings and no new build break
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 26, 2026 with `3` warnings after the import cleanup
	- no real-data runtime validation was run

### Mar 26, 2026 - Shared Alpha MPQ Old-Repo Caller Cutover Landed

- Finished the remaining active old-repo per-asset MPQ caller cutover onto shared `wow-viewer/src/core/WowViewer.Core.IO/Files/AlphaArchiveReader.cs`:
	- `WoWMapConverter.Core.VLM.VlmDatasetExporter`
	- `WoWMapConverter.Core.Converters.WmoV14ToV17Converter`
	- `WoWMapConverter.Core.Converters.WmoV14ToV17ExtendedConverter`
- Deleted the now-dead duplicate `WoWMapConverter.Core/Services/AlphaMpqReader.cs` implementation.
- Added focused shared-library regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AlphaArchiveReaderTests.cs` for:
	- default block selection in per-asset MPQs without explicit internal names
	- companion `.MPQ` fallback using internal-name candidates from the requested path
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `61` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `53` warnings and no new build break
	- no `MdxViewer` build was required because the active viewer already consumed the shared reader and only a comment changed there
	- no runtime validation was run

### Mar 26, 2026 - Dead Old DBC Helper Cleanup Landed

- Tightened the old-repo boundary after the shared `Core.IO` cutovers by deleting the now-dead helper layer from `WoWMapConverter.Core`:
	- `Dbc/DbcReader.cs`
	- `Services/NativeMpqService.cs`
	- `Services/Md5TranslateResolver.cs`
	- `Services/MapDbcService.cs`
	- `Services/GroundEffectService.cs`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Dbc/AreaIdMapper.cs` now uses shared `WowViewer.Core.IO.Dbc.DbcReader`
	- the remaining active DBC-backed seam in `WoWMapConverter.Core` is now explicit instead of being hidden behind dead duplicate helpers
	- `AreaIdMapper` remains live through `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs`
- Validation limits:
	- workspace diagnostics for `WoWMapConverter.Core` reported no errors after the cleanup
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `54` warnings and no new build break
	- no new `wow-viewer` tests were run because the shared library code did not change in this pass
	- this proves cleanup and dependency-boundary tightening, not completion of the remaining area-crosswalk migration seam

### Mar 26, 2026 - Shared DBC Lookup And VLM Archive Cutover Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with the next shared non-PM4 table-backed helper slice:
	- `DbcReader`
	- `DbcHeader`
	- `MapDirectoryLookup`
	- `GroundEffectLookup`
- Scope:
	- re-homes the tiny DBC parser plus the active map-directory and ground-effect lookup helpers out of `WoWMapConverter.Core`
	- expands shared `DbClientFileReader` probing to cover `DBFilesClient`, `DBC`, and root `.dbc` or `.db2` candidates
	- keeps active VLM archive and lookup behavior on shared `Core.IO` seams instead of `NativeMpqService` and old helper ownership
- Added regression coverage for:
	- `Map.dbc` archive-backed directory resolution
	- archive-backed ground-effect doodad lookup resolution
	- expanded shared DBC or DB2 probe ordering
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj` now references `wow-viewer/src/core/WowViewer.Core.IO`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now uses shared `IArchiveCatalog` or `IArchiveReader`
	- `VlmDatasetExporter` now resolves map directories through shared `MapDirectoryLookup`
	- `VlmDatasetExporter` now resolves ground-effect doodads through shared `GroundEffectLookup`
	- `VlmDatasetExporter` now loads MD5 minimap translation through shared callback-based `Md5TranslateResolver`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `59` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run
	- `MdxViewer` was not rebuilt in this slice because the active consumer change targeted VLM in `WoWMapConverter.Core`

### Mar 26, 2026 - Concrete Shared MPQ Catalog Port Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with the concrete standard MPQ implementation used by the active viewer consumer path:
	- `MpqArchiveCatalog`
	- `MpqArchiveCatalogFactory`
	- internal `MpqDiagnostics`
- Scope:
	- re-homes the actual archive loading, hash-table lookup, block-table parsing, decompression, and patch-priority behavior out of `WoWMapConverter.Core.Services.NativeMpqService`
	- keeps the active `MdxViewer` MPQ consumer path on a library-owned implementation instead of a compatibility adapter over the old repo
	- preserves the utility surface that still matters for future shared use, including internal listfile extraction and direct file-0 reads
- Added regression coverage for:
	- higher-priority patch reads winning over base archives
	- patched-delete fallback to base archive content
	- internal listfile extraction
	- direct file-0 reads from a standalone MPQ
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now defaults to shared `MpqArchiveCatalogFactory`
	- deleted `gillijimproject_refactor/src/MdxViewer/DataSources/NativeMpqArchiveCatalog.cs`
	- active `MdxViewer` `.cs` source no longer references `WoWMapConverter.Core.Services.NativeMpqService` in its standard MPQ path
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `57` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run
	- older `NativeMpqService` code still exists for other non-migrated old-repo consumers, so this slice proves active-path ownership cutover rather than full old-repo deletion

### Mar 26, 2026 - Shared Archive Bootstrap And Alpha Wrapper Cutovers Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with two new shared non-PM4 archive seams:
	- `ArchiveCatalogBootstrapper`
	- `ArchiveCatalogBootstrapResult`
	- `AlphaArchiveReader`
	- `PkwareExplode`
- Scope:
	- re-homes the standard archive bootstrap and external listfile parsing path out of `MpqDataSource`
	- re-homes the Alpha per-asset MPQ wrapper reader out of direct `WoWMapConverter.Core.Services.AlphaMpqReader` consumer usage
	- keeps `MpqDataSource` as a consumer of shared `Core.IO` archive helpers instead of an owner of those seams
- Added regression coverage for:
	- external listfile row parsing and bootstrap aggregation
	- Alpha internal-name candidate generation
	- Alpha direct-file fallback behavior
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now uses shared `ArchiveCatalogBootstrapper`
	- `MpqDataSource` now uses shared `AlphaArchiveReader`
	- active `MdxViewer` source no longer directly references `WoWMapConverter.Core.Services.AlphaMpqReader`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `53` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run
	- `NativeMpqService` still remains behind the compatibility adapter; this slice does not prove a full MPQ implementation port

### Mar 26, 2026 - Shared Archive-Reader MPQ Cutover Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with a new shared non-PM4 archive access seam:
	- `IArchiveReader`
	- `IArchiveCatalog`
	- `IArchiveCatalogFactory`
	- `DbClientFileReader`
- Scope:
	- re-homes the standard MPQ reader boundary out of direct `MdxViewer` ownership and onto shared `Core.IO` contracts
	- keeps the current `NativeMpqService` implementation behind a compatibility adapter instead of treating it as the active consumer contract
	- re-homes `DBFilesClient` DBC or DB2 candidate probing into shared `Core.IO`
- Added regression coverage for:
	- DBC or DB2 path candidate ordering
	- first-match table reads through a shared archive reader
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now uses shared archive interfaces for standard MPQ access and prefetch worker creation
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDBCProvider.cs` now uses shared `DbClientFileReader`
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now consumes `ArchiveReader` instead of `MpqService`
	- direct `NativeMpqService` coupling is isolated to `DataSources/NativeMpqArchiveCatalog.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `49` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run
	- Alpha wrapper reads remain on `WoWMapConverter.Core.Services.AlphaMpqReader`; that seam was not ported in this slice

### Mar 26, 2026 - Shared MD5 Minimap Translation Cutover Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with a new shared non-PM4 path-translation seam:
	- `Md5TranslateIndex`
	- `Md5TranslateResolver.TryLoad(...)`
	- `MinimapService.GetMinimapTilePath(...)`
	- `MinimapService.MinimapTileExists(...)`
- Scope:
	- re-homes MD5 minimap translation loading and minimap tile path helpers out of `WoWMapConverter.Core.Services`
	- keeps archive reads abstracted behind callbacks instead of hard-coding `NativeMpqService` into the shared seam
	- retargets the active `MdxViewer` minimap and GLB-export consumers to shared `WowViewer.Core.IO` types
- Added regression coverage for:
	- map-specific archive TRS loading
	- `dir:` context parsing for disk-backed TRS files
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now loads MD5 minimap translation through shared `Core.IO`
	- `Rendering/MinimapRenderer.cs` and `Export/MapGlbExporter.cs` now consume shared `Md5TranslateIndex` and `MinimapService`
	- `MdxViewer.csproj` now references `wow-viewer/src/core/WowViewer.Core.IO`
	- `ViewerApp` no longer uses `WoWMapConverter.Core.Services.DevelopmentMapAnalyzer.DefaultDevelopmentMapDirectory`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `47` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run

### Mar 26, 2026 - Direct MdxViewer PM4 Import Cutover Landed

- Removed the remaining direct `WoWMapConverter.Core.Formats.PM4` dependency from the active `MdxViewer` PM4 consumer path:
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now reads PM4 through `WowViewer.Core.PM4.Services.Pm4ResearchReader`
	- `WorldScene` now aliases PM4 decode or model usage to shared `WowViewer.Core.PM4` document and chunk types instead of `WoWMapConverter` PM4 wrapper types
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now uses the shared PM4 reader for loose-overlay build-hint detection
	- removed the stale PM4 import from `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs`
- Boundary outcome:
	- no direct `WoWMapConverter.Core.Formats.PM4` import remains under `gillijimproject_refactor/src/MdxViewer`
	- `MdxViewer` still keeps a broader `WoWMapConverter.Core` project reference for non-PM4 subsystems; that wider cutover is not part of this slice
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 after the cutover
	- no new automated tests were added or run for this viewer-side refactor
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Linked-Position-Ref Summary Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 placement or research-adjacent seam:
	- `Pm4LinkedPositionRefSummary`
	- `Pm4PlacementMath.SummarizeLinkedPositionRefs(...)`
- Scope:
	- re-homes linked MPRL position-ref summary aggregation from `WorldScene`
	- keeps floor-range, heading-range, and circular-mean summary logic on shared PM4 contracts instead of viewer-local summary code
	- does not change PM4 inspect or report payload shape in this slice
- Added regression coverage for:
	- synthetic linked position refs with mixed normal and terminator entries
	- terminator-only linked position refs preserving NaN heading fallback
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `31` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `45` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-linked-position-ref-summary-hookup/` passed on Mar 26, 2026
	- no PM4 inspect command or viewer runtime validation was run because the slice did not change analyzer or report output

### Mar 26, 2026 - PM4 Placement-Solution Consumer Hookup Landed

- Updated `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` so the active viewer now consumes the existing shared PM4 placement-solution seam:
	- the CK24 overlay path now calls `Pm4PlacementMath.ResolvePlacementSolution(...)`
	- planar transform, world pivot, and world yaw correction now come from one shared typed placement result instead of three separate consumer-owned steps
	- removed the redundant per-piece consumer wrappers for that PM4 placement path
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-placement-solution-hookup/` passed on Mar 26, 2026
	- no new `wow-viewer` tests were added or rerun because the shared library seam was unchanged in this slice
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Connector-Key Consumer Hookup Landed

- Updated `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` so the active viewer now consumes the existing shared PM4 connector-key seam:
	- `BuildCk24ConnectorKeys()` now builds a shared `Pm4PlacementSolution`
	- connector-key derivation now comes from `Pm4PlacementMath.BuildConnectorKeys(...)`
	- removed the redundant viewer-local connector-point conversion and quantization implementation for that PM4 grouping input path
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-connector-key-hookup/` passed on Mar 26, 2026
	- no new `wow-viewer` tests were added or rerun because the shared library seam was unchanged in this slice
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Merge-Map Consumer Hookup Landed

- Updated `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` so the active viewer now consumes the existing shared PM4 merge-map seam:
	- `RebuildPm4MergedObjectGroups()` now maps local overlay groups to shared `Pm4ConnectorMergeCandidate` inputs
	- canonical merged-group resolution now comes from `Pm4PlacementMath.BuildMergedGroupMap(...)`
	- removed the redundant viewer-local union-find and merge-heuristic implementation for that PM4 grouping path
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-merge-map-hookup/` passed on Mar 26, 2026
	- no new `wow-viewer` tests were added or rerun because the shared library seam was unchanged in this slice
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Correlation Geometry-Input Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 correlation seam:
	- `Pm4GeometryLineSegment`
	- `Pm4GeometryTriangle`
	- `Pm4CorrelationGeometryInput`
	- `Pm4CorrelationMath.BuildObjectStatesFromGeometry(...)`
- Scope:
	- re-homes PM4 correlation geometry-input assembly from `WorldScene` into `Core.PM4`
	- keeps PM4 line or triangle transform application and sampled world-geometry point derivation on shared PM4 contracts instead of viewer-local flattening helpers
	- explicitly keeps WMO-facing correlation report payload ownership outside `Core.PM4`
- Added regression coverage for:
	- synthetic PM4 geometry-input object-state construction without viewer-specific world-point assembly
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `29` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `45` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-correlation-geometry-hookup/` passed on Mar 26, 2026
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Correlation Object-State Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 correlation-state seam:
	- `Pm4CorrelationObjectDescriptor`
	- `Pm4CorrelationObjectInput`
	- `Pm4CorrelationObjectState`
	- `Pm4CorrelationMath.BuildObjectStates(...)`
	- public footprint-hull and footprint-area helpers for transformed or precomputed world geometry
- Scope:
	- re-homes PM4 correlation object summarization, bounds derivation, sampled footprint hull construction, and empty-geometry fallback out of `WorldScene`
	- keeps WMO correlation report consumption on shared state and shared hull or metric helpers instead of viewer-local state records and duplicated polygon code
	- does not yet move the full correlation-report payload contract itself into `Core.PM4`
- Added regression coverage for:
	- synthetic object-state bounds and footprint derivation
	- empty-geometry fallback center behavior
	- transformed footprint-hull construction
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `28` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `42` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-correlation-state-hookup/` passed on Mar 26, 2026
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Correlation-Math Library Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 correlation seam:
	- `Pm4CorrelationMetrics`
	- `Pm4CorrelationCandidateScore`
	- `Pm4CorrelationMath.EvaluateMetrics(...)`
	- `Pm4CorrelationMath.CompareCandidateScores(...)`
- Scope:
	- re-homes planar-gap, vertical-gap, overlap-ratio, footprint-distance, polygon-clipping, and correlation-candidate ranking helpers from `WorldScene`
	- keeps future PM4 correlation reports or placement matching on shared library contracts instead of viewer-owned anonymous metric tuples
	- does not yet move the active viewer's correlation-report assembly or object-state construction into `Core.PM4`
- Added regression coverage for:
	- synthetic overlap and footprint-distance metric calculation
	- same-tile ranking precedence over stronger cross-tile overlap
	- footprint-overlap precedence when tile parity matches
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `25` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `39` passing tests
	- no active-viewer compile or runtime validation was run because consumer compatibility did not change

### Mar 26, 2026 - PM4 Connector-Group Merge Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 grouping seam:
	- `Pm4ObjectGroupKey`
	- `Pm4ConnectorMergeCandidate`
	- `Pm4PlacementMath.BuildMergedGroupMap(...)`
- Scope:
	- re-homes connector-overlap, bounds-padding, and center-distance merge heuristics from `WorldScene`
	- keeps canonical merged-group selection in the shared library instead of leaving it viewer-owned
	- does not yet move active-viewer object-group rebuild wiring into `Core.PM4`
- Added regression coverage for:
	- neighbor-tile merge resolution with shared connector keys
	- same-tile non-merge protection even with shared connector keys
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `22` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `36` passing tests
	- no active-viewer compile or runtime validation was run because consumer compatibility did not change

### Mar 26, 2026 - PM4 Connector-Key Library Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 grouping or correlation helper seam:
	- `Pm4ConnectorKey`
	- `Pm4PlacementMath.BuildConnectorKeys(...)`
- Scope:
	- converts `MSUR.MdosIndex` exterior vertices into quantized world-space connector keys through typed `Pm4PlacementSolution`
	- keeps connector dedupe and deterministic ordering in the shared library
	- does not pull renderer-space conversion or group-merge heuristics into this slice
- Added regression coverage for:
	- distinct sorted connector-key extraction
	- yaw-corrected connector placement in world space
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `22` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `36` passing tests
	- no active-viewer compile or runtime validation was run because consumer compatibility did not change

### Mar 26, 2026 - wow-viewer Source-Of-Truth Reset

- Updated the working rule for future `wow-viewer` chats:
	- `WowViewer.Core.PM4`, `WowViewer.Core`, and `WowViewer.Core.IO` are now the canonical implementation targets for new `wow-viewer` work
	- `MdxViewer` is now a reference or compatibility consumer, not the default PM4 source of truth
	- default validation for `wow-viewer` work is `WowViewer.slnx` build or test plus the relevant tool command on the fixed development dataset
	- `MdxViewer` compile validation is now optional and should be run only when a slice changes consumer compatibility or the user explicitly asks for it
- This is a workflow and continuity reset, not runtime proof by itself.

### Mar 26, 2026 - PM4 Handoff State Prepared For Fresh Chat

- Refreshed the PM4 continuity state so the next session can start from the actual current boundary instead of re-deriving it.
- Current PM4 state to carry forward:
	- `wow-viewer` now has the research reader, inspect or audit or report verbs, and the current extracted placement-math stack in `Core.PM4`
	- active `MdxViewer` now consumes shared `Core.PM4` only for the narrow planar-transform, world-yaw, and world-space centroid seams
	- the typed coordinate-mode resolver already exists in `Core.PM4`, but its active-viewer consumer hookup is still the clean next seam rather than a solved problem
- Fresh validation re-run on Mar 26, 2026:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed with `22` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed with `11` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with `36` total tests
- Explicit non-claim preserved for future sessions:
	- PM4 is not "finished"
	- library and compile validation do not equal runtime viewer PM4 signoff
	- renderer-space composition, broader object placement flow, and remaining research semantics are still open
- Recommended next PM4 slice for the next chat:
	- wire `ResolveCoordinateMode(...)` into the active viewer through a narrow adapter follow-up, then keep continuity files synchronized with whatever that slice proves

### Mar 25, 2026 - wow-viewer Tool Inventory And Cutover Plan

- Added planning document:
	- `plans/wow_viewer_tool_inventory_and_cutover_plan_2026-03-25.md`
- Purpose:
	- inventory the old repo tool sprawl and make explicit keep, merge, kill, and archaeology calls for the future `wow-viewer` repo.
- Main decisions captured:
	- keep one interactive app shell, one converter CLI, one inspect CLI, one optional catalog CLI, and a real PM4 library from day one instead of porting every legacy executable.
	- merge `WoWMapConverter.Cli`, `AlphaLkToAlphaStandalone`, and the still-useful conversion seams from `WoWRollback` into one future converter surface.
	- merge `AlphaWdtAnalyzer.Cli` and `AlphaWdtInspector`; keep `DBCTool.V2` behavior only.
	- PM4 correction: current `MdxViewer` behavior is the runtime reference, and `Pm4Research` should be ported into the new repo as the future PM4 library family rather than left behind as a pure archaeology seam.
	- treat `parpToolbox` and `PM4Tool` as supporting PM4 evidence rather than as production app identities.
	- keep poorly scoped or obsolete executables such as `ADTPrefabTool`, `DBCTool`, old WoWRollback GUI or viewer surfaces, and archived WMOv14 tools in `parp-tools` only.
	- follow-up planning docs now exist for the bootstrap layout, the CLI or GUI dual-surface design, and the PM4 library direction.
	- migration emphasis is now `1, 3, 2`: bootstrap layout and skeleton first, dual-surface plan second, deeper PM4 consolidation third.
- Validation limits:
	- planning and documentation only
	- no builds or runtime validation were run because no code changed

### Mar 25, 2026 - wow-viewer Initial Skeleton Scaffolded

- Created a new `wow-viewer/` folder at the workspace root with an initial solution and project graph:
	- `WowViewer.slnx`
	- `src/viewer/WowViewer.App`
	- `src/core/WowViewer.Core`
	- `src/core/WowViewer.Core.IO`
	- `src/core/WowViewer.Core.Runtime`
	- `src/core/WowViewer.Core.PM4`
	- `src/tools-shared/WowViewer.Tools.Shared`
	- `tools/converter/WowViewer.Tool.Converter`
	- `tools/inspect/WowViewer.Tool.Inspect`
- Added first-pass root files and bootstrap placeholders:
	- `Directory.Build.props`
	- `Directory.Packages.props`
	- `eng/Version.props`
	- `README.md`
	- `scripts/bootstrap.ps1`
	- `scripts/bootstrap.sh`
	- `scripts/validate-real-data.ps1`
- The scaffold encodes the current PM4 planning decision directly:
	- `Core.PM4` exists immediately
	- placeholder code names `MdxViewer` as the PM4 runtime reference and `Pm4Research` as the library seed
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- this is still only a skeleton and placeholder-code build, not a real implementation or runtime-validated migration

### Mar 25, 2026 - First PM4 Reader Slice Ported Into Core.PM4

- Ported the first real PM4 code from `gillijimproject_refactor/src/Pm4Research.Core` into `wow-viewer/src/core/WowViewer.Core.PM4`.
- Added:
	- typed PM4 chunk models
	- research document container
	- binary PM4 reader
	- exploration snapshot builder
- Scope boundary:
	- this is a raw research-facing PM4 layer only
	- it does not yet move current `MdxViewer` reconstruction, grouping, transform, or correlation behavior into `Core.PM4`
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after the port
	- no runtime validation or viewer integration was performed in this slice

### Mar 25, 2026 - Single-File PM4 Analyzer And Inspect Verbs Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the first single-file PM4 analyzer or report layer.
- Added working Tool.Inspect PM4 verbs:
	- `pm4 inspect`
	- `pm4 export-json`
- Smoke-tested against the fixed development reference tile `development_00_00.pm4`.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 25, 2026
	- this is still single-file research analysis only, not viewer integration or broad PM4 signoff

### Mar 25, 2026 - PM4 Audit Path And Placement Contracts Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with:
	- decode audit and corpus-audit report models plus analyzer
	- first extracted MdxViewer-facing placement contracts: `Pm4AxisConvention`, `Pm4CoordinateMode`, `Pm4PlanarTransform`, `Pm4CoordinateService`, and `Pm4PlacementContract`
- Added new working Tool.Inspect PM4 verbs:
	- `pm4 audit`
	- `pm4 audit-directory`
- Captured the current research note that CK24 low-16 object values may align with expected `UniqueID` ranges on the development map, but this remains a hypothesis until correlated against real placement data.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit-directory --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026 and scanned `616` PM4 files with no unknown chunks or file-level diagnostics
	- this is still not the full viewer reconstruction or solver migration

### Mar 25, 2026 - First PM4 Tests Added To wow-viewer

- Added `tests/WowViewer.Core.PM4.Tests` as the first test project in the new repo.
- Locked current behavior with real-data assertions against:
	- `development_00_00.pm4`
	- the fixed development PM4 corpus directory
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `6` passing tests
	- this is still narrow fixed-dataset regression coverage, not broad PM4 correctness signoff

### Mar 25, 2026 - PM4 Linkage Report And Placement-Math Helper Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with:
	- linkage report types and corpus analyzer
	- first extracted placement-math helper layer from current `WorldScene`
- Added a new working Tool.Inspect PM4 verb:
	- `pm4 linkage --input <directory> [--output <report.json>]`
- Validated fixed-corpus linkage findings:
	- `616` files scanned
	- `150` files with ref-index mismatches
	- `58` files with bad `MDOS` refs
	- `4553` total ref-index mismatches
- Interpretation boundary preserved:
	- low16 CK24 object values may still sit in plausible `UniqueID` ranges, but the linkage report does not support treating them as globally unique identifiers by range alignment alone.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `7` passing tests

### Mar 25, 2026 - PM4 MSCN Report Family Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with MSCN relationship report types and corpus analyzer.
- Added a new working Tool.Inspect PM4 verb:
	- `pm4 mscn --input <directory> [--output <report.json>]`
- Validated fixed-corpus MSCN findings:
	- `616` files scanned
	- `309` files with MSCN
	- `1,342,410` MSCN points
	- `MSUR.MdosIndex -> MSCN`: `511,891` fits and `6,201` misses
	- raw bounds overlap outperformed swapped-XY overlap by a wide margin in this corpus slice
- Interpretation boundary preserved:
	- MSCN still looks relevant as a companion layer, but this slice does not support a simple swapped-XY explanation as the dominant corpus-wide answer.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `7` passing tests

### Mar 26, 2026 - PM4 Unknowns Report And Normal-Axis Solver Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with:
	- unknowns report types and corpus analyzer
	- the next extracted `WorldScene` solver seam: normal-based axis detection and scoring in `Pm4PlacementMath`
- Added a new working Tool.Inspect PM4 verb:
	- `pm4 unknowns --input <directory> [--output <report.json>]`
- Validated fixed-corpus unknowns findings:
	- `616` files scanned
	- `309` non-empty geometry or link files
	- `1,273,335` sentinel-pattern `MSLK.LinkId` values and no non-sentinel values in this corpus slice
	- `598,882` active `MSLK` path windows: `399,183` indices-only fits and `199,699` dual-fit windows
	- `MSLK.RefIndex -> MSUR` remains partial with `4,553` misses
- Solver-seam consequence:
	- `Core.PM4` now owns both the current range-based axis fallback and the next normal-based axis scoring helpers, reducing how much of the placement heuristic remains marooned inside `WorldScene`.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 26, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `8` passing tests

### Mar 26, 2026 - PM4 Planar-Transform Resolver Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next extracted PM4 solver seam from `WorldScene`:
	- `Pm4PlacementMath.ResolvePlanarTransform`
	- MPRL centroid-distance scoring
	- MPRL footprint scoring
	- MPRL yaw comparison with quarter-turn fallback
- Added regression coverage for:
	- current whole-tile development PM4 resolver behavior
	- a synthetic world-space quarter-turn planar candidate case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `9` passing tests
	- measured whole-tile development-tile result currently resolves to tile-local planar transform `(swap=false, invertU=false, invertV=false)` for the fixed test slice
	- this is still a solver-seam extraction, not full viewer-runtime PM4 integration or final placement correctness signoff

### Mar 26, 2026 - PM4 World-Yaw Correction And First Viewer Consumer Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next extracted PM4 solver seam from `WorldScene`:
	- `Pm4PlacementMath.TryComputeWorldYawCorrectionRadians`
	- signed basis fallback against MPRL heading evidence
- Added regression coverage for:
	- a synthetic non-zero world-yaw correction case
- Started active viewer consumption of shared PM4 solver logic:
	- `MdxViewer.csproj` now references `wow-viewer` `Core.PM4`
	- `WorldScene` now delegates planar-transform resolution and world-yaw correction into shared `Core.PM4` through explicit type adapters
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `10` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-corepm4-hookup/` passed on Mar 26, 2026
	- no automated viewer integration tests were added or run
	- no real-data runtime signoff yet on viewer-visible PM4 behavior after the shared-library hookup

### Mar 26, 2026 - PM4 World-Space Centroid Slice And Second Viewer Consumer Hookup Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next extracted PM4 solver seam from `WorldScene`:
	- `Pm4PlacementMath.ComputeSurfaceWorldCentroid`
	- shared surface-derived pivot computation using the chosen PM4 axis convention, coordinate mode, and planar transform
- Added regression coverage for:
	- a synthetic tile-local world-space centroid case using the real PM4 tile-size mapping
- Extended active viewer consumption of shared PM4 solver logic:
	- `WorldScene.ComputeSurfaceWorldCentroid(...)` now delegates into shared `Core.PM4` through the existing explicit adapters
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `11` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `4` passing placement-focused tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-centroid-hookup/` passed on Mar 26, 2026
	- no automated viewer integration tests were added or run
	- no real-data runtime signoff yet on viewer-visible PM4 behavior after the added shared centroid hook-up

### Mar 26, 2026 - PM4 World-Space Yaw Helper Slice Landed In wow-viewer

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-only PM4 math seam:
	- `Pm4PlacementMath.RotateWorldAroundPivot`
	- `Pm4PlacementMath.ConvertPm4VertexToWorld(...)` overload for corrected world-space conversion around a pivot
- Added regression coverage for:
	- a synthetic world-space pivot rotation case
	- a synthetic tile-local corrected world-position case using the real PM4 tile-size mapping
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `6` passing placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `13` passing tests
	- no new active-viewer code changed in this slice
	- no real-data viewer runtime signoff was performed in this slice

### Mar 26, 2026 - PM4 Placement-Solution Contract Slice Landed In wow-viewer

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the first typed placement-result contract:
	- `Pm4PlacementSolution`
	- `Pm4PlacementMath.ResolvePlacementSolution(...)`
	- `Pm4PlacementMath.ConvertPm4VertexToWorld(Vector3, Pm4PlacementSolution)`
- Added regression coverage for:
	- a synthetic world-space placement-solution case with resolved transform and pivot but no yaw correction
	- a synthetic world-space placement-solution case with resolved transform, pivot, and meaningful yaw correction
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `8` passing placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `15` passing tests
	- no new active-viewer code changed in this slice
	- no real-data viewer runtime signoff was performed in this slice

### Mar 26, 2026 - wow-viewer Copilot Workflow Surface Updated

- Updated `.github/copilot-instructions.md` so `wow-viewer` is now treated as an active primary path, with explicit PM4 library-first guardrails and validation rules.
- Added new reusable project skills:
	- `.github/skills/wow-viewer-pm4-library/SKILL.md`
	- `.github/skills/wow-viewer-migration-continuation/SKILL.md`
- Added a dedicated PM4 continuation prompt:

### Mar 26, 2026 - First Non-PM4 Shared Map Inspect Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core` with the first non-PM4 map-format contracts:
	- `MapChunkIds`
	- `MapFileKind`
	- `MapChunkLocation`
	- `MapFileSummary`
- Extended `wow-viewer/src/core/WowViewer.Core.IO` with the first shared WDT or ADT top-level reader slice:
	- `ChunkedFileReader`
	- `MapFileSummaryReader`
- Added a real non-PM4 inspect consumer on top of shared IO:
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` verb `map inspect --input <file.wdt|file.adt>`
- Added regression coverage for:
	- synthetic WDT summary detection
	- synthetic ADT summary detection
	- fixed-dataset `development.wdt`
	- fixed-dataset `development_0_0.adt`
- Validation limits:
	- this is still only top-level chunk summary behavior, not full ADT or WDT parsing or writing

### Mar 26, 2026 - First Shared Cross-Family Detection Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core` with the first broader file-detection contracts:
	- `WowFileKind`
	- `WowFileDetection`
- Extended `wow-viewer/src/core/WowViewer.Core.IO` with the first shared cross-family detector:
	- `WowFileDetector`
- Refactored `MapFileSummaryReader` to consume that shared detector instead of its own file-kind heuristics.
- Added the first non-placeholder converter command on top of the shared detector:
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter` verb `detect --input <file>`
- Added regression coverage for:
	- synthetic WDT detection
	- fixed-dataset WDT, root ADT, split texture ADT, split object ADT, and PM4 detection
- Validation limits:
	- this is still only shared classification and version detection, not conversion or payload parsing

### Mar 26, 2026 - wow-viewer Shared I/O Workflow Surface Tightened

- Added a dedicated non-PM4 shared-I/O skill:
	- `.github/skills/wow-viewer-shared-io-library/SKILL.md`
- Added a dedicated non-PM4 shared-I/O implementation prompt:
	- `.github/prompts/wow-viewer-shared-io-implementation.prompt.md`
- Added a dedicated shared-I/O continuity plan:
	- `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`
- Updated routing surfaces so future sessions can distinguish:
	- PM4 implementation work
	- shared `Core` or `Core.IO` implementation work
	- broader tool-suite or migration planning
- Updated `.github/copilot-instructions.md` with shared-I/O first reads and guardrails.
- Added a forward-maintenance rule across instructions and continuity surfaces:
	- new `wow-viewer` skills or implementation prompts must also update `.github/copilot-instructions.md`, `wow-viewer/README.md`, the relevant continuity plan, and the memory bank in the same slice
- Validation limits:
	- workflow and continuity updates only
	- no new runtime claim beyond the already-validated shared detector and summary slices
	- `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
- Updated `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md` so implementation-sized `Core.PM4` work now routes to the dedicated PM4 library prompt instead of only the broader migration-planning prompts.
- Updated `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md` so the new `.github` skills or prompts are recorded as the canonical shared workflow surface for the active PM4 migration slice.
- Validation limits:
	- documentation or workflow updates only
	- no code build or runtime validation was needed for this customization slice

### Mar 26, 2026 - PM4 Coordinate-Mode Resolver Slice Landed In wow-viewer

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-only PM4 solver seam:
	- `Pm4CoordinateModeResolution`
	- `Pm4PlacementMath.ResolveCoordinateMode(...)`
	- shared coordinate-mode score evaluation for tile-local versus world-space interpretation using the already-extracted planar-transform, footprint, and centroid scoring helpers
- Added regression coverage for:
	- current development-tile coordinate-mode behavior
	- a synthetic world-space coordinate-mode case
	- the missing-evidence fallback path
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `11` passing placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `18` passing tests
	- no new active-viewer code changed in this slice
	- no real-data viewer runtime signoff was performed in this slice

### Mar 26, 2026 - wow-viewer Bootstrap And Non-PM4 Core Follow-Up

- Corrected two clear plan-adherence gaps in `wow-viewer`:
	- bootstrap scripts are no longer placeholders and now clone the baseline upstream repos from the migration draft into `libs/`
	- the repo now has its first non-PM4 shared-core slice in `WowViewer.Core` and `WowViewer.Core.IO`
- Added core foundation files:
	- `src/core/WowViewer.Core/Chunks/FourCC.cs`
	- `src/core/WowViewer.Core/Chunks/ChunkHeader.cs`
	- `src/core/WowViewer.Core.IO/Chunked/ChunkHeaderReader.cs`
- Added new test project:
	- `tests/WowViewer.Core.Tests`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `22` total passing tests
	- bootstrap scripts were implemented but not executed here because cloning external repos would require network access and would materially change the workspace contents
	- this still does not mean the broader shared I/O and runtime migration phases are complete

### Mar 25, 2026 - Post-v0.4.5 Roadmap Prompt Bundle + Isolated Branch

- Detailed Copilot prompt assets for the larger `wow-viewer` tool-suite/library refactor now live under workspace `.github/prompts/` instead of `gillijimproject_refactor/plans`, because this work is prompt-surface/workflow material rather than just another local markdown note set.
- Added dedicated workspace prompt files:
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-bootstrap-layout-plan.prompt.md`
	- `.github/prompts/wow-viewer-shared-io-library-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-inventory-cutover-plan.prompt.md`
	- `.github/prompts/wow-viewer-cli-gui-surface-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-migration-sequence-plan.prompt.md`
- Created a dedicated follow-on branch for next-version work:
	- `feature/v0.4.6-v0.5.0-roadmap`
- Added new planning prompt files under `plans/`:
	- `post_v0_4_5_plan_set_2026-03-25.md`
	- `v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md`
	- `wowrollback_uniqueid_timeline_prompt_2026-03-25.md`
	- `alpha_core_sql_scene_liveness_prompt_2026-03-25.md`
	- `viewer_performance_recovery_prompt_2026-03-25.md`
	- `v0_5_0_new_repo_library_migration_prompt_2026-03-25.md`
	- `v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`
- Updated existing planning files:
	- `v0_5_0_goal_stack_prompt_2026-03-25.md`
	- `enhanced_terrain_shader_lighting_prompt_2026-03-25.md`
- Planning direction captured:
	- `v0.4.6` is now framed as the first WoWRollback / `UniqueID` timeline integration slice inside the active viewer, plus Alpha-Core SQL caching/fidelity work and a first performance recovery pass.
	- `v0.5.0` is now reframed as the migration into `https://github.com/akspa0/wow-viewer`, with a canonical shared library plus split viewer/tool consumers, instead of just a larger in-place renderer/performance milestone inside `parp-tools`.
	- latest constraint on that migration: fully re-own the first-party read/parse/write/convert stack, including current base libraries such as `gillijimproject-csharp`, while keeping upstream externals like `Warcraft.NET`, `DBCD`, `WoWDBDefs`, `Alpha-Core`, `WoWTools.Minimaps`, and `SereniaBLPLib` under `libs/` and tracking original repos where practical.
	- repository bootstrap should also automate support-material pulls such as `wow-listfile`.
	- possible alpha-era support contributions upstream to `Noggit` / `noggit-red` remain an explicit stretch/outreach track, not the core delivery target for `v0.5.0`.
	- possible secondary integration/evaluation seams include `MapUpconverter`, `ADTMeta`, `wow.export`, and `wow.tools.local`.
	- a first concrete `wow-viewer` repo tree plus migration order draft now exists so future planning can refine a named proposal instead of reopening the repo-shape argument each session.
- Documentation follow-up:
	- root `README.md` now states the documented support range more plainly (`0.5.3` through `4.0.0.11927`) and does a better job surfacing the built-in converters, WMO `v14/v16/v17` support, SQL-driven spawns, PM4 tooling, and screenshot automation reality.
- Validation limits:
	- planning/docs only
	- no automated tests or builds were run for this slice because no code changed

### Mar 25, 2026 - Fullscreen Minimap Transpose Repair + Runtime User Signoff

- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- reverted the over-corrected world-axis swap from the earlier Designer Island follow-up
	- camera tile readout and minimap teleport now stay on the direct world `X/Y` mapping used by the active viewer instead of reinterpreting the entire minimap world orientation
- `src/MdxViewer/MinimapHelpers.cs`
	- reverted the broad POI/taxi overlay world-axis swap
	- kept the narrower camera-marker screen-placement transpose so marker placement matches the already-correct drawn tile grid
- `src/MdxViewer/ViewerApp.cs`
	- restored the legacy `DrawMinimap_OLD()` fallback path to the same final orientation logic so old code does not preserve the over-corrected behavior
- Root cause:
	- the minimap tile grid itself was already oriented correctly after the `ChunkSize` regression repair
	- the first axis patch then over-corrected the world/camera layer; the real remaining seam was the marker/grid screen transposition
- Release outcome:
	- runtime user confirmation after this final patch says the fullscreen minimap is fixed
	- the fullscreen minimap should no longer be treated as an open `v0.4.5` blocker
- Validation limits:
	- build plus targeted runtime user signoff: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-transpose-repair/"` passed on Mar 25, 2026
	- runtime user feedback then confirmed the repaired Designer Island/top-right minimap behavior on the fixed development dataset
	- no automated tests were added or run

### Mar 25, 2026 - World Object Culling And Object-Fog Tuning

- `src/MdxViewer/Terrain/WorldScene.cs`
	- world WMO visibility now uses point-to-AABB distance instead of AABB-center distance for frustum grace and distance culling, which keeps large nearby objects from disappearing when the camera is close to their edge.
	- near-camera frustum-cull grace is now larger and scales with object bounds instead of relying only on a small fixed radius.
	- WMO cull range now expands relative to fog end instead of staying pinned to a short fixed distance.
	- object render passes now use a delayed object-fog start so distant objects are not pushed into fog color as aggressively while still remaining rendered.
	- MDX/taxi-object frustum gating now uses AABB distance for the near-camera exemption path as well.
- `src/MdxViewer/Rendering/WmoRenderer.cs`
	- the separate internal WMO doodad cull distance was raised substantially and now also expands with fog range at runtime.
	- WMO doodad render cap was increased to reduce disappearing interior/attached doodads in dense sets.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-object-culling-fog/"` passed on Mar 25, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on object pop-in reduction or far-fog feel.

### Mar 25, 2026 - Taxi Override Workflow, World Return, And Override Persistence

- `src/MdxViewer/ViewerApp.cs`
	- added world-return capture/restore helpers so opening a standalone model can preserve the current world session path and camera for later restoration.
	- added browser-selection helpers plus taxi override application helpers so a selected browser asset can be applied directly to the active taxi-route override target.
	- added persisted taxi override storage in viewer settings keyed by map name and route ID, with replay when the current world loads.
	- fixed a follow-up compile break in the same slice where helper methods were accidentally inserted inside `LoadFileFromDisk()`; final solution build is the only validation that counts for this slice.
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- file browser now exposes `Open Selected`, `Copy Path`, `Use For Taxi Override`, and `Return To Last World`.
	- taxi inspector now exposes `Use Selected Browser Asset`, `Copy Override Path`, `Open Override Asset`, and `Return To Last World` around the existing override controls.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-taxi-workflow/"` passed on Mar 25, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on map/object swapping, saved taxi overrides, or browser-selected override application.

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

### Mar 25, 2026 - Minimap Tile-Scale Regression Repair

- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- reverted the Mar 24 `TileSize` swap for minimap camera position, pan clamping, and teleport math.
	- docked and fullscreen minimap now both use `WoWConstants.ChunkSize` again for the active viewer's `64x64` world-tile grid.
- `src/MdxViewer/MinimapHelpers.cs`
	- POI markers, taxi route polylines, taxi node markers, and shared minimap click-to-world conversion were restored to the same `ChunkSize`-based grid spacing.
- `src/MdxViewer/ViewerApp.cs`
	- the legacy `DrawMinimap_OLD()` fallback path was restored to the same minimap scale so the bad `TileSize` assumption does not survive outside the shared helper path.
- Root cause:
	- `WoWConstants.TileSize` was the wrong spacing for the active minimap path even though the name looked plausible; the live viewer still uses `WoWConstants.ChunkSize` for the `64x64` world-tile grid, so the prior swap pushed marker placement, overlays, pan bounds, and click-to-teleport out of sync.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-regression-repair/"` passed on Mar 25, 2026.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on docked/fullscreen minimap behavior, marker placement, pan feel, or minimap teleport correctness.

### Mar 25, 2026 - Minimap Axis-Mapping Repair

- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- camera tile readout, pan clamping, and minimap teleport now treat the minimap click result as `(row, column)` and write it into renderer space with the correct axis order.
- `src/MdxViewer/MinimapHelpers.cs`
	- camera marker placement now uses the same row/column orientation as the tile grid instead of drawing the marker with transposed axes.
	- POI and taxi overlays now project with the same minimap axis order as the base tiles.
- `src/MdxViewer/ViewerApp.cs`
	- the legacy `DrawMinimap_OLD()` fallback path was updated to the same axis mapping so old code does not preserve the Designer Island/top-right teleport bug.
- Root cause:
	- the minimap grid is drawn with horizontal screen position from tile column and vertical screen position from tile row, but the teleport path and camera marker were still mixing row/column into renderer `X/Y` in the opposite order.
	- concrete runtime symptom: clicking the top-right Designer Island teleported the marker to the lower-left even though the minimap status text reported the intended tile.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-axis-repair/"` passed on Mar 25, 2026.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on fullscreen/docked minimap marker alignment or top-right teleport correctness.

### Mar 25, 2026 - Fullscreen Minimap Release Blocker Closed

- The earlier fullscreen-minimap blocker status is now historical, not current.
- Final state for `v0.4.5`:
	- the bad `TileSize` minimap hypothesis was reverted
	- the later broad world-axis swap was also reverted
	- the landed fix is the narrower transpose-only repair recorded above, followed by runtime user confirmation that the previously broken top-right Designer Island scenario now behaves correctly
- Planning prompts remain useful as archaeology if the bug regresses again, but they should no longer be read as describing an active unresolved blocker.

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
