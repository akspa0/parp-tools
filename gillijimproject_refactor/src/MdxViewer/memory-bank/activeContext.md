# Active Context — MdxViewer / AlphaWoW Viewer

## Mar 31, 2026 - Live Viewer Still Has A Remaining Adapted-M2 Shaded-Pass Failure After The Build Fix

- latest live screenshots after the `AzjolRoofGiant.m2` build-resolution correction still show a large missing world-M2 set, especially the giant root structures expected to cover the development terrain
- the new evidence is important because it narrows what is still broken:
   - selected-object tooltip text still resolves those root models
   - `Show Bounding Boxes` still draws their world-space bounds from `WorldScene` instance metadata
   - those overlays prove placement and object registration, but they do not prove that the shaded triangle pass rendered successfully
- current working interpretation:
   - stale build selection was one real seam and is now corrected in code
   - the remaining active seam is now inside the adapted-M2 render path itself, most likely around world-pass submission or per-layer material routing
- immediate next files to inspect or instrument:
   - `src/MdxViewer/Terrain/WorldScene.cs` at the adapted-M2 opaque/transparent `RenderWithTransform(...)` submission sites
   - `src/MdxViewer/Rendering/ModelRenderer.cs` inside `RenderGeosets(...)`, especially pass filtering, alpha-cutout detection, and transparent fallback behavior
   - `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` blend-mode / layer-flag mapping for world M2 materials
- immediate next proof goal:
   - add targeted viewer diagnostics or a temporary force-solid adapted-M2 path to determine whether the missing root models fail before draw submission or become invisible because of material-state classification once submitted
- important boundary:
   - no code change for this remaining seam has landed yet
   - do not describe the runtime as fixed based only on the successful `AzjolRoofGiant.m2` probe and the stale-build override correction

## Mar 31, 2026 - M2 Runtime Path Now Corrects Stale Build Selection From The Real Client Root

- targeted follow-up on the remaining invisible standalone/world M2 report used a single concrete asset: `AzjolRoofGiant.m2`
- direct probe evidence showed the active seam was build mismatch, not just renderer/material fallback:
   - the same model + skin + client root adapted correctly under `3.3.5.12340`
   - the same asset collapsed to degenerate geometry under stale `3.0.1.8303`
- landed runtime correction in `src/MdxViewer/Terrain/BuildVersionCatalog.cs`, `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/Terrain/WorldAssetManager.cs`, and `src/MdxViewer/Rendering/WmoRenderer.cs`:
   - M2-family load paths now infer an effective build from the actual client/game path and use that when it disagrees with the stale selected build
   - this correction covers standalone M2 open, world M2 loads, and WMO doodad M2 loads
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - this still needs real viewer runtime confirmation on the saved 3.3.5 client plus the development map overlay

## Mar 31, 2026 - WorldScene Again Forces M2-Adapted World Doodads Through RenderWithTransform

- latest live-user runtime signal after the negative-lookup suppression slice was specific: world M2s were still showing up in tooltips / picking but most remained invisible in-scene
- landed a narrow follow-up in `src/MdxViewer/Terrain/WorldScene.cs`:
   - M2-adapted world doodads now bypass the generic batched `RenderInstance(...)` path again and use `RenderWithTransform(...)` for both opaque and transparent world passes
   - classic MDX doodads stay on the batched path
   - `BeginBatch(...)` now seeds from the first actually batched renderer instead of whichever visible doodad happens to appear first
- concrete reason for the change:
   - current `ModelRenderer.RequiresUnbatchedWorldRender` only flags particle/ribbon cases, not `IsM2AdapterModel`
   - that left adapted world M2s on the shared instanced path even though the user symptom and older continuity both pointed at the world batch path as the likely invisible-model seam
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - no real-data viewer capture or flythrough was run in this chat
   - WMO hiccups and any remaining invisible-model edge cases still need runtime confirmation on the development map

## Mar 31, 2026 - Shared ModelRenderer Now Keeps Adapted M2s Visible On Base-Texture Misses

- latest user report after the world-scene M2 submission fix was narrower and more useful:
   - more objects now render
   - the remaining invisible set also fails in standalone model viewing, which points at the shared M2 render/material path rather than another placement bug
- landed renderer-side correction in `src/MdxViewer/Rendering/ModelRenderer.cs`:
   - adapted M2s now use the neutral white fallback texture even when the missing texture is the base `Load` layer
   - fallback-geoset drawing is no longer suppressed just because an adapted/pre-release layer missed texture resolution
- why this is the likely right seam:
   - the previous logic could leave an adapted M2 with zero rendered layers when its primary texture lookup failed
   - that exactly matches the current symptom: object exists in scene metadata and can be inspected, but the rendered model disappears in both world and standalone paths
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - runtime proof is still pending
   - if some MPQ-backed M2s still show only partial strips or malformed geometry after this, the next investigation seam should move into `WarcraftNetM2Adapter` skin/submesh/material extraction

## Mar 31, 2026 - MdxViewer Now Consumes wow-viewer Core.Runtime For World Render Telemetry

- after the initial in-app renderer stats slice landed, the first stable `WorldScene` seam was moved out into `wow-viewer`
- landed consumer-side follow-up in `src/MdxViewer/Terrain/WorldScene.cs` plus both MdxViewer project files:
   - `WorldScene` now uses `WowViewer.Core.Runtime.World.WorldRenderFrameStats` and `WorldRenderOptimizationAdvisor` instead of defining its public telemetry contracts inline
   - `MdxViewer.csproj` and `MdxViewer.CrossPlatform.csproj` now reference `wow-viewer/src/core/WowViewer.Core.Runtime/WowViewer.Core.Runtime.csproj`
- important boundary:
   - this does not mean the world renderer has been migrated out of `MdxViewer`
   - `WorldScene` still owns render orchestration, culling, submission, and overlay behavior; only the reusable telemetry contract and hint logic moved
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only

## Mar 31, 2026 - WorldScene Now Emits A Real Renderer Stats Frame Instead Of Only Ad-Hoc Counters

- first renderer-first implementation slice is now in the live viewer code, not just in the plan file
- landed in `src/MdxViewer/Terrain/WorldScene.cs` and `ViewerApp_Sidebars.cs`:
   - `WorldScene` now owns a reusable render-frame contract with stage timings and the visible WMO/MDX scratch lists
   - the active world path records per-frame CPU timings for the current major passes plus MDX batched-vs-unbatched submission counts
   - the terrain sidebar now exposes a `Renderer Stats` tree with the last captured world-frame timings and a heuristic `next win` hint
- current purpose:
   - this is the instrumentation and smallest frame-contract seam needed before deeper MDX batching or WMO pass extraction work
   - it does not yet claim that explicit scene-layer ownership is finished
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - this is compile validation only
   - no development-map runtime capture was performed yet for the new stats panel, so the live next-win hint still needs manual readback during real camera movement

## Mar 31, 2026 - Renderer Performance Is The Next Mainline Slice, Not More Overlay Expansion

- current user priority is now explicit: moving the camera on live world maps is still too slow, so renderer architecture comes before more feature work
- roadmap recorded in `gillijimproject_refactor/plans/mdxviewer_renderer_performance_plan_2026-03-31.md`
- concrete planning direction:
   - the active bottleneck is the current monolithic `Terrain/WorldScene.cs` render path
   - `Rendering/RenderQueue.cs` exists but is not yet the active owner of world-scene submission
   - the first implementation slice should add frame timings/counters and an explicit world render-frame contract before deeper renderer rewrites
   - MDX batching/state reduction should come before graveyard overlays or other new world markers
   - `LightService` stays important, but lighting completion follows render-frame cleanup rather than leading it
   - graveyards from `WorldSafeLocs.dbc` should be added later as a sibling lazy-loaded overlay to Area POIs and taxi paths
- important boundary:
   - this is planning/continuity only
   - no renderer-performance code landed in this slice yet

## Mar 31, 2026 - Fixed Sidebar Mode Uses Splitter-Driven Panels Instead Of Pseudo-Resizable Windows

- latest viewer-shell correction was not another PM4 or terrain slice; it was about making the default fixed sidebars actually usable
- landed viewer-shell changes in `src/MdxViewer/ViewerApp.cs` and `ViewerApp_Sidebars.cs`:
   - fixed left/right sidebars now use explicit draggable splitter bars to control width
   - the sidebars remain edge-anchored panels while the splitter updates the stored widths directly
   - fixed sidebars now intentionally disable normal window resizing because the supported resize path is the splitter, not hidden border grabs
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - this is compile validation only
   - development-map runtime still needs manual confirmation that the splitter interaction feels correct across different window sizes

## Mar 31, 2026 - Mouse-Look Regression Came From The Splitter Host Covering The Viewport

- immediate follow-up after the new fixed-sidebar splitter shell was that mouse camera look stopped working
- landed correction in `src/MdxViewer/ViewerApp_Sidebars.cs`:
   - the old one-window splitter host was replaced with narrow splitter-only windows for each active sidebar handle
   - only the actual splitter strip now captures mouse input, so the viewport is no longer treated like it sits under one giant UI window in fixed-sidebar mode
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - this is compile validation only
   - the live viewer still needs manual confirmation that right-mouse camera drag is restored cleanly

## Mar 31, 2026 - Viewer Can Now Silence Hover Cards And Hide Lower UniqueId Layers

- latest runtime/exploration request was not another PM4 slice; it was about making object archaeology easier in the live world viewer:
   - mouse-hover scene tooltips needed a direct off switch
   - the viewer needed a `UniqueId` slider that can hide older/lower-id object layers either map-wide or only in the current camera tile
- landed viewer changes in `src/MdxViewer/ViewerApp.cs` and `Terrain/WorldScene.cs`:
   - `DrawSceneHoverAssetOverlay()` now exits early when `WorldScene.ShowHoveredAssetTooltips` is disabled
   - the `World Objects` panel now exposes `Hover Tooltips`, `Hide UniqueId Layers`, `Per Map` vs `Camera Tile` scope, the current camera tile, explicit `Hide Range Min` / `Hide Range Max` controls, detected archaeology layers for the active scope, and a reset button
   - `WorldScene` now carries a scoped unique-id range filter, gap-based archaeology layer detection, and tile ownership metadata on flattened object instances so per-tile filtering survives instance rebuilds
   - hidden unique-id ranges are removed consistently from rendering, hover hit testing, scene picking, and debug bounds instead of only being visually mislabeled
- validation completed:
   - `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `Terrain/WorldScene.cs`
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - this is compile validation only
   - the development map still needs manual runtime confirmation that the archaeology slider reveals the intended object layers cleanly and that disabling hover cards feels correct in practice

## Mar 31, 2026 - Zone Lighting Keeps DBC Color But No Longer Owns Fog Distance

- user runtime report after the shared `LightService` / `TerrainLighting` frame-state change was that fog could no longer be effectively removed and that farther world visibility regressed
- root cause was concrete:
   - `TerrainLighting.ApplyExternalLighting(...)` overwrote `FogStart` and `FogEnd` every frame
   - terrain sidebar fog sliders therefore stopped controlling the live scene as soon as `LightService` found an active zone
   - `WorldScene` then fed that shortened fog end into WMO and object far-visibility logic, so view distance shrank with it
- landed correction in `src/MdxViewer/Terrain/TerrainLighting.cs` and `Terrain/WorldScene.cs`:
   - zone lighting still drives directional or ambient or fog color and shared time-of-day lighting
   - user-controlled fog start/end now remain authoritative for the live world scene
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - this is compile validation only
   - development-map runtime still needs manual confirmation that the older no-fog / farther-view behavior is effectively restored

## Mar 31, 2026 - WDT Global WMO Terrain Path Fixed; M2 Detail UV Regression Reduced To Adapter Metadata

- latest runtime regressions were not PM4-related; they were in the live world/object path:
   - terrain-backed maps were dropping WDT-level global `MWMO` or `MODF` placements that should appear as over-terrain roof or shell geometry
   - M2 material behavior regressed enough that some detail layers rendered as giant projected leaf-like sheets
- landed terrain fix in `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`:
   - WDT global WMO parsing now runs for terrain maps when `MPHD` indicates global map objects or when `MWMO` and `MODF` are both present
   - terrain-map WDT `MODF` placements now convert from file-space into the same renderer-space convention as ADT placements
- current M2 regression narrowing points at `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`, not `ModelRenderer`:
   - the old adapter effectively forced UV0 for M2 layers
   - the newer adapter resolved dynamic texture-coordinate ids and treated negative ids as reflective/env-mapped behavior
   - active mitigation now clamps negative coord ids back to UV0 and removes the negative-id `SphereEnvMap` inference
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - this is still compile validation only
   - the development map still needs manual runtime confirmation for both restored WDT global WMOs and the M2 oversized-detail regression

## Mar 31, 2026 - Active M2 Runtime Path Is Back On The Conservative Per-Section UV0 Material Mode

- latest M2 follow-up was not a new parser theory pass; it was another live regression report that some trees still rendered as leaves with no trunks
- landed correction in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`:
   - the active runtime material path now keeps only the first batch per section again
   - active runtime layers are forced back to `UV0` for that conservative path
   - this is explicitly a rollback to the old known-good compatibility behavior, not proof that richer multi-layer M2 semantics are correct yet
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- important boundary:
   - this is compile validation only
   - the live viewer still needs manual confirmation that affected tree trunks are visible again

## Mar 30, 2026 - WorldScene Now Collects Visible WMO Instances Before Submission

- renderer follow-up after the shared LightService or TerrainLighting frame-state fix stayed intentionally narrow instead of attempting a full render-graph rewrite in one pass
- landed structural change in `src/MdxViewer/Terrain/WorldScene.cs`:
   - `WorldScene` now builds a reusable visible-WMO scratch bucket before the WMO draw phase instead of mixing frustum or distance culling and submission inline in the opaque pass
   - this mirrors the existing visible-MDX scratch path more closely and gives the world renderer a real WMO visibility layer to build on for later opaque or liquid or transparent pass separation
   - current WMO draw behavior is still the same consumer-wise because `WmoRenderer.RenderWithTransform(...)` remains the owner of WMO-local opaque or liquid or transparent sequencing for now
- important boundary:
   - this is groundwork for explicit layer ownership, not the final WMO pass split yet
   - it should reduce repeated cull logic inside `Render(...)`, but it does not yet batch WMO materials across instances or move WMO liquids into a shared scene-wide liquid layer
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
   - no development-map runtime signoff has been completed yet for this WMO visibility-bucket slice

## Mar 30, 2026 - PM4 Workbench Glossary Now Explains Viewer-Derived Labels

- user called out that PM4 UI labels were opaque enough that even the current maintainer could not reliably tell which terms were raw PM4 fields and which were viewer inventions
- landed clarification in `src/MdxViewer/ViewerApp_Pm4Utilities.cs`, `ViewerApp.cs`, `Terrain/WorldScene.cs`, and `README.md`:
   - the PM4 workbench now exposes a glossary/evidence block that separates raw chunk names, viewer aliases, and viewer-generated structure
   - `part` / `ObjectPartId` is now explicitly documented as a viewer-generated split id assigned after `CK24` grouping, dominant `MSLK` grouping, optional `MDOS` split, and optional connectivity split
   - selected-object text and graph text now repeat that `part` is not a raw PM4 field
- important boundary:
   - this is terminology/help clarification only
   - no PM4 decode semantics or placement behavior changed in this slice

## Mar 30, 2026 - World Lighting Now Uses One Shared Frame State When DBC Light Data Is Active

- current renderer correctness problem was broader than fog alone:
   - `WorldScene` could already pull sky or fog colors from `Light.dbc` / `LightData.dbc` through `LightService`
   - terrain, WDL, liquids, skybox backdrops, WMOs, and MDXs were still primarily reading `TerrainLighting`, which meant one frame could mix DBC fog/sky with fallback ambient or direct light colors and even a mismatched sun time
- landed correction in `src/MdxViewer/Terrain/TerrainLighting.cs` and `Terrain/WorldScene.cs`:
   - `TerrainLighting` now accepts an external per-frame lighting override for direct color, ambient color, fog color, and fog range
   - when `LightService` resolves an active zone, `WorldScene` now maps its current time of day into `TerrainLighting.GameTime`, applies the DBC-driven light/fog override, and updates that shared lighting state before rendering WDL, terrain, liquids, skybox backdrops, WMOs, or MDXs
   - when no active DBC light zone exists, the renderer falls back to the existing procedural `TerrainLighting` path cleanly
- important boundary:
   - this is a lighting-consistency slice, not the full render-layer or shader-bucketing redesign yet
   - it should improve correctness immediately, but it does not by itself remove the pass-order or state-churn costs still embedded in `WorldScene.Render(...)`
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 after the associated capture-automation compile blockers in `ViewerApp_CaptureAutomation.cs` were corrected
- important validation boundary:
   - this is compile validation only
   - no development-map runtime signoff has been completed yet for the shared LightService or TerrainLighting frame-state path

## Mar 30, 2026 - Inspector-First PM4 Workbench Replaces The Old Multi-Window Default

- latest viewer-shell change is explicitly about workflow quality, not another PM4 decode theory pass:
   - PM4 bounds now default off
   - PM4 x-ray now default off
   - fixed sidebars now start on by default so the right inspector remains anchored instead of drifting through docked-panel state unless the user opts back in
   - the normal PM4 workflow now lives in the right sidebar `PM4 Workbench` rather than spreading match/correlation/settings across multiple PM4 floating windows
   - hover cards now stay shorter and act more like loot-tooltips, while click selection owns the detailed PM4 graph or match or correlation workflow
- important boundary:
   - the old PM4 alignment micro-window is still available as an advanced fallback from the workbench because the per-axis transform controls are still too large to inline cleanly in one pass
   - compile validation passed, but no manual runtime signoff has been captured yet for the new inspector-first PM4 shell

## Mar 30, 2026 - PM4 Hover Tooltip Now Uses PM4-First Metadata And Hover-Time Match Cache

- the generic asset hover card is no longer limited to WMO or MDX or WL metadata only
- landed viewer correction:
   - `WorldScene.UpdateHoveredAssetInfo(...)` now detects PM4 overlay objects while the PM4 overlay is visible and returns a hovered PM4 object key alongside the tooltip payload
   - PM4 hover detection now wins first in that mode so the overlay can stay on PM4 geometry instead of falling through to nearby scene assets
   - `ViewerApp.DrawSceneHoverAssetOverlay()` now uses a darker gold-bordered tooltip look with brighter title text and a compact PM4 top-candidate list
   - hovered PM4 match suggestions use a dedicated cache and the existing PM4 match builder so mouse-over can show likely placement candidates without changing selected-object state
- important research note worth preserving:
   - current development-map evidence still points to derived `CK24` low-16 values separating many `WMO`-like PM4 meshes, while `CK24=0x000000` remains an unresolved umbrella bucket that still appears to contain many `M2`-like families
   - the tooltip candidate list is diagnostic evidence only, not proof that PM4 subobject or ownership semantics are solved
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 after the hover-tooltip update
- important boundary:
   - this is compile validation only
   - no development-map runtime signoff has been completed yet for the new PM4 hover tooltip behavior

## Mar 30, 2026 - PM4 Additive Collection Click Path No Longer Yields To Normal Scene Picks

- user runtime feedback on the new PM4 collection workflow showed the first additive click path was still not usable in dense scenes:
   - `Shift + Left Click` could fail silently because PM4 collection still depended on normal scene hit priority
   - collection removal could leave stale highlighted PM4 parts behind
- landed viewer correction:
   - `ViewerApp.PickObjectAtMouse(...)` now treats the `addPm4ToCollection` path as PM4-first and directly toggles collection membership when any PM4 hit exists, instead of letting WMO or MDX selection take the click first
   - failed additive clicks now produce an explicit status message instead of looking like ignored input
   - collection-item `Remove` now calls back into `SyncPm4CollectionHighlight()` immediately so highlight state tracks the list
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 after the click-path fix
- important boundary:
   - this is compile validation only
   - no development-map runtime signoff was completed yet after this correction

## Mar 30, 2026 - PM4 Selection Uses Merged Groups Again After Family-Split Regression

- latest user runtime report on zero-`CK24` PM4 selection is that the temporary selection-family split was making things worse, including visibly splitting one object into partial pieces
- active correction in `src/MdxViewer/Terrain/WorldScene.cs`:
   - removed the separate family-selection grouping path
   - restored selected-object ownership, highlight color, and selected-object graph expansion to the existing merged-group key path
   - left the selected-only PM4 match builder/cache changes in place because they are not tied to the half-object regression
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 after the rollback
- important boundary:
   - this is compile validation only
   - the development-map runtime still needs manual confirmation that zero-`CK24` selection is back to the last non-regressed behavior

## Mar 30, 2026 - Zero-CK24 Seed Buckets Now Use MSLK Ownership Before Connectivity Fallback

- latest PM4 regrouping investigation narrowed the remaining zero-`CK24` fragmentation to one runtime decision in `src/MdxViewer/Terrain/WorldScene.cs`:
   - zero/root seed buckets were still bypassing `SplitSurfaceGroupByMslk(...)`
   - they went straight from `(GroupKey, AttributeMask)` seed grouping into connectivity splitting, which over-fragmented linked families whenever `MSLK.GroupObjectId` tied together disconnected or weakly connected pieces
- landed correction:
   - zero/root seed buckets now take an `MSLK`-first split path via `SplitZeroCk24SeedGroup(...)`
   - subgroups that already resolve a non-zero dominant `MSLK.GroupObjectId` stay intact
   - only the leftover subgroups with no `MSLK` ownership fall back to connectivity splitting
- linked placement-ref collection was also tightened to match the research-side behavior more closely:
   - when a PM4 subgroup already resolves one or more `MSLK.GroupObjectId` values, `CollectLinkedPositionRefs(...)` now gathers `MPRL` refs from all links in those ownership families before falling back to direct surface-linked `RefIndex` lookup
- important boundary:
   - this does not reopen the earlier fake-`MsurIndex` path; current evidence still says active `MSLK` layout is 20-byte and the old 24-byte `MsurIndex` interpretation is not the decoder we want to restore
   - this is file-diagnostics clean, but not runtime-signed off yet on the development map
   - solution build was attempted and failed only because `ParpToolsWoWViewer (16096)` still held the output DLLs open during copy

## Mar 29, 2026 - v0.4.6 Target Freezes PM4 Runtime Wins And Points The Renderer At Real Layers

- latest user runtime report says the recent PM4 runtime fixes were the right ones and that PM4 objects are now almost `100%` correct on the active development-map workflow
- treat that as the current release freeze line for PM4 runtime behavior in `MdxViewer`, not as a signal to immediately reopen another broad PM4 rewrite
- `v0.4.6` is now the active viewer release target
- the PM4 fixes that define this target are:
   - ADT-scale camera-window indexing for PM4 loads
   - PM4 `XX_YY -> YY_XX` terrain tile remap
   - empty-carrier and empty-known-window handling
   - no terrain-AOI slicing of already loaded PM4 structures
   - linked-group placement solving inside non-zero `CK24` families
- next renderer/performance seam requested by user is more structural than the last two micro-optimizations:
   - the viewer needs actual render layers / submission ownership instead of leaving terrain, WMO, MDX, liquids, PM4 overlays, and debug draws mixed inside one large world-scene render routine
   - current `Rendering/RenderQueue.cs` exists but is not the active world-scene path; the next serious performance slice should either activate or replace that idea with layer-aware submission lists
- likely first implementation target for that work:
   - collect visibility once
   - build explicit per-layer submission lists
   - batch compatible opaque/transparent work by renderer/material/state where possible
   - keep PM4/debug/editor overlays as separate late layers so exploration tooling stops polluting main world submission cost
- important boundary:
   - current performance improvements are real but partial
   - the draw-call/state-churn problem is not closed until layer ownership and submission are explicit

## Mar 29, 2026 - Second Rendering Performance Slice Targets WMO Doodad Spikes And Object Fog

- user feedback after the first `WorldScene` optimization was still negative on runtime feel:
   - the viewer could eventually climb toward roughly `30 FPS`, but only after long stabilization
   - tile or data loads still produced hard hitches down to `0 FPS`
   - world objects were visibly fogged in a way the user does not want
- strongest newly confirmed cause for the load spikes:
   - `src/MdxViewer/Rendering/WmoRenderer.cs` eagerly loaded the active doodad set in its constructor
   - that means a newly visible WMO could recursively construct many doodad `MdxRenderer`s immediately on the render thread
- landed behavior:
   - `WmoRenderer` now supports deferred initial doodad loading for world-scene use and drains queued doodad model loads incrementally during render under a tight per-frame budget
   - `src/MdxViewer/Terrain/WorldAssetManager.cs` now opts world-scene WMO loads into the deferred doodad path
   - `src/MdxViewer/Terrain/WorldScene.cs` now lowers the render-thread deferred asset load budget from `24/20 ms` to `6/4 ms`
   - world-scene object fog is now disabled by default through `ObjectFogEnabled`, while WMO cull distance still uses terrain fog end so disabling object fog does not accidentally shorten world draw distance
   - `src/MdxViewer/ViewerApp.cs` now exposes a `Fog Objects` checkbox in the world-objects panel
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings only
- important boundary:
   - this still needs real viewer runtime validation on the fixed development data paths
   - no measured hitch reduction or manual no-fog signoff has been captured yet in this session

## Mar 29, 2026 - First Rendering Performance Slice Targets Duplicate MDX Scene Walks

- user priority has shifted from more PM4 exploration to practical viewer rendering performance, with current map loads still reported around `1-5 FPS`
- first chosen optimization stays in `src/MdxViewer/Terrain/WorldScene.cs` rather than jumping straight into shader/effect parity:
   - collect visible MDX or taxi instances once per frame
   - reuse that result for both opaque and transparent passes
   - avoid duplicate frustum/AABB checks and duplicate `TryGetQueuedMdx(...)` calls inside the same frame
- landed behavior:
   - added a `VisibleMdxInstance` scratch contract plus `_visibleMdxInstances`
   - added `CollectVisibleMdxInstances(...)` to precompute visible renderer references and fade values
   - transparent sorting now works from the already-visible scratch list instead of re-walking `_mdxInstances` and `_taxiActorInstances`
- validation completed:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings only
- important boundary:
   - this reduces obvious per-frame CPU waste but does not yet prove acceptable runtime FPS on real map loads
   - next likely slice should continue with scene culling or submission or batching cost before broader shader/lighting work

## Mar 29, 2026 - PM4 Hierarchy Research Moved Into `wow-viewer` And Back Into `MdxViewer`

- the active PM4 research path in `MdxViewer` is no longer anchored on `src/Pm4Research.Core`; `src/MdxViewer/Terrain/WorldScene.cs` now builds its cached PM4 research context from shared `wow-viewer/src/core/WowViewer.Core.PM4`
- landed shared seam:
   - `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4ResearchHierarchyAnalyzer.cs` now ports the old object-hypothesis split families into `Core.PM4`
   - each hypothesis now also carries shared placement evidence through `Pm4ForensicsPlacementComparison` plus dominant `MSLK.GroupObjectId` / link-group ownership data
   - `WowViewer.Tool.Inspect` now supports `pm4 hierarchy --input <file.pm4> [--output <report.json>]` for real-data scene-graph or placement research outside the viewer
- landed active-viewer consumer update:
   - `WorldScene` now uses shared `Pm4ResearchSnapshotBuilder`, shared decode-audit output, and shared hierarchy analysis instead of the old `Pm4ResearchObjectHypothesisGenerator`
   - the `PM4 Research` section in `ViewerApp_Pm4Utilities.cs` now shows candidate link-group counts, dominant group ids, shared coordinate-mode decisions, planar transform flags, frame yaw, and heading deltas for the selected object
- important boundary:
   - this is a research and diagnostics slice, not a final runtime placement fix
   - the new shared report explains more of the scene-graph evidence, but it does not by itself solve the remaining opposite-side or void-placement PM4 regressions

## Mar 29, 2026 - PM4 Disappear-On-Approach Bug Traced To Additive Loader State Reset

- latest viewer symptom tightened a separate PM4 runtime failure from the CK24 placement work:
   - PM4 objects could disappear as the camera approached them
   - pressing `Reload PM4` did not restore them reliably
- landed root-cause fix in `src/MdxViewer/Terrain/WorldScene.cs`:
   - the additive PM4 loader was still resetting `_pm4LoadedCameraWindow = null` inside `BeginPm4OverlayLoad(...)` before the async load completed
   - `TryFinalizePm4OverlayLoad()` uses `_pm4LoadedCameraWindow.HasValue` to decide whether to merge new tiles into the existing PM4 overlay or clear and replace it
   - that meant every incremental camera-window load was effectively treated as a full replacement, which can drop previously loaded PM4 objects while moving around the map
- current behavior after the fix:
   - normal PM4 background loads keep the previous loaded-window state until finalize merges the new cache data
   - `Reload PM4` now does a real explicit clear of PM4 runtime state first and then starts a cache-bypassing reload
- important boundary:
   - this is build-validated only in this session
   - if PM4 still disappears after this patch, the next likely seam is render/cull gating rather than overlay residency bookkeeping

## Mar 29, 2026 - Same-Tile PM4 Candidate Collisions No Longer Merge By Default

- latest attached non-zero `CK24` graph exports still showed exact paired duplicate parts even when both split toggles were off, which pointed away from viewer-side fragmentation toggles and toward the PM4 file-selection path instead
- landed correction in `src/MdxViewer/Terrain/WorldScene.cs`:
   - both PM4 runtime loading and offline PM4 OBJ export now group candidate `.pm4` files by effective tile first
   - when more than one file maps to the same effective tile, the viewer keeps one preferred canonical candidate instead of rebasing object-part ids and merging the sets together
   - preferred candidates are ranked toward the cleanest `.../World/Maps/<map>/...` path and collisions are logged for follow-up diagnosis
- why this matters:
   - the current best explanation for the `part=0` / `part=495` style duplicate pattern is that two different PM4 files were both being accepted as the same map tile and then merged before graph export or runtime rendering
   - if that hypothesis is right, this should remove a whole class of fake PM4 placement regressions that were really duplicate-content collisions
- important boundary:
   - this is build-validated only in this session
   - no live viewer signoff yet proves that the remaining opposite-corner objects were caused by same-tile candidate collisions rather than another transform seam

## Mar 29, 2026 - Zero-CK24 Objects Stop Inheriting The Global PM4 N/S Mirror

- latest manual viewer evidence tightened the root cause for the remaining misplaced root-bucket PM4 overlays:
   - with `Mirror PM4 N/S` enabled globally, the bad `CK24=0x000000` objects still sat in the wrong place
   - manually pressing `Wind Obj Y` on the selected object corrected the placement
   - that means the zero/root bucket still wanted the opposite Y-handedness from the current global PM4 mirror path
- landed correction:
   - `BuildPm4ObjectTransform(...)` now only applies the global `_pm4FlipAllObjectsY` mirror to non-zero `CK24` objects
   - `CK24=0x000000` objects can still use explicit object-local scale or winding overrides, but they no longer inherit the global north/south mirror by default
- important boundary:
   - this is a narrow runtime-consumer fix for the zero/root bucket only
   - it does not prove that all PM4 placement semantics are solved, and it still needs manual viewer confirmation on the development map

## Mar 29, 2026 - Zero-CK24 Seed Groups Stop Re-Splitting After Their Mandatory Seed Split

- latest PM4 graph/runtime evidence suggests the zero/root `CK24` path was still being fragmented again after the intended seed-level connectivity split
- current viewer behavior before this change:
   - `BuildPm4OverlaySeedGroups(...)` already forces zero-`CK24` surfaces into grouped seed buckets that must split by connectivity once
   - later in `BuildPm4TileObjects(...)`, those same zero/root linked groups could still be re-split by the user-facing `Split CK24 by MdosIndex` or `Split CK24 by Connectivity` stages
   - that made the runtime object list and PM4 graph look like paired or over-fragmented sub-parts even when the first seed-level split had already isolated the root components
- landed correction:
   - zero/root seed groups that already required the seed connectivity split now bypass the later MDOS/connectivity split toggles entirely
   - non-zero `CK24` groups still keep the later toggle-controlled split path unchanged
- important boundary:
   - this only removes the redundant second split stage for zero/root buckets; it does not yet prove the remaining placement basis or frame-rotation semantics are final
   - the fix is build-validated only in this session and still needs live viewer confirmation on the development map

## Mar 29, 2026 - Viewer Shell Resize/Input Sync Hardened Again

- the latest recurrence of the broken shell symptom was not isolated to PM4 controls; the whole viewer UI fell back into bad resize or hit-testing behavior again, with toolbar/sidebar layout obviously wrong and buttons effectively unusable
- root cause was still in the Silk-to-ImGui window metrics bridge in `src/MdxViewer/ViewerApp.cs`:
   - the viewer was invoking the private `ImGuiController.WindowResized(Vector2D<int>)` hook against logical window size only
   - it was not also keeping `ImGui.GetIO().DisplaySize` and `DisplayFramebufferScale` explicitly synchronized from both logical size and framebuffer size
   - it also relied on `FramebufferResize` alone instead of subscribing to the logical `Resize` event directly
- landed hardening:
   - `ViewerApp` now subscribes to both `Resize` and `FramebufferResize`
   - `SyncImGuiWindowMetrics(...)` now updates the private Silk hook only when logical size changes, and also writes `ImGuiIO.DisplaySize` plus `DisplayFramebufferScale` from `window.Size` and `window.FramebufferSize`
   - the per-frame update path now re-syncs both metrics before `_imGui.Update(...)`
- important boundary:
   - this is build-validated only in this session; manual runtime confirmation is still needed to prove the shell is stable again

## Mar 29, 2026 - Zero-CK24 PM4 Root Bucket Placement No Longer Reuses One Mixed Frame

- follow-up runtime evidence showed the latest PM4 changes had drifted some `M2`-aligned PM4 data while nearby WMO-aligned PM4 still looked correct
- current best root cause was the zero-`CK24` / root-style bucket path in `src/MdxViewer/Terrain/WorldScene.cs`:
   - zero-`CK24` seed groups were first formed as mixed buckets by `GroupKey` or `AttributeMask`
   - then connectivity split happened later, but all resulting linked groups still reused one shared placement basis from the whole mixed seed bucket
   - that meant unrelated zero/root pieces could inherit the same planar transform or pivot or frame yaw or connector basis, which is consistent with M2/root-style drift while non-zero WMO-style `CK24` groups stay mostly stable
- landed fix:
   - zero/root-style seed groups that require connectivity splitting now resolve coordinate mode or placement solution or connector keys per linked group instead of inheriting the whole mixed-bucket frame
   - non-zero `CK24` groups still keep the existing shared per-`CK24` frame basis across their split parts
- important semantic note from user evidence:
   - `CK24 = 0x000000` should not be treated as "just M2 data"
   - current better framing is an unresolved root or bucket or umbrella group that can still contain structure relevant to placed-object alignment
- follow-up alignment-control correction:
   - the earlier raw-`CK24` alignment controls were too coarse because they keyed transforms only by `ck24` across all loaded tiles
   - alignment state is now keyed by `(tileX, tileY, ck24)` so experiments on `CK24=0x000000` can rotate or mirror one tile-local bucket without dragging every matching raw `ck24` bucket elsewhere in the map
   - the PM4 alignment window now also exposes explicit tile/object winding toggles that flip axis sign directly for faster handedness experiments
- important boundary:
   - this is compile-validated in `MdxViewer`, not manual runtime signoff yet
   - if runtime testing still shows drift after this fix, the next likely seam is not mixed-bucket placement reuse anymore but base-frame rotation ownership for specific zero/root subfamilies

## PM4 CK24 Frame/Mesh Rotation Ownership Corrected (Mar 29)

- The PM4 CK24 object-generation path in `src/MdxViewer/Terrain/WorldScene.cs` no longer applies the shared `worldYawCorrection` directly to the visible mesh lines and triangles during vertex conversion.
- Reason:
   - current runtime viewer evidence showed CK24 objects whose visible mesh was being rotated into the wrong orientation and side of the tile, while the correction appeared to belong to the object frame or anchor basis instead
   - that indicated the PM4 object pipeline was treating frame yaw as mesh geometry rotation instead of as frame metadata
- Landed behavior:
   - `ComputeSurfaceRendererCentroid(...)` still uses the CK24 frame yaw path for placement-anchor/frame interpretation
   - `BuildCk24ObjectLines(...)` and `BuildCk24ObjectTriangles(...)` now convert PM4 mesh vertices without the frame yaw correction, so the rendered CK24 mesh stays in raw converted orientation while the frame/anchor basis remains separate
- Important boundary:
   - this is compile-validated in `MdxViewer`, not runtime-signed-off on the development map yet
   - if follow-up runtime testing shows the frame itself now needs explicit visualization, add that separately instead of re-baking frame yaw into mesh vertices
   - release publish currently relies on a workflow-side mitigation for duplicate dependency publish outputs because `WoWMapConverter.Core` still reaches into `WoWRollback.PM4Module` as an executable project instead of a pure library seam

## PM4 Raw CK24 Layer Alignment Added For Viewer Investigation (Mar 29)

- The PM4 alignment window in `src/MdxViewer/ViewerApp_Pm4Utilities.cs` now exposes a second transform block for the selected raw `CK24` layer, separate from the existing per-object-group controls.
- Reason:
   - current PM4 selection splits `ck24 == 0x000000` into synthetic per-part groups for object transforms, which blocked the user from rotating the raw `0x000000` family as one exploratory layer
   - the active investigation needs a way to test whether that raw family has a consistent per-tile or quadrant-like orientation rule across the map, not only per-part correction
- Landed behavior:
   - selecting any PM4 object now also exposes raw-`CK24` layer move or rotate or scale controls plus reset and print actions
   - `WorldScene` now keeps raw-`CK24` bounds and transform dictionaries keyed by the original `ck24` value across all loaded tiles
   - PM4 object rendering now applies raw-`CK24` layer transform first and then the existing object-group transform, so whole-layer experiments and per-part tweaks can be combined
   - PM4 interchange JSON now reports the raw-layer transform state alongside the existing object-group transform state for each exported object
- Important boundary:
   - this is compile-validated in `MdxViewer`, not runtime-signed-off placement logic
   - no UI runtime test was completed in this session yet, so the actual `0x000000` orientation hypothesis is still open

## PM4 Graph JSON Export Made JSON-Safe (Mar 28)

- The selected-object `PM4 Graph` export path in `src/MdxViewer/ViewerApp_Pm4Utilities.cs` no longer serializes the raw graph structs directly.
- Root cause:
   - `Pm4LinkedPositionRefSummary` can legitimately carry non-finite heading values when no normal headings exist, and raw `System.Text.Json` serialization rejects those values as invalid JSON
   - that surfaced in the viewer status bar as PM4 graph export failures instead of writing a file
- Landed fix:
   - `ExportSelectedPm4GraphJson(...)` now projects the graph into a JSON-safe anonymous payload
   - linked-position-ref heading values now serialize through a finite-or-null helper instead of emitting raw `NaN` or `Infinity`
   - this keeps the output standards-compliant JSON instead of relying on named floating-point literals
- Important boundary:
   - this fix was compile-validated in `MdxViewer`
   - the export button itself was not re-run through the UI in this session after the patch, so treat runtime confirmation as still pending

## PM4 Overlay North/South Mirror Defaulted (Mar 28)

- `WorldScene` now defaults the existing PM4 object-group mirror path on, so PM4 overlay objects are mirrored around their logical group pivot in renderer space by default.
- Reason:
   - current manual viewer feedback showed the PM4 reference geometry landing on the wrong north/south side of nearby placements, with handedness-related rotation mismatch against neighboring `M2` objects
   - the existing `Pm4FlipAllObjectsY` path already applies the right class of correction for that symptom: renderer-space `scale(1, -1, 1)` around the PM4 object-group pivot, which corresponds to a north/south flip because WoW world `X` is north and renderer `Y` is derived from world `X`
- Related narrow cleanup landed in the same pass:
   - `WorldScene` no longer keeps its own duplicate `Pm4PlanarTransform` contract; it now uses the shared `WowViewer.Core.PM4` contract directly
   - the CK24 coordinate-mode consumer path now keeps the richer shared `Pm4CoordinateModeResolution` result instead of collapsing it to a bool immediately
   - default world-space planar-transform fallback in `WorldScene` now comes from shared `Pm4PlacementContract`
- Important boundary:
   - this is a viewer-behavior correction based on current manual runtime evidence, not a signed-off final PM4 placement solution
   - the PM4 correlation-input builder still stays viewer-local for now; the active issue was overlay handedness/alignment, not downstream correlation state construction

## Default Next Non-MDX Continuation (Mar 28)

- If the next chat is just "move on from MDX", the default target should be `wow-viewer` `Core.PM4` library completion, not another classic `MDX` seam.
- Viewer-side PM4 work should stay narrow and consumer-focused; use the existing shared `Core.PM4` plan as the main continuation route.
- Only fall back to non-`MDX` shared-I/O slices when there is a concrete ADT/WDT/WMO proof target and the slice stays tool-thin.

## MDX Audit Against wow-viewer Shared Work (Mar 28)

- Audit result: do not assume every recent `wow-viewer` classic `MDX` reader slice is direct `MdxViewer` parity.
- Current classification:
   - `GEOS` shared work is grounded in real legacy parser/runtime ownership and is the cleanest parity slice so far.
   - `TXAN` shared payload ownership goes beyond the current classic `MdxFile` parser. The active renderer has texture-transform concepts and the M2 adapter populates them, but the classic MDX parser does not currently read `TXAN`.
   - `HTST` shared payload ownership has no current active classic `MdxViewer` parser/runtime equivalent.
   - `CLID` shared payload ownership also goes beyond the current classic parser; the active viewer currently uses only shared collision summary metadata in probe/model-info surfaces.
- If future work returns to classic `MDX` parity, the hotter missed seam is the already-used `ATSQ` / geoset-animation and material-animation behavior in the active renderer, not more cold chunk-family expansion.

## Viewer UI Resize And Hit-Testing Regression Fixed (Mar 28)

- The active viewer had a real UI-shell regression where panels stopped sizing correctly and buttons became effectively unclickable after resize or maximize.
- Current best root-cause assessment:
   - `ViewerApp` was updating the OpenGL viewport on `FramebufferResize`, but the packaged Silk `ImGuiController` could drift out of sync with the logical window size used for layout and mouse hit-testing on Windows.
   - The visible symptom matched stale `ImGui` display-size or input-space state rather than a single broken panel.
- Landed fix:
   - `src/MdxViewer/ViewerApp.cs` now reflects the packaged private `ImGuiController.WindowResized(Vector2D<int>)` method once and uses it through `SyncImGuiWindowSize(...)`.
   - `SyncImGuiWindowSize(...)` now runs once after controller creation and again before each `_imGui.Update(...)`, using `_window.Size` and a last-synced guard to avoid redundant calls.
   - `OnResize(...)` still keeps `_gl.Viewport(...)` tied to framebuffer resize; the fix only resyncs the `ImGui` logical window size.
- Current verified validation:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 28, 2026 after the patch.
   - a short `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj --` startup smoke launched cleanly on Mar 28, 2026.
   - the user manually retested the resized shell on Mar 28, 2026 and reported that it now seems to be working.
- Important boundary:
   - this is build plus startup smoke plus user manual UI validation only.
   - no automated UI regression test exists yet for this resize or hit-testing path.
   - the current fix depends on the private Silk `ImGuiController` resize method name staying stable; if the package is upgraded and the regression reappears, revisit the controller integration first.

## ViewerApp Shared `MDX` Runtime Metadata Consumer Validation (Mar 28)

- The active standalone viewer now consumes shared `wow-viewer` classic `MDX` summary and `GEOS` payload metadata in the real runtime load path instead of deriving all model-info sidebar counts from `MdxFile.Load(...)` geosets alone.
- Landed pieces:
   - `src/MdxViewer/ViewerApp.cs` now reads shared `MdxSummaryReader` plus `MdxGeometryReader` results during the `MDLX` container route in `LoadModelFromBytesWithContainerProbe(...)`
   - `LoadMdxModel(...)` now prefers shared version or model-name or geoset-count or valid-geoset-count or total-vertex-count or total-triangle-count or pivot-count values when shared reads succeed, while still falling back to the legacy `MdxFile` object when they do not
   - the standalone model-info panel now also surfaces shared `CLID` collision counts when present
   - the actual runtime renderer still uses `MdxFile.Load(...)`; this slice only moves runtime metadata ownership, not render-buffer assembly or animation or material evaluation
- Current verified validation:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 28, 2026 with existing warnings
   - `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --verbose i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` reached the real disk-load path on Mar 28, 2026 and printed `[SharedMDX] Runtime metadata consumer: summary=yes geometry=yes file=Wisp.mdx`
- Important boundary:
   - this is the first non-probe runtime `MDX` consumer cutover in `MdxViewer`, but it is still metadata-only
   - do not describe this as a renderer cutover, a world-scene cutover, or full runtime signoff

## AssetProbe Shared `GEOS` Payload Consumer Validation (Mar 28)

- The active viewer now consumes the first shared `wow-viewer` classic `GEOS` payload seam through the existing non-UI probe path instead of relying on `MdxFile.Load(...)` for probe-side geoset inspection.
- Landed pieces:
   - `src/MdxViewer/AssetProbe.cs` now runs `WowViewer.Core.IO.Mdx.MdxGeometryReader` on the probed model bytes after shared file detection succeeds
   - probe output now reports geoset counts from the shared geometry result when available
   - probe geoset lines now print shared payload-level signals such as `Triangles`, `UvSets`, `PrimaryTexCoords`, `MatrixGroups`, and `MatrixIndices` instead of only the legacy `Vertices` or `Indices` or flat `TexCoords` counts from `MdxFile.Load(...)`
   - the rest of the probe still intentionally uses the legacy model parse for material and texture-path reporting, so this is a narrow geoset consumer cutover rather than a full shared-library model cutover
- Current verified validation:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 28, 2026 with existing warnings
   - `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and now reports shared geoset payload fields on both chest geosets
   - `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "Creature/AncientOfWar/AncientofWar.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and reports all `5` geosets with shared payload counts, confirming the path scales beyond the small chest fixture
   - those same Mar 28 probe runs now also report shared `PIVT` or `CLID` signals when present, for example `SharedPIVT: count=6` on chest and both `SharedPIVT: count=72` plus `SharedCLID: vertices=8 triangles=12` on `Creature/AncientOfWar/AncientofWar.mdx`
- Important boundary:
   - this is probe-only consumer validation
   - `MdxViewer` still uses `MdxFile.Load(...)` for actual runtime model loading and rendering
   - do not describe this as a full shared-model runtime cutover or runtime viewer signoff

## AssetProbe Shared `MDX` Consumer Validation (Mar 28)

- The active viewer now consumes the first shared `wow-viewer` `MDX` model-family seam through the existing non-UI probe path instead of limiting model validation to shared file detection only.
- Landed pieces:
   - `src/MdxViewer/AssetProbe.cs` now runs `WowViewer.Core.IO.Mdx.MdxSummaryReader` on the probed model bytes after shared file detection succeeds
   - probe output now prints `SharedMDX` lines with version, model name, blend time, chunk counts, texture counts, replaceable-texture counts, material counts, material-layer counts, the first few top-level chunk ids, the first shared `TEXS` texture paths, and compact first-layer `MTLS` signals
   - the earlier shared `BLP` texture-summary output remains in place, so the same chest-model probe now surfaces shared model and texture seams together
- Current verified validation:
   - `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "MdxSummaryReaderTests|WowFileDetectorTests"` passed on Mar 27, 2026 with `11` targeted passing tests
   - `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --virtual-path world/generic/activedoodads/chest01/chest01.mdx --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and reported `version=1300`, `model=Chest01`, `textures=2`, `materials=2`, and real `MTLS` layer lines on the archive-backed asset
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 27, 2026 with existing warnings
   - `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and now reports `SharedMDX: ... textures=2 replaceableTextures=0 materials=2 materialLayers=2 ... firstTextures=... firstMaterials=tex0/blend0/alpha1.000,tex1/blend0/alpha1.000`
- Important boundary:
   - this is compile plus non-UI probe validation only
   - `MdxViewer` still uses `MdxFile.Load(...)` for the actual model parse and render path
   - the shared `MTLS` seam is summary-only; it does not replace the legacy material or animation-track parse path
   - do not describe this as runtime viewer signoff or a full shared-library model cutover

## AssetProbe Shared `BLP` Consumer Validation (Mar 27)

- The active viewer now has a narrow non-UI compatibility check for the latest shared `wow-viewer` `BLP` seam instead of leaving that validation only inside `wow-viewer` itself.
- Landed pieces:
   - `src/MdxViewer/AssetProbe.cs` now runs `WowViewer.Core.IO.Files.WowFileDetector` on the probed model bytes before loading them through `MdxFile`
   - the same probe path now also runs `WowViewer.Core.IO.Blp.BlpSummaryReader` on resolved texture bytes when the shared detector classifies them as `Blp`
   - probe output now prints shared `BLP` header-summary signals alongside the existing `SereniaBLPLib` decode-based width or alpha analysis
- Current verified validation:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 27, 2026 with existing warnings
   - `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 27, 2026 and now reports `SharedDetect kind=Mdx` for the model plus per-texture `SharedBLP` lines such as `format=BLP2 version=1 compression=Dxtc pixelFormat=Dxt1 size=128x64 mips=8 inBoundsMips=8 outOfBoundsMips=0`
- Important boundary:
   - this is consumer compile plus non-UI probe validation only
   - `MdxViewer` still uses `SereniaBLPLib` for actual bitmap decode in the probe and main render/export paths
   - do not describe this as a runtime viewer signoff or a full texture-pipeline cutover away from the legacy decode library

## wow-viewer Library Priority Reset (Mar 26)

- Future `wow-viewer` work should no longer treat `MdxViewer` as the default PM4 source of truth.
- The active viewer is now a secondary compatibility consumer and historical reference for `wow-viewer` library work.
- For new `wow-viewer` implementation slices, default validation should happen in `wow-viewer` itself; build `MdxViewer` only when the slice intentionally changes consumer compatibility or the user explicitly asks for that check.
- If a future chat is deciding between completing a library seam in `wow-viewer` and adding another viewer hookup, the default choice is to complete the library seam.

## PM4 Fresh-Chat Handoff (Mar 26)

- The active viewer should now be treated as a partial consumer of shared `wow-viewer` PM4 math, not as the only place where PM4 logic lives.
- Shared seams already consumed by `WorldScene`:
   - `ResolvePlanarTransform(...)`
   - `TryComputeWorldYawCorrectionRadians(...)`
   - `ComputeSurfaceWorldCentroid(...)`
- Shared seams already present in `Core.PM4` but not yet consumed by the active viewer call site:
   - typed coordinate-mode resolution
   - typed placement-solution contract
   - world-space yaw helper layer for pivot rotation and corrected world conversion
- Fresh validation worth carrying into the next chat:
   - `wow-viewer` PM4 test project passed on Mar 26, 2026 with `18` tests
   - placement-focused PM4 tests passed on Mar 26, 2026 with `11` tests
   - full `wow-viewer` solution tests passed on Mar 26, 2026 with `32` tests
- Important boundary for future viewer work:
   - this is still shared-library and compile validation only
   - do not describe current viewer PM4 behavior as runtime-signed-off just because these seams now compile and pass library tests
   - the clean next PM4 viewer follow-up is the coordinate-mode consumer hookup, not a broad rewrite of the full PM4 placement path

## Tool Cutover Planning For wow-viewer (Mar 25)

- Added `plans/wow_viewer_tool_inventory_and_cutover_plan_2026-03-25.md` to make the viewer-tool cutover explicit instead of leaving it as implied bootstrap guidance.
- Viewer-side implications captured there:
   - `MdxViewer` remains the winning interactive surface and should become `WowViewer.App`; older WoWRollback GUI or viewer shells should not be ported as parallel apps.
   - current viewer panels should mostly survive as panels or workflows, but the modal converter utilities should be rebuilt as thin clients over shared converter services rather than keeping viewer-owned business logic.
   - PM4 correction: current `MdxViewer` PM4 behavior is now explicitly treated as the runtime reference implementation for the future repo.
   - `Pm4Research` should be ported into the new repo as `Core.PM4`, while the viewer PM4 workspace becomes a consumer of that library rather than continuing to own PM4 behavior internally.
   - the surviving core viewer panels are the shell itself, Navigator, Inspector, Minimap, runtime inspection tools, and the PM4 workspace.
   - diagnostics windows should likely merge into cleaner docked surfaces instead of surviving as many one-off windows.
- Migration order now favored for the new repo:
   1. shared `Core.IO` plus `Core.PM4` and `Tool.Converter`
   2. the new viewer shell and core panels
   3. `Tool.Inspect` plus PM4 inspect verbs
   4. deeper PM4 consolidation and research promotion work
- Additional planning docs now exist for this follow-up:
   - `plans/wow_viewer_bootstrap_layout_plan_2026-03-25.md`
   - `plans/wow_viewer_cli_gui_surface_plan_2026-03-25.md`
   - `plans/wow_viewer_pm4_library_plan_2026-03-25.md`
- Validation status:
   - planning and documentation only
   - no active viewer code changed in this slice

## wow-viewer Skeleton Follow-Up (Mar 25)

- A first-pass `wow-viewer/` scaffold now exists at the workspace root.
- Viewer-relevant consequence:
   - `WowViewer.App` and `WowViewer.Core.PM4` are now real project identities, not only planning names.
   - the placeholder `Core.PM4` code already encodes the planning rule that current `MdxViewer` behavior is the PM4 runtime reference and `Pm4Research` is the library seed.
- Validation status:
   - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
   - no active viewer code was ported yet; this is scaffold-only validation

## PM4 Library Follow-Up (Mar 25)

- The first PM4 code-port slice now exists in `wow-viewer/src/core/WowViewer.Core.PM4`.
- Current rule remains unchanged:
   - `MdxViewer` is still the runtime reference implementation for PM4 reconstruction behavior.
   - the new `Core.PM4` port is currently a raw research-facing reader layer seeded from `Pm4Research.Core`.
- Ported pieces so far:
   - typed chunk models
   - research document container
   - binary reader
   - exploration snapshot builder
- Validation status:
   - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after this PM4 slice
   - no active viewer PM4 code has been moved yet

## PM4 Inspect Follow-Up (Mar 25)

- The new `wow-viewer` repo now has working single-file PM4 inspect verbs on top of `Core.PM4`.
- This is useful for the viewer migration because the new repo can now inspect PM4 through shared library code without depending on `Pm4Research.Cli` directly.
- Important boundary remains unchanged:
   - current `MdxViewer` runtime reconstruction behavior is still the reference implementation for viewer-facing PM4 semantics.
   - the new inspect verbs are research analysis, not a replacement for current viewer reconstruction logic.

## PM4 Audit And Placement-Contract Follow-Up (Mar 25)

- The new `wow-viewer` repo now also has a first decode-audit path and a first extracted viewer-facing PM4 placement-contract seam.
- Viewer-relevant consequence:
   - `Core.PM4` now exposes the runtime PM4 contract types already used implicitly in `WorldScene`: `Pm4AxisConvention`, `Pm4CoordinateMode`, `Pm4PlanarTransform`, and the current planar candidate set via `Pm4PlacementContract`.
   - this is still only the first seam extraction; the active solver and reconstruction behavior still live in `WorldScene`.
   - the inspect/report layer now preserves the current research note that CK24 low-16 object values may be plausible `UniqueID` candidates, but that remains unverified until correlated against real placed-object data.
   - new audit commands already surfaced real `MDOS.buildingIndex->MDBH` invalid references and `MSLK.RefIndex->MSUR` mismatches in the development corpus, which is more evidence that viewer-facing linkage semantics should stay explicitly research-labeled until the solver port is backed by correlation data.

## wow-viewer PM4 Test Follow-Up (Mar 25)

- The new repo now has first-pass PM4 regression coverage in `tests/WowViewer.Core.PM4.Tests`.
- Viewer-relevant consequence:
   - the current `development_00_00.pm4` reader counts, analysis summary, audit findings, and corpus-audit shape are now locked by executable tests instead of only markdown notes.
   - this is still not a viewer-runtime PM4 rendering test, but it gives the migration a real-data regression floor before deeper solver extraction.

## PM4 Linkage And Placement-Math Follow-Up (Mar 25)

- The new repo now also has a first linkage report family and a first actual `WorldScene` placement-helper extraction.
- Viewer-relevant consequence:
   - `Core.PM4` now owns the current range-based axis selection fallback, tile-local heuristic, and PM4-vertex-to-world conversion helper as a reusable service layer instead of leaving that logic entirely marooned inside `WorldScene`.
   - the new linkage report gives the migration a real corpus-level view of low16 CK24 object-id reuse versus mismatch families, which is directly relevant to the current `UniqueID` hypothesis work.
   - current fixed-corpus result: low16 values can still align with expected ranges, but the corpus does not support treating that range alignment alone as confirmation of globally unique object identity.

## PM4 MSCN Follow-Up (Mar 25)

- The new repo now also has a first MSCN relationship report family.
- Viewer-relevant consequence:
   - the migration now has a real corpus-level read on how `MSUR.MdosIndex` and `MSCN` interact, rather than leaving MSCN as a vague side theory.
   - current fixed-corpus result strongly favors raw MSCN bounds overlap over simple XY-swapped overlap, which means the earlier swapped-XY explanation should not be treated as the dominant answer for this corpus anymore.
   - this still does not make MSCN authoritative for final viewer reconstruction, but it gives the next PM4 research ports a much better factual baseline.

## PM4 Unknowns + Normal-Axis Follow-Up (Mar 26)

- The new `wow-viewer` repo now also has the first unknowns-report family plus the next extracted PM4 solver seam from `WorldScene`.
- Viewer-relevant consequence:
   - `Core.PM4` now exposes normal-based axis scoring and detection helpers on top of the earlier range-based fallback, so more of the existing axis-selection logic is reusable outside `WorldScene`.
   - `pm4 unknowns --input <directory>` now gives the migration a single corpus-level surface for the still-open `MSLK`, `MPRL`, `MPRR`, and header-field questions instead of scattering that evidence across ad hoc notes.
   - current fixed-corpus result: all observed `MSLK.LinkId` values fit the sentinel-tile pattern in the development corpus, but `MSLK.RefIndex -> MSUR` still has `4,553` misses and `MPRR.Value1` remains mixed-domain.
- Important boundary:
   - this still does not close the final solver or coordinate-ownership semantics for viewer reconstruction.
   - active `WorldScene` reconstruction behavior remains the runtime reference implementation until the remaining solver slices are extracted and validated against real placement data.

## PM4 Planar-Transform Resolver Follow-Up (Mar 26)

- The new `wow-viewer` repo now also has the next extracted PM4 solver seam above axis selection: planar-transform resolution against MPRL anchors.
- Viewer-relevant consequence:
   - `Core.PM4` now owns the candidate planar-basis scoring loop instead of only the low-level axis and conversion helpers.
   - the shared solver can now compare planar candidates using centroid proximity, multi-anchor footprint fit, and packed MPRL heading evidence with quarter-turn fallback.
   - current fixed development-tile regression locks a whole-tile tile-local result of `(swap=false, invertU=false, invertV=false)` for this slice, while a synthetic world-space case locks a non-default quarter-turn candidate.
- Important boundary:
   - this is still not the full object-level PM4 placement pipeline and does not yet move world-yaw correction or final renderer alignment out of `WorldScene`.

## PM4 World-Yaw Correction + Shared Solver Consumer Follow-Up (Mar 26)

- The active viewer now consumes the shared `wow-viewer` PM4 solver for the first narrow PM4 object-placement slice.
- Viewer-relevant consequence:
   - `WorldScene.ResolvePlanarTransform(...)` and `WorldScene.TryComputeWorldYawCorrectionRadians(...)` now delegate into `WowViewer.Core.PM4.Services.Pm4PlacementMath` through explicit local-to-shared type adapters.
   - this keeps the current viewer call sites stable while starting the real migration of solver ownership out of `WorldScene`.
   - the shared library now also owns signed world-yaw correction fallback against MPRL heading evidence.
- Important boundary:
   - this is build-validated integration only so far.
   - no runtime real-data signoff has happened yet on viewer-visible PM4 behavior after the hookup.

## PM4 World-Space Centroid + Shared Solver Consumer Follow-Up (Mar 26)

- The active viewer now consumes one more shared `wow-viewer` PM4 solver seam above world-yaw correction.
- Viewer-relevant consequence:
   - `Core.PM4` now owns `Pm4PlacementMath.ComputeSurfaceWorldCentroid(...)`, so the world-space pivot derived from PM4 surface geometry is no longer viewer-owned math.
   - `WorldScene.ComputeSurfaceWorldCentroid(...)` now delegates to shared `Core.PM4` through the same explicit local-to-shared type adapters already used for planar-transform resolution and world-yaw correction.
   - the added synthetic regression locks the real tile-size mapping for a tile-local centroid case, so the pivot helper is no longer only implied by the surrounding solver coverage.
- Important boundary:
   - this is still only the shared world-space centroid seam.
   - renderer-space centroid handling and the broader PM4 object placement or render path still remain in `WorldScene`.
   - validation is still build plus library tests only; no real-data viewer runtime signoff yet after this additional hookup.

## Post-v0.4.5 Viewer Roadmap Split (Mar 25)

- Viewer follow-up planning is now intentionally isolated on branch `feature/v0.4.6-v0.5.0-roadmap` instead of piling the next milestone discussion directly onto `main`.
   - detailed Copilot prompt assets for the `wow-viewer` tool-suite/library refactor now live under workspace `.github/prompts/`, not under `gillijimproject_refactor/plans`.
   - use `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md` to route future planning chats to the right detailed prompt.
   - latest user constraint: the future repo should fully re-own first-party read/parse/write/convert logic, including current base libraries like `gillijimproject-csharp`, instead of carrying them forward as a permanent layered mess.
   - upstream externals such as `Warcraft.NET`, `DBCD`, `WoWDBDefs`, `Alpha-Core`, `WoWTools.Minimaps`, and `SereniaBLPLib` should stay under a `libs/` policy and track their original repos where practical.
   - repo bootstrap should automatically pull support material such as `wow-listfile` instead of relying on manual setup.
   - possible targeted integrations worth evaluating later include `MapUpconverter`, `ADTMeta`, `wow.export`, and `wow.tools.local`, but they should support the owned-library plan rather than replace it.
   - possible upstream alpha-era support work for `Noggit` / `noggit-red` is interesting but should remain stretch work, not the main viewer migration target.
- New viewer-relevant planning prompts now exist under `gillijimproject_refactor/plans/`:
   - `v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md`
   - `wowrollback_uniqueid_timeline_prompt_2026-03-25.md`
   - `alpha_core_sql_scene_liveness_prompt_2026-03-25.md`
   - `viewer_performance_recovery_prompt_2026-03-25.md`
   - `v0_5_0_new_repo_library_migration_prompt_2026-03-25.md`
   - `v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`
   - updated `enhanced_terrain_shader_lighting_prompt_2026-03-25.md`
- Current intended viewer milestone split:
   - `v0.4.6` should focus on first real WoWRollback / `UniqueID` timeline filtering inside the current viewer, Alpha-Core SQL caching/fidelity follow-up, and an initial performance recovery slice.
   - `v0.5.0` should move into `https://github.com/akspa0/wow-viewer` as the new production repo, with one canonical shared library and separate viewer/tool consumers.
- Important boundaries:
   - use the current viewer UI and world-loading methodology; do not treat the older WoWRollback viewer concepts as the primary product target.
   - treat `parp-tools` as the R&D repo and `wow-viewer` as the intended production home for the next major milestone.
   - a concrete first-pass repo tree and migration order now exists in `plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`; future sessions should refine that document rather than re-arguing the basic repo shape.
   - SQL actor equipment correctness, animation-state handling, and pathing are separate seams.
   - pathing/server-like NPC motion is still speculative until real data sources are verified and the current frame-time problem is improved.
- Validation status:
   - planning/documentation only
   - no active viewer code changed in this slice

## Object Culling + Far-Fog Follow-Up (Mar 25)

- The active viewer object-visibility path was adjusted to reduce aggressive pop-in/pop-out for world MDX/M2/WMO placements:
   - `WorldScene` no longer decides near-camera frustum grace and WMO distance visibility from object-center distance alone; it now uses point-to-AABB distance, which keeps large objects visible when the camera is close to their volume but not their center.
   - the near-camera frustum-cull exemption radius was increased and is now scaled by object bounds, which reduces turn-in-place pop-in for nearby WMOs and doodads.
   - world WMO cull distance is no longer a fixed short range; it now expands relative to fog end so objects do not disappear well before the visible world horizon.
   - object renderers now receive a later fog-start distance than terrain so distant objects are not washed into fog color too early while still remaining rendered.
   - `Rendering/WmoRenderer`'s separate internal doodad cull for WMO-contained doodads was loosened as well, including a higher render cap, so interior/attached doodads do not disappear far earlier than the parent WMO.
- Important boundary:
   - this is a viewer-side visibility tuning pass, not proof that the active culling/fog behavior now matches any historical client.
   - terrain fog and object fog are now intentionally less tightly matched than before in order to reduce the object-specific washout the user reported.
- Validation status:
   - build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-object-culling-fog/"` passed after this slice.
   - no automated tests were added or run.
   - no real-data runtime signoff yet on close-range object stability, long-range WMO retention, or fog feel on the fixed development dataset.

## Taxi Override Workflow + Persistence Follow-Up (Mar 25)

- The active viewer taxi prototype now has a usable asset-selection workflow on top of the earlier route-picking work:
   - `ViewerApp_Sidebars` file browser actions now expose `Open Selected`, `Copy Path`, `Use For Taxi Override`, and `Return To Last World` so a browser-selected standalone model can drive taxi actor overrides without forcing manual path retyping.
   - taxi inspector controls now expose `Use Selected Browser Asset`, `Copy Override Path`, `Open Override Asset`, and `Return To Last World` alongside the existing override input.
   - `ViewerApp` now captures the currently loaded world path/camera before opening standalone models and can restore that world session through `ReturnToLastWorldScene()`.
   - taxi actor overrides are now persisted in viewer settings by map name plus route ID and reapplied on world load for the current map.
- Important boundary:
   - this is a workflow/persistence improvement on top of the existing taxi-route actor prototype, not proof that override persistence or world-return behavior is runtime-correct on real data.
   - editor diagnostics were not reliable during this slice; solution build output caught the actual syntax break, and only the final solution build should be treated as validation.
- Validation status:
   - build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-taxi-workflow/"` passed after the follow-up fix.
   - no automated tests were added or run.
   - no runtime real-data signoff yet on the new file-browser-to-override flow, return-to-world restoration, or persisted override replay.

## Fullscreen Minimap Tile-Scale Assumption Reverted (Mar 25)

- The Mar 24 `TileSize` minimap hypothesis was wrong for the active viewer's current world-tile conventions and made the minimap behavior worse instead of better.
   - `ViewerApp_MinimapAndStatus` now maps camera position, pan clamping, and minimap teleport targets back onto `WoWConstants.ChunkSize`, which is the world-tile spacing the active viewer currently uses for the `64x64` minimap grid.
   - `MinimapHelpers` now projects POIs and taxi overlays with that same spacing again so overlay markers share the same coordinate system as the base minimap tiles.
   - the legacy `DrawMinimap_OLD()` path in `ViewerApp.cs` was restored to the same scale so fallback code does not preserve the bad `TileSize` assumption.
- Root cause:
   - the active viewer's naming is misleading here: `WoWConstants.TileSize` is not the minimap's live `64x64` world-tile spacing, so swapping the minimap math from `ChunkSize` to `TileSize` desynchronized camera marker placement, pan bounds, overlay projection, and click-to-teleport.
- Validation status:
   - build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-regression-repair/"` passed after this slice.
   - no automated tests were added or run.
   - no real-data runtime signoff yet on docked/fullscreen minimap placement, panning feel, or teleport correctness.

## Fullscreen Minimap Repair Closed For v0.4.5 (Mar 25)

- The fullscreen/docked minimap repair is now treated as closed for `v0.4.5` after the final transpose-only follow-up and runtime user confirmation on the fixed development minimap dataset.
- Final landed behavior:
   - the bad `TileSize` minimap hypothesis stays reverted; the active minimap grid still uses `WoWConstants.ChunkSize`
   - the later broad world-axis swap also stays reverted
   - the landed fix is the narrower transpose-only repair: direct world/click mapping remains intact while the camera-marker screen placement is aligned with the drawn tile grid
   - the legacy `DrawMinimap_OLD()` path now matches that same final behavior
- Practical release consequence:
   - the fullscreen minimap is no longer an open `v0.4.5` blocker
   - if the minimap regresses again, inspect tile lookup/order and fullscreen interaction parity before re-swapping world axes
- Validation status:
   - build plus targeted runtime user signoff: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-transpose-repair/"` passed after the final patch
   - runtime user feedback then confirmed the repaired Designer Island/top-right minimap behavior on the fixed development dataset
   - no automated tests were added or run

## Taxi Route Actor Prototype + Node Inspector Controls (Mar 25)

- The active viewer now has a first taxi-route actor prototype wired into the live scene:
   - `TaxiPathLoader` resolves taxi mount metadata from the historical node-driven DBC chain instead of hardcoding route birds.
   - `WorldScene` can animate a taxi actor along selected taxi routes using the resolved mount model when a usable node mount path exists.
   - `ViewerApp` now supports viewport taxi-node picking, and taxi selection feeds the same selected-object inspector flow used by other scene objects.
   - `ViewerApp_Sidebars` exposes taxi controls in the inspector, including the requested taxi speed slider and a show/hide toggle for the animated taxi actor.
- Important boundary:
   - this is a viewer-side prototype for inspection and iteration, not proof of client-faithful taxi runtime behavior.
   - current mount selection still resolves from node metadata and endpoint fallback, so any route-specific client nuances beyond that remain open until proven with live data.
   - viewport taxi selection currently uses screen-space node-indicator picking, not scene-depth-tested geometry picking.
- Validation status:
   - build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-taxi/"` passed after this slice.
   - a normal build to the default output path was blocked by the running `ParpToolsWoWViewer` process locking output DLLs.
   - no automated tests were added or run.
   - no real-data runtime signoff yet on taxi actor motion, model correctness, or click-selection ergonomics.

## WMO Vertex-Light Prototype (Mar 24)

- The active viewer now has a first renderer-side object-lighting prototype in `Rendering/WmoRenderer.cs`:
   - WMO group vertex buffers now carry baked vertex-light colors.
   - parsed `MOCV` is used when available and usable.
   - if parsed `MOCV` is absent but preserved v14 lightmap payloads exist, the renderer samples `MOLV` / `MOLD` / `MOLM` into per-vertex baked-light colors during buffer build.
   - the fragment shader now modulates textured WMO lighting by that baked-light color instead of relying only on the generic ambient+directional path.
- Important boundary:
   - this is not a full analogue of the client's `RenderGroupLightmap` / `RenderGroupLightmapTex` path.
   - there is still no recovered batch-local lightmap texture binding path or true object-lightmap render split in the active viewer.
   - treat this as a first prototype that uses already-preserved object light data, not as completed lightmap parity.
- Validation status:
   - build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed after this change.
   - no automated tests were added or run.
   - no runtime real-data signoff yet.

## WoW 0.5.3 Render Fast-Path And Viewer Perf Gap (Mar 24)

- Current viewer/performance work should treat the following as engine-backed guardrails from the symbolized `0.5.3` client:
- durable write-up: `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
   - `CreateRenderLists` (`0x00698230`) is a real terrain precompute step that builds texcoord and batch/render-list tables up front instead of leaving chunk draw setup entirely to the frame loop
   - `RenderLayers` (`0x006a5d00`) and `RenderLayersDyn` (`0x006a64b0`) use locked GX buffers plus prepared chunk batches, and they reduce terrain layer count by distance rather than always drawing the full local layer stack
   - terrain in `0.5.3` already has shader-assisted specialization: the draw path binds `CMap::psTerrain` / `CMap::psSpecTerrain` plus `shaderGxTexture` when terrain/specular shader support is available
   - moving terrain-layer behavior is now directly supported in the terrain draw path itself: runtime layer flag `0x40` triggers an extra texture transform indexed into time-varying world transform tables, so the terrain motion seam is not just `WCHUNKLIQUID`
   - terrain shadows are drawn as a separate modulation pass instead of being flattened into one generic terrain blend loop
   - object lighting/rendering is also more specialized than the active viewer:
      - `RenderMapObjDefGroups` (`0x0066e030`) walks visible `CMapObjDefGroup` lists and dispatches `CMapObj::RenderGroup(...)` at group scope
      - `CreateLightmaps` (`0x006adba0`) allocates per-group lightmap textures and registers `UpdateLightmapTex`
      - `RenderGroupLightmap(...)` and `RenderGroupLightmapTex(...)` show a dedicated group-lightmap render/combine path, not just one generic WMO material path with extra texture sampling
      - `UpdateLightmapTex(...)` exposes CPU lightmap memory and stride on `GxTex_Latch`, which supports a persistent object-lightmap texture workflow
      - `CalcLightColors` (`0x006c4da0`) computes much richer world-light state than the active viewer currently models (direct, ambient, multiple sky/cloud/water channels, fog, storm blending)
- Practical implication for viewer fixes:
   - the active viewer is still structurally flatter than the client in the exact places that matter for both speed and fidelity:
      - `StandardTerrainAdapter` still actively uses `MPHD` only for big-alpha/profile handling and still flattens `MAIN` entries to boolean tile existence
      - `TerrainRenderer` is still a generic base+overlay pass loop that only interprets `MCLY 0x100`; it has no terrain shader-family split, no per-layer motion support, no layer-count LOD collapse, and no specular terrain path
      - `LightService` remains a simplified nearest-zone DBC interpolator instead of a full terrain/object/sky/runtime-light system
      - `WmoRenderer` / `MdxRenderer` still rely on shared generic shader families instead of the client's stronger specialization
      - `WorldScene` hot paths remain heavy, and PM4 forensic budgets are still effectively uncapped when that overlay is enabled
      - `RenderQueue.cs` exists, but it is not yet the active world-scene submission path, so current batching/sorting/state reuse is still mostly renderer-local
- Priority order now supported by evidence:
   1. preserve `MAIN` / `MPHD` / `MCLY` semantics as first-class runtime metadata
   2. split terrain renderer responsibilities into fallback vs client-faithful material/shader path
   3. treat object/lightmap parity as a separate seam from terrain lighting
   4. reduce generic hot-path state churn before layering on more fidelity features
   5. use `WorldAssetManager`'s existing read/path-probe counters to drive an explicit scene residency/prefetch policy
- Validation status:
   - reverse engineering plus code audit only; no viewer code changed and no runtime signoff was produced by this slice

## WoW 2.0.0 Beta Ghidra Recon: 2.x Runtime Risk Map (Mar 24)

- Current `2.x` viewer work should treat the following as engine-backed guardrails from a static Ghidra pass against a beta `2.0.0` client:
- durable write-up: `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
   - `Model2` is not using a generic shared early-model shader path in the client; `FUN_00717b00` explicitly loads `shaders\vertex\Model2.bls` and `shaders\pixel\Model2.bls`.
   - map-object rendering in `FUN_006b3b20` preloads multiple material-specific pixel BLS programs, including `MapObjTransDiffuse.bls` and `MapObjTransSpecular.bls`.
   - `M2Light` handling is spatial/runtime-managed: `FUN_0072d1a0` inserts lights into bucketed structures or general linked lists, and related mutators relink on state/position changes.
   - `ParticleSystem2` is a real engine runtime with pool/bootstrap logic (`FUN_007c26c0`) and runtime `CParticle2` / `CParticle2_Model` object storage/copy paths (`FUN_007ca9d0`, `FUN_007c3180`, `FUN_007c79d0`).
   - the `Light*.dbc` family is loaded through strict schema-checked `WDBC` loaders, so raw light-table ingestion is probably not the risky seam for viewer parity.
   - terrain follow-up is now more precise than the earlier `terrainp* == fast path` shorthand:
      - `terrain1..4` and `terrain1_s..4_s` are the cached one-pass terrain programs chosen by `FUN_006cee30` by chunk layer count
      - `terrainp` / `terrainp_s` belong to the slower manual terrain fallback path
      - `terrainp_u` / `terrainp_us` are loaded in `FUN_006a2360` but still not tied to an active draw branch in this beta pass
      - both terrain draw branches also contain a separate time-varying layer-transform seam: `FUN_006c00f0` copies a source layer flag field into each runtime layer object, `FUN_006cee30` / `FUN_006cf590` apply an extra transform when bit `0x40` is present, and `FUN_006804b0` updates the transform tables every world tick
      - animated `WCHUNKLIQUID` remains a separate engine track from the terrain shader selector
      - `WCHUNKLIQUID` mode is now tied directly to animated family index in `FUN_0069b310`; recovered family entries are `lake_a`, `ocean_h`, `lava`, `slime`, and a duplicate `lake_a`, with higher slots still unresolved
      - interesting likely-dead or unfinished content remains in view: unresolved family slot `6`, unused `XTextures\river\fast_a.%d.blp`, and the terrain-side `_u/_us` shader variants
      - viewer-side terrain flag handling is still much thinner than the known format surface:
         - `StandardTerrainAdapter` only actively uses `MPHD` for big-alpha selection
         - `ReadMainChunk(...)` collapses every non-zero `MAIN` entry into generic tile existence and does not keep per-entry semantics like `has ADT` vs `all water`
         - `TerrainRenderer` only interprets `MCLY 0x100` (`use_alpha_map`) and ignores the rest of the preserved layer flag bits
- Practical implication for future viewer fixes:
   - preserve the current profile-routing split, but do not assume smoke/light parity will come from parser-only changes.
      - if later `2.x` visuals are wrong, inspect material/BLS selection and runtime light/particle interpretation before widening format-profile heuristics further.
      - for moving/sliding terrain visuals specifically, treat `WDT` global flags, terrain-layer runtime flags, and `WCHUNKLIQUID` mode dispatch as separate investigative seams until the file-to-runtime mapping is proven.
- Validation status:
   - reverse engineering only; no viewer code changed and no runtime signoff was produced by this slice.

## Later 2.x M2-Family Routing Follow-Up (Mar 24)

- The active viewer no longer hard-rejects later `2.x` / TBC-era `MD20` models solely because there was no resolved model profile for that build family.
- Current active-tree changes:
   - `Terrain/FormatProfileRegistry.cs` now exposes `M2Profile_20x_Unknown` for `2.x` builds.
   - the routed version window is `0x104..0x107`, which matches later TBC-era `MD20` versions and keeps the structural split threshold at `0x108`.
   - `ViewerApp.cs` fallback build options now include `2.4.3.8606` so the profile can be selected even when `Map.dbd` build metadata is unavailable.
   - `Rendering/ReplaceableTextureResolver.cs` now recognizes the short alias `2.4.3 -> 2.4.3.8606`.
- Important limit:
   - this is profile-routing enablement, not full runtime signoff for all `2.x` assets.
   - no fixed TBC real-data validation path exists in the current workspace notes, so the slice is build-validated only.
- Validation status:
   - file diagnostics were clean on the edited viewer files.
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed after this change.
   - no automated tests were added or run.

## v0.4.5 Branding + MH2O LiquidType Fix (Mar 24)

- Viewer branding now points at `parp-tools WoW Viewer` in the active runtime shell:
   - window title uses the new product name
   - Help -> About is now a real modal with version, author, and credits
   - executable/assembly metadata now emits `ParpToolsWoWViewer`
- Release prep follow-up in the active tree:
   - `src/MdxViewer/MdxViewer.csproj` and `src/MdxViewer/MdxViewer.CrossPlatform.csproj` now carry version `0.4.5`
   - `.github/workflows/release-mdxviewer.yml` now uses the .NET 10 SDK, the new product naming, and the `parp-tools-wow-viewer-<version>-win-x64.zip` artifact name
- Terrain/liquid correction in the active tree:
   - `StandardTerrainAdapter` now resolves `MH2O` liquid family from `LiquidType.dbc.Type` when the active client build has DBC metadata loaded
   - fallback behavior still exists for cases where DBC lookup is unavailable, but it now recognizes the later 3.3.5 / 4.0 IDs used elsewhere in the repo (`13`, `14`, `17`, `19`, `20`)
- Validation status:
   - build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed after this slice
   - no automated tests were added or run
   - no runtime real-data signoff yet on the actual corrected liquid visuals for 3.3.5 / 4.0 maps

## Current Focus: Recovery On v0.4.0 Baseline (Mar 17, 2026)

MdxViewer work has been reset to a v0.4.0-based branch in the main workspace tree.

- Branch: recovery/v0.4.0-surgical-main-tree
- Base commit: 343dadf (tag v0.4.0)
- .github instructions/skills/prompts restored from main and committed (845748b)

### Tool Dialog Path Seeding Follow-Up (Mar 23)

- Viewer tools should reuse the already loaded session roots instead of forcing users to browse back to them repeatedly.
- Current viewer-side behavior:
   - `Generate VLM Dataset` seeds from the active `MpqDataSource.GamePath` plus the current loaded map name.
   - `Terrain Texture Transfer` seeds source/target map directories from the attached loose overlay root and active base client root when those directories exist for the current map.
   - `Map Converter` seeds current-map WDT/map-directory inputs from the loaded session roots when an on-disk path is available.
   - `WMO Converter` continues to seed from the currently loaded standalone WMO file.
- Scope limit:
   - this change reduces UI friction only; it does not prove the downstream converters are correct.
   - after this follow-up, file diagnostics were clean on `src/MdxViewer/ViewerApp.cs`, but no new full build/runtime signoff was recorded yet for the exact slice.

### Unified Terrain/Model/WMO I/O Overhaul Proposal (Mar 23)

- User direction is to stop splitting read/write knowledge across viewer/runtime code and `WoWMapConverter.Core`.
- Desired long-term state:
   - one shared library for Alpha/LK/4.x terrain, WDT/ADT, M2/MDX, and WMO read/write/conversion contracts
   - viewer, converter, dataset exporters, and future tools all call that same contract instead of carrying divergent logic
   - Alpha placement downconversion remains explicitly open until MODF/MDDF writer support is ported and validated
- Planning prompt for the larger effort lives at `plans/unified_format_io_overhaul_prompt_2026-03-23.md`.
- New PM4-specific planning guardrail after the Mar 24 viewer follow-up:
   - the current useful selected-object hierarchy in the viewer is `CK24 -> MSLK-linked subgroup -> optional MDOS subgroup -> connectivity part`
   - `MSUR.AttributeMask` and `GroupKey` should remain exposed as inspectable subgroup labels even while their semantics stay open
   - PM4 centroids are still derived anchors for display/debugging, not proven raw PM4 hierarchy nodes

### PM4 Legend + Selected-Object Graph Follow-Up (Mar 24)

- `WorldScene` now exposes two viewer-side PM4 inspection helpers built from the current overlay state:
   - a categorical color legend for the active PM4 color mode so `MSUR Attr Mask` colors can be identified by explicit values/counts
   - a selected-object graph summary that reflects the current overlay assembly chain: `CK24 -> MSLK-linked group -> MDOS bucket -> part`
- `ViewerApp` / `ViewerApp_Sidebars` now surface those helpers directly in the UI:
   - `World Objects` shows `PM4 Color Legend`
   - selected PM4 objects in the inspector show `PM4 Graph`
- Mar 24 interaction follow-up on the same UI:
   - PM4 graph leaf rows now reselect the exact PM4 part they describe instead of being read-only text
   - the graph panel also supports JSON export of the selected PM4 structural view for later research/planning capture
- Important boundary:
   - the graph is a viewer-derived structural explanation of the current overlay assembly, not proof that PM4 stores matching explicit graph nodes
   - the legend identifies value buckets only; it does not close the semantics of `AttributeMask` or `GroupKey`

### PM4 Overlay Load Contract Change: Full-Map Overlay Restore (Mar 22)

### MCLQ / MDX Transparency Ordering Follow-Up (Mar 23)

- World-scene render ordering now treats terrain liquid as an in-between pass instead of the final pass:
   - opaque terrain / WMO / MDX still establish the depth buffer first
   - terrain liquid now renders before transparent MDX layers
   - batched MDX transparent draws explicitly re-run `BeginBatch(...)` after the liquid pass because liquid rendering changes the active GL program/state
- `ModelRenderer` now honors material `PriorityPlane` for transparent geoset ordering within a model, using the documented lowest-to-highest order instead of raw geoset insertion order.
- `WmoRenderer` no longer renders doodad MDX in a single `RenderPass.Both` block before liquids:
   - doodad opaque layers render before WMO liquids
   - doodad transparent layers render after WMO liquids
- Follow-up regression fix on the same slice:
   - splitting doodad/model rendering into opaque + transparent passes exposed an old `ModelRenderer` fallback seam where transparent-only geosets would draw magenta fallback geometry during the opaque pass.
   - `ModelRenderer.RenderGeosets(...)` now suppresses that fallback only when the current pass skipped every layer because of pass filtering, while still keeping fallback behavior for real in-pass material failures.
- Important boundary:
   - this is a render-order/material-order correction only; it is not a full global transparent-surface sort across terrain liquids, WMO transparent batches, and all doodad layers.
   - runtime real-data validation is still required on reflective / translucent models before claiming full material parity.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the Mar 23 render-order + transparent-only fallback follow-up
   - no automated tests were added or run

### Render Quality Controls Slice (Mar 23)

- `ViewerApp` now exposes a persistent `Render Quality` window from the `View` menu.
- Current landed scope:
   - persistent texture filtering mode (`Nearest`, `Bilinear`, `Trilinear`)
   - runtime multisample toggle only when the current GL window actually provides multisample buffers
   - live sampler-state refresh for already loaded standalone/world renderers instead of applying only to future asset loads
- Active renderer coverage for the live sampler refresh:
   - `ModelRenderer`
   - `WmoRenderer` including cached doodad renderers
   - `TerrainRenderer`
   - world-scene asset caches through `WorldAssetManager`
- Important boundary:
   - this is sampler-quality control, not a full post-processing stack
   - GIF/WebM capture, `.LIT` decode work, and guaranteed object AA via an explicitly multisampled swapchain are still separate follow-up seams
   - the current branch direction does not require explicit MSAA follow-up right now; filtering is already considered the worthwhile practical improvement when the GL context lacks sample buffers
- Documentation follow-up on the same date:
   - the user rewrote `src/MdxViewer/README.md` to be more grounded/truthful after the initial doc pass overstated or guessed at some support/platform details
   - treat the current viewer README as the authoritative published summary for support claims unless newer runtime evidence contradicts it
   - important current README framing to preserve in future edits:
      - support headline: `0.5.3` through `4.0.0.11927`
      - later `4.0.x` ADT support exists
      - later split-ADT support through `4.3.4` exists but remains explicitly untested
      - do not reintroduce Windows-x64-only wording for repo/build claims
      - do not add branch-specific language to published README text
      - asset-catalog screenshot automation exists already, but UI/menu showcase capture is still only a follow-up idea

- PM4 overlay loading in `src/MdxViewer/Terrain/WorldScene.cs` now restores the map-wide PM4 candidate set instead of filtering to the active camera window.
- Current behavior:
   - the loader still computes PM4 camera-window/radius metrics for diagnostics, but candidate selection is no longer restricted by camera position
   - all valid map PM4 files are decoded/read into the overlay candidate set
   - zero-CK24 PM4 surface families are no longer dropped outright; the viewer now seeds separate overlay objects for those type/attr buckets instead of only reconstructing non-zero CK24 groups
   - PM4 decode/cache load now runs on a background task instead of blocking the render thread when the PM4 layer is enabled or reloaded
   - completed PM4 overlay snapshots are applied back on the render thread on the next frame, so the live dictionaries are not mutated from the background worker
   - the loaded PM4 window is pinned to the full tile range `(0..63, 0..63)` so moving the camera no longer forces PM4 reload churn
   - PM4 cache entries are now effectively keyed by the full map-wide PM4 candidate set for the active map instead of a camera-window subset
   - PM4 status text now reports `map-wide` load/cache results instead of `active-window`
- Important boundary:
   - this restores visibility for PM4 outside the upper-left camera window, but it also restores the heavier map-wide load behavior
   - backgrounding should remove the hard UI freeze on PM4 enable, but no runtime real-data signoff yet exists on final load time, responsiveness during load, or memory pressure on the user's dataset
- Validation status:
   - file diagnostics on `src/MdxViewer/Terrain/WorldScene.cs` were clean after the background-load refactor
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 22, 2026 after the background-load refactor
   - no automated tests were added or run
   - no runtime real-data signoff yet; do not over-claim from build success alone

### PM4 Offline OBJ Export Utility (Mar 22)

- `src/MdxViewer/Terrain/WorldScene.cs` now exposes an offline PM4 OBJ export path that scans PM4 files directly from the active data source instead of depending on the live overlay's currently loaded subset.
- `ViewerApp_Pm4Utilities` now exposes `Export PM4 OBJ Set`, which writes:
   - per-tile OBJ
   - per-object OBJ
   - `pm4_obj_manifest.json`
- Intended use:
   - produce stable comparison artifacts for PM4/WMO/debug analysis without adding more runtime PM4 streaming complexity
- Validation status:
   - edited files were diagnostics-clean
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 22, 2026 after this export path was in the active tree
   - no runtime signoff yet on exported geometry correctness

### Minimap Interaction + Cache Follow-Up (Mar 22)

- Floating and fullscreen minimap views no longer teleport on any short click release.
- Current behavior:
   - teleport now requires triple-clicking the same tile within the confirmation window
   - drag-vs-click discrimination uses full drag-origin distance instead of only the last drag delta
   - minimap window visibility, zoom, and pan offset now persist in viewer settings
   - decoded minimap tiles are cached on disk under `output/cache/minimap/<cache-segment>` so they survive across runs
- Validation status:
   - edited files were diagnostics-clean
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 22, 2026 after this minimap follow-up was in the active tree
   - no runtime real-data signoff yet on teleport feel or cache effectiveness

### Terrain Hole Debug Toggle (Mar 22)

- Terrain hole masking is still preserved in source chunk data; the viewer now has a mesh rebuild override for inspection only.
- Current behavior:
   - `TerrainMeshBuilder.BuildChunkMesh(...)` can ignore `HoleMask` at mesh-build time without mutating the underlying `TerrainChunkData`
   - both `TerrainManager` and `VlmTerrainManager` now support a global `IgnoreTerrainHolesGlobally` override during mesh rebuilds
   - the active UI is a single layers-bar `Holes` toggle, not the earlier sidebar/per-tile controls
- Important boundary:
   - this is a viewer-side debug/inspection feature only; it does not edit ADT hole flags or terrain data on disk
- Validation status:
   - edited files were diagnostics-clean
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change
   - no runtime real-data signoff yet on the rebuild behavior while streaming

### PM4 Yaw Decode Guardrail (Mar 22)

- Latest PM4 object-rotation triage showed the active `+90°` / clockwise reinterpretation of `MPRL` low-16 rotation in `WorldScene.TryComputeExpectedMprlYawRadians(...)` was a viewer-side heuristic, not an established PM4 decode fact.
- Current behavior:
   - `MPRL` low-16 rotation is decoded as a raw packed angle only
   - circular averaging still produces the expected-yaw scoring signal
   - sign and quarter-turn ambiguity remain in the downstream yaw-basis fallback path instead of being baked into raw decode
- Important boundary:
   - this removes one hardcoded semantic assumption, but it does not prove that `MPRL.Unk04` is a closed absolute world-yaw field
   - runtime real-data validation is still required before claiming PM4 rotation closure

- PM4 overlay loading in `src/MdxViewer/Terrain/WorldScene.cs` now restores the map-wide PM4 candidate set instead of filtering to the active camera window.
- Current behavior:
   - the loader still computes PM4 camera-window/radius metrics for diagnostics, but candidate selection is no longer restricted by camera position
   - all valid map PM4 files are decoded/read into the overlay candidate set
   - the loaded PM4 window is pinned to the full tile range `(0..63, 0..63)` so moving the camera no longer forces PM4 reload churn
   - PM4 cache entries are now effectively keyed by the full map-wide PM4 candidate set for the active map instead of a camera-window subset
   - PM4 status text now reports `map-wide` load/cache results instead of `active-window`
- Important boundary:
   - this restores visibility for PM4 outside the upper-left camera window, but it also restores the heavier map-wide load behavior
   - no runtime real-data signoff yet on the resulting load time or memory pressure on the user's dataset
- Validation status:
   - file diagnostics on `src/MdxViewer/Terrain/WorldScene.cs` should be kept clean after the full-map restore
   - build/runtime validation should be reported separately; do not over-claim from code inspection alone

### Standalone PM4 Research Library (Mar 21)

- There is now a separate raw-reader path at `src/Pm4Research.Core` for fresh PM4 rediscovery work.
- Use that project when the question is about chunk structure, offsets, raw typed layouts, or whether the current PM4 decoder is making a bad assumption.
- Do not start new format-rediscovery work inside `WorldScene` unless the question is specifically about viewer reconstruction behavior.
- Preferred PM4 reference tile for that research path is `test_data/development/World/Maps/development/development_00_00.pm4`.
- The repo does not contain the matching `00_00` ADT triplet, so viewer-side signoff on that tile still depends on the user's external trusted ADT copy.

### Explicit Base-Build Selection Recovery (Mar 21)

### Archive I/O Performance Slice: Read-Path Probe Reduction + Prefetch Signal (Mar 21)

### ViewerApp Partial-Class Refactor (Mar 21)

- `ViewerApp` was split further along existing partial-class seams instead of continuing to accumulate everything in one file.
- New partials now hold the main extracted UI domains:
   - `ViewerApp_ClientDialogs.cs`
   - `ViewerApp_Pm4Utilities.cs`
   - `ViewerApp_MinimapAndStatus.cs`
   - `ViewerApp_Sidebars.cs`
- The split is intentionally low-risk:
   - no shell rewrite, no dockspace restoration, no intended behavior change.
   - the large world-objects inspector path still remains in `ViewerApp.cs` as `DrawWorldObjectsContentCore()` for now.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the extraction.
   - no automated tests or runtime validation were added for this structural pass.

### Viewer UI / Perf Slice: Hideable Chrome + Clipped Long Lists (Mar 21)

### Viewer UI Follow-Up: Dockspace Host + Dockable Navigator/Inspector (Mar 21)

- Latest user feedback after the clipped-list shell pass: the viewer still lacked real dock panels, and `World Maps` should not start collapsed.
- Current viewer behavior:
   - ImGui docking is now enabled explicitly in `ViewerApp.OnLoad(...)`.
   - `ViewerApp.DrawUI()` now hosts a real central dockspace between the top chrome and the status bar.
   - left/right shell panels can render as normal dockable titled windows (`Navigator`, `Inspector`) when dock panels are enabled.
   - `View` menu now exposes a `Dock Panels` toggle.
   - `World Maps` defaults open again on first draw.
   - scene viewport math no longer assumes fixed sidebar insets.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this follow-up.
   - no automated tests were added or run.
   - no runtime real-data signoff yet on the dock workflow or viewport interaction feel.

- Latest user feedback moved the immediate priority from PM4 transform details to the viewer shell itself: UI clutter and list-heavy panels were making PM4 debugging slower than the geometry work.
- Current change in `ViewerApp` is an incremental shell/perf pass only:
   - `Tab` toggles a hide-chrome mode for menu/toolbar/sidebars/status/floating utility windows.
   - major sidebar sections no longer all default open on first draw.
   - long panel lists now render through clipped child windows instead of walking every row each frame.
- Current clipped lists:
   - file browser
   - discovered maps
   - subobject visibility/group toggles
   - WMO placements
   - MDX placements
   - area POIs
   - taxi nodes
   - taxi routes
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this UI slice.
   - no automated tests were added or run.
   - no runtime signoff yet; do not claim final UI/perf recovery from build success alone.

- Confirmed archive/path hot seam before editing:
   - `WorldAssetManager.ReadFileData(...)` still replayed alias and fallback probes after the viewer already had `MpqDataSource` normalization, Alpha-wrapper resolution, raw-byte caching, and file-set indexes.
   - duplicate lowercase and `.mpq` retries in that method were confirmed redundant for the active MPQ data source path.
- Current code change:
   - `MpqDataSource` now exposes `MpqDataSourceStats` with exact counters for `FileExists`, `ReadFile`, read-cache behavior, read-source buckets (`loose`, `alpha wrapper`, `MPQ`, `miss`), and prefetch queue/read timing.
   - `WorldAssetManager` now exposes `WorldAssetReadStats` and caches the winning resolved read path for each requested asset key so retries do not replay the full candidate chain.
   - `WorldAssetManager.ReadFileData(...)` now dedupes candidate probes and removes the duplicate lowercase and `.mpq` retries that the MPQ data source already handled.
   - model prefetch now warms the canonical resolved root asset first and prefers the best indexed `.skin` path instead of fanning out across all alias + skin permutations by default.
   - `ViewerApp` world stats now surface these counters directly for runtime measurement.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this slice.
   - no automated tests were added or run.
   - no runtime real-data validation has been run yet; do not claim scene-streaming improvement from build success alone.

- `ViewerApp` MPQ load flow no longer treats folder-path inference as the only build-selection mechanism.
- Current viewer behavior:
   - `Open Game Folder (MPQ)...` now opens a build-selection dialog before MPQ load.
   - build options are sourced from `Terrain/BuildVersionCatalog.cs` via `WoWDBDefs/definitions/Map.dbd` when available.
   - fallback build list now explicitly includes Cataclysm-era candidates `4.0.0.11927` and `4.0.1.12304`.
   - selected build is passed directly into `LoadMpqDataSource(...)`.
- Saved base clients now preserve build identity:
   - `KnownGoodClientPath` stores `BuildVersion`
   - viewer settings also store `LastSelectedBuildVersion`
   - reopening a saved base or using `Load Loose Map Folder Against Saved Base` now reuses the saved explicit build when present
- Loose PM4 overlay attach now surfaces build-era mismatch hints:
   - first PM4 version marker found under the overlay can currently map `11927 -> 4.0.0.11927` or `12304 -> 4.0.1.12304`
   - if the overlay hint disagrees with `_dbcBuild`, viewer log/status now says so instead of silently continuing
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change
   - no runtime signoff yet with the development PM4 overlay and a matching Cataclysm-beta base client

### M2 Material Parity Slice: Explicit Env-Map + UV Selector Recovery (Mar 21)

- Current renderer-gap correction is now implementation, not planning only:
   - `WarcraftNetM2Adapter` no longer hardcodes every M2 layer to `CoordId = 0`
   - raw `.skin` batch metadata now preserves `textureCoordComboIndex` and merges it into the Warcraft.NET skin path
   - raw `MD20` vertex supplement now preserves both UV sets for M2-family assets instead of dropping to UV0 only
   - raw `textureCoordCombos` lookup now feeds `MdlTexLayer.CoordId`; `-1` marks reflective `SphereEnvMap`, `1` can select UV1
   - `ModelRenderer` debug traces now show pass + resolved material family for focused M2 batch runs
- Why this slice first:
   - the renderer already had environment-map and UV-set hooks
   - the active flattening seam was source metadata extraction, so this slice improves reflective/env-mapped appearance without broad new transparency heuristics
- Current scope limits:
   - improved family: reflective / env-mapped surfaces and explicit UV1 routing where source data requests it
   - still flattened: texture transform animation, color/transparency tracks, and broader shader-combo parity beyond existing blend/cutout/add routing
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change
   - no automated tests were added or run
   - no runtime real-data signoff yet on reflection-heavy M2 assets; do not over-claim PM4 matching benefit from this slice alone

### M2 Material Parity Follow-Up: 4.0.0.11927 Wrap + Blend Correction (Mar 21)

- Cataclysm-era M2 runtime triage found two concrete material-state mismatches after the env-map / UV recovery slice:
   - `ModelRenderer` was only treating `WrapWidth` / `WrapHeight` as M2 repeat flags for the pre-release `3.0.1` profile; Cataclysm-era M2 was still using the classic MDX clamp-flag interpretation.
   - `WarcraftNetM2Adapter.MapBlendMode(...)` was off by one after mode `2`, so M2 modes `4`..`7` were being translated into the wrong local blend families.
- Current correction:
   - all M2-adapted models now use repeat-flag semantics for wrap X/Y; classic MDX keeps the older clamp-flag behavior.
   - M2 blend ids now map as: `0=Load`, `1=Transparent`, `2=Blend`, `3=Add` (`NoAlphaAdd`), `4=Add`, `5=Modulate`, `6=Modulate2X`, `7=AddAlpha` (`BlendAdd`).
   - note: the local renderer still does not expose distinct `NoAlphaAdd` or `BlendAdd` states, so those cases intentionally collapse into the nearest additive families instead of being shifted accidentally.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this follow-up slice
   - no automated tests were added or run for this slice
   - no runtime real-data signoff yet on `4.0.0.11927` M2 assets; do not claim visual parity from code inspection alone

### 4.0.0.11927 Terrain Blend Recovery (Mar 21)

- 4.0 terrain texturing is now treated as a separate runtime-behavior track, not as a trivial extension of the validated 3.x path.
- Latest wow.exe analysis established the missing model:
   - `CMapChunk_UnpackChunkAlphaSet` builds chunk alpha with linked neighbors, not only local MCAL bytes
   - neighbor layers are matched by texture id
   - 8-bit layers with no direct payload can be synthesized as residual coverage from the other layers
   - final blend textures are rebuilt through the `TerrainBlend` runtime path
- Active viewer-side implementation now ports the first verified subset of that behavior:
   - `FormatProfileRegistry` routes unknown 4.0 ADTs to `TerrainAlphaDecodeMode.Cataclysm400`
   - `TerrainChunkData` stores per-layer `AlphaSourceFlags`
   - `StandardTerrainAdapter` runs Cataclysm400 post-processing after chunk parse:
      - residual alpha synthesis for 8-bit layers with missing direct payload
      - same-tile chunk-edge stitching by neighbor texture id
- Documentation/handoff has been expanded so future sessions start from the runtime-backed model instead of the old shorthand:
   - `documentation/wow-400-terrain-blend-wow-exe-guide.md`
   - `.github/prompts/wow-400-terrain-blend-recovery.prompt.md`
   - updated archive/spec docs for 4.0 terrain behavior
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the implementation
   - no real-data runtime signoff yet for the fixed development dataset
   - do not describe this as full 4.0 terrain parity until viewer output is checked on real data

### WMO Blend And Loose PM4 Overlay Recovery (Mar 21)

- `WmoRenderer` was rendering WMO material blend modes too coarsely for the active branch state.
- Current fix in `src/MdxViewer/Rendering/WmoRenderer.cs`:
   - map raw WMO `BlendMode` values to `EGxBlend`
   - keep `AlphaKey` batches in the opaque pass with alpha-test
   - restrict transparent rendering to `Blend` / `Add` batches only
- Loose overlay PM4 file resolution now honors overlay priority in `src/MdxViewer/DataSources/MpqDataSource.cs`:
   - newest attached loose root wins when duplicate virtual paths exist
   - this is the intended override behavior for base client + loose overlay workflows
   - PM4 loose-path misses now log detailed trace paths, not only WMO misses
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the precedence fix on top of current viewer state
   - runtime signoff is still pending for both the WMO sheen symptom and loose-overlay PM4 loading

### PM4 Picking Follow-Up: Overlay Selection No Longer Loses To WMO/MDX First-Hit Routing (Mar 21)

- PM4 objects in the viewport could be visible but effectively unclickable because `ViewerApp.PickObjectAtMouse(...)` selected WMO/MDX first and returned before PM4 selection ran.
- Current fix:
   - `WorldScene` now exposes hit-test helpers for both scene objects and PM4 objects that return nearest hit distance without mutating selection first.
   - `ViewerApp` now compares the nearest scene-object hit against the nearest PM4 hit from the same ray and selects whichever is closer.
   - this preserves normal WMO/MDX picking when they are actually in front, but allows PM4 alignment work when the PM4 object is the nearest hit.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this picking fix
   - no automated tests were added or run for this slice
   - no runtime signoff yet; selection behavior still needs an in-view click check on real PM4 overlay data

### PM4 Cross-Tile Merge Follow-Up: MSCN Connector Groups (Mar 22)

- Border-spanning PM4 objects are no longer treated as independent runtime groups solely because they were loaded from different ADT tiles.
- Current viewer behavior in `src/MdxViewer/Terrain/WorldScene.cs`:
   - each CK24 family now captures a quantized MSCN connector signature from valid `MSUR.MdosIndex -> MSCN` nodes during PM4 overlay build
   - after all PM4 tiles load, the viewer builds a post-load merge map across neighboring `(tile, ck24)` groups when their MSCN connector sets overlap strongly enough
   - PM4 object selection, object-local transforms, group highlighting, and PM4/WMO correlation candidate dedupe now resolve through that merged runtime-group key instead of raw `(tile, ck24)` only
- Important boundary:
   - this is a post-load runtime merge layer, not a new CK24 reconstruction solver and not proof that MSCN is the sole authoritative object-ownership model
   - PM4 geometry generation still happens per tile/per CK24 first; the merge only unifies groups afterward
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change
   - no automated tests were added or run
   - no runtime real-data signoff yet that the new merge thresholds correctly collapse Dark Portal-style multi-ADT duplicates without false merges

### Viewer Regression Follow-Up: AreaTable + Development Spawn + M2 UV Contract (Mar 22)

- Current viewer behavior:
   - `AreaTableService` now resolves columns from `IDBCDStorage.AvailableColumns` instead of probing a sample row and incorrectly preferring `AreaNumber` / `ParentAreaNum` on all builds.
   - canonical area lookup now indexes by the resolved `ID`/row id first, while still keeping `AreaNumber` aliases and legacy `0.5.x` packed-word aliases as fallbacks for older tables.
   - `TerrainManager` now forces the default initial camera for map `development` to tile `0_0` when that tile exists, instead of averaging all populated tiles.
   - `WarcraftNetM2Adapter` now packs geoset UVs in the renderer's expected layout `[all uv0][all uv1]` instead of per-vertex interleaving, which was causing UV1/env-map layers to read the wrong coordinates.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after these changes on top of the current dirty viewer tree
   - no automated tests were added or run for this slice
   - no runtime real-data signoff yet for `3.3.5` / `0.5.x` / `4.x` AreaTable resolution, development-map spawn behavior, or remaining M2 material-state symptoms beyond the UV layout fix

### Viewer Regression Follow-Up: M2 Foliage AlphaKey Classification (Mar 22)

- Current viewer behavior:
   - `ModelRenderer` no longer restricts M2 alpha-cutout handling to layer 0 only.
   - explicit M2 `AlphaKey` / `Transparent` layers now render as cutouts on any material layer instead of falling into the blended path just because they were not layer 0.
   - texture alpha classification is now tolerant of near-0 / near-255 compressed alpha samples, so binary foliage textures with compression fringes are less likely to be misclassified as truly translucent.
- Why this matters:
   - when binary leaf-card textures fall into the blended pass, the viewer cannot sort per-triangle/per-fragment foliage correctly, so trees/plants look angle-dependent or only "right" part of the time.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change on top of the current dirty viewer tree
   - no automated tests were added or run for this slice
   - no runtime real-data signoff yet that this resolves the remaining M2 foliage transparency failures across representative tree/plant models

### Viewer Regression Follow-Up: PM4 Forced Rebuild Reload + M2 Effect Missing-Texture Fallback (Mar 22)

- Latest user-reported symptoms after the persistent PM4 cache landed were:
   - `Reload PM4` could appear to do nothing because it flowed back through the persisted PM4 overlay cache and restored the same incomplete result.
   - some PM4-heavy tiles could still silently lose known objects because `WorldScene` still kept per-tile PM4 line / triangle / position-ref caps even after the earlier full-map cache change.
   - M2 glow / light-ray / effect-like layers with unresolved textures still fell into the renderer's magenta missing-texture path and showed up as solid pink geometry.
- Current viewer behavior:
   - `ReloadPm4Overlay()` now bypasses disk-cache restore and deletes the current PM4 cache file before rebuilding from source.
   - PM4 overlay cache version is now `2`, so older persisted overlays from the earlier cache behavior are invalidated.
   - remaining per-tile PM4 overlay caps in `WorldScene` are now `int.MaxValue`, matching the earlier removal of the total-map caps.
   - `ModelRenderer` now binds a neutral white 1x1 fallback texture for M2-adapted non-opaque / effect-like base layers when the texture is unresolved, instead of always rendering those layers as magenta missing-texture errors.
- Why this matters:
   - the user expectation for `Reload PM4` is a real rebuild, not a fast restore of the same cached overlay.
   - the earlier persistent cache solved repeat startup cost, but it also made stale or truncated PM4 overlays sticky until the cache was explicitly bypassed.
   - effect-style M2 layers are often visually acceptable with a neutral additive-style fallback, while magenta is only useful for plainly broken ordinary textured geometry.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change; build succeeded with existing workspace warnings only.
   - no automated tests were added or run for this slice.
   - no runtime real-data signoff yet that forced PM4 rebuild now restores the missing objects or that all reported pink M2 effect objects now render acceptably on representative assets.

### PM4 Orientation Follow-Up: World-Space Solver No Longer Forces Mirrored Swap-Only Fits (Mar 21)

### PM4 Render-Derivation Follow-Up: Object-Local Geometry + Baked Base Placement (Mar 21)

### PM4 MPRL Axis Contract Correction (Mar 21)

- Current viewer behavior in `WorldScene`:
   - the common `XY+Zup` PM4 mesh path now preserves the older fixed `MSVT` viewer/world basis `(Y, X, Z)` that matched placed WMO/M2 assets during earlier R&D.
   - PM4 axis convention is now detected once per file and reused across CK24 groups so neighboring PM4 wall/object pieces do not choose different mesh bases.
   - PM4 `MPRL.Position` is now converted to world as `(PositionX, PositionZ, PositionY)` for planar scoring, nearest-ref distance checks, and in-scene PM4 ref markers so the ref data follows that same basis.
   - the earlier viewer-side assumption that `MPRL` was ADT-style planar `X/Z`, vertical `Y` was inconsistent with older PM4 forensic notes on the development dataset.
- Why this matters:
   - if the `MPRL` axis contract is wrong, the PM4 planar solver can pick the wrong swap/inversion basis even when raw `MSVT` geometry is otherwise present and coherent.

- PM4 overlay objects in `WorldScene` no longer exist only as already-placed line/triangle geometry.
- current viewer behavior:
   - PM4 object geometry is localized around a preserved linked-group placement anchor instead of each split fragment center.
   - each `Pm4OverlayObject` carries a baked base placement transform that restores that anchored local geometry into the solved placed frame.
   - split CK24 fragments keep the original pre-split placement anchor so linked-group offsets survive MDOS/connectivity splitting.
   - overlay-wide PM4 transforms and object-local alignment edits now layer on top of that base transform during rendering.
- important limit:
   - this does not change the CK24 solve boundary or claim final PM4 natural-rotation closure.
   - it is structural groundwork so future PM4 placement/container work stops flattening local object geometry into final placed space too early.

### PM4 Link-Decode Follow-Up: Linked `MPRL` Forensics On `development_00_00.pm4` (Mar 21)

- Current runtime-forensics checkpoint for the selected `development_00_00.pm4` object family:
   - raw dump + rollback analyzers show `CK24=0x421809` is one raw CK24/object-id family at the `MSUR` layer (`objId=0x1809`) and the viewer's many `objectPartId` values are reconstruction splits, not separate raw CK24 ids
   - raw `MPRL.Unk04` on this tile spans only about `0°..22.3°`, so do not treat it as already-proven absolute object yaw for this file
   - `Unk06` is constant `0x8000` on this tile
   - `Unk16` still behaves like normal vs terminator typing
   - `Unk14` still behaves like floor/level bucketing
- Active-code fix landed during this forensic pass:
   - `Pm4File.PopulateLegacyView(...)` no longer leaves unsupported legacy `MSLK` fields at zero
   - unsupported fields now use sentinels so `WorldScene` does not accidentally read fake `MsurIndex = 0` data when linking/grouping PM4 surfaces and `MPRL` refs
- Active viewer instrumentation added:
   - selected PM4 object debug info now shows linked `MPRL` summary stats (normal/terminator counts, floor range, heading min/max/mean)
   - PM4 interchange JSON now includes the same summary per object
- Practical implication:
   - before inventing pitch/roll from `MPRL`, inspect the selected object's linked-`MPRL` heading/floor summary in the viewer first
   - if the selected object still needs a large manual rotation while linked `MPRL` headings stay in a narrow low-angle band, the missing orientation is likely not a trivial direct `Unk04 -> absolute yaw` decode

- Authoritative PM4 viewer contract doc: `documentation/pm4-current-decoding-logic-2026-03-20.md`.
- The document was refreshed on Mar 21, 2026 to reflect current `WorldScene` behavior rather than the reverted linked-`MPRL` center-translation experiment.
- Start from that doc before changing PM4 grouping, transform solving, or viewer-side placement rules.

### PM4 Tile-Local Orientation Follow-Up: Quarter-Turn Swap Solve No Longer Rotates Non-Origin Tiles (Mar 21)

- Latest runtime report narrowed a new PM4 regression: tiles beyond `0_0` / `0_1` were coherently rotated about `90°` counter-clockwise while origin-adjacent tiles still looked correct.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
   - the recent quarter-turn planar solver expansion was being applied to tile-local PM4 as well as world-space PM4
   - tile-local PM4 already has a fixed south-west tile basis, so letting the solver choose `swap` candidates per tile could rotate whole tiles once tile coordinates moved away from the origin
- Current correction:
   - tile-local PM4 planar solving now tests only the non-swapped mirror set inside the existing tile basis
   - tile-local PM4 world assembly now uses viewer-world tile ordering (`tileY -> worldX`, `tileX -> worldY`) so non-origin tile-local PM4 no longer lands on the wrong tile grid cell
   - quarter-turn `swap` candidates remain enabled for world-space PM4 only
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change
   - no automated tests were added or run
	- no runtime real-data signoff yet on the reported non-origin tile placement/orientation case

### PM4 Overlay Follow-Up: Full-Map Cache Instead Of AOI Reloading (Mar 22)

- Fresh runtime evidence from the PM4/WMO correlation dump exposed a missing-tile failure mode that was separate from the coordinate-basis regressions:
   - PM4 status showed `100/3290 files` with exactly `400000 lines`, which matched the old hard global overlay line budget in `WorldScene`
   - the old PM4 overlay loader walked the full map file list once and stopped when that shared budget was exhausted
   - this left large portions of the map uncached even before any coordinate-placement logic had a chance to matter
- Short-lived AOI-scoped reload logic was tried, but that made PM4 follow terrain streaming and caused viewer hitching/freezes while camera movement forced PM4 reloads.
- Current viewer behavior in `src/MdxViewer/Terrain/WorldScene.cs`:
   - PM4 overlay now returns to one-time full-map caching instead of AOI-triggered reloading
   - total PM4 line / triangle / position-ref caps are no longer the limiting factor for the one-time cache build
   - per-tile PM4 caps remain in place so single pathological files still cannot explode one tile's overlay geometry indefinitely
   - PM4 file enumeration is sorted deterministically before caching so map-wide cache contents are reproducible across runs
   - PM4 overlay build results are now persisted under `output/cache/pm4-overlay/<data-source-hash>/...` as a gzip-compressed binary cache keyed by ordered PM4 paths plus the active CK24 split flags
   - cache restore rebuilds `Pm4OverlayObject` directly from already-localized geometry, so later viewer runs can skip the expensive PM4 decode/rebuild path instead of only keeping the overlay resident in one process
- Why this matters:
   - the user requirement is to pay the PM4 load cost once and keep the resulting cache stable, rather than reloading PM4 as the camera moves
   - the first cold load still pays the full PM4 decode/build cost, but later runs against the same PM4 corpus and split mode should restore from disk cache instead
   - render-time PM4 visibility still uses existing tile/AOI gating, but the PM4 data itself is now intended to stay resident once built
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change
   - no automated tests were added or run
   - no runtime real-data signoff yet that full-map PM4 cache build time, cache hit speed, and steady-state interaction are acceptable on representative datasets

- Latest runtime evidence from the PM4 alignment window showed mirrored solutions like `swap=True, invertU=False, invertV=False, windingFlip=True` on objects whose real mismatch was a rigid quarter-turn, not a true reflection.
- Root cause in `WorldScene.ResolvePlanarTransform(...)`:
   - world-space PM4 candidate enumeration only tested `identity` and `swap`
   - that meant a world-space object that actually needed a rigid `+/-90` degree basis change could only be approximated by the mirrored `swap` candidate, which reverses object handedness and makes stair/ramp winding run the wrong way around the structure
- Current correction:
   - world-space PM4 now evaluates the full rigid planar set first: identity, 180 degree, +90 degree, and -90 degree basis changes
   - mirrored candidates are no longer part of the active PM4 planar solver, so PM4 stays on rigid candidates only and cannot flip winding by choosing a mirror fit
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this solver fix
   - no automated tests were added or run for this slice
   - no runtime real-data signoff yet on the guardtower / clockwise-staircase PM4 case; do not claim closure from build success alone

### PM4 Bounds Follow-Up: Per-Object PM4 Bounds Can Now Be Rendered In The Scene (Mar 21)

### PM4 MPRL Frame Follow-Up: Linked-Center Translation Experiment Reverted (Mar 21)

- The earlier viewer-side linked-`MPRL` center translation experiment is no longer active.
- Runtime user validation reported that PM4 alignment got materially worse after that change.
- Runtime viewer evidence also does not support the broader `MPRL` bounding-box/container paradigm: reconstructed PM4 geometry is not naturally conforming to that model.
- User/domain correction: `MPRL` itself should be interpreted as terrain/object collision-footprint intersections, not as object-center noise.
- Current viewer behavior in `src/MdxViewer/Terrain/WorldScene.cs`:
   - linked CK24 groups are no longer translated into a linked `MPRL` world-bounds center.
   - PM4 object reconstruction is back on the prior geometry-pivot path with the existing coarse yaw-correction logic.
   - the `12°` suppression of small principal-axis yaw deltas still remains active.
- Working rule:
   - keep using `MPRL` as footprint/collision reference input.
   - do not reintroduce an `MPRL` bounds/container ownership model without fresh evidence.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after reverting the translation path.
   - no automated tests were added or run.
   - no runtime real-data signoff yet on whether PM4 alignment is restored on the development dataset.

### PM4 Yaw Follow-Up: Small Principal-Axis Corrections Are Now Suppressed (Mar 21)

- Runtime user feedback after the earlier PM4 yaw-basis and continuous-yaw-correction work: many PM4 objects were now close, but still looked coherently off by about `5..10` degrees.
- Current viewer-side correction:
   - `WorldScene.TryComputeWorldYawCorrectionRadians(...)` now treats the geometry-derived CK24 yaw correction as coarse-only recovery.
   - residual deltas below `12°` are ignored so MPRL-derived orientation remains authoritative when the object is already near-correct.
- Reasoning:
   - PM4 principal-axis yaw from reconstructed geometry is useful for fixing large basis mistakes.
   - it is not reliable enough to drive small final alignment tweaks across irregular object footprints.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change.
   - no automated tests were added or run.
   - no runtime real-data signoff yet after the threshold change.

- PM4 object bounds already existed internally for culling, picking, and selected-object debug output, but they were not visible in the world, which made nested-object extent triage much harder.
- Current fix:
   - `WorldScene` now exposes a dedicated PM4 bounds render path that draws per-object PM4 AABBs through `BoundingBoxRenderer`.
   - `ViewerApp` now exposes a `PM4 Bounds` checkbox in the PM4 controls next to `PM4 MPRL Refs` and `PM4 Centroids`.
   - selected PM4 groups are highlighted and the exact selected PM4 object gets a white bounds box for click/rotation triage.
- Important scope limit:
   - current PM4 bounds are still computed from the rendered PM4 object geometry path, not directly from `Pm4File.ExteriorVertices` / `MSCN`.
   - this slice makes the current extent source visible for runtime comparison; it does not yet close the MSCN-versus-MSVT bounds question.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this PM4 bounds overlay change
   - no automated tests were added or run for this slice
   - no runtime real-data signoff yet on whether the visible PM4 bounds now explain the reported mismatch

### PM4 Decode Triage And Renderer Parity Queue (Mar 21)

- Current PM4 overlay failure in the viewer is no longer treated as an attach/indexing problem first.
- User-observed runtime state:
   - `PM4: 2674 files found, none decoded into overlay data`
   - interpret this as: PM4 candidates are being found, but every file is currently being rejected before it yields renderable overlay objects
- `WorldScene.LazyLoadPm4Overlay()` now has explicit failure buckets for:
   - tile parse rejection
   - tile range rejection
   - loose/base read failure
   - PM4 parse/decode failure
   - parsed PM4 files that still yield zero overlay objects
- Working explanation for `4.0` versus `3.3.5` behavior:
   - PM4 object reconstruction path itself does not look build-specific
   - viewer map discovery / WDT resolution still is build-specific through `_dbcBuild`
   - the current `2674` PM4 candidate count is likely a signal that the wrong map context or candidate set is being used; fixed development data in `memory-bank/data-paths.md` documents `616 PM4 files`
- Renderer work for PM4 matching is now grouped into one deliberate queue rather than ad hoc fixes:
   1. M2 material, transparency, and reflective parity
   2. lighting DBC expansion beyond current `LightService` coverage
   3. skybox / environment parity so object lighting context is trustworthy
- Planning prompts now exist under workspace `.github/prompts/` for each queue item:
   - `m2-material-parity-implementation-plan.prompt.md`
   - `lighting-dbc-expansion-implementation-plan.prompt.md`
   - `sky-environment-parity-implementation-plan.prompt.md`
- Status correction:
   - this section is planning + handoff only
   - no renderer implementation slice from this queue has landed yet

### Terrain Decode Direction (Current)

- Priority is profile-correct alpha decode behavior before broader feature intake.
- FormatProfileRegistry now carries terrain alpha decode mode per ADT profile.
- StandardTerrainAdapter alpha extraction routes by profile mode:
   - 3.x strict path
   - 0.x legacy sequential path
- Keep terrain renderer topology/shader rewrites out until decode stability is verified.

### Next Steps

1. Validate runtime terrain alpha output with real data on Alpha-era and LK 3.3.5.
2. Continue surgical intake from v0.4.0..main with SAFE-first triage.
3. Keep UI evolution incremental (no drastic layout churn).
4. Bring import/export enhancements in small, build-gated batches after decode path stabilization.
5. Run the renderer parity queue in order for PM4 object-matching work: materials first, lighting second, sky/environment third.

### Current Intake Decision

- Commit queue triage for the current recovery pass:
   - `177f961`: RISKY, skip
   - `37f669c`: RISKY, skip
   - `d50cfe7`, `326e6f8`, `4e2f681`, `39799bf`, `62ecf64`: MIXED, extract only isolated safe slices
- First SAFE batch is limited to the corrected alpha-atlas helper from `62ecf64`.
- Do not pull the earlier `d50cfe7` atlas helper version; it bakes in the old 63->62 edge remap during import/export.
- Do not pull ViewerApp, TerrainRenderer, terrain decode heuristic, or test-project changes in this first batch.
- First SAFE batch has now been applied and the MdxViewer solution build passed.
- Runtime real-data validation is still required before treating the helper as terrain-safe in practice.

### Rendering Recovery Follow-up (Mar 18)

- Main-branch renderer residency fix is now applied in `WorldAssetManager`:
   - do not evict live MDX/WMO renderers by default
   - keep only raw file bytes under LRU pressure
   - retry failed cached model loads instead of pinning permanent nulls
- Minimal skybox support is now present:
   - `WorldScene` classifies skybox-like MDX/M2 placements separately
   - nearest skybox renders as a camera-anchored backdrop before terrain
   - `ModelRenderer.RenderBackdrop(...)` forces no depth test/write for all layers
- Reflective M2 bugfixes were already present on this branch before this batch:
   - no inferred `NoDepthTest` / `NoDepthSet` from unstable Warcraft.NET render flags
   - guarded env-map backface handling in the model shader path
- Build passed again after the rendering batch.
- Runtime verification still required for doodad reload/culling, skybox behavior, and LK MH2O liquids.

### MCCV + MPQ Follow-up (Mar 18)

- Active chunk-based terrain rendering now includes MCCV vertex colors again.
- Implementation path is intentionally minimal:
   - `StandardTerrainAdapter` extracts `MccvData`
   - `TerrainChunkData` stores per-vertex MCCV bytes
   - `TerrainMeshBuilder` uploads RGBA as a new vertex attribute
   - `TerrainRenderer` applies the tint in shader
- Runtime follow-up corrected the semantics further:
   - MCCV bytes are now interpreted as BGRA, not RGBA
   - neutral/no-tint values are treated as mid-gray (`127`) instead of white
   - terrain tint is now derived from RGB remapped around mid-gray; MCCV alpha is preserved but not used as terrain tint strength
- `NativeMpqService` also now carries the isolated patch-reader recovery slice needed for 1.x+ patched clients and later encrypted entries.
- `NativeMpqService.LoadArchives(...)` now also scans recursively so map content in nested/custom `patch-[A-Z].mpq` archives is not skipped during archive discovery.
- Both the converter core project and the MdxViewer solution build passed after this batch.
- Real-data validation is still pending for MCCV appearance and patched MPQ chains.

### 3.x Terrain Alpha Follow-up (Mar 18)

- The incorrect offset-0 LK alpha fallback experiment was reverted after runtime validation showed it was wrong.
- Current terrain recovery direction is now explicitly profile-driven instead of heuristic-driven:
   - 3.0.1 / 3.3.5 ADT profiles treat MPHD `0x4 | 0x80` as the big-alpha mask
   - `Mcal` decode now distinguishes compressed alpha, 8-bit big alpha, and legacy 4-bit alpha while respecting the MCNK do-not-fix-alpha bit
- Build validation passed after this batch, including the alternate-output MdxViewer build used while the live viewer holds `bin/Debug` locks.
- Runtime validation follow-up is now positive on the user's real data:
   - the tested 3.0.1 alpha-build terrain now renders correctly on this path
   - the same recovery line also preserves Alpha 0.5.3 terrain after restoring the legacy edge fix in `AlphaTerrainAdapter`
- Keep broader claims narrow: this is strong evidence that the profile split is correct for the tested samples, not blanket proof for every later-era terrain dataset.

### 3.x Terrain Guardrail Update (Mar 18)

- User direction is now explicit: do not use `*_tex0.adt` split terrain sourcing in the active viewer path for current 3.x alpha recovery work.
- Active viewer profiles for `3.0.1`, `3.3.5`, and unknown `3.0.x` no longer opt into `_tex0` terrain layer/alpha sourcing.
- `StandardTerrainAdapter` also now avoids opening `_tex0` files unless a future profile explicitly re-enables that path.
- The temporary rollback of `MCNK.SizeMcal` / `SizeMcsh` trust caused a major runtime regression and was reverted immediately; the active viewer path still uses the prior 3.x header-size behavior.
- Follow-up parser guardrail: `Mcnk.ScanSubchunks(...)` now treats `MCNK.SizeMcal` / `SizeMcsh` as an optional extension of the declared MCAL/MCSH payload, never a reason to advance less than the declared subchunk size. This avoids landing the FourCC scan inside MCAL/MCSH payload bytes when header sizes are smaller than the chunk-declared span.
- Build validation passed after this parser fix:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- This is a guardrail rollback only. Runtime validation is still required for the remaining chunk-skip / decode-loss issue on 3.x terrain.

### 4.x / 5.x Terrain Profile Direction (Mar 18)

- Keep `_tex0.adt` and `_obj0.adt` parsing as a separate 4.x+/5.x concern, not part of the active 3.x recovery path.
- `FormatProfileRegistry` now has separate provisional `4.x` and `5.x` ADT profiles that opt into split texture and placement sourcing.
- `StandardTerrainAdapter` now routes placement parsing through `_obj0.adt` only when the resolved terrain profile explicitly requests it; 3.x remains on root-ADT placement parsing.
- This is profile scaffolding, not full Cataclysm/MoP correctness. The user requirement is broader MPQ-era support through `5.3.x`; later CASC support is a separate future track.

### 4.x No-MCIN Root Fallback (Mar 19)

- Real-data audit on the fixed `test_data/development/World/Maps/development` source confirmed the active 4.x blocker is structural, not just bad chunk indices:
   - 466 root ADT filenames
   - 114 zero-byte placeholders
   - 352 non-empty roots
   - 0 non-empty roots with `MCIN`
- `StandardTerrainAdapter` now treats missing `MCIN` on later-era root ADTs as a top-level `MCNK` scan fallback instead of an automatic hard failure.
- Scope limit for this fallback:
   - it is intended to recover root geometry/chunk order first
   - it does not by itself prove full `_tex0.adt` texture-layer parity for 4.x data
   - keep 3.x alpha-path guardrails unchanged

### PM4 MSLK-Driven Assembly Follow-up (Mar 20)

- PM4 overlay object assembly in `WorldScene` now consumes `MSLK` linkage to split CK24 buckets into linked sub-groups before optional MDOS/connectivity splitting.
- PM4 object keys now include a per-component `objectPart` id (`tile + ck24 + objectPart`) so per-object selection/offset state does not collide when CK24 is reused by multiple linked components.
- Planar transform solving now uses linked `MPRL` refs at CK24 scope and applies one shared transform per CK24, so split linked/components remain on the same coordinate plane.
- `MSLK` linkage logic now prefers `MsurIndex` for surface association and only falls back to `RefIndex` as a surface id when needed.
- Selected PM4 diagnostics now expose `ObjectPartId`, dominant `MSLK.GroupObjectId`, and linked `MPRL` ref count to aid runtime triage.
- Build status: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only).
- Runtime signoff is still required on the reported real-data PM4 cases (split structures + 90-degree ramp mismatch).

### PM4 Tile Mapping Guardrail Follow-up (Mar 20)

- PM4 overlay tile assignment in `WorldScene` now trusts filename coordinates and maps them into the terrain adapter's row/col tile convention:
   - PM4 filename `map_x_y.pm4` -> viewer tile `(tileX=x, tileY=y)`.
- Removed the prior PM4 tile reassignment heuristic that remapped tiles from `MPRL` centroid/bounds checks.
   - Inter-tile links in sparse PM4 datasets made that heuristic unstable and caused drift/collisions.
- Duplicate PM4 files that resolve to the same viewer tile now merge instead of overwrite.
   - Overlay object lists, tile stats, and PM4 position refs append; object-part ids are rebased for lookup-key uniqueness.
- Practical effect: sparse/missing PM4 or ADT tile sets remain sparse/blank rather than shifting adjacent PM4 geometry into the wrong tile.

### PM4 Reboot Runtime Handoff (Mar 20)

- Next session starts with runtime-only validation, not additional PM4 decode refactors.
- First required checks after restart:
   - verify PM4 tile placement continuity for the reported mismatch path (`00_00`, `01_00`, and `01_01`/`1_1`)
   - confirm missing PM4 tiles remain blank instead of shifting neighboring PM4 geometry
   - confirm no duplicate-tile overwrite symptoms when multiple PM4 files map to one viewer tile
- If mismatch persists, collect one concrete file pair and an on-screen tile reference, then add temporary debug output for `pm4Path -> (tileX,tileY)` mapping before changing transforms again.

### ModelRenderer Follow-up From 39799bf (Mar 18)

- The commit message for `39799bf` bundled terrain and model notes together, but the only remaining model-renderer hunk on top of the already-applied MPQ fix was particle suppression on the world-scene instanced path.
- That hunk is now applied:
   - batched placed-model rendering skips particles
   - standalone model preview/rendering still allows particles
- Keep this split until particle simulation becomes instance-aware.

### World Wireframe Reveal Follow-up (Mar 18)

- World-scene wireframe toggle is now hover-driven instead of a blanket terrain-only toggle:
   - `WorldScene.ToggleWireframe()` now keeps terrain wireframe in sync while also enabling a hover reveal mode for placed WMOs and MDX/M2 doodads
   - ViewerApp refreshes the reveal set every frame from the current scene-viewport mouse position
   - hovered objects render an extra wireframe overlay pass without changing standalone model-viewer wireframe behavior
- Current validation status:
   - alternate-OutDir `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed after restoring terrain wireframe and switching the hover test from a loose ray/AABB heuristic to a screen-space brush
   - `WorldAssetManager` world-model loading now resolves the canonical model path before M2 skin lookup so `.mdx` aliases that actually resolve to `MD20` roots can search for skins relative to the real asset path
   - runtime visual validation is still pending for reveal radius feel and for confirming the remaining world-scene M2 load failures are actually cleared on user data

### M2 Adapter Follow-up (Mar 18)

- `WarcraftNetM2Adapter` now treats raw `MD20` as the primary parse path instead of only using direct `MD21` parsing as a fallback after the Warcraft.NET `Model(...)` wrapper fails.
- Current rationale:
   - the user's active client data is dominated by raw `MD20` roots, not chunked `MD21` containers
   - relying on the wrapper first made the effective parse path sporadic across assets
- Build-only validation passed again on the alternate-OutDir MdxViewer solution build.
- Runtime confirmation is still required for the remaining sporadic world-scene M2 failures.

### World Load Performance Follow-up (Mar 18)

- Northrend load-time investigation confirmed AOI terrain streaming was already the default; the bigger stall was world-object asset loading on tile arrival and first render.
- `WorldScene` no longer eagerly calls blocking `EnsureMdxLoaded` / `EnsureWmoLoaded` for streamed tiles or external spawns.
- `WorldAssetManager` now has deferred MDX/WMO load queues plus a bounded per-frame `ProcessPendingLoads(...)` path.
- `WorldScene.Render(...)` now processes a small per-frame asset budget and only uses loaded renderers in render paths, queueing missing assets instead of force-loading them on the render thread.
- Instance bounds are refreshed after queued model loads complete so culling can converge from temporary fallback bounds to real model bounds.
- Follow-up asset-read recovery after runtime queue investigation:
   - the UI queue counter now reports unique pending assets instead of raw queue-node count
   - repeated `PrioritizeMdxLoad` / `PrioritizeWmoLoad` calls no longer flood the priority queues with duplicate entries every frame
   - `MpqDataSource` now builds file-path and extension indexes once at startup instead of re-filtering the full file list for repeated model/skin lookups
   - `MpqDataSource.ReadFile(...)` now has a bounded global raw-byte LRU cache so repeated model and texture reads reuse already-read archive data instead of hitting MPQ/loose-file resolution again
   - `WorldAssetManager` skin selection now caches best `.skin` matches per resolved model path instead of rescanning the `.skin` file list on retries
   - `MpqDataSource` now also has a bounded background prefetch path with separate read-only `NativeMpqService` workers so queued model bytes can be warmed into the shared raw-byte cache without sharing the primary archive reader across threads
   - `WorldAssetManager` now triggers that prefetch when new MDX/WMO assets are queued, including common extension aliases and M2 skin candidates
- Build validation passed after this change using the alternate output path:
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- No runtime real-data validation has been performed yet for the new loading behavior. Do not claim the Northrend load regression is fixed until startup responsiveness and in-world streaming are checked on real data.
- Parallel MPQ archive reads are now limited to background raw-byte warmup only:
   - GL renderer/material creation remains main-thread work in the current pipeline
   - the primary `MpqDataSource` reader is still not shared across threads; worker threads use separate `NativeMpqService` instances
   - runtime profiling is still required before increasing worker count or pushing texture/material construction off the main thread

### World-Scene M2 Render Follow-up (Mar 18)

- User runtime feedback after the deferred-load change: world M2 doodads appeared to load but remained invisible.
- Current mitigation is targeted, not a full rollback:
   - `MdxRenderer` now tracks whether it was built through the Warcraft.NET M2 adapter
   - `WorldScene` keeps the lighter batched `RenderInstance(...)` path for classic MDX models
   - M2-adapted world doodads now use the proven per-instance `RenderWithTransform(...)` path instead of the batched path
- Rationale:
   - standalone model viewing and WMO doodad rendering already rely on `RenderWithTransform(...)`
   - the invisible-M2 symptom is therefore more likely a world-scene batch-path issue than an asset-read failure
- Build validation passed after this mitigation using the alternate output path.
- Runtime real-data validation is still required to confirm M2 doodads are visible again and to measure whether the selective fallback has an acceptable frame-time cost.

### World-Scene M2 Conversion Follow-up (Mar 18)

- Historical diff review showed the stronger world-side M2 recovery path lives in `main` / `4e9237a`, not in `177f961` alone.
- `WorldAssetManager` now prefers `M2ToMdxConverter` for raw `MD20` world doodads before falling back to `WarcraftNetM2Adapter`.
- `ModelRenderer` also now disables the classic layer-0 `Transparent` hard alpha-cutout heuristic for M2-derived models so their materials follow the blended path used by the working mainline M2 support.
- Latest parity correction versus final `main` commit `62ecf64`:
   - old `main` branch world M2 behavior was simpler than this recovery branch briefly became:
      - direct `M2 + .skin` adaptation was the first-choice world load path
      - world doodads then rendered through the normal `RenderInstance(...)` path with no M2-specific world-scene split
   - recovery branch is now back on that shape:
      - direct Warcraft.NET adaptation is tried first for world M2s
      - byte-level `M2ToMdxConverter` conversion is now only a fallback after adapter failure
      - world-scene rendering no longer special-cases M2-adapted doodads into `RenderWithTransform(...)`; all loaded world doodads use the normal instanced world path again
- Deferred world-model loading now preserves the older retry semantics for failed entries:
   - queued MDX/WMO loads only short-circuit when a non-null renderer is already cached
   - queued `null` entries are allowed back through `ProcessPendingLoads(...)` for retry instead of becoming permanent invisible instances
   - `.mdx` and `.m2` aliases are now both considered during direct reads and file-set resolution so LK-era model-extension mismatches have an exact-path fallback before basename heuristics
- Build-only validation passed after these changes using the alternate output path:
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- No automated tests were added or run for this slice.
- Runtime real-data validation is still the blocker:
   - confirm Northrend or NorthrendBG now shows nonzero MDX/M2 world-object load/render stats
   - confirm the converted M2 path does not regress frame time or material appearance

### WMO Doodad M2 Loader Follow-up (Mar 18)

- Remaining parity gap after the world-scene fixes: `WmoRenderer` doodad-set loading was still on an older MDX-only path.
- Concrete issue:
   - `GetOrLoadDoodadModel(...)` only did raw `MdxFile.Load(...)` after a direct file read
   - it never attempted direct `.m2` / `MD20` / `MD21` adaptation with companion `.skin`
   - it also round-tripped raw bytes through a shared cache filename, which could collide on duplicate doodad basenames across different directories
- Current fix now mirrors the shared world/standalone behavior more closely:
   - `WmoRenderer` resolves canonical doodad paths through the file set before loading
   - WMO doodad M2s now try Warcraft.NET adapter + `.skin` first
   - raw `MD20` doodads then fall back to `M2ToMdxConverter` only after adapter failure
   - non-M2 doodads now load from in-memory streams instead of cache-file writes
   - adapted and converted M2 renderers are explicitly marked as M2-derived so `ModelRenderer` keeps them on the non-cutout transparent-material path
- Same M2-derived renderer flag is now also applied in `WorldAssetManager` and standalone `ViewerApp.LoadM2FromBytes(...)`.
- Build validation passed after this change using the alternate output path:
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime real-data validation is still required:
   - confirm WMO doodad sets now populate visible M2s instead of just the WMO shell
   - confirm world doodads also recover with the restored shared M2 load path

### MPQ Listfile Recovery Follow-up (Mar 18)

- Root-cause follow-up for the latest standalone M2 `.skin` failure:
   - `ViewerApp` UI text already claimed the community listfile was auto-downloaded
   - actual `Open Game Folder` flow still passed `null` into `LoadMpqDataSource(...)`, so `MpqDataSource` never received any external listfile unless one was supplied manually
- Current fix:
   - `ViewerApp.LoadMpqDataSource(...)` now resolves the listfile path before constructing `MpqDataSource`
   - resolution order is: explicit path, bundled repo/runtime `community-listfile-withcapitals.csv`, then cached/downloaded `ListfileDownloader` path
   - if none are available, viewer now logs that it is falling back to archive-internal names only
- Why this matters:
   - many MPQ internal listfiles do not expose `.skin` entries even when `.m2` entries are present
   - without the external listfile, companion `.skin` discovery can fail and surface as `Missing companion .skin for M2`
- Build-only validation passed after this fix using the alternate output path:
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime real-data validation is still required to confirm standalone M2 loading and world/WMO M2 recovery on the user's client data.

### WDL Preview Cache + Model Texture Cache Follow-up (Mar 18)

- The missing `main`-branch WDL spawn-point slice is now ported in a surgical form:
   - added `WdlPreviewCacheService` with memory cache, disk cache, and background warmup
   - `ViewerApp` now initializes a cache root per loaded game path, warms discovered WDL maps after map discovery, and opens previews through the cache-aware path
   - `ViewerApp_WdlPreview` now shows loading/error state while a preview is warming instead of only failing synchronously
   - `WdlPreviewRenderer` now accepts prebuilt `WdlPreviewData` payloads and uses the cached spawn-position math path
- Model-load performance follow-up on the active renderer path:
   - `ModelRenderer` texture diagnostics are no longer always-on; file logging now requires `PARP_MDX_TEXTURE_DIAG=1` or a substring filter value
   - identical BLP/PNG textures are now shared across renderers through a refcounted GL texture cache instead of being decoded/uploaded once per model instance
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this batch
   - no runtime real-data validation has been performed yet for preview warmup behavior or the M2 load-speed improvement
   - do not claim the per-model load regression is fixed until the user verifies the actual world/model load experience on real data

### WDL Parser + Transparency Follow-up (Mar 18)

- The first recovery-port WDL preview cache batch exposed a deeper compatibility issue on real 1.x/3.x data:
   - active `WdlParser` previously rejected every WDL whose version was not `0x12`
   - this matched the reported `0/107 cached, 0 warming, 107 failed` preview state on non-Alpha clients
- Current code path changes:
   - `WdlParser` is now version-tolerant and scans for `MAOF`/`MARE` instead of assuming Alpha-only layout ordering
   - new `WdlDataSourceResolver` unifies `.wdl` / `.wdl.mpq` reads and `FindInFileSet(...)` recovery for both `WdlPreviewCacheService` and `WdlTerrainRenderer`
   - `WmoRenderer` canonical doodad resolution now tries `.m2` aliases in addition to `.mdx`/`.mdl`
   - `ModelRenderer` now inspects decoded texture alpha shape and only keeps the classic layer-0 `Transparent` hard-cutout path for binary-alpha textures; semi-translucent textures stay blended
- Validation status:
   - compile/build only: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
   - no runtime real-data validation yet for the WDL fix, doodad aliasing change, or semi-translucent rendering change

### Standalone 3.x Model Loader Follow-up (Mar 18)

- User runtime feedback: opening individual 3.x `.mdx` files could freeze the viewer and still fail to load the model.
- Current standalone-path fixes in `ViewerApp`:
   - `LoadModelFromBytesWithContainerProbe(...)` now recognizes `MD21` as well as `MD20`, so `.mdx` files with either M2-family root are routed away from the classic MDX parser
   - standalone M2 loading now resolves a canonical MPQ model path before skin lookup, instead of trusting the UI-selected alias path blindly
   - same-basename `.skin` candidates are tried first; the broader `.skin` file-list search is now only a fallback and is cached per session to avoid repeated UI-thread scans
   - standalone `MD20` loads now also get the existing M2->MDX converter fallback when direct adapter + skin loading fails
   - standalone skin-path cache is cleared when the viewer switches to a new MPQ data source
- Validation status:
   - compile/build only: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
   - no runtime real-data validation yet for the standalone 3.x file-open freeze report

### M2 Empty-Fallback Guardrail (Mar 18)

- Latest runtime clue: some M2-family assets can still end up in a blank "loaded" state with `0` geosets / `0` vertices.
- Current interpretation:
   - this is not trustworthy evidence that the model truly loaded
   - one active failure mode is the raw `MD20` converter fallback producing an `MDX` shell with no renderable geometry
- Current code change:
   - shared M2 fallback validation now rejects converted models unless they contain at least one renderable geoset
   - applied consistently in standalone `ViewerApp`, world `WorldAssetManager`, and WMO doodad `WmoRenderer`
   - logs now keep the real failure signal instead of presenting an empty converted model as success
- Validation status:
   - alternate-OutDir `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed
   - no runtime real-data validation yet
   - do not claim pre-release `3.0.1` M2 compatibility is fixed; this change only removes a misleading false-positive load state

### Pre-release 3.0.1 Model Profile Guardrail (Mar 19)

- Live Ghidra work against `wow.exe` build `3.0.1.8303` now confirms the client-side entry gate for this model family:
   - root must be `MD20`
   - accepted versions are `0x104..0x108`
   - parser layout splits at `0x108`
- Active viewer path now uses that evidence as an early guardrail instead of letting the generic adapter infer compatibility:
   - `ViewerApp.LoadM2FromBytes(...)`, `WorldAssetManager.LoadMdxModel(...)`, and `WmoRenderer.LoadM2DoodadRenderer(...)` all validate the resolved model bytes against `FormatProfileRegistry.ResolveModelProfile(...)` before `.skin` selection or converter fallback
   - `WorldScene` now receives the build string during construction so constructor-time manifest/model loads do not miss the profile guard
   - `WorldAssetManager.SetBuildVersion(...)` keeps later lazy loads aligned with `SetDbcCredentials(...)`
- Intentional scope limit:
   - this is a fail-fast compatibility guardrail, not proof that the remaining pre-release `3.0.1` parser differences are fully implemented
   - Track B remains separate: neon-pink transparent surfaces still point at shared `ModelRenderer` / material / texture-binding behavior
- Build validation passed after this slice:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation is still required on real `3.0.1` assets before claiming the guarded load path is correct end-to-end.

### Pre-release 3.0.1 Profile Routing Broadening (Mar 19)

- The active viewer registry now routes the whole `3.0.1.x` family through the pre-release `3.0.1` profile instead of reserving that path for exact build `3.0.1.8303` only.
- Affected routing paths stay unified because they all resolve through `FormatProfileRegistry`:
   - terrain ADT profile selection
   - WMO profile selection
   - M2-family guard/validation in standalone, world, and WMO doodad loaders
- Keep claims narrow:
   - this removes an avoidable fallback to generic `3.0.x` handling for other `3.0.1` builds
   - it does not by itself implement the missing pre-release parser families documented from `wow.exe`
- Validation status:
   - build/runtime validation for this narrow routing change is still pending

### Pre-release 3.0.1 Parser + Fallback Alignment (Mar 19)

- `WarcraftNetM2Adapter` no longer treats pre-release `3.0.1` raw `MD20` files as if they were standard Warcraft.NET `MD21` layouts.
- Current viewer-side path now does two things consistently for standalone, world, and WMO doodad loads:
   - uses the local profiled `MD20` parser for the main adapter path
   - passes the active build version into `M2ToMdxConverter` so fallback conversion can avoid later-layout animation / bone assumptions on pre-release builds
- Fallback converter scope was narrowed intentionally:
   - keep vertex / texture / bounds parsing plus skin index / triangle tables for geometry conversion
   - do not force pre-release `.skin` submesh / texture-unit parsing from unproven fixed strides
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
   - runtime validation on real `3.0.1` assets is still pending

### Pre-release 3.0.1 Texture Mapping Follow-up (Mar 19)

- Latest user runtime feedback narrowed the remaining visible problem from geometry to texture binding: affected `3.0.1` models now appear, but some still render magenta or with the wrong texture.
- `WarcraftNetM2Adapter` now preserves non-file M2 texture semantics instead of discarding them:
   - non-`None` texture types now keep their `ReplaceableId` instead of becoming empty-path textures with replaceable id `0`
   - texture wrap flags now flow through `MdlTexture.Flags`, so renderer-side clamp handling can still work for adapted M2s
- Embedded root-profile batch parsing also now preserves `MaterialIndex` and `TextureComboIndex` instead of forcing every batch to slot `0`.
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this change
   - no automated tests were added or run
   - no new real-data runtime validation has happened yet for the texture fix itself
   - do not claim the magenta/pre-release texture issue solved until the same real client assets are rechecked

### Pre-release 3.0.1 Embedded Submesh Decode Follow-up (Mar 19)

- New runtime evidence showed the embedded root-profile path was still only decoding part of many pre-release `3.0.1` models, with severe spiking/artifact geometry on affected doodads.
- Root cause in the current adapter was concrete:
   - the embedded `0x30` submesh records were being read with the wrong field mapping
   - the parser was effectively treating `Level` as `VertexStart` and later fields as triangle bounds, which can cut sections incorrectly and produce partial/exploded meshes
- Current correction in `WarcraftNetM2Adapter`:
   - embedded root-profile submeshes now use the same `VertexStart` / `VertexCount` / `IndexStart` / `IndexCount` ordering as the known `M2SkinSection` layout
   - replaceable textures also now stay on the renderer's replaceable-resolution path by emitting an empty texture path when a non-file replaceable id is present
- Scope note:
   - this change is isolated to the special pre-release embedded-root-profile path
   - it does not reroute normal `3.3.5` Warcraft.NET parsing and does not affect classic standalone `MDX` handling
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the correction
   - no automated tests were added or run
   - real-data runtime validation is still required before claiming the geometry-loss issue fixed

### Pre-release 3.0.1 Direct Material Metadata Fallback (Mar 19)

- Follow-up after geometry improved but affected pre-release `3.0.1` doodads still rendered magenta / untextured and appeared to miss normal lighting.
- Current adapter change in `WarcraftNetM2Adapter`:
   - the profiled `MD20` path no longer depends only on Warcraft.NET for textures, render flags, and texture lookup
   - if Warcraft.NET does not populate those tables, the adapter now scans the profiled header region and validates direct table candidates for:
      - texture records (`0x10` stride)
      - render flags (`0x04` stride)
      - texture lookup (`0x02` stride)
   - replaceable textures remain on the renderer's replaceable-resolution path instead of being forced through file-path loading
- Scope note:
   - this fallback only runs inside the special pre-release `3.0.1` profiled parser path
   - it does not change normal `3.3.5` Warcraft.NET model parsing or classic `MDX` handling
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the change
   - the build ran while the live viewer held output DLL locks, so MSB3026 copy-retry warnings were expected but non-fatal
   - no automated tests were added or run
   - no new real-data runtime validation has been completed yet for the direct metadata fallback

### Pre-release 3.0.1 Transparent Material Follow-up (Mar 19)

- Latest runtime feedback narrowed the remaining visual issue further: most non-transparent pre-release `3.0.1` M2s now appear to load, while foliage/cutout-style transparent assets still rendered as opaque quads using the texture color in areas that should be alpha-driven.
- Current renderer-side mitigation in `ModelRenderer`:
   - layer-0 M2-adapted materials now derive an effective blend mode from the loaded texture alpha shape when the declared blend mode is still `Load`
   - binary-alpha textures are promoted to `Transparent` so they can use alpha-cutout behavior
   - translucent-alpha textures are promoted to `Blend` so they use standard alpha blending instead of an opaque pass
   - the alpha-cutout path is no longer blanket-disabled for all M2-adapted models; it now keys off actual texture alpha classification
- Scope note:
   - this is a narrow fallback for M2-adapted models with imperfect pre-release blend metadata
   - it does not change classic non-M2 renderer behavior
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the change
   - no automated tests were added or run
   - real-data runtime validation is still required on the same transparent doodads before claiming the issue fixed

### WDL Spawn Fallback + World Load Throughput Follow-up (Mar 19)

- Latest user follow-up added three viewer-facing requirements beyond base pre-release M2 decode:
   - spawn-point selection should only be offered once a WDL preview is actually ready
   - if WDL preview generation/read fails, map load should silently fall back to the default terrain spawn instead of trapping the user in a preview error path
   - deferred world-object loading needs to settle materially faster on large maps
- Current viewer changes:
   - map discovery rows now expose explicit `Load` and `Spawn` actions instead of a generic preview-only path
   - `Spawn` is gated on `WdlPreviewWarmState.Ready`; loading/failed states stay disabled in the UI
   - `OpenWdlPreview(...)` and `DrawWdlPreviewDialog()` now fall back to the normal map load path when WDL preview preparation fails
   - the first attempt to speed up deferred world-object loading used a larger adaptive per-frame load budget and heavier queue-time model prefetch
   - that throughput experiment caused a major runtime regression on real data and was reverted after user feedback
   - active behavior is back to the lighter fixed `ProcessPendingLoads(maxLoads: 24, maxBudgetMs: 20.0)` path plus the simpler alias-based model/skin prefetch
- Transparent follow-up coupled to this pass:
   - `ModelRenderer` no longer renders magenta fallback geometry for M2-adapted layers/geosets whose textures failed to resolve
   - alpha-kind fallback for uncategorized textures now defaults to `Opaque` instead of `Binary`, so the M2 layer-0 blend heuristic does not infer alpha-cutout behavior from unloaded textures
- Validation status:
   - file-level diagnostics are clean after the code change
   - the throughput experiment itself should be treated as rejected, not active
   - no automated tests were added or run

### WDL Spawn Chooser Regression Handoff (Mar 20)

- Latest runtime report: the WDL heightmap spawn chooser does not function on tested versions in the active branch state.
- Treat earlier spawn-fallback notes as historical implementation intent, not proof of current runtime correctness.
- Active investigation slice for a fresh chat:
   - verify map-row `Spawn` enablement versus actual warm-state transitions
   - verify chooser open path and spawn-commit callback execution
   - verify failure fallback still loads map normally when preview warmup/read fails
- Required closure evidence:
   - real runtime confirmation on both Alpha-era and 3.x data
   - explicit user-visible proof that spawn selection applies camera/player spawn rather than silently no-oping
- Validation limits for this note-only handoff:
   - no code changes in this entry
   - no automated tests added or run

### Pre-release 3.0.1 M2 Wrap + Pink Transparency Follow-up (Mar 19)

- Latest runtime feedback after the load-regression revert narrowed the remaining model issues to two specific symptoms:
   - some pre-release `3.0.1` M2 surfaces showed wrong texture addressing consistent with wrap/clamp inversion
   - transparent surfaces still rendered pink, suggesting the renderer was binding fallback state instead of a usable texture path
- Current targeted fixes:
   - `ModelRenderer` now interprets `WrapWidth` / `WrapHeight` as repeat flags; clamp is only used when those flags are absent
   - `WarcraftNetM2Adapter.ToMdlTexture(...)` now preserves any parsed texture filename even when a nonzero replaceable texture type is also present
   - the direct profiled texture-table fallback now reads filenames whenever the record contains a valid string span, instead of discarding names solely because the texture type is nonzero
- Why this matters:
   - the previous sampler logic inverted the MDX-side wrap semantics into GL clamp state
   - the previous adapter logic could strip the only usable texture filename from pre-release records that still also carried replaceable metadata, which is a plausible cause of all-pink transparent layers
- Validation status:
   - file-level diagnostics were clean for `ModelRenderer.cs` and `WarcraftNetM2Adapter.cs`
   - alternate-OutDir build passed: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
   - no automated tests were added or run
   - real-data runtime validation is still required before claiming the clamp and pink-surface issues are fixed

### Pre-release 3.0.1 Transparent Layer Stack Follow-up (Mar 19)

- User runtime evidence after the wrap/path fixes still showed foliage-like transparent doodads rendering as pink crossed planes, plus detached transparent fragments that looked like the wrong layers were bound to the wrong sections.
- Current adapter + renderer change set:
   - `WarcraftNetM2Adapter.BuildMaterialsFromBatches(...)` no longer collapses each skin section to only the first batch/material layer
   - all texture-unit batches for the same skin section now accumulate as layers on a shared material, preserving layered cutout/blended section composition instead of dropping later layers
   - `ModelRenderer.LoadTextures()` now keeps replaceable-texture resolution available as a fallback even when a direct texture filename exists but fails to load
- Why this is the current best root-cause fix:
   - pink transparent quads are consistent with section geometry surviving while the intended transparent layer stack is reduced to an incomplete or wrong first layer
   - pre-release records that carry both a nominal filename and replaceable metadata can still need the replaceable path when the direct filename does not actually resolve on disk/MPQ
- Validation status:
   - file-level diagnostics were clean for the edited files
   - alternate-OutDir build passed: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
   - no automated tests were added or run
   - real-data runtime validation is still required before claiming the pink foliage / detached transparent fragment issue fixed

### Pre-release 3.0.1 Profiled Texture Metadata Preference Follow-up (Mar 19)

- User runtime feedback after the layer-stack change reported the result was better, but broad pink foliage still remained on Northrend.
- Current adapter-side conclusion:
   - the profiled `MD20` path was still treating any non-empty Warcraft.NET texture/render metadata as authoritative
   - that meant a partial or weak Warcraft.NET table could block the direct profiled metadata reader from replacing it, even when the profiled table had more usable filenames and stronger lookup coverage
- Current fix in `WarcraftNetM2Adapter`:
   - direct profiled texture, render-flag, and texture-lookup discovery now always runs for the pre-release path
   - profiled metadata replaces the current table only when it scores higher than the existing metadata, instead of only when the existing list is completely empty
   - texture-table quality now prefers named texture records strongly, which is the right bias for the remaining pink-foliage symptom
- Why this matters:
   - these pre-release shrubs appear to be failing as texture-resolution problems more than geometry problems
   - if Warcraft.NET preserved only replaceable IDs or an incomplete lookup set, the renderer could still end up with unresolved pink-transparent layers even after the earlier replaceable and layer-stack fixes
- Validation status:
   - file-level diagnostics were clean for `WarcraftNetM2Adapter.cs`
   - alternate-OutDir build passed again: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
   - no automated tests were added or run
   - no new runtime real-data validation has been performed yet for this follow-up

### 0.5.3 MDX Replaceable Fallback Regression Follow-up (Mar 19)

- User runtime validation on Alpha `0.5.3` Azeroth showed a clear regression after the latest M2-oriented texture work: classic MDX foliage that previously rendered correctly was now resolving to pink/wrong leaf surfaces.
- Root cause is narrow and renderer-side:
   - `ModelRenderer.LoadTextures()` had been broadened so direct-path textures with a nonzero `ReplaceableId` could fall back through replaceable-texture heuristics after a direct load miss
   - that fallback was intended only for pre-release M2-adapted models that carry both a nominal filename and replaceable metadata
   - applying it to classic MDX leaked M2 recovery logic into the 0.5.3 MDX path and could redirect valid classic foliage materials through the wrong replaceable texture resolution flow
- Current fix:
   - the direct-path replaceable fallback path is now gated behind `_isM2AdapterModel`
   - classic MDX keeps the older working behavior: replaceable resolution is only used when the MDX texture path itself is empty
- Validation status:
   - file-level diagnostics were clean for `ModelRenderer.cs`
   - alternate-OutDir build passed again: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
   - no automated tests were added or run
   - runtime real-data recheck is still required to confirm Alpha `0.5.3` foliage is back to the prior working state

### 0.5.3 MDX Wrap Semantics Regression Follow-up (Mar 19)

- The Alpha `0.5.3` foliage regression still reproduced after scoping the direct-path replaceable fallback back to M2 only.
- Stronger current renderer-side conclusion:
   - the later M2 wrap/clamp fix had also been applied in the shared `ModelRenderer` texture upload path for every model
   - that changed classic MDX sampler behavior from the previously working recovery-branch interpretation
   - for foliage cards, this is a plausible source of broad magenta tree canopies because transparent texels often carry magenta RGB and sampler edge behavior determines whether those texels bleed into visible leaf quads
- Current fix:
   - classic MDX now stays on the earlier working wrap/clamp interpretation in `ModelRenderer.LoadTextures()`
   - the newer wrap interpretation remains scoped to `_isM2AdapterModel`
- Validation status:
   - file-level diagnostics were clean for `ModelRenderer.cs`
   - alternate-OutDir build passed again: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
   - no automated tests were added or run
   - runtime real-data recheck is still required to confirm Alpha `0.5.3` foliage is back to the prior working state

### 0.5.3 MDX Transparent-Layer Alpha-Cutout Regression Follow-up (Mar 19)

- After the wrap fix, standalone `DuskwoodTree07.mdx` still reproduced with magenta canopy cards.
- A new non-UI `--probe-mdx` diagnostic path was added to `MdxViewer` so broken assets can be inspected directly against the real client data source without relying on the live GL viewer.
- Probe result for `World\Azeroth\Duskwood\PassiveDoodads\Trees\DuskwoodTree07.mdx` against `H:\053-client`:
   - the MDX parses correctly: canopy material is `Layer[0] TextureId=0 Blend=Transparent`
   - the canopy BLP decodes correctly: `DuskwoodTreeCanopy11.blp` contains substantial real alpha (`zero=20158`, `full=42316`, `translucent=3062`)
   - this ruled out profile routing, TEXS parsing, and basic BLP decode as the active cause for the remaining 0.5.3 tree failure
- Verified renderer-side root cause:
   - current `ModelRenderer.ShouldUseAlphaCutout(...)` had been generalized so `Layer 0 + Transparent` no longer used alpha-cutout when the texture had any translucent edge pixels
   - recovery-branch classic MDX behavior was simpler: `Layer 0 + Transparent` always rendered as alpha-cutout
   - for classic foliage textures with magenta RGB in low-alpha edge texels, downgrading them to regular blending is a plausible direct cause of the magenta canopy bleed seen in Duskwood trees
- Current fix:
   - classic MDX now restores the recovery behavior in `ShouldUseAlphaCutout(...)`: `Layer 0 + Transparent` always uses alpha-cutout
   - the newer alpha-kind-sensitive heuristic remains scoped to `_isM2AdapterModel`
- Validation status:
   - file-level diagnostics were clean for `ModelRenderer.cs`
   - alternate-OutDir build passed again: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
   - no automated tests were added or run
   - user runtime validation now confirms Alpha `0.5.3` MDX rendering is back to the expected working state
   - remaining model-family rendering issues are now isolated to the pre-release `3.0.1` path

### Model Rendering Status Update (Mar 19)

- User runtime validation now confirms the classic Alpha `0.5.3` MDX regression is fixed.
- Current state after the renderer rollback / probe-guided repair:
   - classic `0.5.3` MDX foliage and transparency behavior are back to the expected working state
   - the non-UI `--probe-mdx` path remains available for direct asset triage against real client data
   - pre-release `3.0.1` rendering is still buggy and should not be described as solved or broadly compatible yet
- Working conclusion for future follow-up:
   - do not re-open classic MDX parser or generic texture-loading suspicion first
   - treat remaining rendering defects as pre-release `3.0.1` parser / material / texture-binding work unless fresh runtime evidence shows a new classic regression

### Standalone Data-Source M2 Read-Path Fix (Mar 19)

- Follow-up after user report that every standalone/browser-loaded M2 showed `Failed to read`.
- Current conclusion:
   - that specific message comes from `ViewerApp.LoadFileFromDataSource(...)`, before any M2 parser or profile guard runs
   - the browser path was still using exact `_dataSource.ReadFile(virtualPath)` semantics instead of the canonical model-path recovery already used deeper in standalone M2 loading
- Current fix:
   - `.mdx` / `.mdl` / `.m2` file-browser loads now resolve through `ResolveStandaloneCanonicalModelPath(...)` and `ReadStandaloneFileData(...)` before failing
   - resolved paths are now carried into the later container-probe stage so M2-family aliases reach the actual parser path
- Build validation passed:

### Standalone Alias Recovery + Unsuffixed Skin Candidates (Mar 19)

- Follow-up after new runtime logs still showed:
   - `DataSourceRead` failure on a standalone/browser `.mdx` alias path that did not resolve through the narrower standalone lookup
   - `Missing companion .skin for M2` on pre-release `3.0.1` model loads where the numbered `00`-`03` guesses may be too narrow
- Current standalone-path changes:
   - `ResolveStandaloneCanonicalModelPath(...)` now uses the same broader candidate family as the world loader: exact path, extension aliases, bare filename aliases, and `Creature\Name\Name.{mdx|m2|mdl}` guesses
   - standalone reads now also probe those guessed candidates directly through `FileExists` / `ReadFile`, so recovery is not blocked only because the MPQ file index is incomplete
   - shared `BuildSkinCandidates(...)` now tries unsuffixed `.skin` alongside the numbered forms
- Keep claims narrow:
   - this improves path and companion-file discovery
   - it does not prove the remaining pre-release `.skin` structure assumptions are fully solved

### Cocoon Optional-Span Parser Follow-up (Mar 19)

- New runtime evidence from `Creature\Cocoon\Cocoon.mdx` narrowed the next parser issue:
   - the loader now reaches the profiled pre-release `MD20` parser
   - failure came from a `colors` span (`0x2C` family) being out of range on a `0x106` file, before the viewer attempted the geometry tables it actually needs
- Current parser adjustment:
   - only required runtime spans remain fatal in `ParseProfiledMd20Model(...)`
   - optional / unresolved families now use a nonfatal validator that logs and skips invalid spans
   - embedded texture-name spans are also optional now
- Keep the interpretation precise:
   - this is not proof that the legacy header mapping is fully correct
   - it is a surgical reduction of false rejects while the unresolved pre-release families are still being mapped
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime follow-up is still needed to see the next real blocker once the read-path failure is gone.

### Ghidra Verification: 3.0.1 Root Profiles, Not Proven External `.skin` (Mar 19)

- Latest `wow.exe 3.0.1.8303` Ghidra trace tightened the active blocker around standalone `NightElfMale.m2`:
   - `FUN_0077d3c0` normalizes accepted model-family requests to `.m2` and performs the confirmed primary file open
   - the traced `CM2Shared` load path then stays on the in-memory `MD20` blob through `FUN_0079bc70` -> `FUN_0079bb30` -> `FUN_0079a8c0` -> `FUN_007988c0`
   - no second external `.skin` file open was confirmed on that traced path
- New high-confidence structure note:
   - `FUN_007988c0` selects a root-contained `0x2C` profile record and stores it at `param_1 + 0x13C`
   - `FUN_00797D20` builds vertex buffers from that selected root profile
   - `FUN_00797AD0` builds index buffers from that selected root profile
   - `FUN_00797A40` shows the root `0x2C` profile header contains typed spans with strides `0x02`, `0x02`, `0x04`, `0x30`, and `0x18`, plus a selector at `+0x28`
- Practical consequence for current viewer work:
   - `Missing companion .skin for M2` was too strong for traced pre-release `3.0.1.8303` failures
   - the real unresolved gap is root-contained profile parsing / geometry-material extraction in `WarcraftNetM2Adapter`, not just companion-file discovery
   - keep claims narrow: this does not yet prove every `3.0.1` caller or every model variant avoids external companion files

### Embedded 3.0.1 Root-Profile Fallback (Mar 19)

- Current implementation pass lifted the traced root-contained profile path into `WarcraftNetM2Adapter`:
   - `BuildRuntimeModel(...)` now accepts a nullable skin payload and can fall back to embedded model-side profile geometry
   - profiled `MD20` parsing now reads vertices from the traced root `0x30` table, reads bounds from the traced root bounds block, and attempts to parse the root `0x2C` profile family at `0x4C`
   - selected root profiles are converted into the adapter's `SkinData` shape using:
      - vertex remap table (stride `0x02`)
      - triangle-index table (stride `0x02`)
      - submesh table (stride `0x30`)
      - optional batch table (stride `0x18`)
- Fallback wiring now exists in all three active M2-family load paths when no external `.skin` resolves:
   - `ViewerApp`
   - `WorldAssetManager`
   - `WmoRenderer`
- Metadata handling remains conservative:
   - textures / render flags / texture lookups are still supplemented opportunistically from Warcraft.NET when available
   - material extraction for the embedded pre-release root profiles is still incomplete, so current root-profile loads may render with fallback material assignment rather than final 3.0.1 section-material parity
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passes after the new fallback path
   - real runtime validation on actual `3.0.1` assets is still pending

### Pre-release 3.0.1 M2 + Shared Pink Transparency (Mar 18)

- User runtime verification now suggests the remaining model-format failures are concentrated in the pre-release `3.0.1` family, not the later `3.3.5` model family.
- Treat pre-release `3.0.1` as a separate compatibility problem:
   - possible hybrid / transitional `MDX` + `M2` semantics
   - do not assume later `MD20` + `.skin` behavior is enough
   - keep `FormatProfileRegistry` / profile-routed model handling as the likely next implementation path
- Separate shared rendering defect still open:
   - neon-pink transparent surfaces remain visible on both `MDX` and M2-family models
   - that points to shared material, texture, blend, or shader behavior rather than only the pre-release parser path
- Next practical split:
   1. model-structure compatibility work for pre-release `3.0.1`
   2. shared transparent-material parity work in renderer/shader code

### Pre-release 3.0.1 wow.exe Guide Handoff (Mar 19)

- Latest Ghidra pass now has a documented viewer-facing handoff in `documentation/pre-release-3.0.1-m2-wow-exe-guide.md`.
- High-confidence facts from `wow.exe` build `3.0.1.8303`:
   - common loader chain is `FUN_0077e2c0` -> `FUN_0077d3c0` -> `FUN_0079bc70` -> `FUN_0079bc50` -> `FUN_0079bb30` -> `FUN_0079a8c0`
   - root gate is `MD20` with version `0x104..0x108`
   - parser layout splits at `0x108`
   - shared span validators use strides `1`, `2`, `4`, `8`, `0x0C`, `0x30`, and `0x44`
   - confirmed nested record families include `0x70`, `0x2C`, `0x38`, `0xD4`, `0x7C`
   - legacy split uses `0xDC` + `0x1F8`; later split uses `0xE0` + `0x234`
- Fresh-chat prompts added for follow-up work:
   - `.github/prompts/pre-release-3-0-1-m2-implementation-plan.prompt.md`
   - `.github/prompts/pre-release-3-0-1-m2-ghidra-followup.prompt.md`
   - `.github/prompts/pre-release-3-0-1-m2-runtime-triage.prompt.md`
- Scope reminder:
   - this is documentation for Track A implementation, not proof that runtime support is now complete
   - Track B pink transparency still needs separate renderer/material work

## Current Focus

**v0.4.0 Release — 0.5.3 Rendering Improvements + Initial 3.3.5 Groundwork** — Major rendering improvements for Alpha 0.5.3 (lighting, particles, geoset animations). Initial 3.3.5 WotLK support scaffolding added but **NOT ready for use** — MH2O liquid and terrain texturing are broken. Only client versions 0.5.3 through 0.12 are currently usable.

## 3.3.5 WotLK Status: IN PROGRESS (NOT USABLE)

**Known broken:**
- MH2O liquid rendering — parsing exists but rendering is broken
- Terrain texturing — alpha map decode not working correctly for LK format
- These must be fixed before 3.3.5 data can be used

## Immediate Next Steps

1. **Fix 3.3.5 MH2O liquid rendering** — Parsing exists but output is broken
2. **Fix 3.3.5 terrain texturing** — Alpha map decode for LK format not working
3. **3.3.5 terrain alpha maps** — Current LK path uses basic Mcal decode; needs full `AlphaMapService` integration without breaking 0.5.3
4. **Light.dbc / LightData.dbc integration** — Replace hardcoded TerrainLighting values with real game lighting data per zone
5. **Skybox rendering** — Minimal backdrop routing is now implemented; real-data runtime verification is still pending
6. **Ribbon emitters (RIBB)** — Parsed but no rendering code yet
7. **M2 particle emitters** — WarcraftNetM2Adapter doesn't map PRE2/particles to MdxFile format yet

## Session 2026-02-13 Summary — WDL/WL/WMO Fixes

### Completed

1. **WDL parser correctness**
   - Strict chunk parsing (`MVER`/`MAOF`/`MARE`) with version `0x12` validation
   - Proper `MARE` chunk header handling before height reads

2. **WDL terrain scale + overlay behavior improvements**
   - WDL cell size corrected to `WoWConstants.TileSize` (8533.3333), not chunk size
   - Existing ADT-loaded tiles hidden from WDL at load-time
   - Polygon offset added to reduce z-fighting with real terrain
   - UI toggle added to fully disable WDL rendering for testing

3. **WDL preview reliability**
   - `.wdl.mpq` fallback path and error propagation (`LastError`)
   - Preview dialog now displays failure reason instead of closing silently

4. **WMO intermittent non-rendering fix**
   - Converted WMO main + liquid shader programs to shared static programs with ref-counted lifetime
   - Prevents per-instance shader deletion race (same class of bug previously fixed in MDX renderer)

5. **WL liquids transform tooling**
   - Replaced hardcoded axis swap with configurable matrix transform (rotation + translation)
   - Added `WL Transform Tuning` controls in UI and `Apply + Reload WL`
   - Added `WorldScene.ReloadWlLiquids()` for fast iteration

## MDX Particle System — IMPLEMENTED (2026-02-15)

Previously deferred issue now resolved. ParticleRenderer rewritten with per-particle uniforms, texture atlas support, and per-emitter blend modes. Wired into MdxRenderer — emitters created from PRE2 data, updated each frame with bone-following transforms, rendered during transparent pass. Fire, glow, and spell effects now visible.

## Session 2026-02-15 Summary — Multi-Version Support + Lighting/Particle Overhaul

### Completed

1. **Partial WotLK 3.3.5 terrain scaffolding** (StandardTerrainAdapter) — **NOT USABLE**
   - Split ADT file loading via MPQ data source
   - MPHD flags detection for `bigAlpha` (0x4)
   - MH2O liquid chunk parsing — **BROKEN, not rendering correctly**
   - LK alpha maps via `hasLkFlags` detection — **texturing BROKEN**
   - Surgical revert of shared renderer code was needed to restore 0.5.3

2. **M2 (MD20) model loading** (WarcraftNetM2Adapter)
   - Converts MD20 format models to MdxFile runtime format
   - Maps render flags (Unshaded, Unfogged, TwoSided), blend modes
   - Texture loading from M2 texture definitions
   - Bone/animation data mapping

3. **Terrain regression fix** (surgical revert)
   - Commit e172907 broke 0.5.3 terrain rendering (grid pattern artifacts)
   - Root cause: `AlphaTextures.ContainsKey` guard skipping overlay layers + edge fix removal in TerrainRenderer.cs
   - Plus StandardTerrainAdapter ExtractAlphaMaps rewrite with broken `spanSuggestsPacked` logic
   - Surgical revert restored 0.5.3 terrain while preserving M2/WMO improvements

4. **Lighting improvements** (TerrainLighting, ModelRenderer, WmoRenderer)
   - Raised ambient values: day (0.4→0.55), night (0.08→0.25) — no more pitch black
   - Half-Lambert diffuse shading: `dot * 0.5 + 0.5` squared — wraps light around surfaces
   - WMO shader: replaced lossy scalar lighting `(r+g+b)/3.0` with proper `vec3` lighting
   - MDX shader: half-Lambert + reduced specular (0.3→0.15)
   - Moderated day directional light (1.0→0.8) to avoid blow-out with higher ambient

5. **Particle system wired into pipeline** (ParticleRenderer, ModelRenderer)
   - Rewrote ParticleRenderer: per-particle uniforms, texture atlas (rows×columns), per-emitter blend mode
   - MdxRenderer creates ParticleEmitter instances from MdxFile.ParticleEmitters2
   - Emitter transforms follow parent bone matrices when animated
   - Particles rendered during transparent pass after geosets
   - Supports Additive, Blend, Modulate, AlphaKey filter modes

6. **Geoset animation alpha** (ModelRenderer)
   - `UpdateGeosetAnimationAlpha()` evaluates ATSQ alpha keyframe tracks per frame
   - Alpha multiplied into layer alpha during RenderGeosets
   - Geosets with alpha ≈ 0 skipped entirely
   - Supports global sequences and linear interpolation

7. **WMO fixes from 3.3.5 work** (preserved)
   - Multi-MOTV/MOCV chunk handling for ICC-style WMOs
   - Strict WMO validation preventing Northrend loading hangs
   - WMO liquid rotation fixes

### Files Modified
- `TerrainRenderer.cs` — Reverted edge fix + ContainsKey guard
- `StandardTerrainAdapter.cs` — Reverted ExtractAlphaMaps to clean hasLkFlags path
- `TerrainLighting.cs` — Raised ambient/light values, better night visibility
- `ModelRenderer.cs` — Half-Lambert shader, particle wiring, geoset animation alpha
- `WmoRenderer.cs` — vec3 lighting instead of scalar, half-Lambert diffuse
- `ParticleRenderer.cs` — Complete rewrite with working per-particle rendering
- `WarcraftNetM2Adapter.cs` — MD20→MdxFile adapter (from e172907, preserved)
- `WorldAssetManager.cs` — MD20 detection + adapter routing (from e172907, preserved)

## Session 2026-02-13 Summary — MDX Animation System Complete

### Three Bugs Fixed

1. **KGRT Compressed Quaternion Parsing** (`MdxFile.cs`, `MdxTypes.cs`)
   - Rotation keys use `C4QuaternionCompressed` (8 bytes packed), not float4 (16 bytes)
   - Ghidra-verified decompression: 21-bit signed components, W reconstructed from unit norm
   - Added `C4QuaternionCompressed` struct with `Decompress()` method

2. **Animation Never Updated** (`ModelRenderer.cs`, `ViewerApp.cs`)
   - `ViewerApp` called `RenderWithTransform()` directly, bypassing `Render()` which was the only place `_animator.Update()` was called
   - Fix: Extracted `UpdateAnimation()` as public method, called from ViewerApp before render

3. **PIVT Chunk Order — All Pivots Were (0,0,0)** (`MdxFile.cs`)
   - PIVT chunk comes AFTER BONE in MDX files. Inline pivot assignment during `ReadBone()` found 0 pivots
   - Fix: Deferred pivot assignment in `MdxFile.Load()` after all chunks are parsed
   - This caused "horror movie" deformation — bones rotating around world origin instead of joints

### Terrain Animation Added (`WorldScene.cs`)
- Added `UpdateAnimation()` calls for all unique MDX renderers before opaque/transparent render passes
- Uses `HashSet<string>` to ensure each renderer is updated exactly once per frame

### Other Improvements
- `MdxAnimator`: `_objectIdToListIndex` dictionary replaces O(n) `IndexOf` calls
- `GNDX`/`MTGC` chunks now stored in `MdlGeoset` for vertex-to-bone skinning
- MATS values remapped from ObjectIds to bone list indices via dictionary lookup

### Key Architecture (MDX Animation)
- `MdxAnimator` — Evaluates bone hierarchy per-frame, stores matrices in `_boneMatrices[]` by list position
- `ModelRenderer.UpdateAnimation()` — Public method to advance animation clock
- `BuildBoneWeights()` — Converts GNDX/MTGC/MATS to 4-bone skinning format
- Bone transform: `T(-pivot) * S * R * T(pivot) * T(translation) * parentWorld`
- Shader: `uBones[128]` uniform array, vertex attributes for bone indices + weights

### Files Modified
- `MdxTypes.cs` — Added `C4QuaternionCompressed` struct
- `MdxFile.cs` — Fixed `ReadQuatTrack`, stored GNDX/MTGC, deferred pivot assignment
- `MdxAnimator.cs` — `_objectIdToListIndex` dict, cleaned diagnostics
- `ModelRenderer.cs` — Extracted `UpdateAnimation()`, ObjectId→listIndex remapping in `BuildBoneWeights`
- `ViewerApp.cs` — Added `mdxR.UpdateAnimation()` before standalone MDX render
- `WorldScene.cs` — Added per-frame animation update for unique MDX doodad renderers

## Session 2026-02-09 Summary

### WMO v16 Root File Loading Investigation
- **Symptom**: WMO v16 root files (e.g., `Big_Keep.wmo`) fail to load with "Failed to read" — group files load but without textures/lighting
- **Root cause chain**: `MpqDataSource.ReadFile` → `NativeMpqService.ReadFile` → `FindFileInArchive` succeeds → `ReadFileFromArchive` returns null
- **Block info**: offset=435912, size=318 (compressed), fileSize=472 (decompressed), flags=0x80000200 (EXISTS|COMPRESSED)
- **Decompression failure**: Compression type byte = `0x08` (PKWARE DCL), but remaining data has dictShift=0 (expected 4/5/6)
- **0.6.0 MPQ structure**: All files in standard MPQ archives (`wmo.MPQ`, `terrain.MPQ`, etc.) — NOT loose files, NOT per-asset `.ext.MPQ` wrappers

### Key Findings About 0.6.0 MPQs
- 11 MPQ archives: base, dbc, fonts, interface, misc, model, sound, speech, terrain(2331), texture(33520), wmo(4603)
- All have internal listfiles (56573 total files extracted)
- Zlib (0x02) works fine for large files (groups extract correctly)
- PKWARE DCL (0x08) fails for small files (root WMOs, possibly some ADTs)
- `FLAG_COMPRESSED (0x200)` = per-sector compression with type byte prefix
- `FLAG_IMPLODED (0x100)` = whole-file PKWARE without type byte (not seen in these archives)

### StormLib Reference Code Available
- `lib/StormLib/src/pklib/explode.c` — Complete PKWARE DCL explode implementation
- `lib/StormLib/src/pklib/pklib.h` — Data structures (`TDcmpStruct`, lookup tables)
- `lib/StormLib/src/SCompression.cpp` — Decompression dispatch (`Decompress_PKLIB`, `SCompDecompress`)
- Key: `explode()` reads bytes 0,1 as ctype/dsize_bits, byte 2 as initial bit buffer, position starts at 3

### WMO Liquid Rendering Added
- MLIQ chunk now parsed in `ParseMogp` sub-chunk switch
- `WmoRenderer` has liquid mesh building + semi-transparent water surface rendering
- Diagnostic logging added for failed material textures

### Ghidra RE Prompts Written
- `specifications/ghidra/prompt-053-mpq.md` — 0.5.3 MPQ implementation (HAS PDB — best starting point)
- `specifications/ghidra/prompt-060-mpq.md` — 0.6.0 MPQ decompression (no PDB, use string refs)

### Files Modified This Session
- `NativeMpqService.cs` — Added diagnostic logging throughout ReadFile/ReadFileFromArchive/ReadFileData/DecompressData
- `MpqDataSource.cs` — Added diagnostic logging to ReadFile and TryResolveLoosePath
- `WmoV14ToV17Converter.cs` — Added diagnostic logging to ParseWmoV14Internal
- `WmoRenderer.cs` — Added WMO liquid rendering, material texture diagnostics
- `PkwareExplode.cs` — New file, PKWARE DCL decompression (needs fixing — current impl fails)
- `AlphaMpqReader.cs` — Wired up PkwareExplode for 0x08 compression
- `StandardTerrainAdapter.cs` — Added ADT loading diagnostics

## Session 2026-02-08 (Late Evening) Summary

### Standard WDT+ADT Support
- **ITerrainAdapter interface** — New common contract for all terrain adapters
- **StandardTerrainAdapter** — Reads LK/Cata WDT (MAIN/MPHD) + split ADT files from MPQ via IDataSource
- **TerrainManager refactored** — Accepts `ITerrainAdapter` (was hardcoded to `AlphaTerrainAdapter`)
- **WorldScene refactored** — New constructor accepts pre-built `TerrainManager`
- **ViewerApp detection** — File size ≥64KB → Alpha WDT, <64KB → Standard WDT (requires MPQ data source)

### Format Specifications Written
- `specifications/alpha-053-terrain.md` — Definitive WDT/ADT/MCNK/MCVT/MCNR/MCLY/MCAL/MCSH/MDDF/MODF spec
- `specifications/alpha-053-coordinates.md` — Complete coordinate system documentation
- `specifications/unknowns.md` — 13 prioritized format unknowns needing Ghidra investigation

### Ghidra LLM Prompts Created
- `specifications/ghidra/prompt-053.md` — 0.5.3 (HAS PDB! Best starting point)
- `specifications/ghidra/prompt-055.md` — 0.5.5 (diff against 0.5.3)
- `specifications/ghidra/prompt-060.md` — 0.6.0 (transitional format detection)
- `specifications/ghidra/prompt-335.md` — 3.3.5 LK (reference build, well-documented)
- `specifications/ghidra/prompt-400.md` — 4.0.0 Cata (split ADT introduction)

### Converter Master Plan
- `memory-bank/converter_plan.md` — 4-phase plan: LK model reading → format conversion → PM4 world support with CK24 aggregation and coordinate validation → unified project

## Session 2026-02-08 (Evening) Summary

### What Was Fixed

#### MCSH Shadow Blending (TerrainRenderer.cs)
- **Problem**: Shadow map (MCSH) was only applied on the base terrain layer. Alpha-blended overlay texture layers drawn on top would cover/wash out the shadows.
- **Root cause**: Both the C# render code and GLSL shader had `isBaseLayer` guards on shadow binding/application.
- **Fix**: Removed `isBaseLayer` condition from both:
  - C# `RenderChunkPass()`: Changed `bool hasShadow = isBaseLayer && chunk.ShadowTexture != 0` → `bool hasShadow = chunk.ShadowTexture != 0`
  - GLSL fragment shader: Changed `if (uShowShadowMap == 1 && uIsBaseLayer == 1 && uHasShadowMap == 1)` → `if (uShowShadowMap == 1 && uHasShadowMap == 1)`
- **Result**: Shadows now darken all texture layers consistently.

#### MDX Bounding Box Pivot Offset (WorldScene.cs, WorldAssetManager.cs)
- **Problem**: MDX model geometry is offset from origin (0,0,0). The MODL bounding box describes where geometry actually sits. MDDF placement position targets origin, but geometry center is elsewhere, causing models to appear displaced.
- **Fix**: Pre-translate geometry by negative bounding box center before scale/rotation/translation:
  - Added `WorldAssetManager.TryGetMdxPivotOffset()` — returns `(BoundsMin + BoundsMax) * 0.5f`
  - Transform chain: `pivotCorrection * mirrorX * scale * rotX * rotY * rotZ * translation`
  - `pivotCorrection = Matrix4x4.CreateTranslation(-pivot)`
  - Applied in both `BuildInstances()` and `OnTileLoaded()` in WorldScene.cs
- **WMO models**: Do NOT need pivot correction — their geometry is already correctly positioned relative to origin.

#### VLM Terrain Rendering (Previous session, 2026-02-08 afternoon)
- **GLSL shader em-dash**: Replaced unicode em-dash with ASCII hyphen in shader comment.
- **NullReferenceException**: Fixed null-conditional access in `DrawTerrainControls`.
- **VLM coordinate conversion**: Fixed `WorldPosition` in `VlmProjectLoader.cs` — swapped posX/posY, removed MapOrigin subtraction.
- **Minimap for VLM projects**: Refactored `DrawMinimap()` to work with either `_terrainManager` or `_vlmTerrainManager`. Added `IsTileLoaded()` to `VlmTerrainManager`.

#### Async Tile Streaming (TerrainManager.cs, VlmTerrainManager.cs)
- Both terrain managers now queue tile parsing to `ThreadPool` background threads.
- Parsed `TileLoadResult` objects enqueued to `ConcurrentQueue`.
- `SubmitPendingTiles()` runs on render thread each frame, uploading max 2 tiles/frame to avoid GPU stalls.
- `_disposed` flag prevents background threads from accessing disposed resources.

#### Thread Safety (VlmProjectLoader.cs, AlphaTerrainAdapter.cs, TerrainRenderer.cs)
- `TileTextures` → `ConcurrentDictionary` in both adapters.
- `_placementLock` protects dedup sets (`_seenMddfIds`, `_seenModfIds`) and placement lists in both adapters.
- `TerrainRenderer.AddChunks()` parameter widened from `Dictionary` to `IDictionary` to accept both.

#### VLM Dataset Generator (ViewerApp.cs)
- New menu item: `File > Generate VLM Dataset...`
- Dialog UI: client path (folder picker), map name, output dir, tile limit, progress log.
- Runs `VlmDatasetExporter.ExportMapAsync()` on `ThreadPool` with `IProgress<string>` feeding real-time log.
- "Open in Viewer" button after export completes.

#### Loose Map Overlay Workflow (ViewerApp.cs, MpqDataSource.cs, MapDiscoveryService.cs) (Mar 19, 2026)
- Base 3.3.5 MPQ clients can now be extended with loose custom-map content after initial load.
- Workflow:
   - `File > Open Game Folder (MPQ)...`
   - `File > Attach Loose Map Folder...`
- Supported overlay expectation:
   - selected folder contains `World\Maps\...` directly, or is itself under `World\Maps\<mapDir>`
- `MpqDataSource` overlay behavior:
   - overlay roots are indexed into the same virtual-path file set used by terrain/model loading
   - loose overlay scan now includes `.wdt`, `.adt`, `.pm4`, `.wlw`, `.wlq`, and `.wlm` in addition to existing model/texture extensions
   - raw-byte read cache is cleared on overlay attach so old misses do not hide newly added files
- `MapDiscoveryService` behavior:
   - loose `World\Maps\<dir>\<dir>.wdt` paths are merged into the discovered map list even when no `Map.dbc` row exists
   - custom loose maps are shown with synthetic IDs / custom labels in the UI
   - custom loose maps intentionally skip `Map.dbc` lighting IDs
- Scope boundary:
   - this slice initially improved loading/discovery for converted loose maps and PM4 sidecars
   - follow-up PM4 viewer rendering work now exists in `WorldScene`/`ViewerApp` (see Mar 20 PM4 overlay note below)
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
   - no automated tests were added or run
   - no runtime real-data validation has been completed yet for this loose overlay workflow

#### PM4 Overlay Diagnostics + Grouping/Winding Iteration (WorldScene.cs, ViewerApp.cs) (Mar 20, 2026)
- PM4 sidecars now render in-viewer as a debug overlay instead of only being indexed/discovered.
- Added PM4 visualization controls:
   - color-by mode (`CK24` type/object/key, tile, dominant group key, dominant attribute mask, height)
   - optional solid overlay + wireframe edge overlay
   - optional 3D pins for `MPRL` refs and PM4 object centroids
- Added PM4 grouping controls for disjoint geometry:
   - split CK24 groups by shared-vertex connectivity
   - optional split by dominant `MSUR.MdosIndex` before connectivity split
- Added PM4 orientation/winding diagnostics path:
   - per-object planar transform solve (swap/invert U/V candidates, scored against nearest `MPRL` refs)
   - winding parity correction flips triangle index order when chosen transform mirrors orientation
   - selected-object panel now shows dominant group key, attribute mask, `MdosIndex`, and planar/winding flags
- Scope boundary:
   - this is still viewer-side debug reconstruction, not a finalized cross-tile PM4 object identity contract
   - CK24 aggregation across full map and MSCN semantics remain open beyond this slice
- Validation status for this PM4 slice:
   - repeated `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
   - no automated tests were added or run
   - runtime real-data visual signoff is still pending for the merged/disjoint object edge cases

#### Split ADT Auto-Promotion For Loose Maps (StandardTerrainAdapter.cs) (Mar 19, 2026)
- `StandardTerrainAdapter` now detects `*_tex0.adt` / `*_obj0.adt` companions from the actual tile set before locking the ADT profile.
- If the loaded base client build resolves to a non-split terrain profile but the map data is visibly split, the adapter promotes only the terrain parser to provisional `AdtProfile_40x_Unknown`.
- Scope boundary:
   - this keeps `_dbcBuild` unchanged for model, WMO, and DBC-driven systems
   - the goal is to let a 3.3.5 base client load loose 4.x+ split terrain without reclassifying the whole client as 4.x
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` still required after this slice
   - no runtime real-data validation has been completed yet for `test_data/development/World/Maps/development`

#### Development Map Tile Coverage Follow-up (StandardTerrainAdapter.cs) (Mar 19, 2026)
- Real-data check on `test_data/development/World/Maps/development` showed the loose map files and `development.wdt` disagree materially:
   - WDT `MAIN` advertises 1496 tiles
   - loose files on disk cover 613 tile coordinates across root / `_obj0` / `_tex0`
   - only 352 root ADTs are both present on disk and flagged by `MAIN`
- Active viewer consequence before the fix:
   - 114 root filenames on disk were zero-byte placeholders paired with `_obj0` / `_tex0`, so they could not contribute terrain geometry through the root-ADT path
   - 147 `_obj0` tiles without a root ADT returned early and lost their placements entirely
- Current adapter behavior:
   - tile discovery now merges `MAIN` with indexed loose split-ADT filenames for the current map and drops `MAIN` entries that have no backing tile files
   - rootless `_obj0` tiles now still load placement data even when no terrain root exists
- Current dataset interpretation:
   - tiles that load as terrain are the 352 non-empty root ADTs
   - many of the remaining “real Blizzard” split tiles appear to be placement/texture sidecars around zero-byte root placeholders, so they need a different terrain source (for example WDL-derived geometry) if the goal is to render ground there
- Validation status:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after this slice
   - no automated tests were added or run
   - no runtime real-data validation has been completed yet for the updated development-map load path

### Key Technical Decisions
- **Coordinate system**: Renderer X = WoW Y, Renderer Y = WoW X, Z = height. MapOrigin = 17066.66666f, ChunkSize = 533.33333f.
- **MDX pivot**: Bounding box center, NOT PIVT chunk (PIVT is for per-bone skeletal animation pivots).
- **Shadow blending**: Apply to ALL layers, not just base. Overlay layers must also be darkened.
- **Thread safety**: `ConcurrentDictionary` for shared tile data, `lock` for placement dedup sets.

## What Works

| Feature | Status |
|---------|--------|
| Alpha WDT terrain rendering + AOI | ✅ |
| **Standard WDT+ADT terrain (WotLK 3.3.5)** | ✅ Partial — terrain + M2 models + WMO loading |
| Terrain MCSH shadow maps | ✅ (all layers, not just base) |
| Terrain alpha map debug view | ✅ (Show Alpha Masks toggle) |
| Async tile streaming | ✅ (background parse, render-thread GPU upload) |
| Standalone MDX rendering | ✅ (MirrorX, front-facing) |
| MDX skeletal animation | ✅ (standalone + terrain, compressed quats, GPU skinning) |
| MDX pivot offset correction | ✅ (bounding box center pre-translation) |
| MDX doodads in WorldScene | ✅ Position + animation + particles working |
| WMO v14 rendering + textures | ✅ (BLP per-batch) |
| WMO v17 rendering | ✅ Partial (groups + textures, multi-MOTV/MOCV) |
| M2 model rendering | ✅ MD20→MdxFile adapter (WarcraftNetM2Adapter) |
| Particle effects (PRE2) | ✅ Billboard quads, texture atlas, bone-following |
| Geoset animation alpha (ATSQ) | ✅ Per-frame keyframe evaluation |
| WMO rotation/facing in WorldScene | ✅ |
| WMO doodad sets | ✅ |
| MDDF/MODF placements | ✅ (position + pivot correct) |
| Bounding boxes | ✅ (actual MODF extents) |
| VLM terrain loading | ✅ (JSON dataset → renderer) |
| VLM minimap | ✅ |
| VLM dataset generator | ✅ (File > Generate VLM Dataset) |
| Live minimap + click-to-teleport | ✅ (WDT + VLM) |
| AreaPOI system | ✅ |
| GLB export (Z-up → Y-up) | ✅ |
| Object picking/selection | ✅ |
| Format specifications | ✅ (specifications/ folder) |
| WMO liquid rendering (MLIQ) | ✅ (semi-transparent water surfaces) |
| Object picking/selection | ✅ (ray-AABB, highlight, info) |
| Camera world coordinates | ✅ (WoW coords in status bar) |
| Left/right sidebar layout | ✅ (docked panels) |
| Ghidra RE prompts (5+2 versions) | ✅ (specifications/ghidra/) |
| 0.6.0 MPQ file extraction | ❌ PKWARE DCL (0x08) decompression fails |
| Half-Lambert lighting | ✅ Softer shading on MDX + WMO models |
| Improved ambient lighting | ✅ Day/night cycle with WoW-like brightness |

## Key Files

- `Terrain/WorldScene.cs` — Object instance building, pivot offset, rotation transforms, rendering loop
- `Terrain/WorldAssetManager.cs` — Model loading, bounding box/pivot queries
- `Terrain/AlphaTerrainAdapter.cs` — MDDF/MODF parsing, coordinate conversion, thread-safe placement dedup
- `Terrain/VlmProjectLoader.cs` — VLM JSON tile loading, thread-safe TileTextures/placements
- `Terrain/VlmTerrainManager.cs` — VLM terrain AOI, async streaming
- `Terrain/TerrainManager.cs` — WDT terrain AOI, async streaming
- `Terrain/TerrainRenderer.cs` — Terrain shader, shadow maps on all layers, alpha maps, debug views
- `Rendering/WmoRenderer.cs` — WMO geometry, textures, doodad sets
- `Rendering/ModelRenderer.cs` — MDX rendering, MirrorX, blend modes, textures
- `ViewerApp.cs` — Main app, UI, DBC loading, minimap, VLM export dialog
- `Export/GlbExporter.cs` — GLB export with Z-up → Y-up conversion

## Dependencies (all already integrated)

- `MdxLTool` — MDX file parser
- `WoWMapConverter.Core` → `gillijimproject-csharp` — Alpha WDT/ADT/MCNK parsers, WMO v14 parser, VLM dataset export
- `SereniaBLPLib` — BLP texture loading
- `Silk.NET` — OpenGL + windowing + input
- `ImGuiNET` — UI overlay
- `DBCD` — DBC database access
