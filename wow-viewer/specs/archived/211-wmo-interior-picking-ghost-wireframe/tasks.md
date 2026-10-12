# Spec 211: Tasks — WMO Interior Ray Picking, Doodad Selection & Ghost Transparent Wireframes

## Phase 1: WMO Interior Ray Picking & Doodad Selection

- [x] 211-T101: Add `ObjectType.WmoDoodad` to `ObjectType` enum in [`src/viewer/WoWViewer/Terrain/WorldScene.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs).
- [x] 211-T102: Implement `TryPickDoodadsByRay` in [`src/viewer/WoWViewer/Rendering/WmoRenderer.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs) to test ray intersections against doodads using their local transforms and bounds.
- [x] 211-T103: Update `CollectSceneObjectPickHits` in [`src/viewer/WoWViewer/Terrain/WorldScene.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs) to test WMO doodads and implement WMO container fall-through (prioritizing interior objects over enclosing container WMO AABBs).
- [x] 211-T104: Wire `ObjectType.WmoDoodad` candidate handling, selection state, and inspector highlighting in [`src/viewer/WoWViewer/ViewerApp_ClickSelection.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/ViewerApp_ClickSelection.cs) and [`ViewerApp.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/ViewerApp.cs).
- [x] 211-T105: Gate: Compile and verify with `dotnet build WoWViewer.csproj -c Debug`.

## Phase 2: Ghost Transparent Wireframe Rendering

- [x] 211-T201: Implement dual-pass ghost wireframe (33% alpha fill + offset wireframe line overlay) in [`src/viewer/WoWViewer/Rendering/ModelRenderer.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs).
- [x] 211-T202: Implement dual-pass ghost wireframe (33% alpha fill + offset wireframe line overlay) in [`src/viewer/WoWViewer/Rendering/WmoRenderer.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs).
- [x] 211-T203: Implement dual-pass terrain wireframe (textured fill pass + offset wireframe line overlay) in [`src/viewer/WoWViewer/Terrain/TerrainRenderer.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/TerrainRenderer.cs).
- [x] 211-T204: Gate: Compile and verify clean build, zero errors, test suites green.

## Phase 3: Registration, Memory Bank & Verification

- [x] 211-T301: Register Spec 211 in [`specs/STATUS.md`](file:///I:/parp/parp-tools/wow-viewer/specs/STATUS.md).
- [x] 211-T302: Update [`memory-bank/activeContext.md`](file:///I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md).
- [x] 211-T303: Commit SpecKit files to git before implementation.
- [x] 211-T304: Complete implementation and verification.

## Phase 4: WMO Doodad Hover Tooltip (operator follow-up 2026-09-04)

- [x] 211-T401: Route the precise-ray hover path ([`WorldScene.TryBuildHoveredSceneInfoByRay`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs)) through the existing Spec 211 click picker (`CollectSceneObjectPickHits`) so WMO doodads participate in hover with the same container fall-through semantics as click.
- [x] 211-T402: Apply visibility gates (WMO/doodad toggles) before container fall-through and reuse [`WmoContainerFallThroughFilter`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.Runtime/World/WmoContainerFallThroughFilter.cs) instead of duplicating the rule.
- [x] 211-T403: Carry `ParentWmoIndex`/`ParentSourcePath` through [`HoveredAssetInfo`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs) and pass it in `SelectSceneObject` so "Left-click to inspect" selects the exact hovered doodad.
- [x] 211-T404: Render MODD definition index, MODN offset, active MODS doodad set, MODR group references, and parent WMO identity in the hover overlay; distinguish the transient active-set index from the stable MODD definition.
- [x] 211-T405: Add `ApplyFallThrough_PreservesWmoDoodadCandidateAndUnrelatedRayOrder` to [`WmoContainerFallThroughFilterTests`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/World/WmoContainerFallThroughFilterTests.cs).
- [x] 211-T406: Gate: focused Spec 211 tests 7/7 green; viewer Debug build 0 errors.
- [ ] 211-T407: Operator interactive verification — hover a doodad inside a placed WMO and confirm the tooltip shows set/group/parent context and left-click selects it.

## Phase 5: Doodad selection bounds & 3D selection aids (operator follow-up 2026-09-04)

- [x] 211-T501: [`WmoRenderer.TryGetDoodadLocalBounds`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs) exposes the doodad model's local geometry bounds; `RequestDoodadModelLoad` queues the doodad's model for deferred loading on selection so real bounds resolve within a few frames instead of a permanent centroid cube.
- [x] 211-T502: [`TryBuildSelectedWmoDoodadInstance`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs) now feeds `LocalBoundsMin/Max` + `SelectionLocalBounds*`/`SelectionBoundsResolved` from the model geometry so the selection overlay takes the oriented tight-box path through the doodad's world transform instead of an axis-aligned centroid cube.
- [x] 211-T503: Selection overlay draws 3D aids on selected WMO doodads: cyan origin octahedron + gold position pin at the MODD placement point, and an RGB (X/Y/Z) axis tripod through the placement transform, all sized from the box extents.
- [x] 211-T504: Placeholder bounds render with an ORANGE accent so an unloaded model's centroid box is never readable as the object's extent; resolves to real geometry + normal accent once the model streams in.
- [ ] 211-T505: Operator interactive verification — select a barrel in Ironforge with WMOs loaded and confirm the box wraps the barrel geometry, the pin/jewel/axes mark the placement, and the box snaps from orange placeholder to real bounds as the model loads.
