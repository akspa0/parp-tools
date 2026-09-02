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
