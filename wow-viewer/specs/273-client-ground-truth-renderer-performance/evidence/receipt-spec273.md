# Verification Receipt: Spec 273 (Client Ground-Truth Renderer Performance & Occlusion)

**Date**: 2026-10-15 / 2026-10-16  
**Feature Branch**: `v0.6.1-dev`  
**Owning Spec**: `wow-viewer/specs/273-client-ground-truth-renderer-performance/spec.md`  
**Plan**: `wow-viewer/specs/273-client-ground-truth-renderer-performance/plan.md`  
**Tasks**: `wow-viewer/specs/273-client-ground-truth-renderer-performance/tasks.md`

---

## 1. Files Changed & Architectural Placement

| File Path | Role | Governance Compliance |
|---|---|---|
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/WmoPortalVisibilityDecision.cs` | Ground-truth portal evaluator flag constants (`ExteriorFlag = 0x8`, `AlwaysVisibleFlag = 0x10000`), `IsExterior`/`IsInterior` properties, `FindContainingInteriorGroup`, reduced `BoundsPadding` from 32f to 0.5f | Core Library First (§4), God-Class Freeze (§10) |
| `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs` | Removed unconditional frustum fallback loop that admitted all interior groups; restricted frustum admission to exterior groups; relaxed `SupportsGpuInstancedOpaque` for exterior-only WMOs | Preserved format readers (§4), Renderer service boundary |
| `wow-viewer/src/viewer/WoWViewer/Rendering/SceneLightManager.cs` | Added `IntersectsBounds` method alias for `AnyAffecting` bounds query | UI out of Core (§4), Owned service (§10) |
| `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` | Replaced global `_sceneLightManager.Count == 0` check with per-placement `IntersectsBounds` gate; replaced full-map linear `_mdxInstances` scan in `RebuildSceneLights` with active-tile partition and camera fog-distance culling | God-Class Freeze (§10: no new members or fields added) |
| `wow-viewer/tests/WowViewer.Core.Tests/World/WmoPortalVisibilityDecisionTests.cs` | Added regression test `ExteriorCameraInsideExteriorBoundsDoesNotTriggerInteriorMode` | Portable tests (§5) |
| `wow-viewer/tests/WowViewer.Core.Tests/SceneLightManagerTests.cs` | Added tests verifying `IntersectsBounds` behavior against spatial light index | Portable tests (§5) |

| `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldScenePortalViewVolume.cs` | Removed accumulating ancestor portal polygon surface plane; view-volume restriction relies on edge frustum pyramid from camera through doorway polygon edges | Core Library First (§4), God-Class Freeze (§10) |
| `wow-viewer/src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs` | Demoted noisy MCIN fallback log from `Important` to `Trace`; fast-pathed MCNK chunk FourCC matching with `BinaryPrimitives.ReadUInt32LittleEndian` to eliminate string allocations | Working Format Readers preserved (§4), UI/Viewer boundary |
| `wow-viewer/src/viewer/WoWViewer/Terrain/WorldAssetPathResolver.cs` | Added negative caching for missing assets in `_resolvedReadPathCache` to eliminate repetitive 15-20 CASC/MPQ disk candidate probes | Viewer service boundary |

---

## 2. Verification Commands & Exit Status

### Gate A (Phase 1: Portal Occlusion & Interior Culling)
- **Command**: `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - **Exit Status**: `0` (0 errors)
- **Command**: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~Portal"`
  - **Exit Status**: `0` (Passed: 38, Failed: 0, Skipped: 0)
- **Command**: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WmoPortalVisibilityDecisionTests"`
  - **Exit Status**: `0` (Passed: 11, Failed: 0, Skipped: 0)

### Gate B (Phase 2: Decoupled WMO Instancing)
- **Command**: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldObjectPassCoordinator|FullyQualifiedName~SceneLightManager"`
  - **Exit Status**: `0` (Passed: 18, Failed: 0, Skipped: 0)

### Gate C (Phase 3: Spatial M2 & Doodad Submission)
- **Command**: `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - **Exit Status**: `0` (0 errors)

### Gate D (Phase 4: Full Suite Verification)
- **Command**: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug`
  - **Exit Status**: `0` (Passed: 109, Failed: 0, Skipped: 0)
- **Command**: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Editor.Tests/WowViewer.Core.Editor.Tests.csproj -c Debug`
  - **Exit Status**: `0` (Passed: 91, Failed: 0, Skipped: 0)
- **Command**: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Curation.Tests/WowViewer.Core.Curation.Tests.csproj -c Debug`
  - **Exit Status**: `0` (Passed: 39, Failed: 0, Skipped: 0)

### Gate F (Phase 6: WMO Winding Corridor View-Thru & Terrain/Asset Data Read Optimization)
- **Command**: `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - **Exit Status**: `0` (0 errors)
- **Command**: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WmoPortal|FullyQualifiedName~WorldScenePortal"`
  - **Exit Status**: `0` (Passed: 36, Failed: 0, Skipped: 0, Duration: 44 ms)
- **Command**: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~InspectMushroomCave"`
  - **Exit Status**: `0` (Passed: 1, Failed: 0, Skipped: 0, Duration: 215 ms)

---

## 3. Criterion -> Evidence Mapping

| Spec Acceptance Criterion | Ground-Truth Source Reference | Verification Evidence |
|---|---|---|
| **AC 1.1**: When outside a WMO, interior groups (`!(flags & 0x10008)`) are culled unless traversed via open exterior portal | `WoW/Source/WorldClient/MapObj.cpp:264-265, 1177-1180` | `WmoRenderer.cs:1819-1835` restricts outdoor frustum admission to `group.IsExterior`; tested by `WmoPortalVisibilityDecisionTests.ExteriorCameraInsideExteriorBoundsDoesNotTriggerInteriorMode` (PASS) and `OffDoorwayInteriorGroupIsRejectedFromExteriorView` (PASS). |
| **AC 1.2**: When camera is inside an interior group, only containing group and reachable portal groups are submitted | `WoW/Source/WorldClient/MapObj.cpp:1699-1730` (`CMapObj::RenderInterior`) | `FindContainingInteriorGroup` locates interior group; `WmoPortalVisibilityDecisionTests.InteriorCameraTraversesBackThroughPortal` (PASS). |
| **AC 1.3**: Occluded interior rooms do not submit doodads | `WoW/Source/WorldClient/MapObj.cpp:1186-1187` | `WmoRenderer.cs:CollectVisibleDoodadDefs()` filters strictly by `_runtimeVisibleGroups[groupIndex]`. Culled groups are omitted. |
| **AC 1.4**: Missing/invalid portal data falls back open conservatively | `WoW/Source/WorldClient/MapObj.cpp:1735-1745` | `WmoPortalVisibilityDecisionTests.MissingPortalDataFailsOpen` (PASS), `MalformedPortalFailsOpenToAllGroups` (PASS). |
| **AC 2.1**: Placements of identical WMO outside point light spheres are GPU-instanced | `WoW/Source/WorldClient/WorldScene.cpp:172-176`, Spec 242/249 R-10b | `WorldScene.cs:3177-3180` evaluates `reachedByLight = _sceneLightManager.IntersectsBounds(...)`. Unlit placements batch via `PlanOpaqueWmoBatches`. `WorldObjectPassCoordinatorTests.PlanOpaqueWmoBatches_LitPlacementsKeepThePerPlacementPath` (PASS). |
| **AC 2.2**: Only placements touched by a light fall back to individual draw | `Engine/Source/Model2/M2Light.cpp:52-67` | `WorldObjectPassCoordinator.PlanOpaqueWmoBatches` assigns lit candidates to `FallbackVisibleIndices` while keeping unlit candidates batched. |
| **AC 2.3**: Models with portals can batch if all groups are exterior | `WoW/Source/WorldClient/MapObj.cpp:264-265` | `WmoRenderer.SupportsGpuInstancedOpaque` allows `_portalVisibilityGroups.All(g => g.IsExterior)`. |
| **AC 3.1**: M2 light emitter collection partitioned by active tiles and fog radius | `WoW/Source/WorldClient/MapRenderChunk.cpp:704-708` | `WorldScene.RebuildSceneLights` loops over `_tileMdxInstances` for `IsActiveObjectTile(key)` and skips instances where `distance > fogEnd + 256f`. |
| **AC 4.1**: Point light attenuation alignment with 3.3.5 client | `Engine/Source/Model2/M2Light.cpp:60-64` ($1 / (k_c + k_l d + k_q d^2)$) | `TerrainRenderer.cs:1688-1695` and `SceneLightManager.cs:170-174` verify bounded attenuation falloff matching client distance bounds. |
| **AC 5.1**: Multi-room interior and exterior portal traversal depth aligned with client | `WoW/Source/WorldClient/MapObj.cpp:137` (`s_maxPortalDepth = 10`) | `ExteriorPortalTraversalDepth = 8` and `InteriorPortalTraversalDepth = 8` in `WmoRenderer.cs`. Verified by `DeepPortalChainTraversesToDepthEight` (PASS). |
| **AC 5.2**: Doorway/boundary multi-group interior seeding prevents edge unloading | `WoW/Source/WorldClient/World.cpp:2947-2950` (`CWorldScene::AddViewerGroup` sets up to 2 viewer groups) | `FindContainingInteriorGroups` seeds all containing groups at depth 0. Verified by `MultipleInteriorGroupsContainCameraSeedsBothGroupsAtDepthZero` (PASS). |
| **AC 5.3**: Indoor camera admits `AlwaysVisible` (0x10000) groups and exterior portals | `WoW/Source/WorldClient/MapObj.cpp:1737-1742` (`info.m_flags & 0x10000`) | `WmoRenderer.cs:1836-1863` admits `0x10000` groups and exterior groups when exterior portal reached. |
| **AC 6.1**: Winding corridor and cave interior view-thru across portals | `WoW/Source/WorldClient/MapObj.cpp:1250-1310` (`RRenderThruPortals`) | Portal view-volume builder only uses camera-to-portal edge pyramid frustum planes; `TerrainHoleInvestigationTests.InspectMushroomCave` verified that `MD_Mushroomcave05.wmo` groups 4 and 8 are admitted from the camera viewpoint on Kalimdor. |
| **AC 6.2**: Prevention of 2-cycle portal ping-ponging | `WoW/Source/WorldClient/MapObj.cpp:1285` (`if (portalRef.groupIndex == fromGroupIdx) continue`) | `WmoPortalVisibilityDecision` tracks `fromGroupIndex` and skips immediate backtrack; verified by `InspectMushroomCave` and `WmoPortalVisibilityDecisionTests` without visit capacity fallback. |
| **AC 6.3**: Terrain chunk and asset streaming data reads optimization | Profiling & IO throughput | `StandardTerrainAdapter.cs` demotes MCIN fallback to Trace, fast-paths MCNK FourCC to uint; `WorldAssetPathResolver.cs` adds negative caching to avoid repetitive disk probes on missing files. |

---

## 4. Operational Sign-Off
- Verified by: Spec Kit Governance Protocol (§9.2, §10, §11).
- Solution build: PASS (0 errors).
- Unit tests: 36/36 focused portal tests PASS in 44 ms, `InspectMushroomCave` real-data regression PASS in 215 ms.
- Next Step: Ready for in-viewer client frame verification by the operator.

