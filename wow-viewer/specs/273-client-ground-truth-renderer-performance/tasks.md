# Implementation Tasks: Client Ground-Truth Renderer Performance & Occlusion

**Feature Branch**: `v0.6.1-dev`
**Owning Spec**: `wow-viewer/specs/273-client-ground-truth-renderer-performance/spec.md`
**Plan**: `wow-viewer/specs/273-client-ground-truth-renderer-performance/plan.md`

---

## Phase 1 — WMO Portal Occlusion & Interior Group Culling

- [x] **T001**: Audit `WmoPortalVisibilityEvaluator.cs` against `WoW/Source/WorldClient/MapObj.cpp` and `CPortalView.cpp` in `wowdecomp`.
- [x] **T002**: Classify WMO groups as Exterior (`(flags & 0x2000) == 0`) or Interior (`(flags & 0x2000) != 0`) in `WmoRenderer.cs` and `WmoPortalVisibilityGroup`.
- [x] **T003**: In `WmoRenderer.UpdateRuntimeVisibility`, remove the unconditional fallback loop (`lines 1813–1819`) that marks all frustum-intersecting groups as visible; restrict frustum-only admission to Exterior groups.
- [x] **T004**: Ensure Interior groups are admitted strictly when the camera is inside the group or reached via valid portal traversal from an active view.
- [x] **T005**: Verify that doodad collection (`DoodadRefs`) strictly adheres to admitted groups so that culled interior rooms do not submit doodads.
- [x] **Gate A**: Run `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` (0 errors) and run focused unit tests.

---

## Phase 2 — Decouple WMO Shell Instancing from Scene Lights (`R-10`)

- [x] **T010**: Implement `SceneLightManager.IntersectsBounds(Vector3 min, Vector3 max)` to test whether a specific object placement is touched by any active local point light.
- [x] **T011**: In `WorldScene.cs` WMO batch candidate planning, replace `_sceneLightManager.Count == 0` with a per-placement test: only placements touched by a light fall back to individual unbatched draws.
- [x] **T012**: Relax `WmoRenderer.SupportsGpuInstancedOpaque` so models with portals can still be instanced when only their exterior shell is admitted.
- [x] **Gate B**: Run `dotnet build` and `dotnet test`; verify WMO instanced batching executes even when scene lights are present.

---

## Phase 3 — Spatial M2 & Doodad Light Collection

- [x] **T020**: In `WorldScene.RebuildSceneLights`, replace linear scan over all global `_mdxInstances` with a tile-partitioned query over active/resident tiles.
- [x] **T021**: Ensure M2 light emitters outside the camera fog/view radius are not evaluated.
- [x] **Gate C**: Verify reduced CPU time in `RebuildSceneLights` and `CollectSceneLights` via rolling frame diagnostics.

---

## Phase 4 — Terrain Shader & Calibration Verification

- [x] **T030**: Verify point light attenuation in terrain shader matches the 3.3.5 client formula ($1 / (k_c + k_l d + k_q d^2)$).
- [x] **T031**: Run full solution test suite `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- [x] **T032**: Prepare verification receipt under `specs/273-client-ground-truth-renderer-performance/evidence/receipt-spec273.md`.

---

## Phase 5 — Portal Traversal Depth & Multi-Group Seeding Tuning (Address Premature Unloading)

- [x] **T040**: Increase `ExteriorPortalTraversalDepth` and `InteriorPortalTraversalDepth` in `WmoRenderer.cs` from 1 / 4 to 8, aligning with 3.3.5 client ground truth (`s_maxPortalDepth = 10`) so inner rooms are not culled when looking from outside or navigating multi-room interiors.
- [x] **T041**: Multi-group interior seeding at doorways & boundary padding in `WmoPortalVisibilityDecision.cs`: support multiple containing interior groups at depth 0 when camera is in doorways/boundary zones; increase `BoundsPadding` to 2.0f; add 1.0f tolerance to `IsOnPortalSide`.
- [x] **T042**: Ground-truth `AlwaysVisible` (0x10000) and exterior portal admission when `Mode == Interior`: in `WmoRenderer.cs`, admit groups with `AlwaysVisibleFlag` (0x10000) that intersect the camera frustum when indoors (`MapObj.cpp:1737`), and admit exterior groups when an exterior portal was reached.
- [x] **T043**: Add unit tests in `WmoPortalVisibilityDecisionTests.cs` for multi-group seeding at doorways and depth-8 linear room chain traversal.
- [x] **Gate E**: Run focused portal test suite `dotnet test tests/WowViewer.Core.Tests -c Debug --filter "FullyQualifiedName~Portal"` (40/40 PASS).

---

## Phase 6 — WMO Winding Corridor View-Thru & Terrain/Asset Data Read Optimization

- [x] **T050**: Stop accumulating ancestor portal polygon surface plane in `WorldScenePortalViewVolumeBuilder.BuildChild`. Portal clipping strictly uses the camera-to-portal edge frustum cone; infinite 3D half-space planes cull winding corridors/caves (e.g. `MD_Mushroomcave05.wmo` groups 4 and 8) whose rooms bend behind previous doorway planes.
- [x] **T051**: Track `fromGroupIndex` in `WmoPortalVisibilityDecision` BFS queue and skip immediate backtracking (`if (destinationGroup == fromGroupIndex) continue;`), matching 3.3.5 `MapObj::RRenderThruPortals`. Eliminates 2-cycle ping-pong and prevents false `portal_visit_capacity_reached` fallback.
- [x] **T052**: Demote repetitive synchronous `ViewerLog.Important` in `StandardTerrainAdapter.cs` (line 1311) to `ViewerLog.Trace`, eliminating console I/O blocking hitches during tile streaming on split-ADT maps.
- [x] **T053**: Fast-path chunk FourCC comparisons in `StandardTerrainAdapter.cs` using `BinaryPrimitives.ReadUInt32LittleEndian` against `McnkFourCc` instead of allocating ASCII strings across every chunk in every tile.
- [x] **T054**: Add negative caching for missing assets in `WorldAssetPathResolver.cs` so failed model/texture candidates are remembered, preventing repetitive 15-20 candidate disk probes per missing file.
- [x] **T055**: Update `WorldScenePortalViewVolumeTests` expected plane count (5 planes) and add automated regression test in `TerrainHoleInvestigationTests.cs` verifying groups 4, 5, 8 are admitted from the camera viewpoint on `MD_Mushroomcave05.wmo`.
- [x] **Gate F**: Run `dotnet build` (0 errors) and run focused portal test suite (36/36 PASS in 44 ms; `InspectMushroomCave` PASS in 215 ms).

---

## Phase 7 — Composite City WMO Group Unloading & False Interior Culling (Stormwind Park)

- [x] **T060**: Add `UsesExteriorLightingFlag` (0x40), `IndoorFlag` (0x2000), `ShowSkyboxFlag` (0x40000), and `HasPortals` parameter to `WmoPortalVisibilityGroup`. Update `IsExterior` to admit groups with exterior/sky/lighting flags or groups without portals that are not flagged indoor (`(!HasPortals && (Flags & IndoorFlag) == 0)`).
- [x] **T061**: In `WmoRenderer.cs`, compute `groupHasPortals` in `BuildPortalVisibilityGroups()` from `_wmo.Groups[i].PortalCount > 0` and `_wmo.PortalRefs`.
- [x] **T062**: In `WmoRenderer.cs` interior admission pass, replace `(flags & ExteriorFlag) != 0` with `_portalVisibilityGroups[groupIndex].IsExterior`.
- [x] **T063**: Add unit tests in `WmoPortalVisibilityDecisionTests.cs` for outdoor groups without portals, outdoor groups with exterior lighting, indoor groups with indoor flag, and multi-group city layout view admission.
- [x] **Gate G**: Run `dotnet test tests/WowViewer.Core.Tests -c Debug --filter "FullyQualifiedName~Portal"` (48/48 PASS in 147 ms).

