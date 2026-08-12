# Tasks: World Scene Graph and Spatial Partitioning

**Input**: Design documents from `/wow-viewer/specs/142-world-scene-graph/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`,
`contracts/synthetic-world-workload.schema.md`, `quickstart.md`

**Current execution rule**: The graph foundation, conservative traversal, default-on `WorldScene`
object adapter, nested WMO group mounting, graph-only portal adjacency contract, and runtime
residency diagnostics are validated and committed. The WMO portal read-model adapter and bounded
graph-side portal view-volume contract are also complete; runtime nested portal traversal, doorway
parity, pass, query, and performance promotion tasks remain unchecked.

## Phase 1: Setup

**Purpose**: Establish the runtime-library ownership and contract surface.

- [x] T001 Confirm `WowViewer.Core.Runtime` and `WowViewer.Core.Tests` are the owners for the graph contract in `wow-viewer/specs/142-world-scene-graph/plan.md`.
- [x] T002 [P] Add the synthetic workload schema and replay rules to `wow-viewer/specs/142-world-scene-graph/contracts/synthetic-world-workload.schema.md`.

## Phase 2: Foundational Graph Contract

**Purpose**: Provide the single nested node model needed by every later renderer story.

- [x] T003 [P] Define node kinds and render-pass flags in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneNodeKind.cs`.
- [x] T004 [P] Define stable node metadata, transforms, bounds, and parent/child ownership in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneNode.cs`.
- [x] T005 Implement attach, detach, lookup, depth-first enumeration, and cycle/duplicate-parent rejection in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneGraph.cs`.
- [x] T006 Implement graph counts, stable ordered IDs, and invariant validation in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneGraphSnapshot.cs`.
- [x] T007 [P] Add graph identity, transform propagation, bounds containment, cycle, duplicate-ID, and detach tests in `wow-viewer/tests/WowViewer.Core.Tests/World/WorldSceneGraphTests.cs`.

**Checkpoint**: A nested graph can be built and detached without a renderer or client root.

## Phase 3: User Story 1 — One Node Model For Every World Element (Priority: P1) 🎯 MVP

**Goal**: Represent map, tile, chunk, WMO, M2, PM4, and overlay elements in one inspectable graph.

**Independent Test**: Build a nested fixture, enumerate it from the root, verify stable identity and
parent ownership, then detach one tile and prove no descendant remains reachable.

- [x] T008 [US1] Add explicit asset-key, renderability, queryability, update-required, and non-rejectable policies in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneNode.cs`.
- [x] T009 [US1] Add node-kind count and render-pass count assertions to `wow-viewer/tests/WowViewer.Core.Tests/World/WorldSceneGraphTests.cs`.

**Checkpoint**: The graph contract independently satisfies the node identity and subtree ownership
scenario without integrating `WorldScene`.

## Phase 4: User Story 6 — Synthetic Stress Is Grounded In The Real Renderer (Priority: P1)

**Goal**: Produce deterministic synthetic world-scene workloads that exercise future runtime
interfaces and cannot be confused with image-only minimap previews.

**Independent Test**: Build the same manifest twice from the same seed, compare graph snapshots and
manifest hashes, and reject image-only workload input.

- [x] T010 [P] [US6] Define workload counts, camera, pass mix, portal metadata, and serialized node inventory in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/SyntheticWorldWorkload.cs`.
- [x] T011 [US6] Implement deterministic sparse-region, nested-asset, repeated-asset, and overlay fixture generation in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/SyntheticWorldWorkloadBuilder.cs`.
- [x] T012 [P] [US6] Add deterministic replay, manifest validation, sparse-map, nested-asset, and image-only rejection tests in `wow-viewer/tests/WowViewer.Core.Tests/World/SyntheticWorldWorkloadTests.cs`.
- [ ] T013 [US6] Add a dry-run serialization/replay example and explicit non-renderer-benchmark warning to `wow-viewer/specs/142-world-scene-graph/quickstart.md`.

**Checkpoint**: The fixture manifest and graph snapshot are replayable and prove only graph/workload
identity; no FPS or GPU claim is made.

## Phase 5: User Story 2 — Cull By Subtree, Not By Instance (Priority: P1)

- [x] T014 [US2] Define a shared hierarchical traversal result and rejection attribution contract in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneTraversal.cs`.
- [x] T015 [US2] Implement a first conservative region traversal over `WorldSceneGraph` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneTraversal.cs`.
- [x] T016 [US2] Add fixed-camera resident-scale traversal tests in `wow-viewer/tests/WowViewer.Core.Tests/World/WorldSceneTraversalTests.cs`.

## Phase 6: User Story 3 — Interiors Cull Through Portals (Priority: P2)

- [x] T017 [US3] Add graph-only portal adjacency metadata and bounded nested-view diagnostics in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldScenePortalGraph.cs`; reject malformed links and report cycle, missing-entry, absent-data, and depth-limit fallback.
- [x] T018 [US3] Integrate existing `WmoRenderDocument` portal vertices, geometry, and group references into the graph adapter without changing format readers in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldScenePortalAdapter.cs`.
- [ ] T019 [US3] Add runtime doorway geometry and existing-renderer parity coverage in `wow-viewer/tests/WowViewer.Core.Tests/World/WorldScenePortalTests.cs`; graph and view-volume tests now cover the library-only malformed, cyclic, absent-data, depth-limited, and geometry fallback contracts.
- [x] T030 [US3] Add the bounded parent-plane/portal-edge/destination-side view-volume contract and fallback diagnostics in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldScenePortalViewVolume.cs`.

## Phase 7: User Story 4 — Ordered Visibility Results (Priority: P2)

- [ ] T020 [US4] Define visible, updating, and per-pass queue records in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneVisibilityFrame.cs`.
- [ ] T021 [US4] Route graph traversal output into existing world pass coordinators in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs`.
- [ ] T022 [US4] Add pass membership, state ordering, and shared-update-count tests in `wow-viewer/tests/WowViewer.Core.Tests/World/WorldSceneVisibilityFrameTests.cs`.

## Phase 8: User Story 5 — Shared Spatial Queries (Priority: P3)

- [ ] T023 [US5] Define content-mask ray and volume query requests/results in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneQuery.cs`.
- [ ] T024 [US5] Implement graph-backed query pruning and visibility-state filtering in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneQuery.cs`.
- [ ] T025 [US5] Add picking parity and residency-scaling tests in `wow-viewer/tests/WowViewer.Core.Tests/World/WorldSceneQueryTests.cs`.

## Phase 9: Integration and Evidence

- [x] T026 Integrate graph ownership behind a default-on runtime selector in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`, mount client-backed `WmoGroup` children when summaries are available, and retain a reversible legacy fallback.
- [x] T031 Add a build-checked opt-in bridge from loaded `WmoRenderer` portal read models to placement-keyed `WorldScenePortalAdapter` results in `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs` and `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`; do not change runtime visibility behavior.
- [x] T032 Add graph-side portal visibility evaluation through nested view volumes and fail-open diagnostics in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldScenePortalVisibilityEvaluator.cs`; apply it only to opt-in `WmoGroup` graph traversal in `WorldScene.cs`.
- [x] T033 [US2] Add optional spatial-bucket metadata and nested tile-to-region attachment in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneGraphObjectAdapter.cs`; preserve deterministic ordering and fail-open bounds.
- [x] T034 [US2] Assign resident non-skybox ADT M2 placements to terrain chunk buckets from the existing `WorldScene` coordinate convention; keep external, skybox, WMO, and WMO-internal doodad-set paths unchanged.
- [x] T035 [US2] Add stable-ID, unknown-bounds, and rejected-chunk descendant-skip coverage in `wow-viewer/tests/WowViewer.Core.Tests/World/WorldSceneGraphObjectAdapterTests.cs`.
- [x] T036 [US2] Attribute individually tested nodes, rejected subtree roots, and skipped descendants by node kind in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneTraversal.cs`, with focused chunk/M2 assertions.
- [x] T037 [US2] Add an opt-in traversal policy that defers ordinary ADT M2 leaf visibility under Chunk nodes to the existing collector, with deferred-by-kind diagnostics and focused proof.
- [x] T038 [US2] Add a partitioned graph-set build result with one independent `Tile`-rooted graph per resident ADT and a separate external graph path in `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneGraphObjectAdapter.cs`.
- [x] T039 [US2] Switch opt-in `WorldScene` graph rebuild, portal lookup, visibility traversal, and snapshot aggregation to the per-ADT graph set without changing the legacy path.
- [x] T040 [US2] Add focused tests proving independent ADT roots, no cross-tile descendants, external-graph separation, deterministic partitioning, and per-graph chunk rejection.
- [x] T041 Promote hierarchical traversal to the default `WorldScene` path with a state-invalidating runtime toggle in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`.
- [x] T042 Expose graph, AOI, and last WMO-bearing ADT unload diagnostics in `wow-viewer/src/viewer/WoWViewer/Terrain/TerrainManager.cs`, `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`, `wow-viewer/src/viewer/WoWViewer/ViewerApp_Investigation.cs`, and `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`.
- [x] T043 Keep per-residency graph rebuilds metadata-only by using cached WMO summaries in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldAssetManager.cs` and `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`; preserve fail-open behavior when optional group metadata is not cached.
- [x] T044 Advance deferred WMO doodad model loads once per scene frame through
      `WorldAssetManager`; do not perform that synchronous client read from each WMO placement
      render call.
- [x] T045 Bound minimap client-data reads to one background reader while retaining the existing
      render-thread texture-upload queue.
- [x] T046 Add `profile-render` to `WowViewer.Tool.ValidationCapture`, executing the production
  path with client-coherent WDT/tile validation and an on-disk progress report so a stalled stage is visible.
- [x] T049 Attribute `WorldScene` pre-pass maintenance separately from draw/pass timing so scene-graph
  rebuild work cannot be misreported as unexplained frame time.
- [x] T050 Gate normal per-ADT/chunk/liquid/placement console diagnostics behind verbose mode before
  full-map profiling; retain malformed-data diagnostics at the default level.
- [x] T051 [US2] Keep each per-ADT root rejectable from authoritative native tile bounds while
  streamed child bounds remain unresolved; expand the root around resolved placements and prove an
  off-camera root skips its unresolved descendant in
  `wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldSceneGraphObjectAdapter.cs` and
  `wow-viewer/tests/WowViewer.Core.Tests/World/WorldSceneGraphObjectAdapterTests.cs`.
      `WorldScene.Render` path in a hidden OpenGL context and writing a versioned per-stage JSON
      report with workload, queue, and client-read counters.
- [x] T047 Add focused `WorldRenderDiagnostics` report-contract tests covering stage inventory,
      dominant-stage attribution, unsettled queues, and uncovered object paths.
- [ ] T048 Add timer-query-backed GPU and driver-wait attribution to the production report; until
      then the CPU-stage report MUST label that gap explicitly.
- [x] T052 [US2] Add serializable per-owner overlay frame records and aggregate report summaries in
  `src/core/WowViewer.Core.Runtime/World/WorldRenderDiagnostics.cs` with focused coverage in
  `tests/WowViewer.Core.Tests/World/WorldRenderDiagnosticsTests.cs`.
- [x] T053 [US2] Instrument the existing object-wireframe, bounds, PM4, POI/taxi, area-trigger,
  and remaining overlay blocks in `src/viewer/WoWViewer/Terrain/WorldScene.cs` without moving their
  rendering behavior; every owner must emit a disabled or measured record each frame.
- [x] T058 [US2] Add a report-contract proof that owner durations reconcile to coarse `overlay`,
  disabled owners do no work, and a dominant owner is named in
  `tests/WowViewer.Core.Tests/World/WorldRenderDiagnosticsTests.cs`.
- [x] T059 [US2] Build the validation-capture seam and hand off the owner-attribution capture from
  `specs/142-world-scene-graph/phase-8j-overlay-recovery.md`; record only user-run evidence.
- [ ] T060 [US2] After T052-T059 identify the real dominant owner, extract only that owner behind a
  narrow invalidation/cache seam in `src/viewer/WoWViewer/Rendering/` or `Terrain/`, with focused
  no-change-frame reuse proof.
- [ ] T061 [US2] Add owner-specific bounded preparation and viewport/tile culling only after T060;
  report deferred work and retain the last complete valid batch.
- [ ] T054 [US2] Define and test index-only/CPU-decoded/GPU-ready/retained tile residency records
  in the runtime library.
- [ ] T055 [US2] Implement map-wide index-first discovery and camera-prioritized budgeted tile
  promotion without synchronous all-ADT normal startup.
- [ ] T056 [US2] Attribute explicit full-residency stress mode by decode, mesh, upload, object, and
  graph materialization substage; keep it distinct from normal streaming evidence.
- [x] T062 [US2] Partition WL* liquid bodies into terrain-tile GPU fragments and enumerate only
  camera-window buckets in `src/viewer/WoWViewer/Terrain/LiquidRenderer.cs`, retaining an explicit
  external bucket for data without a safe 64x64 tile assignment.
- [x] T063 [US2] Keep resident WDL meshes available as a far-field underlay, extend their bounded
  residency to the existing horizon projection window, and retain detailed ADT hide/show events
  as residency metadata rather than permanent WDL suppression in `src/viewer/WoWViewer/Terrain/WdlTerrainRenderer.cs`.
- [x] T064 [US2] Add procedural sun/moon discs to the existing shared sky-dome pass, consuming the
  final active TerrainLighting direction while keeping synthesized minimap output terrain-only
  in `src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs`.
- [x] T065 [US2] Resolve the active client sky model from the exact-build LightSkybox DBC record,
  discover the client `Environments/Stars/Stars` asset only when that record is unavailable, and
  render the selected M2/MDX/MDL asset as a night-only camera-anchored backdrop in
  `src/viewer/WoWViewer/Terrain/WorldScene.cs`.
- [ ] T057 [US4] Define a capability-gated modern static-instance submission contract and legacy
  fallback proof, coordinated with Spec 138.
- [ ] T027 Add synthetic workload replay and stage-level report output in `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/ValidationWorldSceneAdapter.cs`.
- [ ] T028 Add current-vs-new parity and performance report tests in `wow-viewer/tests/WowViewer.Core.Tests/World/WorldScenePerformanceReportTests.cs`.
- [ ] T029 Run the focused and full test commands from `wow-viewer/specs/142-world-scene-graph/quickstart.md` and record evidence in `wow-viewer/memory-bank/progress.md`.

## Dependencies and Execution Order

- Phase 1 documentation/setup precedes all implementation.
- Phase 2 is foundational and blocks every user story.
- US1 and US6 form the MVP foundation and must pass before traversal integration.
- US2 precedes US3, US4, and US5 because portals, pass queues, and queries consume shared traversal.
- Phase 9 begins only after the earlier story checkpoints are validated.
- T026/T041/T042 activate and instrument the selector but do not establish portal/pass/query
  parity, change terrain AOI policy, or authorize a heavy capture.
- T043 makes graph rebuild metadata-only for WMO group summaries; it does not change WMO asset
  loading or renderer submission ownership.

## Implementation Strategy

1. Complete the graph contract and tests without touching the existing renderer.
2. Add deterministic synthetic workload generation and replay proof.
3. Stop and review the graph snapshot/invariant evidence.
4. Add conservative traversal behind a selector, then portal/pass/query layers one at a time.
5. Integrate with `WorldScene` only after the library contracts are stable.
6. Run user-owned synthetic and real-client captures only after reports can identify the limiting
   stage and preserve workload provenance.
