# Tasks: World Scene Graph and Spatial Partitioning

**Input**: Design documents from `/wow-viewer/specs/142-world-scene-graph/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`,
`contracts/synthetic-world-workload.schema.md`, `quickstart.md`

**Current execution rule**: The graph foundation, conservative traversal, opt-in `WorldScene`
object adapter, nested WMO group mounting, and the graph-only portal adjacency contract are
validated and committed. The WMO portal read-model adapter and bounded graph-side portal
view-volume contract are also complete; runtime nested portal traversal, doorway parity, pass,
query, and performance promotion tasks remain unchecked.

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

- [x] T026 Integrate graph ownership behind an opt-in runtime selector in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`, mount client-backed `WmoGroup` children when summaries are available, and keep the legacy path as the default until parity evidence exists.
- [ ] T027 Add synthetic workload replay and stage-level report output in `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/ValidationWorldSceneAdapter.cs`.
- [ ] T028 Add current-vs-new parity and performance report tests in `wow-viewer/tests/WowViewer.Core.Tests/World/WorldScenePerformanceReportTests.cs`.
- [ ] T029 Run the focused and full test commands from `wow-viewer/specs/142-world-scene-graph/quickstart.md` and record evidence in `wow-viewer/memory-bank/progress.md`.

## Dependencies and Execution Order

- Phase 1 documentation/setup precedes all implementation.
- Phase 2 is foundational and blocks every user story.
- US1 and US6 form the MVP foundation and must pass before traversal integration.
- US2 precedes US3, US4, and US5 because portals, pass queues, and queries consume shared traversal.
- Phase 9 begins only after the earlier story checkpoints are validated.
- T026 is a bounded ownership seam that may land after the shared traversal contract; it does not
  promote the selector, establish portal/pass/query parity, or authorize a heavy capture.

## Implementation Strategy

1. Complete the graph contract and tests without touching the existing renderer.
2. Add deterministic synthetic workload generation and replay proof.
3. Stop and review the graph snapshot/invariant evidence.
4. Add conservative traversal behind a selector, then portal/pass/query layers one at a time.
5. Integrate with `WorldScene` only after the library contracts are stable.
6. Run user-owned synthetic and real-client captures only after reports can identify the limiting
   stage and preserve workload provenance.
