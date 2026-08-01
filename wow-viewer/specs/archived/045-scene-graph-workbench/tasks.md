---
description: "Task list for spec 045 — scene graph workbench for WoWViewer"
---

# Tasks: 045 — Scene Graph Workbench

**Input**: Design documents from `/specs/045-scene-graph-workbench/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/scene-graph-snapshot.schema.json`

**Tests**: Focused contract and projection tests are required because selection identity, PM4 hierarchy projection, and lazy scene-tree behavior are easy to regress.

**Organization**: Tasks are grouped by user story so the first signed-off slice can deliver a usable world-session outliner before later usability and reuse follow-ups.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the feature structure and shared scene-graph contracts.

- [ ] T001 Create `wow-viewer/src/core/WowViewer.Core.Runtime/SceneGraph/` for shared scene-graph contract types.
- [ ] T002 Create `wow-viewer/src/viewer/WoWViewer/SceneGraph/` for viewer-only scene-graph controller, filter state, and tree rendering helpers.
- [ ] T003 [P] Create `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/` for focused scene-graph contract and projection tests.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define the reusable graph contract and stable node identity before any domain projection or UI binding starts.

**⚠️ CRITICAL**: No user story work should begin until the shared contract exists and passes focused tests.

- [ ] T004 [P] Implement `SceneGraphNodeId`, `SceneGraphSelectionTarget`, and `NodeKind` in `wow-viewer/src/core/WowViewer.Core.Runtime/SceneGraph/`.
- [ ] T005 [P] Implement `SceneGraphNode` and `SceneGraphSnapshot` in `wow-viewer/src/core/WowViewer.Core.Runtime/SceneGraph/`.
- [ ] T006 [P] Implement `ISceneGraphDomainProvider` and availability/status helpers in `wow-viewer/src/core/WowViewer.Core.Runtime/SceneGraph/`.
- [ ] T007 [P] Add `SceneGraphNodeIdTests` in `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/` covering canonical path stability and equality.
- [ ] T008 [P] Add `SceneGraphSnapshotTests` in `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/` covering root creation, child composition, and schema-compatible serialization shape.

**Checkpoint**: Shared scene-graph contract is stable and testable; domain projection can now begin.

---

## Phase 3: User Story 1 - Browse The Whole Loaded Scene As A Hierarchy (Priority: P1) 🎯 MVP

**Goal**: Project one loaded world session into a unified root tree with terrain, object, and PM4 branches.

**Independent Test**: A loaded world session produces one scene snapshot with terrain, object, and PM4 top-level branches when those domains are available.

### Tests for User Story 1

- [ ] T009 [P] [US1] Add `TerrainSceneGraphProjectorTests` in `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/` covering map, tile, and chunk projection.
- [ ] T010 [P] [US1] Add `Pm4SceneGraphProjectorTests` in `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/` covering region, object, and sub-object projection from current PM4 hierarchy owners.
- [ ] T011 [P] [US1] Add `WorldObjectSceneGraphProjectorTests` in `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/` covering placed WMO/M2/MDX instance projection.

### Implementation for User Story 1

- [ ] T012 [P] [US1] Implement a terrain domain projector in `wow-viewer/src/core/WowViewer.Core.Runtime/SceneGraph/` that projects loaded map, tile, and chunk structure into scene-graph nodes.
- [ ] T013 [P] [US1] Implement `Pm4SceneGraphProjector` in `wow-viewer/src/core/WowViewer.Core.PM4/SceneGraph/` using the existing region -> object -> sub-object PM4 model.
- [ ] T014 [P] [US1] Implement a placed-object domain projector in `wow-viewer/src/core/WowViewer.Core.Runtime/SceneGraph/` for loaded world instances.
- [ ] T015 [US1] Implement `ViewerSceneGraphController` in `wow-viewer/src/viewer/WoWViewer/SceneGraph/` to compose the active domain providers into one `SceneGraphSnapshot`.
- [ ] T016 [US1] Add `ViewerApp_SceneGraph.cs` in `wow-viewer/src/viewer/WoWViewer/` and wire a `Scene Graph` panel into the active shell with a right-side default placement plus dockable availability.
- [ ] T017 [US1] Implement `ViewerSceneGraphTreeRenderer` in `wow-viewer/src/viewer/WoWViewer/SceneGraph/` to render the unified tree with collapsed roots by default.

**Checkpoint**: A world session can be browsed as one whole-scene hierarchy from one panel.

---

## Phase 4: User Story 2 - Selection Sync Between Tree And Viewer (Priority: P1)

**Goal**: Make the scene graph drive and reflect live viewer selection for supported node types.

**Independent Test**: Selecting supported nodes from the tree changes viewer selection, and selection from existing viewer surfaces highlights the matching tree node.

### Tests for User Story 2

- [ ] T018 [P] [US2] Add `ViewerSceneGraphSelectionTests` in `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/` covering node-to-selection mapping and reverse lookup from selection target to node id.
- [ ] T019 [P] [US2] Add focused controller tests for unavailable/unloaded node selection fallback behavior in `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/`.

### Implementation for User Story 2

- [ ] T020 [P] [US2] Implement selection-target mapping and reverse lookup support in `wow-viewer/src/viewer/WoWViewer/SceneGraph/ViewerSceneGraphController.cs`.
- [ ] T021 [US2] Wire tree-node selection into the existing viewer selection/inspector flow in `wow-viewer/src/viewer/WoWViewer/`.
- [ ] T022 [US2] Wire reverse selection highlighting from existing viewer selection changes back into the scene graph panel in `wow-viewer/src/viewer/WoWViewer/`.
- [ ] T023 [US2] Add optional node actions for camera framing/highlighting where those capabilities already exist for the target type.

**Checkpoint**: The scene graph is now an actual navigation surface, not a passive dump.

---

## Phase 5: User Story 3 - Large Scene Usability (Priority: P2)

**Goal**: Keep the scene graph interactive on heavier scenes through filtering, lazy expansion, and stable expansion state.

**Independent Test**: A heavier world session can be filtered and expanded progressively without forcing the panel to eagerly materialize every leaf.

### Tests for User Story 3

- [ ] T024 [P] [US3] Add `ViewerSceneGraphFilterTests` in `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/` covering text query matching against labels and canonical ids.
- [ ] T025 [P] [US3] Add `ViewerSceneGraphLazyExpansionTests` in `wow-viewer/tests/WowViewer.Core.Tests/SceneGraph/` covering deferred child realization and preserved expansion state.

### Implementation for User Story 3

- [ ] T026 [P] [US3] Implement `ViewerSceneGraphFilterState` in `wow-viewer/src/viewer/WoWViewer/SceneGraph/`.
- [ ] T027 [US3] Extend `ViewerSceneGraphTreeRenderer` to support text filtering, ancestor reveal, and branch summaries.
- [ ] T028 [US3] Add lazy child materialization / incremental expansion support in the controller and domain providers so large branches do not fully realize on initial panel open.
- [ ] T029 [US3] Preserve expansion state by `SceneGraphNodeId` across snapshot refreshes where the same logical nodes still exist.

**Checkpoint**: The scene graph remains usable on large loaded scenes instead of becoming another giant static dump.

---

## Phase 6: User Story 4 - Reuse The Same Graph Contract Across Scene Types (Priority: P3)

**Goal**: Prove the contract is reusable beyond the world-session path without inventing a second tree UI.

**Independent Test**: A reduced-content or alternate scene type can produce a valid scene graph root and present unavailable branches clearly.

### Implementation for User Story 4

- [ ] T030 [P] [US4] Add a reduced-content scene-graph provider path in `wow-viewer/src/core/WowViewer.Core.Runtime/SceneGraph/` that can emit a valid root when only some domains are available.
- [ ] T031 [US4] Update `ViewerSceneGraphController` so absent domains surface as unavailable status instead of stale placeholders or hidden failures.
- [ ] T032 [US4] Document the standalone-scene adoption path in `wow-viewer/docs/architecture/` or the relevant future spec cross-link once the first world-session slice is landed.

**Checkpoint**: The contract is proven reusable even before full standalone asset adoption is implemented.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, validation, and continuity sync.

- [ ] T033 [P] Update `wow-viewer/specs/044-viewer-shell-usability/spec.md` or its follow-up note to cross-reference the new scene-graph workbench as the next shell-facing inspection slice.
- [ ] T034 [P] Update the relevant `wow-viewer/docs/architecture/` viewer-shell ownership note if file placement or ownership details change during implementation.
- [ ] T035 [P] Add snapshot/export validation against `contracts/scene-graph-snapshot.schema.json` if a debug export is implemented in the slice.
- [ ] T036 Run focused scene-graph tests and a bounded `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- [ ] T037 Update `gillijimproject_refactor/memory-bank/activeContext.md` and `progress.md` with the landed scene-graph owner, proof surface, and remaining gaps.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on foundational contract completion.
- **User Story 2 (Phase 4)**: Depends on the world-session tree from User Story 1.
- **User Story 3 (Phase 5)**: Depends on the initial tree and selection model from User Stories 1 and 2.
- **User Story 4 (Phase 6)**: Depends on the shared contract and controller but can remain a lighter follow-up slice.
- **Polish (Phase 7)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: MVP; first independently useful slice.
- **User Story 2 (P1)**: Builds directly on US1 to make the tree actionable.
- **User Story 3 (P2)**: Builds on US1/US2 to make the workbench scale.
- **User Story 4 (P3)**: Reuse/extension proof after the world-session slice is working.

### Parallel Opportunities

- T004, T005, T006 can run in parallel in different contract files.
- T007 and T008 can run in parallel once the foundational contract files exist.
- T012, T013, and T014 can run in parallel as separate domain projectors.
- T018 and T019 can run in parallel as separate selection-sync tests.
- T024 and T025 can run in parallel as usability/perf tests.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational contract
3. Complete Phase 3: World-session hierarchy workbench
4. **STOP and VALIDATE**: Open a real world session and confirm terrain, objects, and PM4 can all be reached from one tree

### Incremental Delivery

1. Deliver US1 as the first usable outliner
2. Add US2 selection sync to make it the main navigation surface
3. Add US3 filtering/lazy expansion to make it scale
4. Add US4 reuse proof so later scene types can adopt the same contract

### Parallel Team Strategy

With multiple developers:

1. One developer owns the shared contract and controller
2. One developer owns terrain/object projection
3. One developer owns PM4 projection
4. One developer owns selection/filter UI once the contract stabilizes

---

## Notes

- The feature is explicitly viewer-owned and should not drift back into legacy `MdxViewer` shell architecture.
- The first slice is inspection-first and read-only by design.
- PM4 field discoveries such as `TypeFlags` remain metadata in this workbench unless the PM4 ownership model itself is later re-specified.
- The scene graph contract is intended to become a long-range inspection primitive for more than one scene type, but the first signoff surface is the active world session.
