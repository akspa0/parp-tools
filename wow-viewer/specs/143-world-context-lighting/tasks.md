---

description: "Implementation tasks for World Context And Lighting Parity"
---

# Tasks: World Context And Lighting Parity

**Input**: Design documents from `/specs/143-world-context-lighting/`

**Branch status**: Feature branch creation is blocked by the shared repository's read-only
`.git/index.lock`; these tasks are currently attached to `142-world-scene-graph` and must not be
represented as branch-isolated until that environment issue is resolved.

**Execution rule**: Complete and validate one phase before starting the next. Real-client captures,
whole-map loads, native OpenGL runtime tests, and GPU profiling are user-run operations.

## Phase 1: Setup and evidence lock

**Purpose**: Freeze the source/profile evidence and keep implementation from guessing fields.

- [ ] T001 [P] [SETUP] Record the active Spec 106/138 lighting ownership and build/profile matrix in `specs/143-world-context-lighting/research.md`.
- [ ] T002 [P] [SETUP] Add the current AreaTable load/lookup diagnostics and MCNK source paths to `specs/143-world-context-lighting/research.md`.
- [ ] T003 [SETUP] Audit existing WMO root/group fixtures and reference readers for a profile-proven WMOAreaID source; record unresolved profiles in `specs/143-world-context-lighting/research.md`.
- [ ] T004 [SETUP] Add a focused validation inventory for early, 1.x/3.x, and 4.x client roots in `specs/143-world-context-lighting/quickstart.md` without embedding machine-local paths.

## Phase 2: Foundational contracts

**Purpose**: Establish immutable, diagnostic-bearing contracts before viewer wiring.

- [ ] T005 [P] [FOUNDATION] Add `WorldContextSnapshot`, `AreaResolution`, `TerrainAreaContext`, `WmoAreaContext`, and `WmoAreaIdEvidence` contracts under `src/core/WowViewer.Core/World/`.
- [ ] T006 [P] [FOUNDATION] Add `CameraHeadState` and `LightingSelection` contracts under `src/core/WowViewer.Core.Runtime/World/` with explicit version/profile fields.
- [ ] T007 [FOUNDATION] Add deterministic reason/confidence enums and JSON diagnostic serialization under `src/core/WowViewer.Core/World/`.
- [ ] T008 [FOUNDATION] Add focused contract tests in `tests/WowViewer.Core.Tests/WorldContextContractTests.cs` for unresolved-state distinction, explicit source provenance, and snapshot invariants.

**Checkpoint**: Contracts compile and focused tests distinguish missing, zero, malformed, map-mismatch, unavailable-profile, and resolved states.

## Phase 3: User Story 1 - Resolve the current world area (Priority: P1)

**Goal**: Resolve the area under the camera from resident ADT data and active DBC/DBD data without collapsing failures into an empty string.

**Independent Test**: Synthetic Alpha/standard chunk metadata plus an AreaTable fixture produces a resolved name, while missing/zero/map-mismatch cases produce explicit reasons.

- [ ] T009 [P] [US1] Add failing unit cases for Alpha MCNK `Unknown3` area extraction, standard MCNK `AreaId`, coordinate-to-chunk selection, and map mismatch in `tests/WowViewer.Core.Tests/WorldAreaContextTests.cs`.
- [x] T010 [US1] Refactor `src/viewer/WoWViewer/Terrain/AreaTableService.cs` to return `AreaResolution` plus native-style `ZoneText`/`SubzoneText` roles while preserving DBCD logical-column detection and parent-chain provenance.
- [ ] T011 [US1] Add a profile-aware resident terrain context evaluator under `src/core/WowViewer.Core.Runtime/World/` that records camera coordinates, tile/chunk source, raw ID, table key, and unresolved reason.
- [x] T012 [US1] Correct the `src/viewer/WoWViewer/ViewerApp.cs` camera-to-chunk/map-context call site to consume the structured evaluator result instead of treating all misses as an empty `_currentAreaName`.
- [x] T013 [US1] Update `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs` to display `SubzoneText` as the primary area label plus compact raw-ID/source diagnostics and explicit unresolved state.
- [ ] T014 [P] [US1] Add lookup diagnostics assertions for `AreaTableService.DescribeLoadContext` and `DescribeLookup` in `tests/WowViewer.Core.Tests/AreaTableServiceTests.cs`.
- [x] T015 [US1] Run the focused area-context tests and isolated viewer build; record the proof and remaining real-client gap in `specs/143-world-context-lighting/quickstart.md`.

**Checkpoint**: The status bar can distinguish no resident chunk, zero/malformed ADT ID, AreaTable row miss, map mismatch, and a valid localized AreaName.

## Phase 4: User Story 2 - Identify WMO interior context (Priority: P1)

**Goal**: Resolve a profile-proven WMO/group area ID when the camera is contained by a WMO and deterministically fall back to ADT context otherwise.

**Independent Test**: Fixture-backed WMO candidates produce stable WMO-first selection, and missing/unavailable WMO evidence falls back to the ADT result with a reason.

- [ ] T016 [P] [US2] Add fixture tests for every WMOAreaID profile proven by Phase 1 in `tests/WowViewer.Core.Tests/WmoAreaIdEvidenceTests.cs`; include an unavailable-profile case.
- [ ] T017 [US2] Extend the existing shared WMO read model under `src/core/WowViewer.Core.IO/Wmo/` only at the evidence-backed chunk/offset/profile identified in `research.md`.
- [ ] T018 [US2] Carry WMO identity, group index/name, source chunk/offset, raw area ID, and confidence through `src/core/WowViewer.Core.IO/Wmo/WmoRenderDocument.cs` or its current equivalent.
- [ ] T019 [US2] Add a resident WMO containment evaluator under `src/core/WowViewer.Core.Runtime/World/` using existing bounds/portal candidates and stable tie-breaking.
- [ ] T020 [US2] Integrate WMO-first/ADT-fallback selection into the same world-context snapshot consumed by `src/viewer/WoWViewer/ViewerApp.cs`.
- [ ] T021 [US2] Add WMO context and transition diagnostics to `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs` and the existing diagnostics output path.
- [ ] T022 [P] [US2] Add deterministic candidate ordering, overlap, enter, and exit tests in `tests/WowViewer.Core.Tests/WmoWorldContextTests.cs`.

**Checkpoint**: WMO context is source-attributed and deterministic; no guessed WMO area ID is accepted.

## Phase 5: User Story 3 - Navigate as a player head (Priority: P1)

**Goal**: Make eye position/orientation/mode/offset explicit and provide one same-frame camera state to all context and render consumers.

**Independent Test**: Camera movement and mode switching produce a stable serialized head state used consistently by view, context, fog, and lighting consumers.

- [ ] T023 [P] [US3] Add `CameraHeadState` construction and validation to `src/viewer/WoWViewer/Rendering/Camera.cs` without removing existing free-fly controls.
- [ ] T024 [US3] Add explicit `PlayerHead` and `Museum` mode/offset controls to the existing viewer camera/session surface under `src/viewer/WoWViewer/`.
- [ ] T025 [US3] Add a same-frame world-context snapshot step to the existing render/frame coordination path under `src/viewer/WoWViewer/ViewerApp.cs` or `src/core/WowViewer.Core.Runtime/World/`.
- [ ] T026 [US3] Route WMO containment, terrain area lookup, view construction, fog, and lighting selection through the snapshot's eye state.
- [ ] T027 [US3] Add camera save/restore serialization coverage in `tests/WowViewer.Core.Tests/CameraHeadStateTests.cs`.
- [ ] T028 [US3] Add same-frame position/orientation/mode assertions in `tests/WowViewer.Core.Tests/WorldContextSnapshotTests.cs`.

**Checkpoint**: No consumer uses a hidden head offset or a prior-frame camera state.

## Phase 6: User Story 4 - Restore WMO and MDX/M2 lighting (Priority: P1)

**Goal**: Select attributable profile-scoped lighting inputs and remove generic flat-lit behavior where source inputs are available.

**Independent Test**: WMO and M2 fixtures report their selected ambient, directional, baked/vertex, local-light, lightmap, fog, and shader/effect sources, including explicit fallback reasons.

- [ ] T029 [P] [US4] Add `LightingSelection` fixture tests for WMO root/group inputs, vertex/baked/lightmap inputs, M2 scene inputs, and equivalent fallback in `tests/WowViewer.Core.Tests/LightingSelectionTests.cs`.
- [ ] T030 [US4] Add a profile-scoped lighting input selector under `src/core/WowViewer.Core.Runtime/World/` that consumes existing Spec 106/138 contracts and rejects unproven BLS claims.
- [x] T030a [US4] Correct LIT list-header spatial decoding in the shared core contract: divide fixed-point
  XZY values by 36, expose decoded WoW XYZ, and apply the map-origin transform for viewer/minimap
  consumers with focused regression coverage.
- [ ] T031 [US4] Wire selected WMO root/group ambient, light references, vertex colors, baked weights, lightmap data, and fog inputs through `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`.
- [ ] T032 [US4] Replace only the WMO shader fallback behavior required by the selected inputs in `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`; preserve batching, alpha, portal, liquid, and transparent paths.
- [ ] T033 [US4] Wire attributable directional, ambient, local-light, fog, and effect-route inputs through `src/viewer/WoWViewer/Rendering/M2Renderer.cs` without changing animation or placement ownership.
- [ ] T034 [US4] Add renderer diagnostics for native/evidence-backed/equivalent-fallback/unsupported shader routes in the existing viewer diagnostics output.
- [ ] T035 [US4] Add focused WMO/M2 shader-input contract tests and run an isolated viewer build before any user-run visual capture.

**Checkpoint**: Lighting is attributable and non-flat where inputs are proven; unsupported paths remain visibly labeled fallback.

## Phase 7: User Story 5 - Preserve cross-era and performance boundaries (Priority: P2)

**Goal**: Prove the feature across eras without reintroducing full-map loading or graph-path frame stalls.

**Independent Test**: Focused tests/builds pass, then the user-run matrix records context/lighting correctness and p95 frame stages against the flat baseline.

- [ ] T036 [P] [US5] Add cross-era synthetic contract fixtures for Alpha, 1.x/3.x, and 4.x AreaTable/MCNK layouts in `tests/WowViewer.Core.Tests/WorldContextEraTests.cs`.
- [ ] T037 [US5] Add context and lighting counters to the existing render diagnostics without scanning nonresident placements in `src/viewer/WoWViewer/`.
- [ ] T038 [US5] Add p95 context-evaluation and lighting-selection timing fields to the existing diagnostics schema under `src/viewer/WoWViewer/Diagnostics/` or its current owner.
- [ ] T039 [US5] Run focused tests and the isolated viewer build; record results in `specs/143-world-context-lighting/quickstart.md`.
- [ ] T040 [US5] Hand the user the PowerShell real-client matrix from `specs/143-world-context-lighting/quickstart.md` and review returned ADT/WMO/camera/lighting/frame-stage evidence.
- [ ] T041 [US5] Fix any evidence-backed cross-era or performance regression before release documentation is updated.

**Checkpoint**: The user-run evidence meets the spec acceptance criteria or identifies a bounded failure with no parity claim.

## Phase 8: Polish and release hygiene

- [ ] T042 [P] Update `specs/143-world-context-lighting/spec.md`, `plan.md`, `research.md`, and `data-model.md` with the final proven profiles and known gaps.
- [ ] T043 [P] Update `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` with implementation status and proof level.
- [ ] T044 [P] Add the world-context and lighting diagnostics to the lower status-bar contract without reviving deprecated panels.
- [ ] T045 Run `git diff --check` and review `git status --short`; do not stage unrelated worktree changes.
- [ ] T046 Run the quickstart focused commands and record whether the feature is library/build proven, viewer runtime proven, or real-client proven.

## Dependencies and execution order

- Setup tasks T001-T004 precede all implementation.
- Foundational tasks T005-T008 block every user story.
- US1 (T009-T015) must pass before US2 because WMO fallback depends on the ADT context contract.
- US2 (T016-T022) and US3 (T023-T028) share the snapshot boundary; complete both before US4.
- US4 (T029-T035) consumes the camera/context and Spec 106/138 lighting inputs.
- US5 (T036-T041) is the release gate; it does not authorize whole-map loading or a graph-path default.
- Polish tasks T042-T046 follow the final evidence review.

## Parallel opportunities

- T001-T003 can run in parallel because they update separate evidence sections.
- T005-T006 and T009/T016/T023/T029 test scaffolds can run in parallel once the contract names are fixed.
- US1, US2, and US3 implementation files are mostly separable, but integration checkpoints remain ordered.
- T042-T044 can run in parallel after the proof review.
