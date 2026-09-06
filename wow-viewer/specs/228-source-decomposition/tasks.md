# Tasks: Source Decomposition — God-Class Split

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md), and [quickstart.md](quickstart.md)

**Governing rule**: a task may be checked only with an owning receipt that names files changed,
commands and exit status, before/after line counts, and criterion-to-evidence mapping. Build/test
output does not close a visual, input, rendering, capture, or performance gate.

## Phase 1: Baseline and containment

**Purpose**: establish a reproducible source-size and dependency baseline without editing product
code.

- [ ] T001 Record baseline counts, all present over-budget files, and the first selection candidate's callers in `specs/228-source-decomposition/evidence/t001-source-baseline.md`.
- [ ] T002 Record the proposed selection boundary's current public routes and UI-authority dependency on Spec 227 T004 in `specs/228-source-decomposition/evidence/t002-selection-boundary.md`.

**Checkpoint**: No selection source changes begin until T001–T002 have receipts and Spec 227 T004
is checked with its own source and operator evidence.

## Phase 2: External UI-authority gate

**Purpose**: confirm that a refactor cannot silently select a new Inspector or selection UI home.

- [ ] T003 Verify and cite the accepted Spec 227 T004 selection/Inspector authority before any Spec 228 source change in `specs/228-source-decomposition/evidence/t003-ui-authority-gate.md`.

**Checkpoint**: The UI authority is recorded. A pure selection contract may now be implemented
without moving a presentation route.

## Phase 3: User Story 3 — Context-window safety (Priority: P0)

**Purpose**: make selection policy testable without introducing a back-reference into a god class.

- [ ] T004 [US3] Create immutable request, snapshot, candidate, and result records in `src/core/WowViewer.Core.Runtime/World/Selection/` per `data-model.md`.
- [ ] T005 [P] [US3] Add deterministic ranking, WMO container fall-through, distance-limit, and invalid-input tests in `tests/WowViewer.Core.Tests/World/Selection/WorldSceneSelectionServiceTests.cs`.
- [ ] T006 [US3] Implement the pure `WorldSceneSelectionService` in `src/core/WowViewer.Core.Runtime/World/Selection/WorldSceneSelectionService.cs` without a `WorldScene`, GL, ImGui, or parent-delegate reference.
- [ ] T007 [US3] Run `dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldSceneSelection"` and record the result in `specs/228-source-decomposition/evidence/t007-selection-contract.md`.

**Checkpoint**: The Core selection service is independently testable. It does not yet change the
viewer route or claim an extraction.

## Phase 4: User Story 1 — WorldScene decomposition (Priority: P1)

**Goal**: move one UI-authorized hover/click selection algorithm out of `WorldScene` while
preserving existing public route behavior.

**Independent test**: automated tests preserve policy ordering; an operator uses the existing
hover/click route on real scene data and records identical target behavior for the selected cases.

- [ ] T008 [US1] Add the viewer-edge snapshot/result mapper in `src/viewer/WoWViewer/Terrain/Scene/Selection/WorldSceneSelectionAdapter.cs` using only explicit resident-scene inputs and outputs.
- [ ] T009 [US1] Replace the selected hover/click algorithm in `src/viewer/WoWViewer/Terrain/WorldScene.cs` with the narrow adapter/delegation seam; remove the moved helpers and add no partial-class file or new UI route.
- [ ] T010 [US1] Run focused `WorldSceneSelection` tests, `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`, and `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`; record exit status and before/after counts in `specs/228-source-decomposition/evidence/t010-worldscene-selection-extraction.md`.
- [ ] T011 [US1] Obtain and record the operator hover/click smoke on a configured client root in `specs/228-source-decomposition/evidence/t011-worldscene-selection-smoke.md`; leave the task open if a target identity, terrain occlusion, or Inspector route differs.

**Checkpoint**: `WorldScene` has a net source reduction, the moved owner is below the file budget,
and no rendering/input acceptance is inferred without T011.

## Phase 5: User Story 2 — ViewerApp decomposition (Priority: P1)

**Goal**: select exactly one post-UI-audit ViewerApp feature for a later owned-service extraction.

**Independent test**: the candidate has a current authority decision, behavior matrix, and a
bounded public contract before any `ViewerApp` implementation moves.

- [ ] T012 [US2] Choose one candidate only after the relevant Spec 223 runtime gate and Spec 227 authority row are recorded; write the behavior matrix, current owner, and intended service path in `specs/228-source-decomposition/evidence/t012-viewerapp-candidate.md`.
- [ ] T013 [US2] Add the candidate-specific implementation/test/operator-smoke tasks as a dated approved amendment to `specs/228-source-decomposition/tasks.md`; do not implement capture, camera, sidebar, editor, or PM4 extraction from an assumed future plan.

**Checkpoint**: ViewerApp remains unchanged until a real candidate receipt establishes a safe
single-feature extraction.

## Phase 6: Cross-cutting context-safety receipt

**Goal**: record the completed owner so future work reads an owned service and its contract rather
than the whole legacy class.

**Independent test**: a fresh-session handoff can name the owning service, its tests, and the
legacy adapter seam in no more than the linked artifacts.

- [ ] T014 Update the current Spec 228 handoff in `memory-bank/activeContext.md` and record the completed owner, seam, counts, proof status, and next bounded candidate in `memory-bank/progress.md` after each accepted extraction receipt.
- [ ] T015 Re-run the source-size inventory and record that no new source file exceeds the budget and no legacy breach grew in `specs/228-source-decomposition/evidence/t015-context-safety-audit.md`.

## Dependencies and execution order

`T001 → T002 → T003 → T004–T007 → T008–T011 → T012–T013 → T014–T015`.

- T005 can be prepared in parallel with T004 because it is a separate test file, but both block T006.
- T003 is externally blocked by Spec 227 T004; no Phase 3 source edit is allowed before it.
- T012 is externally blocked by the relevant Spec 223 and Spec 227 evidence; it does not authorize
  a ViewerApp code change on its own.

## Implementation strategy

1. Complete the source baseline and the pure Core contract.
2. Stop at the Spec 227 gate rather than guessing a UI home.
3. Extract only selection; validate and obtain the operator smoke.
4. Select the next `ViewerApp` owner only from recorded evidence, then add a separate bounded task
   amendment.
