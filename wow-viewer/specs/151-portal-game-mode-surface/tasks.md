# Tasks: Portal-Aware Rendering, Game Mode, and Simple Viewer Surface

**Input**: Design documents from `/specs/151-portal-game-mode-surface/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `contracts/`

**Execution rule**: Implement one phase, run its focused validation, and stop at the checkpoint
before starting the next phase. `[P]` tasks have disjoint write scopes and may run in parallel.

## Phase 1: Portal Visibility MVP (first implementation slice)

**Purpose**: Replace the current center-distance-only heuristic where portal evidence is trustworthy,
while preserving conservative rendering for unknown or malformed client-era data.

- [ ] T001 [P] [US1] Add focused portal-volume fixture builders and expected visibility cases in `tests/WowViewer.Core.Tests/World/WmoPortalVisibilityDecisionTests.cs` covering exterior rejection, interior reachability, camera-on-boundary, cycles, and depth/capacity limits.
- [ ] T002 [P] [US1] Add a pure `WmoPortalVisibilityDecision`/diagnostics model in `src/core/WowViewer.Core.Runtime/World/WmoPortalVisibilityDecision.cs` using existing decoded portal/group read models and no file reads.
- [ ] T003 [US1] Implement bounded portal plane/clip-volume traversal in `src/core/WowViewer.Core.Runtime/World/WmoPortalVisibilityEvaluator.cs`, documenting the native 0.5.3 evidence mapping and returning a conservative visible set for invalid transforms, missing side data, malformed geometry, or overflow.
- [ ] T004 [US1] Reconcile `src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldScenePortalVisibilityEvaluator.cs` with the shared decision contract or make its bridge explicit so graph diagnostics and final renderer admission cannot silently disagree.
- [ ] T005 [US1] Integrate the decision into `src/viewer/WoWViewer/Rendering/WmoRenderer.cs` at `UpdateRuntimeVisibility`, replacing center-distance admission only when the decision reports trusted portal clipping and retaining the existing fail-open path otherwise.
- [ ] T006 [US1] Apply the final group decision consistently to WMO group geometry, doodad definitions, and liquid buffers in `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`; record portal-tested, fallback, and admitted-group counts in existing render stats.
- [ ] T007 [US1] Add/extend renderer-independent tests in `tests/WowViewer.Core.Tests/World/WorldScenePortalVisibilityEvaluatorTests.cs` for invalid/singular transforms, missing geometry, malformed edges, and conservative fallback.
- [ ] T008 [US1] Run focused portal tests and a Debug build; record the result and remaining real-client proof boundary in `specs/151-portal-game-mode-surface/quickstart.md` and the continuity files.

**Checkpoint**: Portal visibility is unit-tested and structurally integrated. Do not begin game-mode
input work until this checkpoint passes.

## Phase 2: Game-Mode Runtime Core

**Purpose**: Provide deterministic character-head camera and bounded movement physics without
replacing the editor camera.

- [ ] T009 [P] [US3] Add `GameModeState`, `PhysicsBodyState`, movement settings, and anchor-source records in `src/core/WowViewer.Core.Runtime/World/GameMode/GameModeState.cs`.
- [ ] T010 [P] [US4] Add deterministic physics/grounding tests in `tests/WowViewer.Core.Tests/World/GameModePhysicsTests.cs` for walk/run, gravity, jump gating, clamped delta, ground contact, collision fallback, and finite output.
- [ ] T011 [US3] Implement the pure bounded integrator in `src/core/WowViewer.Core.Runtime/World/GameMode/GameModePhysics.cs`, consuming movement intent and an injected collision/ground resolver.
- [ ] T012 [US3] Implement the model-owned head-anchor provider in `src/core/WowViewer.Core.Runtime/World/GameMode/CharacterHeadAnchorProvider.cs`, using recognized model attachment data when present and a finite model-height fallback otherwise.
- [ ] T013 [US4] Add a viewer-facing collision adapter in `src/viewer/WoWViewer/Terrain/WorldScene.cs` or a focused partial/helper that reuses `TryResolveCameraPathCollision` semantics without changing free-fly behavior.
- [ ] T014 [US3] Add opt-in state transition and editor-pose preservation in `src/viewer/WoWViewer/ViewerApp.cs` and a focused partial `src/viewer/WoWViewer/ViewerApp_GameMode.cs`.
- [ ] T015 [US4] Wire bounded keyboard movement, run modifier, jump, mouse look, and camera projection in `src/viewer/WoWViewer/ViewerApp_GameMode.cs`; clamp frame delta before physics and leave existing editor input unchanged when disabled.
- [ ] T016 [US3] Add game-mode status/control presentation to the simple-surface owner and save only explicit user settings in `src/viewer/WoWViewer/ViewerApp.cs`/`ViewerSettings`.
- [ ] T017 [US4] Run focused GameMode tests and Debug build; record that real-client visual movement and collision proof remains user-owned.

**Checkpoint**: Game mode can be toggled without corrupting editor camera state and its pure physics
tests pass.

## Phase 3: Simple Interactive Surface and Diagnostic Budget

**Purpose**: Make ordinary viewing cheap and approachable while preserving the advanced data explorer.

- [ ] T018 [P] [US2] Add pure `ViewerSurfaceProfile` and `DiagnosticProfile` policy types in `src/core/WowViewer.Core.Runtime/World/ViewerSurfaceProfile.cs` or the existing shared runtime policy owner.
- [ ] T019 [P] [US5] Add policy tests in `tests/WowViewer.Core.Tests/Viewer/ViewerSurfaceProfileTests.cs` proving simple defaults hide raw workbench refresh while errors/counters remain available and advanced/forensic restores access.
- [ ] T020 [US5] Add explicit interactive/forensic gating to `src/viewer/WoWViewer/Logging/ViewerLog.cs`, avoiding history-lock/formatting work for suppressed low-level messages while retaining errors and a bounded counter path.
- [ ] T021 [US2] Add the simple interactive surface in a focused partial `src/viewer/WoWViewer/ViewerApp_SimpleSurface.cs`, exposing load/camera/game/audio/region controls and a reversible advanced-surface action.
- [ ] T022 [US2] Gate raw data explorer panels, correlation workbench refreshes, per-frame emitter/area diagnostics, and verbose overlay refreshes in `src/viewer/WoWViewer/ViewerApp.cs`/existing sidebar partials based on the surface/profile policy; do not delete those routes.
- [ ] T023 [US5] Connect the simple surface to existing scene timing/cull counters and expose concise WMO portal fallback/admission status without dumping raw payloads.
- [ ] T024 [US2] Persist the selected surface/profile only through the existing settings contract and default new sessions to `SimpleInteractive` with audio triggers off unless explicitly enabled by the user.
- [ ] T025 [US5] Run policy tests, focused viewer build, and the quickstart controlled comparison; record stage/counter evidence without claiming user-owned FPS proof.

**Checkpoint**: The simple surface is usable without the data explorer and advanced/forensic tools
remain reachable.

## Phase 4: Documentation, Integration, and Handoff

- [ ] T026 [P] Update `wow-viewer/specs/STATUS.md` with Spec 151 phase/proof status and the next bounded task.
- [ ] T027 [P] Update `wow-viewer/memory-bank/activeContext.md` with the active branch, phase, Ghidra evidence, proof owner, and out-of-scope boundaries.
- [ ] T028 [P] Add a newest-first entry to `wow-viewer/memory-bank/progress.md` for each completed implementation checkpoint.
- [ ] T029 Run `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` after the bounded slice is complete.
- [ ] T030 Prepare the user-owned real-client validation handoff with exact client root/build, WMO fixture, camera positions, surface/profile, and expected counters in `specs/151-portal-game-mode-surface/quickstart.md`.

## Dependencies & Execution Order

### Phase Dependencies

- Phase 1 is independent of game mode and is the first implementation checkpoint.
- Phase 2 depends on the existing camera/collision seams and may begin only after Phase 1 validation.
- Phase 3 depends on game-mode controls existing but must preserve the advanced shell; it follows Phase 2.
- Phase 4 depends on the completed bounded implementation slice and records proof/continuity.

### Parallel Opportunities

- T001 and T002 can run in parallel because they write different files.
- T009 and T010 can run in parallel.
- T018 and T019 can run in parallel.
- T026-T028 can run in parallel after implementation decisions are settled.

### Implementation Strategy

1. Complete Phase 1 portal visibility MVP.
2. Run focused tests/build and inspect the diff; stop if the conservative fallback changes.
3. Complete Phase 2 game-mode core and stop for focused validation.
4. Complete Phase 3 simple surface/diagnostic policy and stop for controlled comparison.
5. Finish continuity docs and hand off real-client proof; do not claim visual/FPS/audio validation from build/tests.
