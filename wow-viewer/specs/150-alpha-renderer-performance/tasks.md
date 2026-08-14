# Tasks: Alpha 0.5.3 Renderer Performance Evidence and Optimization

## Phase 1 — Evidence and baseline

- [ ] T001 Read the existing `profile-render` report path and record the current stage/workload fields in `specs/150-alpha-renderer-performance/research.md`.
- [ ] T002 [P] Add the 0.5.3 native renderer evidence ledger to `memory-bank/workstream-alpha053-renderer-performance.md`, recording anchors for world, terrain, object, resource/state, and LOD behavior or explicit unknowns.
- [ ] T003 [P] Prepare one fixed 0.5.3 control-scene identity in `specs/150-alpha-renderer-performance/quickstart.md` without committing client data or reports.
- [ ] T004 Run two unchanged-source `profile-render` captures on the user-configured 0.5.3 client and record the dominant-owner decision and variance; do not select an optimization before this gate.

**Checkpoint**: Native evidence and a repeatable current-renderer baseline exist; no source behavior
has changed.

## Phase 2 — Attribution contract

- [ ] T005 [P] Extend `src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs` only with counters required by the selected owner, keeping CPU and GPU/driver timing separate.
- [ ] T006 [P] Extend `src/core/WowViewer.Core.Runtime/World/WorldRenderDiagnostics.cs` and its report tests with explicit unavailable/failed GPU timing and dominant-owner findings.
- [ ] T007 Thread the new counters through the owning production renderer/`WorldScene` path without changing render behavior.
- [ ] T008 Update the existing `profile-render` JSON output and Runtime Stats presentation only for the new attribution fields.
- [ ] T009 Add focused coverage in `tests/WowViewer.Core.Tests/World/WorldRenderDiagnosticsTests.cs` for schema stability, workload/counter consistency, and dominant-owner selection.

**Checkpoint**: Repeated profiles identify the same owner with the counters needed for one A/B test.

## Phase 3 — One reversible optimization

- [ ] T010 Choose exactly one measured owner and document the expected counter movement and stop condition in `research.md`.
- [ ] T011 [P] Add the old/new path switch or fallback boundary in the measured owner under `src/viewer/WoWViewer/` or `src/core/WowViewer.Core.Runtime/World/`, keeping unsupported content on the existing path.
- [ ] T012 Implement one bounded optimization: scratch reuse, admission ordering, compatible opaque grouping, state/uniform reduction, retained resource reuse, or build-scoped LOD.
- [ ] T013 Add focused tests for the changed decision and fallback behavior in `tests/WowViewer.Core.Tests/`.
- [ ] T014 Run baseline/candidate `profile-render` captures with the same control-scene identity and record the result in the experiment contract.

**Checkpoint**: The candidate either meets the measurable gate or is disabled/rejected with the
negative result preserved.

## Phase 4 — Optional build-scoped follow-through

- [ ] T015 Only if Phase 3 passes, add one separately measured visibility/LOD or resource-state slice using the corresponding 0.5.3 evidence row.
- [ ] T016 Keep Alpha terrain holes/liquids, WMO, M2/MDX, and tile residency in the same focused visual/counter matrix.
- [ ] T017 Do not generalize a 0.5.3 result to later builds without a separate evidence row and control scene.

## Phase 5 — Validation and handoff

- [ ] T018 [P] Update `specs/150-alpha-renderer-performance/quickstart.md` with final focused test and profile commands.
- [ ] T019 [P] Update `memory-bank/activeContext.md`, `memory-bank/progress.md`, and `specs/STATUS.md` only after the phase proof changes the handoff.
- [ ] T020 Run focused tests, `git diff --check`, and `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --no-restore`.
- [ ] T021 Hand off user-owned interactive native-client/viewer visual and FPS comparison with build, client root, scene, resolution, and timing classifications.

## Dependencies and execution order

- Phase 1 blocks all source changes.
- Phase 2 depends on the selected owner from Phase 1.
- Phase 3 depends on Phase 2 and contains exactly one optimization.
- Phase 4 is optional and depends on an accepted Phase 3 experiment.
- Phase 5 follows the desired accepted/rejected phase and never converts build proof into FPS proof.

## Out of scope

- Original client code porting, renderer backend replacement, Vulkan, compute shaders, full-map loading,
  game mode, PM4/audio work, and cross-era performance claims.
