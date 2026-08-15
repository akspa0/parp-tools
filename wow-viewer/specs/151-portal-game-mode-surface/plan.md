# Implementation Plan: Portal-Aware Rendering, Game Mode, and Simple Viewer Surface

**Branch**: `151-portal-game-mode-surface` | **Date**: 2026-08-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/151-portal-game-mode-surface/spec.md`

## Summary

Add a bounded native-informed WMO portal visibility decision at the renderer's final group
submission seam; add a pure, testable game-mode/physics contract that can project a character head
anchor onto the existing camera; add a reversible simple interactive surface/profile; and make
interactive diagnostics avoid raw/per-frame work while preserving a forensic path. The first coding
phase is the portal visibility decision and focused tests, because it is independently testable and
has direct Ghidra evidence.

## Technical Context

**Language/Version**: C# on the repository's existing .NET 9 projects

**Primary Dependencies**: Existing `WowViewer.Core.Runtime` world/scene graph contracts, WMO decoded
read models, Silk.NET OpenGL, ImGui.NET, existing `Camera`, `WorldScene`, `WmoRenderer`, and
`ViewerLog` owners

**Storage**: Runtime client/archive data plus existing persisted viewer settings; no new database

**Testing**: Existing xUnit-style `WowViewer.Core.Tests`; focused `dotnet test` followed by solution
build. Runtime visual/FPS/audio proof is user-owned.

**Target Platform**: Windows desktop OpenGL viewer

**Project Type**: Desktop viewer plus shared core/runtime libraries

**Performance Goals**: Reduce WMO interior group/doodad/liquid submissions where portal evidence
proves they are unreachable; reduce interactive raw logging/diagnostic refresh work; preserve
bounded counters for comparison. No fixed FPS target is claimed before a controlled fixture is
measured.

**Constraints**: Library-first; no duplicate readers; no original client code port; fail-open on
unknown/malformed portal data; game mode off by default; advanced surface remains available; no
hardcoded client root; all loops and scratch buffers bounded.

**Scale/Scope**: One loaded world/WMO placement per visibility decision, one local game-mode body,
and one shell profile. This slice does not implement multiplayer or full game simulation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Repository independence**: PASS. New contracts remain in `wow-viewer`; no reference-tree edits.
- **Library-first ownership**: PASS. WMO visibility/physics contracts belong in core/runtime where
  pure; renderer and shell integration remain in their existing owners.
- **Real-data validation**: PASS. Portal tests use decoded read models and a recorded 0.5.3 Ghidra
  evidence note; no invented archive data is used.
- **No hardcoded client path**: PASS. `H:\CLIENTS` appears only as an operator validation example,
  never as a source default.
- **One bounded phase at a time**: PASS. Phase 1 is portal visibility and tests; later game-mode,
  surface, and diagnostics phases depend on explicit checkpoints.
- **Preserve working routes**: PASS. Existing editor/free camera, advanced explorer, readers, and
  forensic diagnostics remain available.

## Project Structure

### Documentation

```text
specs/151-portal-game-mode-surface/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── portal-visibility.md
│   ├── game-mode.md
│   └── surface-diagnostics.md
├── checklists/requirements.md
└── tasks.md
```

### Source Code

```text
src/core/WowViewer.Core.Runtime/World/
├── SceneGraph/                         # existing portal graph contracts/evaluator
├── WmoPortalVisibilityDecision.cs      # planned shared decision model/evaluator
└── GameMode/                            # planned pure body/anchor/input contracts

src/viewer/WoWViewer/
├── Rendering/WmoRenderer.cs             # final WMO group admission integration
├── Rendering/Camera.cs                  # existing camera projection target
├── Terrain/WorldScene.cs                # collision and frame counters
├── Logging/ViewerLog.cs                 # interactive/forensic policy seam
└── ViewerApp*.cs                        # surface, input, and persisted settings integration

tests/WowViewer.Core.Tests/
├── World/                               # portal/physics contract tests
└── Viewer/                              # only if shell policy has a testable pure owner
```

**Structure Decision**: Extend existing core/runtime WMO and game-mode owners and keep UI wiring in
the viewer shell. Do not add another renderer, reader, or data-explorer implementation.

## Phase Plan

### Phase 0 - Research and baselines (complete in this planning pass)

- Record the live Ghidra native portal anchors and the current renderer gap in `research.md`.
- Confirm existing test/render timing seams and the conservative fallback boundary.

### Phase 1 - Portal visibility MVP (first implementation slice)

- Add a shared pure portal-volume/decision contract that can use existing decoded geometry.
- Replace center-distance-only admission for trusted portal data with bounded plane/clip-volume
  traversal at the WMO group seam.
- Keep an explicit conservative fallback for missing side data, malformed geometry, singular
  transforms, near-root ambiguity, and capacity/depth overflow.
- Apply the result to group geometry, doodads, and liquids and expose bounded counters.
- Add focused tests for exterior rejection, interior reachability, cycles/depth, malformed data, and
  camera boundary behavior.
- Stop and validate before starting game mode.

### Phase 2 - Game-mode runtime core

- Add pure `GameModeState`/`PhysicsBodyState`/head-anchor contracts and deterministic tests.
- Add model-owned head-anchor provider with finite height fallback.
- Integrate opt-in input/update projection in `ViewerApp` while preserving the editor pose.
- Reuse `WorldScene` collision seams and document their conservative limits.
- Stop for focused tests/build and user-owned visual proof.

### Phase 3 - Simple surface and diagnostic budget

- Add `ViewerSurfaceProfile` and `DiagnosticProfile` with simple surface default behavior.
- Gate expensive/raw workbench refresh and verbose logging at the existing UI/diagnostic owners.
- Add concise game/audio/region/camera controls and reversible advanced-surface entry.
- Keep raw diagnostics accessible only after explicit advanced/forensic action.
- Add pure policy tests and a controlled stage-counter comparison path.

### Phase 4 - Documentation and handoff

- Update `STATUS.md`, `activeContext.md`, and `progress.md` with implementation/proof status.
- Run focused tests, solution build, and the prescribed quickstart checks.
- Hand off exact real-client visual/FPS/audio proof commands and stop before user-owned heavy runs.

## Risks and Mitigations

- **Portal side semantics differ by client-era data**: retain conservative fail-open fallback and
  test known 0.5.3 fixtures before tightening admission.
- **Existing graph and renderer authorities diverge**: share the pure decision/volume contract or
  explicitly bridge the graph result; do not silently maintain two independent algorithms.
- **Game-mode collision is not a player mesh**: label AABB/heightfield behavior and keep editor/free
  camera fallback rather than pretending to provide full game physics.
- **UI profile hides a needed tool**: preserve advanced entry and profile switch while hiding only
  default presentation/refresh work.
- **Logging cost is at call sites, not only output**: avoid formatting/raw collection before the
  profile gate; retain errors and bounded counters.

## Complexity Tracking

No constitution violations. The feature spans multiple existing owners because the request combines
renderer visibility, camera/runtime state, and shell presentation, but each phase has one primary
owner and an explicit validation checkpoint.
