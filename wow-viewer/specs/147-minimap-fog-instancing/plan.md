# Implementation Plan: Minimap, Fog-Bounded Residency, and Doodad Instancing

**Branch**: `147-minimap-fog-instancing` | **Date**: 2026-08-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `wow-viewer/specs/147-minimap-fog-instancing/spec.md`

## Summary

Repair the full-screen minimap interaction contract first, then make the active lighting-resolved
`fogEnd` the shared source for normal detailed tile/object coverage, and finally extend the existing
renderer batch paths into a compatibility-aware doodad asset/instance contract. The work is staged
so a bad interaction fix cannot be confused with a renderer regression, and so residency diagnostics
exist before dense doodad submission changes are judged.

The existing code already provides useful owners: `ViewerApp_MinimapAndStatus` owns minimap gesture
state, `WorldScene` resolves active fog, `TerrainManager` owns ADT streaming, the runtime directional
selector owns bounded tile ordering, and `WorldObjectPassCoordinator`/renderer interfaces own
submission grouping. This plan connects those owners with small shared contracts instead of adding a
second reader, a second lighting truth, or a second whole-map scene graph.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET OpenGL, ImGui.NET, existing `WowViewer.Core.Runtime` world and
visibility contracts, existing ADT/WMO/MDX/M2 readers and renderer interfaces

**Storage**: Runtime memory for interaction, residency, and batch state; existing client/cache
storage remains authoritative

**Testing**: Focused `WowViewer.Core.Tests` unit tests, source/build checks, and user-owned real
client capture/visual/FPS proof

**Target Platform**: Windows desktop viewer first; preserve the existing cross-platform project
surface where the shared contracts permit it

**Project Type**: Desktop OpenGL world viewer with shared core/runtime libraries

**Performance Goals**: Normal detailed residency must be bounded by effective fog coverage rather
than map size; repeated compatible doodads must use shared asset resources and grouped instance
submissions; no numeric FPS claim is closed until a named real-client capture proves it

**Constraints**: Preserve current coordinate conventions, Alpha/Standard terrain separation, WMO
interior/fallback behavior, capture-path preload leases, full-load diagnostics, and correctness
fallbacks for transparent/animated/effect-heavy doodads

**Scale/Scope**: Full-screen minimap input, ADT/object streaming admission, normal and WMO-internal
doodad submission, focused diagnostics, and their specs/tests; no WDL/audio/shader/reader rewrite

## Constitution Check

*GATE: Pass before implementation. Re-check after each implementation phase.*

- **Repo independence**: PASS. All implementation and tests remain under `wow-viewer/`; no client
  path is embedded.
- **Library-first**: PASS. Interaction remains viewer-owned; fog-window selection and batch keys are
  shared/runtime contracts where they can be tested without ImGui or OpenGL.
- **Real-data validation**: PASS with a user gate. Focused tests use deterministic fixtures; the
  final visual/FPS/capture proof must name the configured client root, build, map, and camera path.
- **Format ownership**: PASS. Existing ADT, WMO, MDX/M2, DBC, LIT, and minimap readers are reused.
- **Streaming-first**: PASS. The plan bounds runtime work and does not introduce a data-harvester or
  intermediate asset pipeline.
- **One phase at a time**: PASS. Each phase ends with focused validation before the next phase.
- **Memory-bank discipline**: PASS. The continuity dashboard and progress ledger are updated only
  after the planning artifact is complete and the next implementation slice is selected.
- **Workspace exception**: The Spec Kit branch helper could not create the new branch because the
  shared workspace denied `.git/index.lock`; this planning artifact is traceable on the current
  branch and preserves the unrelated `wow-viewer/imgui.ini` change.

## Phase 0 — Evidence and contract recovery

Status: complete as planning input; no production code changed.

1. Confirm the fullscreen and docked minimaps share the same interaction helper and document the
   drag/click classification boundary.
2. Confirm the fullscreen overlay is only drawn once. The current call sites in `ViewerApp.cs`
   can render it both in the shell pass and in the post-shell overlay pass, creating duplicate
   ImGui windows and duplicate interaction IDs.
3. Confirm `WorldScene` owns the effective fog range while `TerrainManager` currently ignores the
   supplied `fogEnd` for streaming target computation.
4. Inventory current directional/retained tile selectors, capture preload exceptions, object fog
   admission, and WMO containment rules.
5. Inventory existing `IGpuInstancedModelRenderer`, `IGpuInstancedWmoRenderer`, WMO doodad grouping,
   and MDX/M2 fallback capabilities.

Exit evidence: [research.md](research.md) records current owners, constraints, and decisions.

## Phase 1 — Minimap interaction repair (US1)

1. Define a viewer-owned gesture state contract that receives pointer-down, pointer-move,
   pointer-up, hover/active, and map-coordinate conversion results from the shared minimap surface.
2. Ensure the fullscreen overlay has one draw owner and one unique interaction surface ID; do not
   allow the shell and post-shell paths to process the same pointer event twice.
3. Make drag classification exclusive: panning updates the persisted offset, releases cancel any
   teleport sequence, and fullscreen parent-window state cannot consume the surface event first.
4. Make triple-click classification explicit and shared: same-target clicks advance the sequence,
   target changes/timeouts reset it, and the third click calls the existing camera/map transform.
5. Add deterministic interaction tests using synthetic pointer events/coordinates; keep ImGui and
   OpenGL out of the pure state machine tests.

Exit evidence: focused interaction tests pass and a source/build check shows both fullscreen and
docked surfaces use the same contract. Real-client click/drag proof remains user-owned.

## Phase 2 — Fog-bounded coverage contract (US2)

1. Define the effective fog coverage window in renderer world units, using the active `WorldScene`
   fog value and a conservative tile-bounds intersection policy.
2. Order fog-window candidates with the existing near-field and directional selector, but do not
   let direction reject a nearby side/rear tile whose bounds intersect the fog window.
3. Resolve the ordering between `WorldScene` fog evaluation and `TerrainManager.UpdateAOI` so the
   stream request consumes the same effective fog snapshot used by the frame, not a stale value.
4. Route normal `TerrainManager` desired-tile admission and unload protection through the fog window;
   keep retained residency, capture leases, and full-load diagnostics separately visible.
5. Apply the same normal coverage gate to tile-owned WMO/MDX admission and detailed liquid/terrain
   submission without changing WDL underlay ownership or WMO interior fallback rules.
6. Add hysteresis/reason codes and focused tests for fog changes, tile edges, camera movement,
   invalid fog, capture preload, and no-whole-map fallback.

Exit evidence: selector/residency tests pass, diagnostics distinguish selected/retained/resident/
drawable/preloaded sets, and the viewer builds. Real-client movement/capture proof remains
user-owned.

## Phase 3 — Doodad asset and instance batching (US3)

1. Define a deterministic doodad compatibility key covering asset identity, render pass, material/
   texture state, alpha/fade, animation/effect requirements, and WMO placement context.
2. Prepare one immutable geometry/material resource per asset and collect placement transforms into
   compatible buckets for the fog-visible set.
3. Connect normal MDX/M2 and WMO-internal doodad submission to the existing GPU/batch interfaces,
   preserving transparent ordering and named fallbacks for particles, ribbons, animation, and
   unsupported states.
4. Deduplicate asset preparation and animation work where the existing renderer semantics permit
   it; do not merge placement-local state accidentally.
5. Add deterministic batch grouping tests and per-frame counters for unique assets, buckets,
   instances, fallbacks, animation updates, and draw submissions.

Exit evidence: focused grouping/capability tests pass and the viewer builds. Dense real-client
visual/performance proof remains user-owned.

## Phase 4 — Diagnostics and proof handoff (US4)

1. Emit a compact structured frame record joining effective fog, tile state/reason, object admission,
   doodad batch/fallback counters, and per-stage CPU/submission timing.
2. Add invariant checks for near-field eviction, out-of-window normal admission, incompatible batch
   merges, duplicate fullscreen surfaces, and capture-lease accounting.
3. Update the quickstart with focused checks and a PowerShell-ready real-client capture recipe that
   records configured client root/build, map/tile, camera path, resolution, warm-up, and controls.
4. Run the solution build and focused tests; stop before user-owned long capture or FPS/audio/GPU
   proof.

Exit evidence: diagnostics are attributable, focused tests/build pass, and the user has an exact
runtime validation handoff.

## Project Structure

### Documentation

```text
wow-viewer/specs/147-minimap-fog-instancing/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── minimap-interaction.md
│   ├── fog-residency.md
│   └── doodad-instancing.md
├── checklists/requirements.md
└── tasks.md
```

### Source and tests

```text
wow-viewer/src/core/WowViewer.Core.Runtime/World/
├── Minimap/                         # pure interaction/coordinate contracts if extracted
├── Terrain/                         # fog coverage and tile selection contracts
└── Passes/                          # shared object/batch planning contracts

wow-viewer/src/viewer/WoWViewer/
├── ViewerApp_MinimapAndStatus.cs    # ImGui surface adapter only
├── Terrain/TerrainManager.cs        # desired/resident detailed tile ownership
├── Terrain/WorldScene.cs            # active fog and world-pass integration
└── Rendering/                       # asset/batch capability integration

wow-viewer/tests/WowViewer.Core.Tests/
├── Minimap*Tests.cs
├── *Tile*Tests.cs
└── *Batch*Tests.cs
```

**Structure Decision**: Keep UI adapters in the viewer shell, put pure state/selection/grouping
contracts in the existing core/runtime owners, and extend current renderer interfaces. Do not add a
new UI framework, parser, or parallel scene graph.

## Validation Gates

1. Requirements checklist has no unresolved items.
2. Phase 1 focused minimap state tests pass before any ImGui visual gate.
3. Phase 2 focused fog-window/residency tests pass before doodad submission changes.
4. Phase 3 batch grouping/capability tests pass before dense-scene runtime proof.
5. Solution Debug build and focused tests pass.
6. User-run real-client proof names client root/build and reports fog, tiles, objects, batches, and
   frame stages; no source-only FPS or visual claim is accepted.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Manual artifact continuation on the current branch | Spec Kit could not create `.git/index.lock` in the shared workspace | Retrying branch creation would not add evidence and could disturb the user's dirty UI settings |
| Three contracts in one feature | The minimap, residency, and doodad failures share a frame/residency boundary but have independent tests | Three unrelated specs would hide the required ordering and make the renderer phase start without a stable coverage contract |
