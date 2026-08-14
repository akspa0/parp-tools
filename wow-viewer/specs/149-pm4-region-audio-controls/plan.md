# Implementation Plan: PM4 Region Navigation and Audio Trigger Controls

**Branch**: `149-pm4-region-audio-controls` | **Date**: 2026-08-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/149-pm4-region-audio-controls/spec.md`

## Summary

Replace the PM4 workbench's correlation-first interaction with a decoded region browser backed by
the existing `WorldScene` PM4 overlay objects. Aggregate region identity, surface/object totals, and
finite bounds from the resident PM4 object set; let the viewer-owned camera focus a selected region
through the normal residency path. Remove user-facing matching/correlation controls and fields after a
caller audit, while retaining any genuinely non-UI research owner.

Add an explicit world-audio trigger policy to `WorldAudioRuntime`. Keep MCSE and current-area/ZoneMusic
diagnostics visible, add legacy MCNK environmental/liquid candidates for 0.5.3 maps with no MCSE, and
require a default-off master gate and per-trigger enablement before any world source can start. Normalize
MCSE local coordinates against the owning tile/chunk before range checks or OpenAL placement. Leave
explicit SoundEntries preview, decoder ownership, AreaNumber resolution, and MIDI/DLS capability
reporting separate.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Existing `WowViewer.Core.PM4` models, `WorldScene`, `TerrainChunkData` /
`LiquidChunkData`, ImGui.NET, Silk.NET OpenAL wrapper, existing DBC/DB2 and MCSE readers

**Storage**: In-memory resident PM4/audio state; no new persistent storage

**Testing**: Focused `WowViewer.Core.Tests` plus Debug solution build; user-owned real-client visual,
streaming, and audible proof

**Target Platform**: Windows desktop viewer with configured client data source

**Project Type**: Desktop application with shared core libraries

**Performance Goals**: Region aggregation and trigger-control changes must not add whole-map scans or
per-frame UI allocations; world updates remain bounded by existing resident tile/object sets.

**Constraints**: Reuse canonical readers and coordinate transforms; add only the tile/chunk-local audio
normalization required by the proven MCSE record frame; no guessed PM4 matching or audio asset pairing;
world-trigger playback is opt-in and default-off; no proprietary assets in the repo.

**Scale/Scope**: Current PM4 resident/loaded session and current audio residency set; one active world
session; three independently testable P1 stories.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

* **PASS — Repo independence**: all code and tests remain under `wow-viewer/`; client roots stay
  runtime configuration.
* **PASS — Library-first**: PM4 aggregation/region contracts and audio enablement remain shared/core or
  runtime owners; ImGui only renders and forwards user intent.
* **PASS — Real-data validation**: fixtures and configured client-backed proof are named separately;
  this plan does not claim real-client proof from unit tests.
* **PASS — Format ownership**: no PM4, MCSE, DBC, DB2, or decoder rewrite is planned.
* **PASS — Streaming-first**: region and audio lists use resident data and do not trigger whole-map
  loading.
* **PASS — One phase at a time**: implement and validate region contracts before the UI retirement and
  audio control follow-through are treated as complete.

## Project Structure

### Documentation (this feature)

```text
specs/149-pm4-region-audio-controls/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
└── tasks.md
```

### Source Code (repository root)

```text
src/core/WowViewer.Core.PM4/Models/     # existing PM4 region/object contracts
src/core/WowViewer.Core/Audio/          # audio trigger control contracts if shared
src/viewer/WoWViewer/Terrain/            # WorldScene PM4/audio runtime facades
src/viewer/WoWViewer/Audio/              # WorldAudioRuntime playback policy
src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs
src/viewer/WoWViewer/ViewerApp_Audio.cs
src/viewer/WoWViewer/ViewerApp_Sidebars.cs
src/viewer/WoWViewer/ViewerApp_Workspaces.cs
tests/WowViewer.Core.Tests/              # focused contract/state coverage
specs/149-pm4-region-audio-controls/      # design and implementation handoff
```

**Structure Decision**: Extend the existing library-first PM4 and audio owners. `WorldScene` exposes
resident decoded snapshots and forwards selection/control operations; `ViewerApp` owns ImGui row
interaction and authoritative camera mutation; `WorldAudioRuntime` owns trigger gating and source
lifecycle. No new project or persistence layer is warranted.

## Implementation Phases

1. **Foundation and contracts**: Add/extend region navigation and audio enablement data contracts,
   expose deterministic resident snapshots, and add focused tests for default states and finite region
   aggregation.
2. **PM4 region navigation**: Build the region list, selected-region state, double-click focus request,
   camera framing, and bounded residency handoff. Validate this story before deleting UI paths.
3. **PM4 presentation retirement**: Remove correlation tabs/windows, match actions, saved-match display,
   stale sidebar/workspace copy, and matching text from PM4 tooltips. Audit non-UI callers before removing
   orphaned state/services.
4. **Audio trigger controls**: Project MCNK flags/liquid data and MCSE into a typed resident trigger
   list, normalize MCSE positions before range evaluation, gate MCNK/MCSE/area/ZoneMusic starts behind
   default-off master/per-instance state, and prove stop/no-duplicate behavior in focused tests.
5. **Resident area overlay**: Expose a revisioned snapshot of resident chunk metadata, aggregate
   map-aware Zone/SubZone groups through `AreaTableService`, render chunk-footprint bounds and pins from
   `WorldScene`, and project one labeled name tag per group in `ViewerApp`. Keep the visualization
   opt-in and omit unresolved rows from geometry while reporting their count.
6. **Cross-cutting validation and handoff**: Run focused tests, Debug build, source audit, and prepare
   the configured-client/user-run visual and audible proof checklist. Update the active dashboard only
   when the implementation handoff changes.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| None | N/A | The change stays inside existing PM4, viewer, and audio owners. |
