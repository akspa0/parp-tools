# Implementation Plan: Scene Graph Workbench

**Branch**: `045-scene-graph-workbench` | **Date**: 2026-06-03 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/045-scene-graph-workbench/spec.md`

## Summary

Introduce a viewer-owned scene graph workbench that projects the active loaded scene into a unified, Blender-like hierarchical outliner. The implementation should use a reusable snapshot/adapter contract spanning terrain, placed objects, and PM4, then bind that contract into a right-sidebar/dockable panel with selection sync and lazy filtering behavior.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Existing `WoWViewer` ImGui shell, `WowViewer.Core.Runtime`, `WowViewer.Core.PM4`, shared viewer selection and world-session surfaces

**Storage**: In-memory scene graph snapshots; optional JSON snapshot/export contract for debugging

**Testing**: `dotnet test` focused viewer/core tests plus bounded real-data viewer/manual validation on staged clients

**Target Platform**: Windows desktop viewer (`WoWViewer`) with the active dockable shell path

**Project Type**: Desktop app + shared runtime/library projection layer

**Performance Goals**: Opening the panel should avoid whole-scene eager expansion; branch expansion and filter application should stay interactive on normal loaded world sessions

**Constraints**:

- `wow-viewer` is the implementation owner; no new design work in `gillijimproject_refactor`
- Reuse existing decoded/runtime data; do not add duplicate parsers
- Read-only first slice; no editing/reparenting semantics
- Work must fit the active dockable shell restored by spec 044

**Scale/Scope**:

- Primary target is active world sessions with terrain, placed objects, and PM4 loaded
- First slice must handle large but normal interactive sessions, not every possible offline corpus at once
- Standalone asset scenes are a follow-up consumer of the same contract, not the first signoff surface

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Repo Independence**: Pass. All planned work lives under `wow-viewer/`.
- **Library-First**: Pass. The graph contract and domain projections belong in shared/runtime surfaces first; the viewer shell is only the host.
- **Real-Data Validation**: Pass. Manual validation targets staged clients and loaded real scenes; tests cover contract behavior.
- **No Duplicate Readers**: Pass. The workbench consumes already-decoded terrain/object/PM4 data and introduces no new parser stack.
- **Bite-Sized Planning**: Pass. Tasks are split into foundational contract work, domain projections, UI binding, and usability/perf slices.

## Project Structure

### Documentation (this feature)

```text
specs/045-scene-graph-workbench/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── scene-graph-snapshot.schema.json
└── tasks.md
```

### Source Code (repository root)

```text
wow-viewer/src/core/WowViewer.Core.Runtime/
└── SceneGraph/
    ├── SceneGraphSnapshot.cs
    ├── SceneGraphNode.cs
    ├── SceneGraphNodeId.cs
    ├── SceneGraphSelectionTarget.cs
    └── ISceneGraphDomainProvider.cs

wow-viewer/src/core/WowViewer.Core.PM4/
└── SceneGraph/
    └── Pm4SceneGraphProjector.cs

wow-viewer/src/viewer/WoWViewer/
├── SceneGraph/
│   ├── ViewerSceneGraphController.cs
│   ├── ViewerSceneGraphFilterState.cs
│   ├── ViewerSceneGraphPanelState.cs
│   └── ViewerSceneGraphTreeRenderer.cs
├── ViewerApp_SceneGraph.cs
└── [existing partials that host shell wiring and selection sync]

wow-viewer/tests/WowViewer.Core.Tests/
└── SceneGraph/
    ├── SceneGraphNodeIdTests.cs
    ├── SceneGraphSnapshotTests.cs
    ├── TerrainSceneGraphProjectorTests.cs
    ├── Pm4SceneGraphProjectorTests.cs
    └── ViewerSceneGraphSelectionTests.cs
```

**Structure Decision**: Put the reusable graph contract in `WowViewer.Core.Runtime`, PM4-specific projection in `WowViewer.Core.PM4`, and viewer-only filter/render/selection plumbing in `WoWViewer`. This preserves library-first ownership and keeps shell code from becoming the canonical data model.

## Complexity Tracking

No constitution violations currently require justification.
