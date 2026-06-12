# Implementation Plan: PM4 Scene Graph Semantics and Panel

**Branch**: `058-pm4-scene-graph-semantics-and-panel` | **Date**: 2026-06-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/058-pm4-scene-graph-semantics-and-panel/spec.md`

## Summary

Promote the PM4 scene graph from an embedded `CollapsingHeader` inside the PM4 Workbench sidebar to a standalone dockable panel (Blender-outliner style, expanded by default) with type-bucket grouping, an image preview of the resolved M2/WMO mesh, and a research-grade CK24 byte-pair bond-stats analyzer. The data-model fields (`Ck24HighByte`, `Ck24LowByte`) already exist on `Pm4MsurEntry` — the spec's FR-007 references `Pm4MslkEntry` by mistake; the plan corrects this.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: ImGui (via Silk.NET), existing PM4 research lib (`WowViewer.Core.PM4`), existing viewer shell (`WoWViewer`), existing M2/WMO render path for mesh preview

**Storage**: N/A (read-only PM4 data, cached in-memory graph/preview per selection)

**Testing**: `dotnet test` on `WowViewer.Core.PM4.Tests`; real-data validation against `test_data/development/World/Maps/development`

**Target Platform**: Windows desktop (ImGui/Silk.NET.OpenGL)

**Project Type**: desktop-app viewer + research library

**Performance Goals**: 60+ FPS with panel open on a 1000+ part container; panel render < 1s from click; bond-stats < 60s on full development corpus; mesh preview updates without rebuild

**Constraints**: Per-frame cost MUST stay O(cached) — no per-frame graph rebuilds (commit `42e83488` fix). No `H:\CLIENTS` references.

## Constitution Check

| Gate | Status | Notes |
|------|--------|-------|
| I. Repo Independence | PASS | All new code in `wow-viewer/` |
| II. Library-First | PASS | Data-model + analyzer in `Core.PM4`; panel in viewer |
| III. Real-Data Validation | PASS | Bond-stats validated against development corpus |
| IV. Residual Model Chain | N/A | No ML model work |
| V. Streaming-First Dataset | N/A | No dataset work |
| VI. No Game Client Path Assumptions | PASS | No client paths needed |

## Panel Design Decision

The right sidebar already has 9 registered `ShellPanelId` values. Adding a 10th is viable because:
- Registration ≠ visibility. Most users have 3-4 panels open at once.
- The scene graph is PM4-specific and belongs near the PM4 Workbench in the BottomRight quadrant.
- A separate `ShellPanelId` lets the user dock it wherever they want, including as a tab alongside the PM4 Workbench.
- The spec explicitly requests "separate dockable panel, expanded by default, with the layout of a Blender outliner."

## Project Structure

```text
wow-viewer/src/core/WowViewer.Core.PM4/
├── Models/
│   └── Pm4ResearchChunkModels.cs        # Ck24HighByte/Ck24LowByte ALREADY EXIST (line 83-85)
├── Research/
│   ├── Pm4ResearchLinkageAnalyzer.cs    # Extend with bond-stats metrics
│   ├── Pm4BondStatsReport.cs            # NEW: bond-stats report model
│   └── Pm4BondStatsAnalyzer.cs          # NEW: dedicated bond-stats analyzer
└── Matching/
    └── (existing PM4 matching code)

wow-viewer/src/viewer/WoWViewer/
├── ViewerApp.cs                         # ShellPanelId enum (add Pm4SceneGraph)
├── ViewerApp_Pm4Utilities.cs            # DrawPm4SceneGraph() extraction + type-bucket grouping
├── Terrain/WorldScene.cs                # Pm4SelectedObjectGraphInfo (add TypeBuckets)
└── Pm4MeshPreviewCache.cs               # NEW: per-selection mesh preview texture cache

wow-viewer/tools/inspect/WowViewer.Tool.Inspect/
└── Program.cs                           # NEW: pm4 bond-stats subcommand

wow-viewer/tests/WowViewer.Core.PM4.Tests/
├── Pm4BondStatsAnalyzerTests.cs         # NEW: bond-stats unit tests
└── (existing PM4 tests)
```

## Implementation Phases

### Phase 1: Data-Model Correction + Forensics Schema (Library)

**Goal**: Correct the spec's FR-007 type reference, update the XML docs, and add `ck24HighByte`/`ck24LowByte` to the forensics export.

**Approach**:
1. Correct spec FR-007 to reference `Pm4MsurEntry` (not `Pm4MslkEntry`). The fields already exist at `Pm4ResearchChunkModels.cs:83-85`.
2. Verify `Ck24ObjectId` XML doc includes "lossy flattening" label (partially done at lines 77-82).
3. Add `ck24HighByte` and `ck24LowByte` to the existing `pm4 forensics` export. Additive schema change.

**Validation**: Forensics export contains new fields; existing fields unchanged.

### Phase 2: Bond-Stats Analyzer + CLI Subcommand (Library + CLI)

**Goal**: Create the bond-stats analyzer with high×low cross-tabulation, unknowns/subtype correlation, and expose it through a CLI subcommand.

**Approach**:
1. Create `Pm4BondStatsReport` model in `Core.PM4/Research/`:
   - Per-file `Ck24HighByte.ReuseCountPerFile` and `Ck24LowByte.ReuseCountPerFile`
   - High×Low cross-tabulation: for each high-byte value, distribution of low-byte values
   - Low×High cross-tabulation (reversed)
   - Per-type-bucket breakdown
   - Correlation table: `Ck24HighByte` vs unknowns/subtype field indices, `Ck24LowByte` vs unknowns/subtype field indices
   - This answers the spec's research question: do the two bytes index into the same table or different tables?
2. Create `Pm4BondStatsAnalyzer` in `Core.PM4/Research/`:
   - `AnalyzeFile(string pm4Path)` — single-file analysis
   - `AnalyzeDirectory(string directory)` — directory-wide aggregation
   - Reuse `Pm4ResearchReader` for file loading
3. Add `pm4 bond-stats --input <file.pm4|directory> [--output <report.json>]` subcommand in `Tool.Inspect/Program.cs`.
4. Emit Markdown report alongside JSON (consistent with 046 match-report pattern).
5. Unit tests on real `development_00_00.pm4`.

**Validation**: `dotnet test` passes. CLI emits bond-stats report on development corpus with cross-tabulation and correlation data.

### Phase 3: Type-Bucket Grouping in Graph Data Model

**Goal**: Add type-bucket grouping to `Pm4SelectedObjectGraphInfo` so the scene graph can group by `Ck24Type` first, matching the user's mental model: "we have to start splitting up the scene graph by using the 0xAA value as a bucket for types of objects."

**Approach**:
1. Add `TypeBuckets` property to `Pm4SelectedObjectGraphInfo` in `WorldScene.cs:613`:
   - New record `Pm4SelectedObjectGraphTypeBucket` with `Ck24Type`, `TypeLabel`, `LinkGroups`
   - Keep existing `LinkGroups` property for backward compatibility; `TypeBuckets` is additive
2. Modify `TryGetSelectedPm4ObjectGraphInfo()` in `WorldScene.cs:11363` to partition link groups by `Ck24Type` before building the tree. Use MSLK TypeFlags lookup per surface via RefIndex.
3. Update cached-graph invalidation key to include type-bucket shape.

**Validation**: Existing graph tests pass; new type-bucket structure visible in debug output on `development_00_00.pm4`.

### Phase 4: Dockable PM4 Scene Graph Panel

**Goal**: Extract `DrawPm4SceneGraph()` from the PM4 Workbench sidebar into its own `ShellPanelId`-hosted dockable panel, with type-bucket grouping in the tree and per-user dock preferences that persist across sessions.

**Approach**:
1. Add `ShellPanelId.Pm4SceneGraph` to the enum in `ViewerApp.cs:62`.
2. Register in panel definition array: `new(ShellPanelId.Pm4SceneGraph, "PM4 Scene Graph", ShellPanelLane.Right, 420f, ...)`. Add to `BottomRightQuadrantPanels`.
3. Move `DrawPm4SceneGraph()` render call from the PM4 Workbench sidebar path to the shell-panel render path. Gate on `_showRightSidebar && _worldScene != null`.
4. Default-expanded when a PM4 object is selected; placeholder "Select a PM4 object" when no selection.
5. Tree rendering shows `Ck24Type` headers at the top level with semantic labels (`0x03 = M2 top`, `0x10 = interior WMO floor`, `0x12 = exterior WMO solid`). Link-group hierarchy nested under each type bucket. Per-entry `0xAA` value always visible.
6. Use cached `Pm4SelectedObjectGraphInfo` — no per-frame rebuild.
7. Per-user dock preferences (US3): persist panel position, expanded/collapsed state, and dock node across sessions via the existing `ShellPanel` persistence surface. The user can move the panel, close it, reopen it — state is restored on next viewer start.
8. Add PM4 Scene Graph to Tools menu toggle.

**Validation**: Panel docks/undocks/closes/reopens. Multi-type object shows grouped headers. Single-type shows one header with nested tree. 60+ FPS with 1000+ part container. No freeze on multi-instance selection. Panel position persists across restarts.

### Phase 5: Mesh Preview Cache + Panel Integration

**Goal**: Show a small image preview of the resolved M2/WMO mesh in the scene graph panel, giving the user a visual anchor while reading the graph.

**Approach**:
1. Create `Pm4MeshPreviewCache` in `WowViewer/src/viewer/WoWViewer/`:
   - Per-selection cache keyed by the same selection key as the graph
   - `GetOrCreatePreview(selectionKey, resolvedAssetPath)` → texture handle
   - Cleared on selection change
2. Resolve the selected object's M2/WMO via the existing asset-corpus surface (`Pm4AssetMatchScorer` matched candidates or validation placements) to find the asset path.
3. Render the resolved mesh to a small texture (256x256 or comparable) using the existing M2/WMO render path. Reuse the existing render infrastructure — no new OpenGL machinery needed beyond what the standalone MDX/M2 inspector already uses.
4. Show placeholder with "no preview available" when mesh cannot be resolved.
5. Cache is invalidated on selection change, not per-frame.

**Validation**: Preview appears for objects with resolved meshes. Placeholder shown for unresolved. No freeze on selection change. Preview updates on selection change.

### Phase 6: Polish + Cross-Cutting

**Goal**: Final integration, doc sync, and validation.

**Approach**:
1. Update spec 058 `spec.md` to correct FR-007 type reference and record any discoveries from implementation.
2. Memory bank update.
3. Full build + test pass.

**Validation**: `dotnet build` green. `dotnet test` green. Panel works end-to-end on real data.
