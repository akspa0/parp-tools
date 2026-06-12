# Tasks: 058 PM4 Scene Graph Semantics and Panel

## Phase 1: Data-Model Correction + Forensics Schema

- [ ] T001 [P] Correct spec 058 FR-007: change `Pm4MslkEntry` to `Pm4MsurEntry` in `spec.md`. The fields `Ck24HighByte` and `Ck24LowByte` already exist at `Pm4ResearchChunkModels.cs:83-85`.

- [ ] T002 [P] Verify `Ck24ObjectId` XML doc at `Pm4ResearchChunkModels.cs:75` includes "lossy flattening" label. Update if needed.

- [ ] T003 Add `ck24HighByte` and `ck24LowByte` to the existing `pm4 forensics` export in `Program.cs`. Additive schema change — existing fields unchanged.

## Phase 2: Bond-Stats Analyzer + CLI Subcommand

- [ ] T004 Create `Pm4BondStatsReport` in `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4BondStatsReport.cs`:
  - Per-file `Ck24HighByte.ReuseCountPerFile` and `Ck24LowByte.ReuseCountPerFile`
  - High×Low cross-tabulation (for each high-byte value, distribution of low-byte values)
  - Low×High cross-tabulation (reversed)
  - Per-type-bucket breakdown
  - Correlation table: high-byte vs unknowns/subtype field indices, low-byte vs unknowns/subtype field indices

- [ ] T005 Create `Pm4BondStatsAnalyzer` in `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4BondStatsAnalyzer.cs`:
  - `AnalyzeFile(string pm4Path)` — single-file analysis
  - `AnalyzeDirectory(string directory)` — directory-wide aggregation
  - Reuse `Pm4ResearchReader` for file loading
  - Correlation with unknowns/subtype field indices from MSLK (Subtype, SystemFlag) and MPRL/MSUR relationship data

- [ ] T006 Add `pm4 bond-stats --input <file.pm4|directory> [--output <report.json>]` subcommand in `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs`.

- [ ] T007 [P] Add Markdown report output alongside JSON (consistent with 046 match-report pattern via `Pm4MatchReportWriter`).

- [ ] T008 Create `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4BondStatsAnalyzerTests.cs`:
  - Single-file analysis on real `development_00_00.pm4`
  - Cross-tabulation shape validation
  - Per-type-bucket breakdown
  - Zero-CK24 exclusion
  - Correlation table presence (even if empty/unknown for dev corpus)

- [ ] T009 Validate CLI on real development corpus. Write smoke report to `output/tmp/pm4-bond-stats-smoke.json`.

## Phase 3: Type-Bucket Grouping in Graph Data Model

- [ ] T010 Add `TypeBuckets` property to `Pm4SelectedObjectGraphInfo` in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs:613`:
  - New record `Pm4SelectedObjectGraphTypeBucket` with `Ck24Type`, `TypeLabel`, `LinkGroups`
  - Keep existing `LinkGroups` at top level for backward compatibility

- [ ] T011 Modify `TryGetSelectedPm4ObjectGraphInfo()` in `WorldScene.cs:11363` to partition link groups by `Ck24Type` before building the tree. Use MSLK TypeFlags lookup per surface via RefIndex.

- [ ] T012 [P] Update cached-graph invalidation key to include type-bucket shape.

## Phase 4: Dockable PM4 Scene Graph Panel

- [ ] T013 Add `ShellPanelId.Pm4SceneGraph` to enum in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs:62`.

- [ ] T014 Register panel: `new(ShellPanelId.Pm4SceneGraph, "PM4 Scene Graph", ShellPanelLane.Right, 420f, ...)`. Add to `BottomRightQuadrantPanels` at `ViewerApp.cs:339`.

- [ ] T015 Move `DrawPm4SceneGraph()` render call from PM4 Workbench sidebar path to shell-panel render path. Gate on `_showRightSidebar && _worldScene != null`.

- [ ] T016 Default-expanded when PM4 object selected. Placeholder "Select a PM4 object" when no selection.

- [ ] T017 Tree rendering: `Ck24Type` headers at top level with semantic labels (`0x03 = M2 top`, `0x10 = interior WMO floor`, `0x12 = exterior WMO solid`). Link-group hierarchy nested under each type bucket. Per-entry `0xAA` value always visible.

- [ ] T018 Per-user dock preferences (US3): persist panel position, expanded/collapsed state, and dock node across sessions via existing `ShellPanel` persistence surface.

- [ ] T019 Add PM4 Scene Graph to Tools menu toggle.

- [ ] T020 Validate: 60+ FPS with 1000+ part container. No per-frame rebuild (commit `42e83488` cache). Multi-type shows grouped headers. Single-type shows one header. Panel position persists across restarts.

## Phase 5: Mesh Preview Cache + Panel Integration

- [ ] T021 Create `Pm4MeshPreviewCache` in `wow-viewer/src/viewer/WoWViewer/`:
  - Per-selection cache keyed by same selection key as graph
  - `GetOrCreatePreview(selectionKey, resolvedAssetPath)` → texture handle
  - Cleared on selection change

- [ ] T022 Resolve the selected PM4 object's M2/WMO via the existing asset-corpus surface. Use `Pm4AssetMatchScorer` matched candidates or validation placements from spec 046 to find the asset path.

- [ ] T023 Render the resolved mesh to a small texture (256x256) using the existing M2/WMO render path. Reuse existing render infrastructure — no new OpenGL machinery beyond what the standalone MDX/M2 inspector already uses.

- [ ] T024 Show placeholder with "no preview available" when mesh cannot be resolved. Placeholder is a clear status, not an error and not a freeze.

- [ ] T025 Integrate preview into the PM4 Scene Graph panel — show above or beside the tree. Sized to fit panel without dominating.

- [ ] T026 Validate: Preview renders for resolved objects. Placeholder for unresolved. Updates on selection change without rebuilding panel layout. No freeze.

## Phase 6: Polish + Cross-Cutting

- [ ] T027 [P] Update spec 058 `spec.md` to correct FR-007 type reference and record any discoveries from implementation.

- [ ] T028 [P] Full build + test: `dotnet build` and `dotnet test` green.

- [ ] T029 [P] Memory bank update: `activeContext.md` and `progress.md`.

## Dependencies

- Phase 1: No dependencies. Start here.
- Phase 2: Depends on Phase 1 (forensics schema adds fields used by CLI).
- Phase 3: No dependency on Phase 1/2. Can run in parallel.
- Phase 4: Depends on Phase 3 (needs TypeBuckets in graph data model).
- Phase 5: Depends on Phase 4 (needs panel to integrate preview into). Can partially parallel with Phase 4.
- Phase 6: Depends on all.
