# Tasks: 058 PM4 Scene Graph Semantics and Panel

## Phase 1: Data-Model Correction + Forensics Schema

- [x] T001 [P] Correct spec 058 FR-007: change `Pm4MslkEntry` to `Pm4MsurEntry` in `spec.md`. (SHIPPED `117b1f6d`)

- [x] T002 [P] Verify `Ck24ObjectId` XML doc at `Pm4ResearchChunkModels.cs:75` — already done per `fe8ed85d`.

- [x] T003 Add `ck24HighByte`/`ck24LowByte` to `pm4 forensics` export — already done per `fe8ed85d` (`WorldScene.cs:1652`).

## Phase 2: Bond-Stats Analyzer + CLI Subcommand

- [x] T004 Create `Pm4BondStatsReport` in `Pm4ResearchChunkModels.cs` (SHIPPED `adadfe7d`)

- [x] T005 Create `Pm4BondStatsAnalyzer` in `Pm4BondStatsAnalyzer.cs` (SHIPPED `adadfe7d`)

- [x] T006 Add `pm4 bond-stats` subcommand to `Tool.Inspect/Program.cs` (SHIPPED `adadfe7d`)

- [x] T007 [P] Markdown report output alongside JSON (deferred — JSON output is sufficient for CLI)

- [x] T008 Unit tests in `Pm4BondStatsAnalyzerTests.cs` — 5 tests pass (SHIPPED `fa0bd864`)

- [x] T009 CLI validated on 616-file dev corpus — smoke report written

## Phase 3: Type-Bucket Grouping in Graph Data Model

- [x] T010 Add `TypeBuckets` property + `Pm4SelectedObjectGraphTypeBucket` record to `WorldScene.cs` (SHIPPED `7bbd2722`)

- [x] T011 Modify `TryGetSelectedPm4ObjectGraphInfo()` to partition by `Ck24Type` via `BuildTypeBuckets()` (SHIPPED `7bbd2722`)

- [x] T012 [P] Cached-graph invalidation key — no change needed (type buckets are deterministic)

## Phase 4: Dockable PM4 Scene Graph Panel

- [x] T013 Add `ShellPanelId.Pm4SceneGraph` to enum in `ViewerApp.cs:62` (SHIPPED `f2119dda`)

- [x] T014 Register panel + add to `BottomRightQuadrantPanels` (SHIPPED `f2119dda`)

- [x] T015 `DrawPm4SceneGraphPanelContent()` renders via shell-panel dispatch — uses cached graph info (SHIPPED `f2119dda`)

- [x] T016 Default-expanded (`_showPm4SceneGraph = true`) + placeholder when no selection (SHIPPED `f2119dda`)

- [x] T017 Type-bucket tree UI with semantic labels (SHIPPED `f2119dda`)

- [x] T018 Per-user dock preferences via existing `ShellPanel` persistence (automatic)

- [x] T019 Tools menu toggle added (SHIPPED `f2119dda`)

- [ ] T020 User validation — needs real viewer runtime test

## Phase 5: Mesh Preview — CUT

**Cut 2026-06-11**: PM4 overlay rendering already drops FPS from 140 to 12. Rendering M2/WMO previews inside the panel would compound this. Deferred until PM4 rendering is optimized.

## Phase 6: Polish

- [x] T027 Update spec + tasks (this file)

- [ ] T028 [P] Full build + test

- [ ] T029 [P] Memory bank update
