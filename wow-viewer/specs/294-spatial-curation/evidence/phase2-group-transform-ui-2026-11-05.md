# Spec 294 Phase 2 — Group Transform UI Receipt

**Date:** 2026-11-05  
**Scope:** T007–T012. The viewer UI is implemented and builds; operator runtime/visual proof remains owed.

## Files changed

| File | Change |
|---|---|
| [`ArchaeologyPanelService.cs`](../../src/viewer/WoWViewer/Workbench/Services/Archaeology/ArchaeologyPanelService.cs) | Owns the new group-transform workbench service instance. |
| [`ArchaeologyPanelService.PhaseLayers.cs`](../../src/viewer/WoWViewer/Workbench/Services/Archaeology/ArchaeologyPanelService.PhaseLayers.cs) | Places group controls adjacent to layer rotation controls and marks project state dirty on accepted transforms. |
| [`DonorGroupTransformWorkbenchService.cs`](../../src/viewer/WoWViewer/Workbench/Services/WorkbenchPanels/DonorGroupTransformWorkbenchService.cs) | Adds arbitrary-angle/translation inputs, pivot and coordinate preview table, validation display, accept/cancel, stale-preview guard, and bounded per-layer undo/redo. |
| [`PhaseComposition.cs`](../../src/core/WowViewer.Core/Maps/PhaseComposition.cs) | Carries exact authored group transform metadata in layer state and clones it. |
| [`DonorGroupTransformService.cs`](../../src/core/WowViewer.Core/Maps/DonorGroupTransformService.cs) | Stores the exact normalized transform on the cloned preview layer. |
| [`ui-controls.json`](../../eng/ui/ui-controls.json) | Registers the group section and preview/accept/cancel/undo/redo actions. |
| [`UI-REFERENCE.md`](../../docs/WoWViewer/UI-REFERENCE.md) | Regenerated from the UI control registry. |
| [`plan.md`](../plan.md), [`tasks.md`](../tasks.md), [`STATUS.md`](../../STATUS.md), [`activeContext.md`](../../../memory-bank/activeContext.md), [`progress.md`](../../../memory-bank/progress.md) | Spec/task/status/continuity updates. |

## Verification

| Command | Exit | Result |
|---|---:|---|
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~DonorGroupTransformServiceTests" --nologo` | 0 | 13 passed, 0 failed, 0 skipped. |
| `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | Inventory 140 rows / 128 distinct labels; 128 registry entries; 0 errors, 0 warnings. |
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug --no-restore --nologo -v minimal` | 0 | Build succeeded, 0 errors, 2 package vulnerability warnings (NU1902). |

Earlier implementation attempts failed before correction: the first build reported three ref-property compiler errors and stale registry entries; a subsequent build reported stale registry entries only. Both were corrected; the final command above is the passing result.

## Requirement → evidence

| Requirement | Evidence |
|---|---|
| FR-001 / FR-002 — group manipulation | The workbench treats every placed tile in a donor layer as one group, reports member count, previews all target coordinate mappings, and only mutates layer state on Accept. |
| FR-003 / FR-004 — arbitrary angle and preview | The UI accepts arbitrary finite degree values (three decimals), uses the service's normalized angle and resolved centroid pivot, shows every donor target before/after coordinate in a scrollable preview table, and blocks acceptance on validation errors. |
| FR-005 — cancel/undo/redo | Cancel drops the clone preview. Accepted snapshots are pushed to a bounded undo stack; Undo/Redo restore placement and exact group-transform state. Runtime input behavior is not proven by build/tests. |
| FR-006 — quarter-turn workflow | Existing exact quarter-turn combo remains unchanged; group controls are separate and adjacent to the rotation section. |
| FR-007 — non-destructive | Only the project-layer clone is previewed; the donor data/source files are not mutated by this UI path. Real-client source-file verification remains an operator gate. |
| AC-015 UI registry contract | Registry and generated reference verify with 0 errors and 0 warnings. |

## Operator gate owed

Run the viewer on a real map. Select a multi-tile donor layer; preview a fractional angle and translation; inspect the pivot and every before/after target coordinate; accept; cancel another preview; undo and redo; save/reopen after Phase 4 persistence lands. Record build, client fingerprint, observed behavior, and artifacts. This receipt does not claim visual/runtime proof.
