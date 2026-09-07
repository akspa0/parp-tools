# Receipt — Phase 2: Data I/O consolidation (T020, T021, T022, T023)

**Date**: 2026-09-07 · **Spec**: [231 Editor/Archaeology UI Overhaul](../spec.md) · Phase 2 (D1, D6)

## Files changed

| File | Change |
|---|---|
| [Workbench/Pages/ViewerAppContext.cs](../../../src/viewer/WoWViewer/Workbench/Pages/ViewerAppContext.cs) | `IEditorPageHost` gains `DrawPm4Exports()` |
| [Workbench/Pages/EditorDataIoPage.cs](../../../src/viewer/WoWViewer/Workbench/Pages/EditorDataIoPage.cs) | New collapsed-by-default "PM4 Exports" section (T020 group default) above Imports & Exports |
| [ViewerApp_Editor.cs](../../../src/viewer/WoWViewer/ViewerApp_Editor.cs) | Explicit `IEditorPageHost.DrawPm4Exports()` delegate → `DrawPm4ExportCommandSet()` |
| [ViewerApp_Pm4Utilities.cs](../../../src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs) | **D1**: `DrawPm4ExportCommandSet()` is now the single draw site for Dump PM4 Objects JSON / Export PM4 OBJ Set / Export PM4 LLM Bundle / Export Visible PM4 Report / Dump PM4/WMO Correlation JSON; the 5 former duplicate clusters are `Open Data I/O` links (`OpenWorkbenchTab(WorkbenchTab.Editor, 2)`). **D6**: `ExportPm4ObjectsJson` + `ExportPm4WmoCorrelationJson` moved to `Task.Run` with the `_pm4JsonExportRunning` re-entrancy guard (pattern per `ExportPm4ObjectsObjSet`) |

## D1 — SC-1 grep receipt (each label = 1 draw site)

`Select-String` over `ViewerApp_Pm4Utilities.cs` after the change:

| Button label | Draw sites |
|---|---|
| "Dump PM4 Objects JSON" | **1** (in `DrawPm4ExportCommandSet`) |
| "Export PM4 OBJ Set" | **1** |
| "Export PM4 LLM Bundle" | **1** |
| "Export Visible PM4 Report" | **1** |
| "Dump PM4/WMO Correlation JSON" | **1** |
| "Dump PM4 JSON" (old short label) | 0 (merged into "Dump PM4 Objects JSON") |
| "Open Data I/O" links | 6 buttons (2 selection-workbench branches, 2 transform-panel branches, 2 info-panel branches) replacing the 5 former export clusters |

The id-suffixed LLM-bundle buttons in the correlation panels (~2615/2662 pre-change) are part of
the correlation-panel dedupe **D2 (Phase 3)**, not D1, and were left untouched.

## D6 — background-execution audit

| Command | Before | After |
|---|---|---|
| PM4 objects JSON dump | sync `BuildPm4OverlayInterchangeJson(includeGeometry: true)` + write on render thread | `Task.Run` + `_pm4JsonExportRunning` guard + status |
| PM4/WMO correlation JSON dump | sync `BuildPm4WmoPlacementCorrelationJson()` + write on render thread | `Task.Run` + same guard + status |
| PM4 OBJ set export | already backgrounded (2026-09-07 pattern) | unchanged |
| LLM evidence bundle / visible report | visible-scoped summary serialization (bounded); below the >1s threshold | unchanged; re-audited when the corpus grows |
| Corpus-wide object-match report build | `RefreshPm4ObjectMatchReport` currently has **no callers** (orphaned after Spec 176 tile-scoped reconcile); the live per-selection path (`TryBuildPm4ObjectMatch`) is object-scoped | no change possible/needed; dead path noted for Phase 3 cleanup |

New god-class field: `_pm4JsonExportRunning` (required by the directed re-entrancy-guard pattern;
counted and noted — Spec 228 budget effect below).

## Verification (T023 gate)

- `dotnet build wow-viewer/WowViewer.slnx -c Debug` — **exit 0, 0 Errors** (654 warnings, all pre-existing; artifact `cmd-1788755212886.txt`).
- `dotnet test wow-viewer/WowViewer.slnx -c Debug` — 11 failures **all pre-existing** in
  `WowViewer.Core.Tests` (10) and `WowViewer.Core.PM4.Tests` (1, dev-corpus dependent, 5m1s).
  These test projects do not reference the viewer UI project changed by this phase, so their
  binaries are unaffected by this diff (failures reproduce independent of it). Passed: 1483 + 102 + 91 + 39.
- UI smoke (capture automation, Shadowfang / 0.5.3.3368):
  [screenshots/phase2-editor-2-data-io-pm4-exports/](screenshots/phase2-editor-2-data-io-pm4-exports/)
  shows the Data I/O page with the collapsed "PM4 Exports" section + Imports & Exports dashboard.
- **Operator-owned remainder of the smoke**: executing an actual export requires interacting with
  the in-app path picker; the background-execution behavior (UI responsive during export) needs a
  real interactive pass.

## Interpretation notes (§9.4 spec-sync)

1. **"VLM/ML harvest launchers" (T020)**: the VLM/MK dataset workflow is retired
   (ViewerApp.cs:892 comment; the `Build ML Dataset` / `Train V7 Terrain Model` dialogs have no
   launcher buttons). Nothing to move; no UI resurrected.
2. **"Delete the Archaeology-hosted imports page" (T021)**: the old Editor-tab imports route is
   gone (replaced by the Data I/O page). Removing the *Archaeology-tab* hosting of the editor
   imports/tasks routes is T042 (Phase 4), as the phase table assigns.
3. **File budget**: `ViewerApp_Pm4Utilities.cs` is 4,516 lines (was 4,477): D1 removed ~55 lines
   of button clusters but D6 added ~95 lines of background wrappers. The spec's shrink outcome is
   measured at the P4 gate (T043); P3/P4 dedupes (D2/D3/D5) carry the bulk of the reduction.

## Open

- Operator: interactive export smoke (background responsiveness), Phase 5 navigation smoke.
