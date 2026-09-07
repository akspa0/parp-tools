# Receipt — Phase 1: Page skeletons + migration map (T010, T011, T012)

**Date**: 2026-09-07 · **Spec**: [231 Editor/Archaeology UI Overhaul](../spec.md) · Phase 1

## Files changed

| File | Change |
|---|---|
| [Workbench/Pages/ViewerAppContext.cs](../../../src/viewer/WoWViewer/Workbench/Pages/ViewerAppContext.cs) | **New** — narrow `ViewerAppContext` (constructor-injected) + `IEditorPageHost` draw contract |
| [Workbench/Pages/EditorPlacementObjectsPage.cs](../../../src/viewer/WoWViewer/Workbench/Pages/EditorPlacementObjectsPage.cs) | **New** — page 1 (Tasks & Workspace / 3D Object Library / Population sections) |
| [Workbench/Pages/EditorTerrainToolsPage.cs](../../../src/viewer/WoWViewer/Workbench/Pages/EditorTerrainToolsPage.cs) | **New** — page 2 (Terrain Lab section) |
| [Workbench/Pages/EditorDataIoPage.cs](../../../src/viewer/WoWViewer/Workbench/Pages/EditorDataIoPage.cs) | **New** — page 3 (Imports & Exports section) |
| [Workbench/Pages/EditorConvertersPage.cs](../../../src/viewer/WoWViewer/Workbench/Pages/EditorConvertersPage.cs) | **New** — page 4 (Converters section) |
| [Workbench/Pages/EditorWorkbenchPages.cs](../../../src/viewer/WoWViewer/Workbench/Pages/EditorWorkbenchPages.cs) | **New** — page aggregate + `MigrateLegacyEditorPageIndex` (0→0, 1→3, 2→0, 3→2, 4→1, 5→0) |
| [Workbench/WorkbenchNavigator.cs](../../../src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs) | `GetEditorWorkbenchLabels()` → the 4 new labels |
| [ViewerApp.cs](../../../src/viewer/WoWViewer/ViewerApp.cs) | `ViewerApp : IDisposable, Workbench.Pages.IEditorPageHost`; `CurrentWorkbenchNavigationVersion` 3→4; settings load relaxes the top-tab gate to `Enum.IsDefined` and remaps the saved Editor page index when `WorkbenchNavigationVersion` is pre-231 |
| [ViewerApp_Editor.cs](../../../src/viewer/WoWViewer/ViewerApp_Editor.cs) | One field `_editorPages` + `EnsureEditorPages()` + 6 explicit `IEditorPageHost` delegates (Spec 228 pattern); `DrawArchaeologyEditorContent` switch remapped to new page order |
| [ViewerApp_Sidebars.cs](../../../src/viewer/WoWViewer/ViewerApp_Sidebars.cs) | `DrawEditorWorkbenchSubTabContent` → one-line delegation to `EnsureEditorPages().Draw(...)`; `NormalizeWorkbenchStateAfterLoad` Experimental→Editor indices emitted in new space (Converters→3, Population→0, Terrain Lab→1); legacy `OpenWorkbenchTab(Editor, …)` call sites remapped (Terrain 4→1, Converters 1→3) |

## Verification

- `dotnet build wow-viewer/WowViewer.slnx -c Debug` — **exit 0, 0 Errors** (695 warnings, all
  pre-existing; artifact `cmd-1788752238474.txt`).

## Criterion → evidence

| Criterion (tasks.md T012 gate) | Evidence |
|---|---|
| `dotnet build` 0 errors | Exit 0, "0 Error(s)" in build output above |
| Old content reachable from new pages | Each page section delegates through `IEditorPageHost` to the unchanged legacy draw methods (`DrawArchaeologyEditorTasksSubTab`, `DrawRosettaObjectLibrarySubTab`, `DrawPopulationSubTabContent`, `DrawTerrainLabSubTab`, `DrawArchaeologyEditorImportsSubTab`, `DrawConvertersSubTabContent`) |
| Zero behavior change (structural) | God-class methods are untouched; pages only re-host them under `SharedUiWidgets.SectionHeader`; the only routing changes are the label list, the delegation switch, and index remaps |
| Old-index remap (T011) | `MigrateLegacyEditorPageIndex` maps all six old indices per plan; applied at settings load when `WorkbenchNavigationVersion < 4` and at the three legacy `OpenWorkbenchTab(Editor, …)` call sites; `NormalizeWorkbenchStateAfterLoad` now emits new-space indices for Experimental→Editor routes |

## Implementation notes (spec-sync per AGENTS.md §9.4)

1. **Top-tab gate relaxation**: the plan said to remap inside `NormalizeWorkbenchStateAfterLoad`;
   the remap actually runs at the settings-load boundary (`ViewerApp.cs`), because that is the only
   place that knows the persisted `WorkbenchNavigationVersion` — a per-session flag would
   double-map indices saved by post-231 builds. Side effect: settings with an older navigation
   version now restore their (valid) top tab instead of resetting to Quick; the version bump to 4
   plus the Editor-index remap preserves the intended destination.
2. **One-time remap semantics**: with the version gate, pre-change settings files remap exactly
   once (verified by construction: post-231 saves write version 4, which skips the remap). The
   operator's "verify with a pre-change settings file" runtime check remains operator-owned.
3. **Phase 2+ contract**: legacy `OpenWorkbenchTab(WorkbenchTab.Editor, n)` call sites now pass
   new-space indices; Phase 2 `Open Data I/O` links must use new indices (2 = Data I/O).

## Not yet done (open items)

- T001 baseline screenshots — operator-owned, will be supplied later (acknowledged 2026-09-07).
- Runtime navigation smoke — operator-owned (Phase 5 T050).
