# Receipt — Phase 6 wiring audit (T060) + removal batch 1

**Date**: 2026-09-07 · **Spec**: [231](../spec.md) — operator amendment "Tools curation & removal
pass" (recorded verbatim in [spec.md](../spec.md))

## Method

PowerShell scan over `ViewerApp*.cs`: every `private void|bool Draw*(...)` declaration was counted
against all textual references in the same file set; methods with ≤1 occurrence (definition only)
have no callers and are dead UI. Orphan iteration then re-ran to a fixpoint for newly unreferenced
helpers.

## Batch 1 — removed this pass (all had 0 callers; removal is behavior-free)

| File | Removed |
|---|---|
| ViewerApp_Editor.cs | `DrawReconciliationPathField` |
| ViewerApp_PhaseLayers.cs | `DrawPhaseLayersPanel` |
| ViewerApp_Pm4Utilities.cs | `DrawPm4SelectedObjectMatchSuggestions`, `DrawSelectedPm4ObjectGraph` |
| ViewerApp_Sidebars.cs | `DrawCenteredTerrainToolbarWindow`, `DrawUnifiedSelectionSidebarContent`, `DrawSelectedPm4ContextSummary`, `DrawCompactTerrainContextSummary`, `DrawUnifiedWorldToolsSidebarContent`, `DrawRightSidebarSection`, `DrawViewerInspectSidebarContent`, `DrawViewerDiagnosticsSidebarContent`, `DrawWorldObjectsContent`, `DrawModelSubTabContent`, `DrawExperimentalSubTabContent`, `DrawTerrainClipboardSubTab`, `DrawTerrainAnalysisSubTab`, `DrawTerrainMcnkSubTab`, `DrawTerrainWeakSignalSubTab`, `DrawSceneSubTabContent`, `DrawWorldTilesSubTab`, `DrawTerrainExportSubTab`, `DrawWorldSelectionToolsSubTab` |
| ViewerApp_Workspaces.cs | `DrawEditorWorkspaceNavigator` |
| ViewerApp.cs | `DrawMinimap_OLD` |
| (orphan sweep) | `DrawViewerSelectionSummary`, `DrawModelInfoSubTab` |

Notable: the entire dead legacy sub-tab family (Model/Experimental/Scene/World Tiles/Selection
Tools/Terrain sub-tabs) and the old minimap were still present and unreachable — this is the
"never fully wired" material the operator named.

## Size delta (line counts)

| File | Before (Phase 3+4 commit) | After | Δ |
|---|---|---|---|
| ViewerApp_Sidebars.cs | 6,013 | 5,648 | −365 |
| ViewerApp_Pm4Utilities.cs | 4,368 | 4,178 | −190 |
| ViewerApp.cs | 16,750 (pre-Phase-3) | 16,556 | −194 |
| ViewerApp_Editor.cs | 1,930 | 1,907 | −23 |
| ViewerApp_PhaseLayers.cs / _Workspaces.cs | — | — | orphan methods removed |

Verification: `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` — **0
Errors** after removals (exit 0).

## Batch 2 — operator-named live surfaces, wiring status (removal slices pending)

| Surface | Wiring status | Notes for the removal slice |
|---|---|---|
| Weak-signal amplifier / restore | **live**: 84 refs to `_terrainWeakSignalRestore*` across ViewerApp.cs + Sidebars (settings load/save + toolbar/sidebar UI). Live draw surface is in the terrain restore controls | Removal touches settings schema fields + UI + any terrain-manager hooks; largest batch-2 slice |
| Terrain task blocks inside Editor Tasks & Workspace | live: Chunk Clipboard + Terrain Import/Export embedded in the task inspector | Belong on Terrain Tools / Data I/O per the IA; relocation or removal per operator |
| Selection tools | the dead legacy `DrawWorldSelectionToolsSubTab`/`DrawUnifiedSelectionSidebarContent` are now gone; remaining selection functionality lives in Inspector/Placement pages | Confirm no further operator-named selection surface remains |
| Quick-panel convergence (T064) | pending | Apply the Data I/O / Quick pattern (SharedUiWidgets, collapsed groups) to remaining pages |

Batch 2 slices (T061–T064) are scoped but not yet executed; each needs its own removal receipt
per T065.
