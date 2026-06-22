# Active Context — wow-viewer

**Last updated**: 2026-06-21 | **Focus**: Spec 069 Phase 16 — workbench sub-tabs use headless content variants (no nested windows)

## Current State

Viewer UI in flux. Spec 049 (sidebar consolidation) abandoned as wrong approach. Spec 060 (cleanup) marked complete. Spec 069 (tab system) in progress on branch `069-viewer-ui-overhaul`.

### UI direction
**Top + bottom tab bars (master window chrome) → FAILED.** Looked like a "Debug window" overlay, tabs popped new popouts (window sprawl). User hated it.

**Sub-tab popouts (one per sub-tab) → FAILED.** Window sprawl. User hated the X buttons that didn't work.

**Current approach (Phase 14-16): single Workbench panel** — one resizable popout on the right side of the master window, with internal top tabs + sub-tabs. All data inside one panel. Sub-tab content uses headless Draw*Content variants (no nested ImGui windows).

### What works in 069
- Cells overlay (8x8 per chunk, 66.666 units, green)
- Tab data model (TopTab/WorldBottomTab/SceneBottomTab/etc enums)
- Per-top-tab sub-tab content methods (DrawWorldSourceSubTab, DrawArcheologyRangeSubTab, etc)
- Archeology playback (UpdateArcheologyPlayback, Play/Pause/Stop, capture integration)
- Sticky archeology settings (ViewerSettings.Archeology* fields)
- 3D viewport full size when tab system on (TryGetSceneViewportRect early-return)
- World > Source sub-tab (file browser + map discovery + workspace bars)
- Scene > Quick sub-tab (camera/lighting/layer controls)

### What still missing/broken
- Native multi-window (per-map workbench windows) — spec'd in 070, not built
- Sidebar support still tied to legacy shell panel system (marked [Obsolete] but present)

### Next
- 070 workbench window spec: each loaded map = native window, multiple workbenches, master becomes launcher
- 070 is a real architectural rewrite (per-map state, render, UI)

### Branch state
- `069-viewer-ui-overhaul` is the active dev branch
- 14 phases of commits, all pushed
- Build clean on every commit

## Open Questions

1. Should torn-off workbench windows be fully independent processes (070)?
2. Per-workbench state: persist per-map-path or per-build-version?
3. Multi-map: tabs in master or fully separate OS windows?

## Files Touched Recently

- `src/viewer/WoWViewer/ViewerApp.cs` - TopTab enums, fields, _useTabUi flag, _workbenchOpen
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` - DrawWorkbenchPopout, DrawQuickControlsContent, per-top-tab sub-tab dispatch
- `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs` - DrawMinimapContent (headless)
- `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs` - ApplyArcheologyPlayback fields, DrawCaptureAutomationContent
- `src/viewer/WoWViewer/ViewerApp_Investigation.cs` - DrawMcnkExplorerContent
- `src/viewer/WoWViewer/ViewerApp_LogViewer.cs` - DrawLogViewerContent
- `src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs` - DrawPerfContent
- `src/viewer/WoWViewer/ViewerApp_RenderQuality.cs` - DrawRenderQualityContent
- `src/viewer/WoWViewer/ViewerApp_TerrainAnalysis.cs` - DrawTerrainAnalysisContent
- `specs/069-viewer-ui-overhaul/{spec,plan,tasks}.md` - 16 phases
- `specs/070-map-workbench-window/spec.md` - workbench window spec (draft)

## Test Data

- `wow-viewer/test_data/development/World/Maps/development/` — primary test map set
- See `memory-bank/data-paths.md` for full list
