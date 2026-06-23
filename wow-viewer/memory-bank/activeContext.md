# Active Context — wow-viewer

**Last updated**: 2026-06-22 | **Focus**: Spec 071 — left/right sidebar split + Model Viewer mode (Phase D done, Phase E next)

## Current State

Viewer UI in flux. 069 (tab system) hit 16 phases. User feedback on 069:
- File browser / World Maps should be in a separate LEFT sidebar, not in the workbench
- Right sidebar = workbench (existing single panel)
- **No useful model inspection panels** when loading a model (M2/MDX/WMO). Need Model Viewer mode with info, animation list, Play/Pause/Stop
- All popups should be tabs in the workbench, not floating windows

### Spec 071 (branch `071-left-right-sidebar-split`)

**Goal:** Two-side layout + Model Viewer mode. 8 phases A-H.

- **Left sidebar** (~360px): file browser + world maps + workspace bars
- **Right sidebar** (~480px): workbench with 3 top tabs (Model/World/Tools)
- **Center**: 3D viewport (full size, no chrome overlap)
- **Model Viewer mode**: Info / Animations / Actions / LOD sub-tabs
- All Tools menu items become tab switchers

**Phase A (done)**: `TryGetSceneViewportRect` now subtracts `_leftSidebarWidth` and `_rightSidebarWidth` when `_useTabUi` is active.

**Phase B (done)**: New `DrawLeftSidebar()` renders a fixed left panel in tab mode with `DrawWorkspaceBarsPanelContent`, `DrawFileBrowserContent`, and `DrawMapDiscoveryContent`. Legacy shell-panel left sidebar renamed to `DrawLegacyLeftSidebar()`.

**Phase C (done)**: `DrawWorkbenchPopout` renamed to `DrawRightSidebar`; anchored to right edge with `_rightSidebarWidth`. Added `DefaultRightSidebarWidth = 480f` and wired it through init/reset/load/save. Legacy shell-panel right sidebar renamed to `DrawLegacyRightSidebar()`.

**Phase D (done)**: Replaced 069's 6-value `TopTab` with 3-value `WorkbenchTab` (Model/World/Tools) in new `WoWViewer.Workbench` namespace. Added `WorkbenchNavigator` with sub-tab enums and labels. World tab has Source/Placements/Tiles/Overlays/Selection Tools. Tools tab has Quick/Archeology/PM4/Terrain/Utilities. Model tab has Info/Animations/Actions/LOD. Tools menu items now call typed `OpenWorkbenchTab` overloads. Old `SceneBottomTab` removed; Quick controls live under Tools > Quick.

**Phase E (done)**: Model > Info sub-tab reuses `DrawModelInfoPanelContent` → `DrawModelInfoContent`. Added `Path:` line to `_modelInfo` for MDX, M2 runtime, WMO, and M2 camera path loads. Info panel shows type, version, name, vertices/triangles, materials/textures, plus existing animation/WMO controls. Placeholder shown when no model is loaded.

**Phase F (done)**: Extracted animation controls from Info into new `DrawModelAnimationControls` method used by Model > Animations sub-tab. Added `PlaybackSpeed` and `Loop` to `IAnimationController`, implemented in `MdxAnimator` and `M2RuntimeAnimator`. Animations tab now has sequence combo, large Play/Pause/Stop buttons, Previous/Next Key, Loop checkbox, speed buttons (0.25x/0.5x/1x/2x), timeline slider, and debug tree. SQL GameObject animation controls reused at the bottom when a SQL-spawned MDX object is selected.

**Phase G (done)**: Split Info/Actions content. Added `DrawModelInfoCoreContent` for pure info; `DrawModelInfoContent` keeps full legacy behavior for other callers. Model > Actions sub-tab has Auto-frame toggle, Frame Model button, and WMO doodad set selector. Model > LOD sub-tab shows placeholder guidance and renderer stats. Selecting an MDX/WMO object in the world now auto-switches to Model > Info and displays `_selectedObjectInfo` there.

**Phase H (done)**: Updated `spec.md` to reflect actual implementation (`WorkbenchTab.Model`, typed `OpenWorkbenchTab` overloads, timeline slider in scope, LOD placeholder). Updated `activeContext.md` and `progress.md` with full 8-phase history. Final build clean, all commits pushed to `071-left-right-sidebar-split`.

**Status**: Spec 071 complete. Branch `071-left-right-sidebar-split` ready for merge or next work.

**Cut branch:** `071-left-right-sidebar-split` from `069-viewer-ui-overhaul`.

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
4. Model Viewer animations: which APIs for M2 vs MDX? (M2 may have different animation format)
5. Model Viewer LOD: what level-of-detail controls exist in the renderer? (deferred to 071 phase G)

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
- `specs/071-left-right-sidebar-split/{spec,plan,tasks}.md` - new spec, 8 phases A-H

## Test Data

- `wow-viewer/test_data/development/World/Maps/development/` — primary test map set
- See `memory-bank/data-paths.md` for full list
