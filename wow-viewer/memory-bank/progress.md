# Progress — wow-viewer

## 2026-06-22 — Spec 071 Phase C: Right sidebar / workbench (complete)

### What landed

- Renamed `DrawWorkbenchPopout` → `DrawRightSidebar`; anchored at `x = displayWidth - _rightSidebarWidth`, full viewport height, width `_rightSidebarWidth`.
- Added `DefaultRightSidebarWidth = 480f`; wired into `_rightSidebarWidth` init, `ResetShellLayoutToDefaults`, `LoadViewerSettings`, and `ViewerSettings.RightSidebarWidth`.
- Renamed legacy shell-panel right sidebar to `DrawLegacyRightSidebar()` and gated it to `!_useTabUi`.
- Removed unused `_popoutDockFrame` field.
- Build: 0 errors.
- Commit `be92b40f` pushed to `071-left-right-sidebar-split`.

### Next

- Phase D: collapse `TopTab` to Model/World/Tools, remap sub-tabs, route Tools menu to tab switches.

## 2026-06-22 — Spec 071 Phase B: Left sidebar (complete)

### What landed

- New tab-mode `DrawLeftSidebar()` in `ViewerApp_Sidebars.cs`: fixed window at x=0, full viewport height, width `_leftSidebarWidth`.
- Content: `DrawWorkspaceBarsPanelContent` (source + open buttons), `DrawFileBrowserContent`, `DrawMapDiscoveryContent`.
- Legacy shell-panel left sidebar renamed to `DrawLegacyLeftSidebar()` and gated to `!_useTabUi`.
- `DrawUI()` now calls `DrawLeftSidebar()` before `DrawWorkbenchPopout()` when `_useTabUi` is active.
- Build: 0 errors.
- Commit `78ec3275` pushed to `071-left-right-sidebar-split`.

### Next

- Phase C: rename `DrawWorkbenchPopout` → `DrawRightSidebar`, anchor to right edge with `_rightSidebarWidth`.

## 2026-06-22 — Spec 071 Phase A: 3D viewport math (complete)

### What landed

- `TryGetSceneViewportRect` updated to squeeze the 3D viewport between the left and right sidebars when `_useTabUi` is active.
- Uses existing `_showLeftSidebar` / `_showRightSidebar` flags and `_leftSidebarWidth` / `_rightSidebarWidth`.
- Legacy (`!_useTabUi`) path unchanged — no regression in dockable shell-panel mode.
- Build: 0 errors (`dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug`).
- Commit `94b25e52` pushed to `071-left-right-sidebar-split`.

### Next

- Phase B: add `DrawLeftSidebar` with workspace bars + file browser + world maps.
- Phase C: rename `DrawWorkbenchPopout` → `DrawRightSidebar` and anchor to right edge.

## 2026-06-21 — Spec 071: left/right sidebar split + Model Viewer mode (drafted)

### What landed

**Spec + plan + tasks** at `specs/071-left-right-sidebar-split/`:
- Two-side layout: left sidebar (file browser + maps), right sidebar (workbench)
- 3 top tabs: Model / World / Tools (collapses 069's 6)
- Model Viewer mode: Info / Animations / Actions / LOD sub-tabs
- All Tools menu items become tab switchers (no floating windows)
- 8 phases A-H, branch `071-left-right-sidebar-split` cut from `069-viewer-ui-overhaul`

### User feedback driving 071
- File browser / World Maps should be in a separate LEFT sidebar
- Right sidebar = workbench
- No useful model inspection panels in 069 workbench when loading a model
- All popups should be tabs

### Salvaged from 069
- Tab data model + dispatch
- Sub-tab content methods (Draw*Content variants)
- Archeology playback, sticky settings
- Headless content variants (no nested ImGui windows)
- Single workbench panel concept

## 2026-06-21 — Spec 069: Viewer UI overhaul (tab system → workbench)

### What landed

**Cells overlay (Phase 0)**:
- `ShowCellGrid` property on `TerrainRenderer` (8x8 per chunk, 66.666 world units, green)
- `uShowCellGrid` uniform in both per-chunk + tile shaders
- Cells checkbox in toolbar + workspace bars sidebar
- Commit `a086a29d`

**Tab system foundation (Phase 1)**:
- `TopTab` enum (Scene/World/Terrain/PM4/Archeology/Utilities)
- 6 per-top-tab `*BottomTab` enums
- `_useTabUi` bool (default `true`), `_activeTopTab`, `_activeBottomTabIndex` fields
- `DrawTopTabBar`, `DrawBottomTabBar`, `DrawMainViewport`, `DrawTopTabContent`, `DrawBottomTabContent` stubs
- Commit `41420ed4`

**Scene + Utilities tabs (Phase 2)**:
- Scene sub-tabs: Selection/Camera/Settings/Themes
- Utilities sub-tabs: Minimap/Log/Perf/RenderQuality/Taxi/CaptureAutomation/AssetCatalog/RuntimeStats
- All call existing draw methods (headless `DrawMinimapContent` extracted)
- View > Tab System toggle (legacy mode)
- Commit `42cdcb38`

**World + Terrain merge (Phase 3)**:
- World sub-tabs: Source/Placements/Tiles/Overlays/SelectionTools
- Tiles sub-tab calls `DrawTerrainWorkbenchSelectionContent` + `DrawTerrainControlsAdjustmentContent`
- Overlays sub-tab: layer/grid/shadow/MCCV/contour toggles
- Commit `b8d735fc`

**Terrain + PM4 tabs (Phase 4)**:
- Terrain sub-tabs: Layers/Clipboard/Analysis/MCNK/WeakSignal/Export
- PM4 sub-tabs: Overlay/Selection/Correlation/Info/Match/Alignment
- Archeology stub: Range/Layers/Playback/Capture
- Commit `497a8155`

**Archeology tab + sticky settings (Phases 5+6)**:
- Split `DrawUniqueIdArchaeologyContent` into per-sub-tab methods
- Sticky range + scope persistence in ViewerSettings
- `UseTabUi`/`ActiveTopTab`/`ActiveBottomTab` persisted
- Commit `4ec074bb`

**Archeology playback (Phase 7)**:
- `_archeologyPlaybackActive`/`Speed`/`Loop` fields
- `UpdateArcheologyPlayback` advances UniqueIdFilterMax per frame
- Slider touch pauses playback
- Capture integration: `PendingCaptureRequest.ApplyArcheologyPlayback`, `ActiveVideoRecording.ApplyArcheologyPlayback`
- Video recording starts/stops playback real-time
- Commit `a54e4481`

**Cleanup + doc (Phase 8)**:
- `ShellPanelId` marked `[Obsolete]`
- New doc: `docs/architecture/viewer-ui-tab-system-2026-06-21.md`
- Commit `a16c6058`

**Critical fixes (Phase 9-10)**:
- World > Source sub-tab with file browser + map discovery + workspace bars
- Removed "Debug window" wrap in `DrawTopTabContent`
- Emptied Tools menu (only modal/dialog entries)
- 3D viewport full size fix in `TryGetSceneViewportRect`
- Commit `69966d3c`, `3f6e918e`

**Popout positioning (Phase 11)**:
- Source popout docks on right edge of master
- Commit `6471a7bb`

**Quick Controls + per-sub-tab popouts (Phase 12-13)**:
- Quick Controls popout: camera/lighting/layer/overlay/reset
- Per-sub-tab popouts: 14 sub-tab popouts
- Click sub-tab to toggle popout, `●` indicator
- Commit `4600f9f1`, `929bfd36`

**Single Workbench panel (Phase 14)**:
- All sub-tab popouts collapsed into ONE big Workbench popout
- Workbench has internal top tab bar + sub-tab bar + content area
- No more window sprawl
- Added Scene > Quick sub-tab (replaces standalone Quick Controls popout)
- Removed per-sub-tab popout state fields
- Commit `4d00f2af`

### What we learned

1. **Tab bars at top/bottom of master = wrong.** Looked like a debug overlay. User wanted workbench-feel.
2. **One popout per sub-tab = wrong.** Window sprawl. User wanted single panel.
3. **Correct: one Workbench panel with internal tabs.** Single resizable window. All data inside.
4. **Spec 049 (sidebar consolidation) was wrong approach from the start.** Should have gone to 070 (workbench windows) directly.

### Spec 070 (next)

Per-map workbench windows. Each loaded map = its own window with its own 3D viewport, tab UI, minimap. Master becomes thin launcher.

Estimated: 2-3 weeks focused work, multi-session.

## Previous progress (June 18-20)
- Spec 068: fractal-aware height loss + curation hardening (V21c)
- Spec 067: V20 multi-modal terrain intent
- Spec 066: V19 minimal-signal height regressor
- PM4 surface correlation matcher (collision fingerprints)
- PM4 simplification algorithm reverse-engineered
- V20 Multi-Modal Chained Terrain Intent — segmentor training

## Branch summary

- `069-viewer-ui-overhaul` — 14 commits, all pushed, active dev
- Build: 0 errors on every commit
- 14 phases of UI iteration
- Salvaged: tab data model, archeology playback, sticky settings
- Pending: 070 workbench window rewrite

## Out-of-Phase Work (Future Specs)

- 070: Per-map workbench windows
- Per-workbench capture automation (capture is per-workbench)
- Per-workbench minimap state persistence
