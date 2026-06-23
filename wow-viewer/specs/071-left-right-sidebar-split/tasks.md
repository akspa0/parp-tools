# Tasks: 071 Left/Right Sidebar Split + Model Viewer Mode

## Phase A: 3D viewport math

- [x] T001 Update `TryGetSceneViewportRect` to subtract both `_leftSidebarWidth` and `_rightSidebarWidth` when both sidebars are visible.
- [x] T002 Verify `UpdateShellLayout` and `ClampFixedSidebarLayout` work with the new two-sidebar math.
- [x] T003 Build clean, no regression in legacy mode.

## Phase B: Left sidebar

- [x] T004 Add `DrawLeftSidebar()` method in `ViewerApp_Sidebars.cs`.
- [x] T005 Position: x=0, y=topOffset, width=_leftSidebarWidth, height=viewport_height.
- [x] T006 Content: `DrawWorkspaceBarsPanelContent` + `DrawFileBrowserContent` + `DrawMapDiscoveryContent`.
- [x] T007 Call from `DrawUI()` before `DrawRightSidebar` (or current workbench popout).
- [x] T008 Suppress old DrawLeftSidebar logic (it was for the shell-panel sidebar system).

## Phase C: Right sidebar (rename workbench)

- [x] T009 Rename `DrawWorkbenchPopout` → `DrawRightSidebar` throughout.
- [x] T010 Position: x=displayWidth - _rightSidebarWidth, y=topOffset, width=_rightSidebarWidth, height=viewport_height.
- [x] T011 Update 3D viewport rect to subtract both sidebars.
- [x] T012 Suppress old `DrawWorkbenchPopout` shell-window flags.

## Phase D: 3 top tabs (Model/World/Tools)

- [x] T013 Replace `TopTab` enum (6 values) with 3 values: `Model`, `World`, `Tools`.
- [x] T014 Update `GetBottomTabLabels` to return sub-tab sets per new top tab.
- [x] T015 Map old 6-tab enums to new 3-tab. World tab gets: Source, Placements, Tiles, Overlays, Selection Tools. Tools tab gets: Quick, Archeology (range/layers/playback/capture), PM4 (overlay/selection/correlation/info/match/alignment), Terrain (layers/clipboard/analysis/MCNK/weak signal/export), Utilities (minimap/log/perf/render quality/taxi/capture automation/asset catalog/runtime stats). Model tab gets: Info, Animations, Actions, LOD.
- [x] T016 Update `DrawTopTabButton` and related dispatch.
- [x] T017 Update `DrawWorkbenchContent` sub-tab dispatch (now per new top tab).
- [x] T018 Tools menu integration: clicking Tools > Log Viewer calls `OpenWorkbenchTab(ToolsBottomTab.Utilities)` instead of `_showLogViewer = true`. Same for Perf, Render Quality, Capture Automation, Taxi, Asset Catalog, UniqueIdArcheology, ChunkClipboard, TerrainAnalysis, McnkExplorer, WeakSignal, ModelInfo.

## Phase E: Model Viewer — Info sub-tab

- [x] T019 Add `ModelBottomTab` enum (Info, Animations, Actions, LOD).
- [x] T020 Reuse `DrawModelInfoContent` for Info sub-tab.
- [x] T021 Show model path, type, vertex/triangle count, materials, textures.
- [x] T022 If no model loaded: show "No model loaded" placeholder.

## Phase F: Model Viewer — Animations sub-tab

- [x] T023 Add animation list view (sequence names from MdxAnimator/M2Animator).
- [x] T024 Add Play/Pause/Stop buttons (large, prominent).
- [x] T025 Add frame slider (current frame / total frames).
- [x] T026 Add speed control (0.25x, 0.5x, 1x, 2x).
- [x] T027 Reuse `DrawSelectedSqlGameObjectAnimationControls` for SQL-spawned objects.
- [x] T028 Add loop checkbox.

## Phase G: Model Viewer — Actions + LOD sub-tabs

- [x] T029 Actions sub-tab: Frame Model button, Auto-frame toggle, WMO doodad set selector.
- [x] T030 LOD sub-tab: distance/quality controls (best-effort, defer if too complex).
- [x] T031 Wire selected object → model viewer: when user clicks a model in the world, Model > Info tab auto-shows that model.

## Phase H: Memory bank + spec sync

- [x] T032 Update `memory-bank/activeContext.md` for Phase A-G completion.
- [x] T033 Update `memory-bank/progress.md` with 8-phase history.
- [x] T034 Update `specs/071-left-right-sidebar-split/spec.md` if any design changes during build.
- [x] T035 Final build + commit.

## Build State

- Build: 0 errors expected on every commit.
- 8 phases A-H, all on `071-left-right-sidebar-split` branch.
- Phases A-D = layout restructure.
- Phases E-G = Model Viewer feature.
- Phase H = memory + spec sync.

## Notes

- 071 inherits 069's 16-phase UI cleanup. Salvages: tab data model, sub-tab content methods, archeology playback, headless content variants, single workbench panel concept.
- 070 (per-map native windows) is still drafted but deferred. 071 is the next step because left/right sidebar + Model Viewer is what the user actually needs now.
- Continue from `069-viewer-ui-overhaul` branch. Cut 071 branch from there.
