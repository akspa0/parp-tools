# Inventory — Spec 292 (Phase 0)

**Date:** 2026-11-03  
**Status:** COMPLETE (Gate A Review)

---

## 1. Bottom-Bar Controls & Target Placement (T001)

| Current Control | Location in Code | Current Behavior | Target Home | Target Behavior / Persistence |
|---|---|---|---|---|
| `preset-Viewer` | `ViewerUiManager.cs:507` | Navigates to `WorkbenchTab.Viewer` | Mode switch: **Viewer** (Left dock switch) | Activates Viewer mode; persists in layout |
| `preset-Editor` | `ViewerUiManager.cs:507` | Navigates to `WorkbenchTab.Editor` | Mode switch: **Editor** (Right dock switch) | Activates Editor mode; persists in layout |
| `preset-Archaeology` | `ViewerUiManager.cs:507` | Navigates to `WorkbenchTab.Archaeology` | Mode switch: **Archaeology** (Right dock switch) | Activates Archaeology mode; persists in layout |
| `bottom-spawns` | `ViewerActionBarsService.cs:106` | Opens `MapSpawnsPopupId` popup | Right Dock: **Spawns** panel | Opens Spawns panel in right dock; persistent dock |
| `bottom-pages` | `ViewerUiManager.cs:509` | Dead measurement item | **DELETED** | Removed entirely from code and layout |
| `bottom-options` | `ViewerUiManager.cs:534` | Toggles `LeftPanel.Options` | Mode switch: **Options** (Left dock switch) | Opens Options in Left Dock; persistent dock |
| Profile buttons | N/A | Non-functional / obsolete | **DELETED** | Removed entirely |
| *(New)* **Sky** toggle | Bottom bar | None | Bottom bar: `[x] Sky` | Toggles `WorldScene.ShowSky`; persists in `ViewerSettings` |
| *(New)* **Fog** toggle | Bottom bar | None | Bottom bar: `[x] Fog` | Toggles `WorldScene.ObjectFogEnabled` & atmosphere fog override; persists in `ViewerSettings` |
| *(New)* **Water** toggle | Bottom bar | None | Bottom bar: `[x] Water` | Toggles `LiquidRenderer.ShowLiquid` & `WorldScene.ShowWlLiquids`; persists in `ViewerSettings` |
| *(New)* **Doodads** toggle | Bottom bar | None | Bottom bar: `[x] Doodads` | Toggles `WorldScene.DoodadsVisible`; persists in `ViewerSettings` |
| *(New)* **WMOs** toggle | Bottom bar | None | Bottom bar: `[x] WMOs` | Toggles `WorldScene.WmosVisible`; persists in `ViewerSettings` |
| *(New)* **Terrain** toggle | Bottom bar | None | Bottom bar: `[x] Terrain` | Toggles `TerrainRenderer.ShowTerrain`; persists in `ViewerSettings` |
| *(New)* **PM4** toggle | Bottom bar | None | Bottom bar: `[x] PM4` | Toggles `WorldScene.Pm4Overlay.ShowPm4Overlay`; persists in `ViewerSettings` |
| *(New)* **Grid** toggle | Bottom bar | None | Bottom bar: `[x] Grid` | Toggles `TerrainRenderer.ShowTileGrid`; persists in `ViewerSettings` |
| *(New)* **Mute** toggle | Bottom bar | None | Bottom bar: `[x] Mute` | Toggles `WorldScene.AudioMuted`; persists in `ViewerSettings` |

---

## 2. Audio Settings Inventory & Consolidation (T002)

| Audio Control / Feature | Current Location | API / State Property | Target Home |
|---|---|---|---|
| Master Gain | `AudioPanelService.cs:212` (Options nested) | `WorldScene.SetAudioMasterGain` | Left Dock: **Audio** panel |
| Emitters Gain | `AudioPanelService.cs:214` (Options nested) | `WorldScene.SetAudioEmitterGain` | Left Dock: **Audio** panel |
| Audio Mute Output | `AudioPanelService.cs:59`, `MinimapAndStatusService.cs:510` | `WorldScene.SetAudioMuted` / `AudioMuted` | Left Dock: **Audio** panel & Bottom bar `[x] Mute` |
| Area Music Playback | `AudioPanelService.cs:47-55` (Options nested) | `AreaMusicPlaybackEnabled`, `PlayAreaMusicNow`, `StopAreaMusicNow` | Left Dock: **Audio** panel |
| MCSE/MCNK Triggers | `AudioPanelService.cs:67` (Options nested) | `AudioWorldTriggersEnabled`, `SetAudioWorldTriggersEnabled` | Left Dock: **Audio** panel |
| 3D Emitter Markers | `AudioPanelService.cs:72` (Options nested) | `ShowAudioEmitterMarkers` | Left Dock: **Audio** panel |
| Emitter Diagnostics | `AudioPanelService.cs:85-153` (Options nested) | `RefreshAudioEmitterDiagnostics`, `AudioEmitterDiagnostics`, `AudioMcseFrame` | Left Dock: **Audio** panel |
| SoundEntries Preview | `AudioPanelService.cs:156-209` (Options nested) | `TryPreviewAudioSoundEntry`, `StopAudioPreview`, `ResidentAudioSoundEntryIds` | Left Dock: **Audio** panel |
| Backend Status | `AudioPanelService.cs:79` (Options nested) | `AudioBackendReady` | Left Dock: **Audio** panel |
| Options > Audio header | `ViewerChromeService.Options.cs:38` | `SharedUiWidgets.DrawSectionHeader("Audio")` | **DELETED** (consolidated to Left Dock Audio panel) |
| Left Rail Icon | N/A | None | Left Rail: **Audio** icon (`IconKind.Audio` / Speaker) |

---

## 3. Empty Sub-Tabs & Scroll-Reset Root Cause (T003)

### Empty / Defective Sub-Tabs Identified
1. **Options > Simulation** (`ViewerChromeService.Options.cs:241`):
   - Only contains static text referring the user to Chat Auto-Stream controls and a disabled checkbox if world is null.
   - *Fix*: Remove redundant header; fold active simulation options directly into World/Investigation or provide proper controls.
2. **Options > Objects** (`ViewerChromeService.Options.cs:182`):
   - When no world or model is loaded, renders blank without feedback.
   - *Fix*: Provide consistent empty-state guidance via `SharedUiWidgets.DrawEmptyState`.
3. **Viewer > World > LOD & Budget** (`WorkbenchPanelsService.cs:55`):
   - Only had 3 toggles and a redirect button to Exports.
   - *Fix*: Enrich with actual resident tile counts, LOD draw distance, and budget readouts.
4. **Pages Button**:
   - `bottom-pages` measured on bottom bar, but never wired to any actual target.
   - *Fix*: Completely removed.

### Scroll-Reset Root Cause
1. **Shared Child ID across different panels**:
   - In `ViewerUiManager.cs`:
     - `DrawLeftDock` calls `ImGui.BeginChild("##LeftDockBody", ...)`.
     - `DrawRightDock` calls `ImGui.BeginChild("##RightDockBody", ...)`.
   - Because the same child ID was used for every panel in LeftDock and RightDock, switching between panels or sub-tabs inherited the scroll offset from previously drawn panels.
   - When switching to a shorter panel, ImGui clamped `ScrollY` to `MaxScrollY`. When returning to a longer panel, scroll position remained clamped at the shorter offset, making the panel appear to jump to top / lose scroll position.
2. **Missing `AlwaysVerticalScrollbar`**:
   - Toggling collapsible sections caused scrollbars to dynamically appear/disappear, changing available content width by ~14px and causing ImGui to reflow and scroll jumping.
3. **Header ID Stability**:
   - `SharedUiWidgets.DrawSectionHeader` needs a stable scoped ID so that toggling a collapsible does not re-initialize on scope shifts.

