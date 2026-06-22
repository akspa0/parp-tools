# Viewer UI — Tab System Architecture (069)

**Status**: Active (default UI as of 069 Phase 7)
**Date**: 2026-06-21
**Related**: spec 069 (Viewer UI Overhaul), spec 044, 049, 060

## TL;DR

The viewer UI was reorganized from a sidebar+panel sprawl into a clean two-band tab system. Six top-level tabs at the top of the viewport, each with its own context-sensitive sub-tabs at the bottom. All scattered floating windows, sidebars, dock panels, and shell panels route into a single, predictable location.

The user can toggle back to the legacy sidebar/panel system via **View → Tab System (069)** if they prefer the old layout.

## Top Tabs

| Top Tab | Sub-tabs | What it does |
|---------|----------|--------------|
| **Scene** | Selection, Camera, Settings, Themes | Selection summary, camera controls, app settings, theme picker |
| **World** | Placements, Tiles, Overlays, Selection Tools | MDDF/MODF/WMO placements + tile/chunk workbench + grid/shadow/alpha overlays |
| **Terrain** | Layers, Clipboard, Analysis, MCNK, Weak Signal, Export | Texture layers, chunk copy/paste, terrain analysis, MCNK explorer, weak signal restore, scoped export |
| **PM4** | Overlay, Selection, Correlation, Info, Match, Alignment | PM4 workbench + object match + WMO correlation + alignment tools |
| **Archeology** | Range, Layers, Playback, Capture | UniqueId range filter + detected layers + playback animation + capture integration |
| **Utilities** | Minimap, Log, Perf, Render Quality, Taxi, Capture Automation, Asset Catalog, Runtime Stats | All utility tools consolidated |

## How it Works

- `_useTabUi` (default `true`): new tab UI. Set to `false` to fall back to sidebars + panels.
- `_activeTopTab`: which top tab is selected.
- `_activeBottomTabIndex`: which sub-tab within the active top tab.
- All three persist across viewer restarts (`ViewerSettings.UseTabUi/ActiveTopTab/ActiveBottomTab`).

## Top Tab Content Band

A slim context band sits between the top tab bar and the bottom tab bar, showing key info for the active top tab:
- **Scene**: target + camera target
- **World**: map name + loaded tile count
- **Terrain**: chunk count
- **PM4**: object count + visible count
- **Archeology**: scope status + filter status
- **Utilities**: current sub-tab name

## Sub-tab Wiring

Each sub-tab routes to an existing content method (extracted from sidebar or window). Where possible, the "headless" content method is called directly inside a child region. For methods that open their own `ImGui.Begin(...)` (like `DrawLogViewer`), the result is a nested window inside the sub-tab — works, but visually a bit busy.

## Sticky Archeology Settings

Archeology tab has full state persistence:
- Visible Range Start / End (the actual range)
- Scope (Per-Map / Camera Tile)
- Playback speed
- Loop checkbox
- "Apply to next capture" / "Apply to video recording" flags

All persisted to `viewer_settings.json`. Restart viewer → all settings restored.

## Archeology Playback

The killer feature: open the **Archeology** tab, set Visible Range, hit **Play**. The scene animates `Visible Range End` from min to max at the configured speed. Simulates the world being built up over time.

- **Speed**: 1-5000 uniqueIds/sec (slider)
- **Loop**: restarts at min when reaching max
- **Slider touch pauses playback** (per user decision)
- **Capture integration**: `Apply to next capture` advances end per shot. `Apply to video recording` starts playback during recording, captures full progression at real-time.

## Migration Path

The old UI is still there (sidebars, panels, floating windows). They are:
- Suppressed when `_useTabUi = true`
- Accessible when `_useTabUi = false` (toggle in View menu)
- Marked for deprecation (1 release cycle)
- Will be removed in 070 (next cleanup pass)

## Files Touched

| File | Change |
|------|--------|
| `src/viewer/WoWViewer/ViewerApp.cs` | TopTab/BottomTab enums, fields, save/load integration, DrawUI wiring, UpdateArcheologyPlayback |
| `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` | DrawTopTabBar, DrawBottomTabBar, per-top-tab sub-tab dispatchers |
| `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs` | DrawMinimapContent (headless variant) |
| `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs` | PendingCaptureRequest.ApplyArcheologyPlayback, ActiveVideoRecording.ApplyArcheologyPlayback, per-shot advance, video start/stop hooks |
| `specs/069-viewer-ui-overhaul/spec.md` | Full spec |
| `specs/069-viewer-ui-overhaul/plan.md` | 8-phase plan |
| `specs/069-viewer-ui-overhaul/tasks.md` | 79 tasks, all done |

## Out of Scope (Future)

- Replace ImGui with Avalonia (deferred per 060)
- Extract Archeology playback into a separate PlaybackOrchestrator service (currently inline in ViewerApp)
- Replace the sidebars in the legacy fallback with a proper `IShellPanelHost` abstraction
