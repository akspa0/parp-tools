# UI Audit — 2026-10-20

Inventory of the current viewer UI, the functionality behind each surface, and the disposition
recommended by Spec 280. Status values: `works`, `dead`, `duplicate`, `misplaced`, `unknown`.

## 0. Deep dive: menu and sidebar nesting

The operator's complaint is accurate: the File menu and the sidebars are deeply nested and
duplicate each other. The tables below list the actual paths and their depth.

### 0.1 File menu tree

| Path | Depth | Notes |
|---|---|---|
| File > Open File... | 2 | Opens a text-path popup, not the shared picker |
| File > Open Alpha WDT (loose map)... | 2 | |
| File > Open Loose PM4 / PD4 File... | 2 | |
| File > Open Game Folder (MPQ)... | 2 | |
| File > Open CASC Install (local)... | 2 | |
| File > Open CASC Install (local + CDN fill)... | 2 | |
| File > DAT Terrain (v22/23/26) > Open DAT Terrain Folder... | 3 | |
| File > DAT Terrain (v22/23/26) > Export format checkboxes | 3 | |
| File > DAT Terrain (v22/23/26) > Export Loaded DAT Map... | 3 | |
| File > DAT Terrain (v22/23/26) > Export Nearby Tiles as DAT v26 | 3 | |
| File > DAT Terrain (v22/23/26) > Height scale | 3 | |
| File > Open Saved Game Folder > [client] | 3 | |
| File > Attach Loose Map Folder... | 2 | |
| File > Load Loose Map Folder Against Saved Base > [client] | 3 | |
| File > Save Current Game Folder As Known-Good Base | 2 | |
| File > Forget Known-Good Base > [client] | 3 | |
| File > Settings... | 2 | |
| File > Quit | 2 | |

**Issues**: 18 entries, 3 levels deep, and the DAT submenu mixes open, export, and format
selection. The "Open Saved Game Folder" and "Load Loose Map Folder Against Saved Base" submenus
repeat the same client list. "Open File..." uses a bespoke text popup while every other open action
uses the shared path picker.

### 0.2 View menu tree

| Path | Depth | Notes |
|---|---|---|
| View > Wireframe | 2 | |
| View > Reset Camera | 2 | |
| View > Hide UI Chrome | 2 | |
| View > Tab System (069) | 2 | Legacy toggle |
| View > Dockable Shell Panels | 2 | Legacy toggle |
| View > Left Sidebar | 2 | |
| View > Right Sidebar | 2 | |
| View > Log Console... | 2 | Duplicates Tools > Panels > Log Viewer |
| View > Performance & Profiling... | 2 | Duplicates Tools > Panels > Perf |
| View > Lighting Diagnostics... | 2 | Duplicates Tools > Panels |
| View > PM4 Workbench | 2 | Duplicates the PM4 tab and the PM4 Actions popup |
| View > Reset Shell Layout | 2 | |
| View > File Browser | 2 | Duplicates the left sidebar's File Browser section |
| View > Model Info | 2 | Duplicates Tools > Panels > Model Info |
| View > Asset Catalog | 2 | Duplicates Tools > Panels > Asset Catalog |

**Issues**: 15 entries, 8 of which duplicate a Tools > Panels entry or a sidebar section. The two
legacy shell toggles ("Tab System (069)", "Dockable Shell Panels") are implementation switches
exposed as user-facing menu items.

### 0.3 Tools menu tree

| Path | Depth | Notes |
|---|---|---|
| Tools > Taxi Routes... | 2 | Duplicates Tools > Panels > Taxi |
| Tools > Audio Settings... | 2 | |
| Tools > Converters > Map Converter... | 3 | |
| Tools > Converters > WMO Converter... | 3 | |
| Tools > Panels > Model Info | 3 | |
| Tools > Panels > Log Viewer | 3 | |
| Tools > Panels > Perf | 3 | |
| Tools > Panels > Settings... | 3 | Duplicates File > Settings |
| Tools > Panels > Asset Catalog | 3 | |
| Tools > Panels > Capture Automation | 3 | |
| Tools > Panels > Camera Path | 3 | |
| Tools > Panels > Taxi | 3 | |
| Tools > Panels > Weak Signal & Stratigraphy | 3 | |
| Tools > Panels > UniqueId Archaeology | 3 | |
| Tools > Panels > PM4 Analysis | 3 | Redirects to the PM4 workbench |
| Tools > Panels > Cartography | 3 | |
| Tools > Panels > Editor Workbench | 3 | |
| Tools > Export > Synthesized Terrain Minimap... | 3 | |
| Tools > Export > GLB > Export GLB... | 4 | |
| Tools > Export > GLB > Export GLB (Collision Only)... | 4 | |
| Tools > Export > GLB > Map Tiles > Current Tile | 5 | |
| Tools > Export > GLB > Map Tiles > Loaded Tiles Folder | 5 | |
| Tools > Export > GLB > Map Tiles > Whole Map Folder | 5 | |
| Tools > Export > Terrain > Alpha Masks > Current Tile Atlas | 5 | |
| Tools > Export > Terrain > Alpha Masks > Current Tile Chunks Folder | 5 | |
| Tools > Export > Terrain > Alpha Masks > Loaded Tiles Folder | 5 | |
| Tools > Export > Terrain > Alpha Masks > Whole Map Folder | 5 | |
| Tools > Export > Terrain > Heightmaps > Current Tile | 5 | |
| Tools > Export > Terrain > Heightmaps > Loaded Tiles Folder | 5 | |
| Tools > Export > Terrain > Heightmaps > Whole Map Folder | 5 | |
| Tools > Export > Terrain > MCCV > Current Tile PNG | 5 | |
| Tools > Export > Terrain > MCCV > Loaded Tiles Folder | 5 | |
| Tools > Export > Terrain > MCCV > Whole Map Folder | 5 | |
| Tools > Import > Terrain > Alpha Masks > From Folder | 5 | |
| Tools > Import > Terrain > Heightmaps > From Folder | 5 | |
| Tools > Import > Terrain > MCCV > From Folder | 5 | |

**Issues**: 36 entries, up to 5 levels deep. The Panels submenu is a second navigation system that
duplicates the workbench tabs. The Export/Import trees are the worst offenders: a user must go
Tools > Export > Terrain > Alpha Masks > Whole Map Folder to export one thing.

### 0.4 Left sidebar tree

| Path | Depth | Notes |
|---|---|---|
| Left sidebar > Workspace bars | 1 | |
| Left sidebar > World Overview | 1 | Collapsing header |
| Left sidebar > World Overview > Export Map | 2 | DAT-only |
| Left sidebar > File Browser | 1 | Collapsing header; duplicates View > File Browser |
| Left sidebar > World Maps | 1 | Collapsing header |
| Left sidebar > Chat | 1 | Separate window, now the population control bar |

**Issues**: the sidebar is shallow, but it duplicates the View menu's File Browser and the Tools
menu's map actions. The "World Overview" section mixes the minimap, the map export, and the map
list.

### 0.5 Right sidebar tree

| Path | Depth | Notes |
|---|---|---|
| Right sidebar > Inspector | 1 | |
| Right sidebar > World | 1 | |
| Right sidebar > Model | 1 | |
| Right sidebar > Settings | 1 | |
| Right sidebar > PM4 | 1 | Eight PM4 pages |
| Right sidebar > PM4 > Overlay / Selection / Correlation / Info / Reconcile / Alignment / Outliner / Object Groups | 2 | |

**Issues**: the right sidebar is the cleanest surface. The PM4 tab is the model to follow. The
"World" and "Model" tabs overlap with the left sidebar's file browser and the Tools > Panels
entries.

### 0.6 Cross-surface duplicates

| Action | Surfaces |
|---|---|
| File Browser | View menu, left sidebar |
| Model Info | View menu, Tools > Panels, right sidebar Model tab |
| Asset Catalog | View menu, Tools > Panels |
| Log Viewer | View menu, Tools > Panels |
| Perf | View menu, Tools > Panels |
| Settings | File menu, Tools > Panels, right sidebar Settings tab |
| PM4 | View menu, Tools > Panels, right sidebar PM4 tab, PM4 Actions popup |
| Taxi | Tools menu, Tools > Panels |
| Archaeology pages | Tools > Panels, Archaeology destination |
| Editor Workbench | Tools > Panels, Editor destination |

### 0.7 Recommended flattened structure

```text
File
  Open
    File...
    Game Folder (MPQ)...
    CASC Install...
    DAT Terrain Folder...
    Loose Map Folder...
    Loose PM4 / PD4 File...
  Saved Bases
    Open [client]
    Attach Loose Map Folder...
    Save Current As Known-Good Base
    Forget [client]
  Settings...
  Quit

View
  Wireframe
  Reset Camera
  Hide UI Chrome
  Left Sidebar
  Right Sidebar
  Reset Layout

Help
  Keyboard Shortcuts
  About
```

- **All export/import actions move to the destination that owns the data**: terrain export/import
  to the Terrain destination, model export to the Model/Editor destination, PM4 export to the PM4
  workbench.
- **All panel navigation moves to the bottom bar**: Viewer, Editor, Archaeology, Options.
- **The Tools > Panels submenu is deleted**; its entries become destination surfaces.
- **The legacy shell toggles are removed** from the user-facing menu.
- **The left sidebar keeps only**: Workspace, World Overview (minimap + map list), File Browser,
  Chat.
- **The right sidebar keeps only the current destination's inspector.**

### 0.8 Audit status

| Area | Status |
|---|---|
| File menu | audited; flattening proposed |
| View menu | audited; duplicates identified |
| Tools menu | audited; 5-level export/import tree identified |
| Left sidebar | audited; duplicates identified |
| Right sidebar | audited; cleanest surface |
| Bottom bar | audited; options toolbar to move |
| Status bar | audited; keep |
| Chat bar | audited; keep, add Options anchor |
| Editor pages | partially audited; needs a per-page control pass |
| Archaeology pages | partially audited; needs a per-page control pass |
| Popups | partially audited; overlays/layers/PM4 actions identified |

## 1. Surface inventory

| Surface | Owner | Current content | Status | Disposition |
|---|---|---|---|---|
| Top menu bar | `MainMenuBarService` | File, View, Tools, Help; PM4 Workbench, Reset Shell Layout, File Browser, Model Info, Asset Catalog, capture, converters, datasets | `duplicate` | Reduce to File / View / Help; move destination actions to the bottom bar |
| Bottom bar | `ViewerChromeService.DrawBottomBar` | Render toggles (Base, Lit, Holes, Alpha, Shadows, MCCV, Contours, Liquid, WL, BBS, PM4, Inspect, Opacity, WMO Tex, Walls, Terrain WF, M2/WMO WF, Anim, 3D Cursor, Settings), overlays popup, layers popup, transparency | `misplaced` | Keep a slim destination bar; move the toggles into an Options sub-menu next to the chat |
| Status bar | `MinimapAndStatusService.DrawStatusBar` | FPS, area, CPU, tiles, chunks, M2/WMO counts, audio, pending | `works` | Keep; it is the one stable readout |
| Left navigator | `NavigatorPanelService` | Workspace, World Overview, minimap, WC3 data, map list, chat | `works` | Keep; make it destination-aware; rename WC3 to WC3 Beta |
| Right inspector | `WorkbenchPanelsService` | Tabs: Inspector, World, Model, Settings, PM4 | `works` | Keep as the Viewer destination's primary surface |
| Workbench tabs | `WorkbenchTab` + `WorkbenchNavigator` | Quick, Inspect, Scene, Utilities, Experimental, Editor, Archaeology; `Canonicalize` redirects Scene→Inspect, Utilities→Quick, Experimental→Editor | `misplaced` | Replace with three destinations: Viewer, Editor, Archaeology |
| PM4 workbench | `Pm4WorkbenchService` | Overlay, Selection, Correlation, Info, Reconcile, Alignment, Outliner, Object Groups | `works` | Keep; it is the model for a destination-specific surface |
| Chat bar | `FakeChatWindowService` | Chat log, Auto-Scroll, Hide, Auto-Stream, Spawn BBs Only | `works` | Keep; anchor the Options sub-menu to it |
| Editor pages | `EditorPlacementObjectsPage` and siblings | Placement & Objects, Terrain Tools, Data I/O, Converters | `works` | Keep; remove viewer-only inspection surfaces |
| Archaeology pages | `ArchaeologyPanelService` | Weak Signal & Stratigraphy, UniqueId Timeline, Layers & Provenance, Playback & Capture, PM4 Analysis (redirect), Cartography | `works` | Keep; remove unrelated editor/viewer tools |
| Overlays popup | `ViewerChromeService` | Overlay toggles | `duplicate` | Move into Options sub-menu |
| Layers popup | `ViewerChromeService` | Layer toggles | `duplicate` | Move into Options sub-menu |
| PM4 Actions popup | `WorldObjectsPanelService` | PM4 Workbench, Reload PM4 | `duplicate` | Keep one entry point in the PM4 workbench |
| Object transparency | `ViewerChromeService` | Opacity sliders | `misplaced` | Move into Options sub-menu |

## 2. Known dead or broken controls

| Control | Surface | Symptom | Disposition |
|---|---|---|---|
| PM4 export buttons | PM4 Actions / Data I/O | Opened a picker that did not produce files | Removed 2026-10-20; re-add only with a tested path |
| PM4 selection | 3D viewport | Clicks passed through to WLW planes | Fixed 2026-10-20: pick uses the full geometry transform |
| PM4 culling | 3D viewport | Objects vanished when the camera was close | Fixed 2026-10-20: cull uses the full geometry transform |
| Population double-click | Population directory | Did not move the camera | Fixed 2026-10-20: check the item hover and mouse state directly |
| PM4 selection box | 3D viewport | Box spanned the map | Fixed 2026-10-20: bounds localized and clamped to the originating tile |

## 3. Styling inconsistencies

- Some panels use `SharedUiWidgets.SectionHeader`; others use raw `ImGui.CollapsingHeader`,
  `ImGui.SeparatorText`, or `ImGui.TextColored` with local colors.
- Font sizes and text wrapping vary between the navigator, inspector, PM4 workbench, and editor
  pages.
- The PM4 workbench glossary, the population panel, and the archaeology panels each have their own
  header and body-text conventions.
- The theme is global, but several panels hard-code colors that do not follow the theme.

## 4. Target layout

```text
┌──────────────────────────────────────────────────────────────────────┐
│ File   View   Help                                                   │  top menu (global only)
├───────────────┬──────────────────────────────────────┬───────────────┤
│ Navigator     │                                      │ Inspector     │
│ (destination) │              viewport                │ (destination) │
│               │                                      │               │
├───────────────┴──────────────────────────────────────┴───────────────┤
│ Chat bar: [General] [Clear] [Auto-Scroll] [Auto-Stream] [Spawn BBs]  │
│           [Options ▾]  (sub-menu opens upward, next to the chat)     │
├──────────────────────────────────────────────────────────────────────┤
│ Viewer | Editor | Archaeology | Options | FPS / stats                │  bottom bar
└──────────────────────────────────────────────────────────────────────┘
```

- **Viewer** shows the Inspector and the PM4 workbench.
- **Editor** shows editor menus and editor tools.
- **Archaeology** shows the archaeology analysis surfaces.
- **Options** opens the render/overlay/layer/transparency toggles that currently live in the bottom
  toolbar.
- The FPS/stats readout stays in the bottom bar in every destination.

## 5. Recommended sequence

1. **Audit closure**: confirm every row above against the running viewer; add any missing control.
2. **Bottom bar**: add the destination buttons and the Options button; keep the FPS/stats readout.
3. **Options sub-menu**: move the render toggles, overlays, layers, and transparency into it.
4. **Destination separation**: make the left and right sidebars destination-aware; remove
   cross-destination controls.
5. **Styling pass**: route every section through `SharedUiWidgets`; remove local colors and
   bespoke headers.
6. **Naming**: rename WC3 data support to "WC3 Beta" in the UI and documentation.
7. **Dead-control sweep**: remove or wire every remaining `dead` row.

## 6. Open questions for the operator

- Should the top menu keep File / View / Help only, or also keep a Tools menu for global tools?
- Should the Options sub-menu be a popup anchored to the chat, or a collapsible strip above the
  chat?
- Should the destination buttons be text labels or icon buttons with tooltips?
