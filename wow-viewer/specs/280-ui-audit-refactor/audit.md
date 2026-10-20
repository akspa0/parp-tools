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

### 0.8 Editor destination tree

The Editor destination is a 4-page strip (`GetEditorWorkbenchLabels`), and each page is a stack of
collapsing sections that call back into the shell through `IEditorPageHost`. Depth is
destination → page → section → content, and the content itself opens further tabs.

| Path | Depth | Notes |
|---|---|---|
| Editor > Placement & Objects | 2 | `EditorPlacementObjectsPage` |
| Editor > Placement & Objects > Tasks & Workspace | 3 | Default open; `DrawArchaeologyEditorTasksSubTab` |
| Editor > Placement & Objects > PM4 Placement Tools | 3 | Collapsed; one button to the PM4 Workbench |
| Editor > Placement & Objects > 3D Object Library | 3 | Collapsed; `DrawRosettaObjectLibrarySubTab` |
| Editor > Placement & Objects > Population | 3 | Collapsed; `DrawPopulationSubTabContent` |
| Editor > Terrain Tools | 2 | `EditorTerrainToolsPage` |
| Editor > Terrain Tools > Terrain Lab | 3 | Default open; `DrawTerrainLabSubTab` |
| Editor > Data I/O | 2 | `EditorDataIoPage` |
| Editor > Data I/O > PM4 Exports | 3 | Collapsed; `DrawPm4ExportCommandSet` |
| Editor > Data I/O > New Map Creator | 3 | Default open; `NewMapCreatorService.Draw` |
| Editor > Data I/O > Imports & Exports | 3 | Default open; `DrawArchaeologyEditorImportsSubTab` |
| Editor > Converters | 2 | `EditorConvertersPage` |
| Editor > Converters > Converters | 3 | Default open; `DrawConvertersSubTabContent` |

**Issues**:
- The page names are clean, but the section names are not: "Tasks & Workspace" is the archaeology
  editor task list, "3D Object Library" is the Rosetta library, and "Imports & Exports" is the
  archaeology import surface. Three of the four Editor pages host archaeology-named content.
- "PM4 Placement Tools" is a section whose entire body is one navigation button. It is a menu entry
  wearing a section header.
- "PM4 Exports" is a second PM4 export home; the PM4 Workbench also exposes export commands. This is
  the same duplicate-home defect as the Tools menu.
- `EditorWorkbenchPages.MigrateLegacyEditorPageIndex` still maps six legacy page indices onto four
  pages, so saved layouts silently land on a different page than the user left.
- The Editor destination has no visible relationship to the Editor menu entries in the top menu bar;
  the two are separate navigation systems for the same work.

### 0.9 Archaeology destination tree

The Archaeology destination is a 6-page strip (`GetArchaeologyWorkbenchLabels`). It is the deepest
surface in the application because it carries **four parallel navigation systems** for the same
content.

| Path | Depth | Notes |
|---|---|---|
| Archaeology > Weak Signal & Stratigraphy | 2 | `DrawTerrainControlsAdjustmentWeakSignalContent` |
| Archaeology > UniqueId Timeline | 2 | `DrawArcheologyRangeSubTab` |
| Archaeology > Layers & Provenance | 2 | `DrawArcheologyLayersSubTab` |
| Archaeology > Playback & Capture | 2 | `DrawArcheologyPlaybackSubTab` + `DrawArcheologyCaptureSubTab` |
| Archaeology > Playback & Capture > Camera Paths & Video Capture | 3 | `DrawCapturePanelContent` |
| Archaeology > PM4 Analysis | 2 | Redirect button to the PM4 Workbench |
| Archaeology > Cartography | 2 | `DrawArchaeologyCartographyContent` |
| Archaeology > Cartography > Layers | 3 | `DrawCartographyLayersSubTab` |
| Archaeology > Cartography > Layers > Layer stack | 4 | Per-layer expand: channels, offsets, rotation, mirror, Z, edge-snap |
| Archaeology > Cartography > Layers > DBC phase discovery | 4 | `Map.dbc` child maps |
| Archaeology > Cartography > Layers > Add phase map | 4 | Combo + manual directory + DAT folder |
| Archaeology > Cartography > Layers > Donor Tile Grid Picker | 4 | Collapsing header; donor/target tile inputs |
| Archaeology > Cartography > Synthesized Minimap | 3 | `DrawSynthesizedMinimapExportContent` |
| Archaeology > Cartography > Save Merged Output | 3 | `DrawCartographySaveMergedSubTab` |

**Parallel navigation systems for the same content**:

| System | Entry point | Pages | Status |
|---|---|---|---|
| Destination strip | `GetArchaeologyWorkbenchLabels` | 6 | Canonical |
| Legacy sub-tab strip | `DrawArcheologySubTabContent` + `GetArcheologyBottomTabLabels` | 5 (Range, Layers, Playback, Capture, Stratigraphy) | `duplicate` |
| Floating window | `DrawUniqueIdArchaeologyWindow` | Stacks Range + Layers + Playback vertically | `duplicate` |
| Quick section | `DrawArchaeologyQuickSection` | Launchers into pages 1, 2, 5 | `duplicate` |

**Issues**:
- The 6-page destination and the 5-tab legacy strip cover the same five concepts with different
  names ("UniqueId Timeline" vs "Range", "Layers & Provenance" vs "Layers", "Playback & Capture" vs
  "Playback" + "Capture"). A user cannot tell which one is authoritative.
- The floating "UniqueId Archaeology" window stacks three of the same sub-tabs vertically, so the
  same controls can be on screen twice.
- The Quick section's "Archaeological Launchers" is a fourth entry point into the same pages.
- Cartography is a 3-level tree inside a 2-level destination, and its "Layers" page is itself a
  stack of four independent tools (layer stack, DBC discovery, add controls, donor picker).
- `DrawNestedSubTabStrip` exists only to let a nested section select its own index; the comment in
  the source says the parent strip "only covers the first level". That is the nesting defect stated
  in the code itself.

### 0.10 Popup and floating-window tree

| Path | Depth | Notes |
|---|---|---|
| View > Log Console... | 2 | Floating window; duplicates Tools > Panels > Log Viewer |
| View > Performance & Profiling... | 2 | Floating window; duplicates Tools > Panels > Perf |
| View > Lighting Diagnostics... | 2 | Floating window |
| View > PM4 Workbench | 2 | Focuses the right sidebar PM4 tab |
| Tools > Panels > Model Info | 3 | Floating window |
| Tools > Panels > Log Viewer | 3 | Floating window |
| Tools > Panels > Perf | 3 | Floating window |
| Tools > Panels > Settings... | 3 | Floating window; duplicates File > Settings |
| Tools > Panels > Asset Catalog | 3 | Floating window |
| Tools > Panels > Capture Automation | 3 | Floating window |
| Tools > Panels > Camera Path | 3 | Floating window |
| Tools > Panels > Taxi | 3 | Floating window |
| Tools > Panels > Weak Signal & Stratigraphy | 3 | Floating window; duplicates Archaeology page 0 |
| Tools > Panels > UniqueId Archaeology | 3 | Floating window; duplicates Archaeology pages 1–3 |
| Tools > Panels > PM4 Analysis | 3 | Redirect to the PM4 Workbench |
| Tools > Panels > Cartography | 3 | Floating window; duplicates Archaeology page 5 |
| Tools > Panels > Editor Workbench | 3 | Redirect to the Editor destination |
| Tools > Export > Synthesized Terrain Minimap... | 3 | Dialog |
| Tools > Converters > Map Converter... | 3 | Dialog |
| Tools > Converters > WMO Converter... | 3 | Dialog |
| Bottom bar > Overlays | 2 | Popup |
| Bottom bar > Layers | 2 | Popup |
| Bottom bar > Transparency | 2 | Popup |
| PM4 Actions popup | 2 | `WorldObjectsPanelService`; PM4 Workbench + Reload PM4 |
| File > Open File... | 2 | Bespoke text-path popup |
| File > Settings... | 2 | Dialog |
| File > DAT Terrain > Export Loaded DAT Map... | 3 | Dialog |
| File > DAT Terrain > Export Nearby Tiles as DAT v26 | 3 | Dialog |
| VLM export dialog | 2 | `PrepareVlmExportDialogInputs` |
| Terrain texture transfer dialog | 2 | `PrepareTerrainTextureTransferDialogInputs` |
| Synthesized minimap export dialog | 2 | `PrepareSynthesizedMinimapExportDialogInputs` |

**Issues**:
- 12 of the 17 Tools > Panels entries are floating windows that duplicate a destination page or a
  sidebar section. They are a second, invisible navigation system: the user can open the same
  content as a docked page or as a floating window, with different state.
- The bottom bar's three popups (Overlays, Layers, Transparency) are the only render controls that
  are not in a destination; they belong in the Options sub-menu.
- The PM4 Actions popup is a third PM4 entry point beside the right sidebar tab and the Tools menu.
- Four separate export dialogs are launched from three different menus.

### 0.11 Audit status

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
| Editor pages | audited; 4 pages, 13 sections, archaeology-named content |
| Archaeology pages | audited; 6 pages, 4 parallel navigation systems, 4-level Cartography tree |
| Popups | audited; 12 duplicate floating windows, 3 bottom-bar popups, 4 export dialogs |

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
| Editor pages | `EditorPlacementObjectsPage` and siblings | Placement & Objects, Terrain Tools, Data I/O, Converters; 13 sections, 3 of which host archaeology-named content | `misplaced` | Keep the 4 pages; rename the sections to their real owners; delete the one-button "PM4 Placement Tools" section |
| Archaeology pages | `ArchaeologyPanelService` | Weak Signal & Stratigraphy, UniqueId Timeline, Layers & Provenance, Playback & Capture, PM4 Analysis (redirect), Cartography | `duplicate` | Keep the 6-page strip; delete the legacy 5-tab strip, the floating UniqueId window, and the Quick launchers |
| Archaeology legacy strip | `DrawArcheologySubTabContent` | Range, Layers, Playback, Capture, Stratigraphy | `duplicate` | Delete; the 6-page destination is authoritative |
| UniqueId Archaeology window | `DrawUniqueIdArchaeologyWindow` | Stacks Range + Layers + Playback vertically | `duplicate` | Delete; it is a second copy of three destination pages |
| Archaeology Quick launchers | `DrawArchaeologyQuickSection` | Buttons into pages 1, 2, 5 | `duplicate` | Delete; the destination strip is the only navigation |
| Cartography tree | `DrawArchaeologyCartographyContent` | Layers, Synthesized Minimap, Save Merged Output; Layers is a 4-tool stack | `misplaced` | Keep as a page; split the Layers stack into named sections |
| Floating tool windows | `MainMenuBarService` + `ViewerApp` | 12 Tools > Panels windows duplicating destination pages | `duplicate` | Delete the duplicates; keep only genuinely global windows (Log, Perf) |
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
5. **Archaeology dedupe**: delete the legacy 5-tab strip, the floating UniqueId window, and the
   Quick launchers; the 6-page destination strip becomes the only navigation. Split the Cartography
   "Layers" page into named sections.
6. **Editor section rename**: rename "Tasks & Workspace", "3D Object Library", and "Imports &
   Exports" to their real owners; delete the one-button "PM4 Placement Tools" section; remove the
   second PM4 export home.
7. **Floating-window dedupe**: delete the 12 Tools > Panels windows that duplicate destination
   pages; keep only Log and Perf as global windows.
8. **Styling pass**: route every section through `SharedUiWidgets`; remove local colors and
   bespoke headers.
9. **Naming**: rename WC3 data support to "WC3 Beta" in the UI and documentation.
10. **Dead-control sweep**: remove or wire every remaining `dead` row.

## 6. Open questions for the operator

- Should the top menu keep File / View / Help only, or also keep a Tools menu for global tools?
- Should the Options sub-menu be a popup anchored to the chat, or a collapsible strip above the
  chat?
- Should the destination buttons be text labels or icon buttons with tooltips?
