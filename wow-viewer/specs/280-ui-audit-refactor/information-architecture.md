# Information Architecture — 2026-10-21

The single logical map for the UI rework. Every visible surface gets **exactly one home**. If a
surface appears in two places today, one place is the home and the other is deleted, not hidden.

This document answers the operator's question directly: **yes, all 10 Utilities pages are covered**
(§5), and every one of them is reassigned to a logical home.

## 1. Why the UI is a mess

The audit found three generations of UI layered on top of each other:

1. **Legacy floating windows** (`_showLogViewer`, `_showPerfWindow`, `_showModelInfo`,
   `_showFileBrowser`, `_showSettingsWindow`, the UniqueId Archaeology window).
2. **The 069 tab system** (`WorkbenchTab`, `ToolsBottomTab`, `UtilitiesBottomTab`,
   `ModelBottomTab`, `WorldBottomTab`, `TerrainBottomTab`, `Pm4BottomTab`, `ArcheologyBottomTab`).
3. **The 231/280 destination model** (`WorkbenchTab.Editor`, `WorkbenchTab.Archaeology`, the
   bottom bar, the PM4 workbench).

All three are still live. `Canonicalize` and the `OpenWorkbenchTab` overloads translate between
them, which is why the same page can be reached four ways. The fix is to keep generation 3 and
delete generations 1 and 2.

## 2. The rule

> One surface, one home. A second entry point is a navigation shortcut, never a second copy.

A navigation shortcut is a button that switches destination and page. It does not draw the content.

## 3. Destinations

The bottom bar is the only destination switcher:

```text
Viewer | Editor | Archaeology | Options | FPS/stats
```

| Destination | Purpose | Right sidebar | Left sidebar |
|---|---|---|---|
| **Viewer** | Inspect the loaded world, models, and PM4 | Inspector, World, Model, PM4 | Workspace, World Overview, File Browser, Scene Tools, Assets, Diagnostics, Chat |
| **Editor** | Author and convert maps | Placement & Objects, Terrain Tools, Data I/O, Converters | Editor Workspace, Project Output, File Browser, Diagnostics, Chat |
| **Archaeology** | Analyse UniqueId, layers, cartography | Weak Signal, UniqueId Timeline, Layers & Provenance, Playback & Capture, Cartography | Archaeology Workspace, Layer Projects, File Browser, Diagnostics, Chat |
| **Options** | Global display, audio, and simulation settings | — (sub-menu) | — |

`Options` is a sub-menu anchored to the chat bar, not a full destination. It opens upward next to
the chat and contains the toggles that currently live in the bottom toolbar.

## 4. Left sidebar map

The left sidebar is **destination-aware**. It shows the common sections plus the current
destination's sections.

### Common (every destination)

| Section | Content | Notes |
|---|---|---|
| Workspace | Workspace bars, open game folder, open file | The only place to open data |
| World Overview | Minimap, map list, map export | The minimap is the canonical one; the Utilities Minimap page is deleted |
| File Browser | Data-source tree | The only file browser; the View menu toggle is deleted |
| Diagnostics | Log, Perf, Runtime Stats | Collapsible; the three Utilities pages move here |
| Chat | Chat log, Auto-Scroll, Auto-Stream, Spawn BBs, Options ▾ | The Options anchor |

### Viewer only

| Section | Content | Notes |
|---|---|---|
| Scene Tools | Taxi, Capture, Camera Path | The three Utilities pages move here |
| Assets | Asset Catalog | The Utilities Asset Catalog page moves here |

### Editor only

| Section | Content | Notes |
|---|---|---|
| Editor Workspace | Editor session, undo, output folder | |
| Project Output | Current project folder, new folder | |

### Archaeology only

| Section | Content | Notes |
|---|---|---|
| Archaeology Workspace | Current map, scope, playback state | |
| Layer Projects | Save/load layer project | |

## 5. Utilities coverage — all 10 pages

The Utilities tab currently has 10 pages. Every one is reassigned:

| # | Utilities page | Current content | New home | Action |
|---|---|---|---|---|
| 1 | Minimap | Minimap viewer | Left sidebar > World Overview | Delete the Utilities page; the sidebar minimap is canonical |
| 2 | Log | Log viewer | Left sidebar > Diagnostics | Move |
| 3 | Perf | Performance panel | Left sidebar > Diagnostics | Move |
| 4 | Render Quality | Render quality settings | Options > Display | Move |
| 5 | Taxi | Taxi route tools | Viewer > Scene Tools | Move |
| 6 | Capture | Capture automation + camera path | Viewer > Scene Tools | Move |
| 7 | Asset Catalog | Asset browser | Viewer > Assets | Move |
| 8 | Runtime Stats | Runtime counters | Left sidebar > Diagnostics | Move |
| 9 | Lighting | Lighting diagnostics/settings | Options > Display | Move |
| 10 | Audio | Audio settings | Options > Audio | Move |

**Result**: the `UtilitiesBottomTab` enum and the `Utilities` workbench tab are deleted. No
Utilities page survives as a Utilities page; each becomes a named section in its logical home.

## 6. Right sidebar map

### Viewer

| Tab | Pages | Notes |
|---|---|---|
| Inspector | Context, Placements, LOD & Budget | The primary Viewer surface |
| World | World context, tiles, selection tools | |
| Model | Model info, animations, actions | |
| PM4 | Overlay, Selection, Info, Outliner, Object Groups | Inspection only; matching is offline (Spec 281) |

### Editor

| Page | Sections |
|---|---|
| Placement & Objects | Population, Rosetta Object Library, Archaeology Tasks |
| Terrain Tools | Terrain Lab, Terrain Export, Terrain Import |
| Data I/O | New Map Creator, Archaeology Imports |
| Converters | Map Converter, WMO Converter, M2/MDX converters |

### Archaeology

| Page | Sections |
|---|---|
| Weak Signal & Stratigraphy | Weak-signal tooling |
| UniqueId Timeline | Range, time machine, quantiles |
| Layers & Provenance | Layer table |
| Playback & Capture | Playback, capture integration, camera paths |
| Cartography | Layers (layer stack, DBC discovery, add map, donor picker), Synthesized Minimap, Save Merged Output |

## 7. Options sub-menu map

Anchored to the chat bar, opens upward.

| Group | Controls |
|---|---|
| Display | Render Quality, Lighting, Theme |
| Overlays | Alpha, Shadows, MCCV, Contours, Liquid, WL, BBs, PM4 |
| Layers | Terrain layer toggles, WDL |
| Grids | Tiles, Chunks, Cells |
| Objects | Object transparency, Terrain WF, M2/WMO WF, Anim, 3D Cursor |
| Audio | Audio settings |
| Simulation | Auto-Stream, Spawn BBs Only (also in the chat bar) |

## 8. Full surface → home table

Every surface from [audit.md](audit.md) and [functionality.md](functionality.md), with its single
home.

| Surface | Home | Fate |
|---|---|---|
| Open File... | Left sidebar > Workspace | Keep; shared picker |
| Open Game Folder... | Left sidebar > Workspace | Keep; shared picker |
| Open Alpha WDT | Left sidebar > Workspace | Keep |
| Open Loose PM4/PD4 | Left sidebar > Workspace | Keep |
| Open CASC Install | Left sidebar > Workspace | Keep |
| DAT Terrain open/export | Left sidebar > Workspace | Keep |
| Saved game folders | Left sidebar > Workspace | Keep |
| Settings | File menu | Keep |
| Quit | File menu | Keep |
| Wireframe | Options > Objects | Move; rename "Model Wireframe" |
| Reset Camera | View menu | Keep |
| Hide UI Chrome | View menu | Keep |
| Tab System (069) | — | Delete |
| Dockable Shell Panels | — | Delete |
| Left/Right Sidebar | View menu | Keep |
| Log Console | Left sidebar > Diagnostics | Move |
| Performance | Left sidebar > Diagnostics | Move |
| Lighting Diagnostics | Options > Display | Move |
| PM4 Workbench | Viewer > PM4 | Keep one entry |
| Reset Shell Layout | View menu | Keep |
| File Browser | Left sidebar > File Browser | Keep one home |
| Model Info | Viewer > Model | Keep one home |
| Asset Catalog | Viewer > Assets | Move |
| Taxi Routes | Viewer > Scene Tools | Move |
| Audio Settings | Options > Audio | Move |
| Map Converter | Editor > Converters | Move |
| WMO Converter | Editor > Converters | Move |
| Tools > Panels (12 entries) | — | Delete |
| Synthesized Minimap Export | Archaeology > Cartography | Move |
| GLB export | Owning destination | Move |
| Map GLB tiles | Editor > Terrain Tools | Move |
| Terrain export (10 kinds) | Editor > Terrain Tools | Move |
| Terrain import (3 kinds) | Editor > Terrain Tools | Keep if proven, else delete |
| Bottom-bar toggles | Options | Move |
| Overlays popup | Options > Overlays | Move |
| Layers popup | Options > Layers | Move |
| Transparency popup | Options > Objects | Move |
| PM4 Actions popup | — | Delete |
| Editor Placement & Objects | Editor | Keep; rename sections |
| Editor Terrain Tools | Editor | Keep |
| Editor Data I/O | Editor | Keep; remove PM4 exports |
| Editor Converters | Editor | Keep |
| Archaeology 6 pages | Archaeology | Keep |
| Archaeology legacy 5-tab strip | — | Delete |
| UniqueId Archaeology window | — | Delete |
| Archaeology Quick launchers | — | Delete |
| PM4 Correlation/Reconcile/Alignment | Spec 281 offline tool | Move offline |
| PM4 match curation | Spec 281 offline tool | Move offline |
| 12 duplicate floating windows | — | Delete |

## 9. What gets deleted

- `UtilitiesBottomTab` and the `Utilities` workbench tab.
- `ToolsBottomTab`, `ModelBottomTab`, `WorldBottomTab`, `TerrainBottomTab`, `Pm4BottomTab`,
  `ArcheologyBottomTab`.
- `Canonicalize` and the `OpenWorkbenchTab` overloads that translate between generations.
- `Tab System (069)` and `Dockable Shell Panels`.
- Tools > Panels.
- The legacy Archaeology 5-tab strip, the floating UniqueId window, and the Quick launchers.
- The 12 duplicate floating windows.
- The PM4 Actions popup.
- The second PM4 export home in Data I/O.
- The PM4 matching/alignment/reconciliation tabs and `Pm4ObjectMatchStore` (Spec 281).

## 10. Implementation order

1. **Options sub-menu** (P2): move the bottom-bar toggles, overlays, layers, transparency, render
   quality, lighting, and audio. This removes the largest duplication first.
2. **Left sidebar** (P4): add Diagnostics, Scene Tools, and Assets; delete the Utilities pages.
3. **Right sidebar** (P4): destination-scoped tabs; delete the compatibility enums.
4. **Top menu** (P3): reduce to File / View / Help; delete Tools > Panels.
5. **Archaeology dedupe** (P5).
6. **Editor rename** (P6).
7. **Floating-window dedupe** (P7).
8. **Styling and naming** (P8).

## 11. Open questions for the operator

- Should Diagnostics be a left-sidebar section (proposed) or a fifth bottom-bar button?
- Should Scene Tools (Taxi, Capture, Camera Path) be a left-sidebar section (proposed) or a
  Viewer right-sidebar tab?
- Should the Asset Catalog be a left-sidebar section (proposed) or a Viewer right-sidebar tab?
- Should Options be a popup anchored to the chat (proposed) or a full destination page?
