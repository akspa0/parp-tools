# UI Audit — 2026-10-20

Inventory of the current viewer UI, the functionality behind each surface, and the disposition
recommended by Spec 280. Status values: `works`, `dead`, `duplicate`, `misplaced`, `unknown`.

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
