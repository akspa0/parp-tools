# Visual Analysis — Every UI Part, 2026-10-21

**Purpose**: Ground the v0.7.0 plan in what the UI actually looks like and does today, surface by
surface. This is the evidence the plan is built on, not a wish list.

**Inputs**: the operator's 4K screenshot of the running viewer (2026-10-21), the source tree at
`wow-viewer/src/viewer/WoWViewer/`, and the Spec 280 audit.

**Verdict key**: `keep` · `reshape` · `move` · `delete` · `build`

---

## 0. What the screenshot shows

The 4K capture is the single most useful artifact. Reading it top to bottom:

| Observation | Consequence |
| --- | --- |
| Text lines overlap vertically across the left sidebar and the minimap panel. | `FontGlobalScale` scaled glyphs but not `ItemSpacing`/`FramePadding`, so line height stayed at 1080p while the font doubled. Fixed in code on 2026-10-21; the plan must keep style metrics scaled with the font. |
| The left sidebar is a single column of ~20 stacked section headers and paragraphs. | It is a document, not a tool panel. Nothing is reachable without scrolling. |
| The right sidebar is dominated by the minimap panel and a tab strip, with a "Context" block of prose. | The minimap is not a minimap; it is a panel with controls and text. |
| The chat strip spans the full width and is mostly empty. | The chat is a bar, not a frame, and its right side is dead space. |
| The bottom bar is four text buttons. | It is a menu, not a preset switcher. |
| The status bar is a long path string. | It is a log line, not a status. |
| The scene is a thin band between two walls of text. | The early-WoW reference is the opposite: scene first, chrome second. |

The screenshot is the "before" state for every acceptance criterion in this spec.

---

## 1. Top bar

**What it is**: `##UiTopBar` with a menu bar: File, View, Help.

**What is wrong**:
- It duplicates actions that also exist in the sidebar and the bottom bar (Open File, Settings,
  UI Reference).
- It is the only place some actions live (About, Keyboard Shortcuts), which makes it mandatory
  chrome even in Explore.

**Preset ownership**: shared, but Explore should not need it.

**Verdict**: `reshape` — keep File/View/Help, but make the top bar hideable and make every entry
also reachable from the preset that owns it.

---

## 2. Left sidebar

**What it is**: `NavigatorPanelService.DrawContent()` — Workspace (with Camera, Fog, Sidebar width),
then a destination switch (Scene Tools/Assets, Editor Workspace/Project Output, Archaeology
Workspace/Layer Projects), then Diagnostics, World Overview, File Browser.

**What is wrong**:
- **It is the same sidebar in every preset.** The destination switch changes two sections; the
  other four are identical. That is why the presets do not feel like presets.
- **It leads with prose.** "Unified workspace", the workspace target summary, the save status
  summary, "Analysis and reconstruction pages are in the right sidebar", "Map layers and their
  project actions have one home: Cartography" — all sentences where a control belongs.
- **It mixes inspection and editing.** Scene Tools and Assets sit next to Project Output and
  placement tools.
- **It has no visual hierarchy.** Every section is the same weight; the eye has nowhere to land.
- **It scrolls forever.** The map list and file browser are nested scroll regions at the bottom.

**Preset ownership**: split. Workspace/File Browser are shared; Scene Tools/Assets are Viewer;
Editor Workspace/Project Output are Editor; Archaeology Workspace/Layer Projects are Archeology.

**Verdict**: `reshape` + `move` — one sidebar per preset, controls first, prose in tooltips, no
nested scroll regions.

**Operator correction 2026-10-21**: replace the permanent left text column with a narrow anchored
icon bar. Each icon opens a small closeable window for the existing Workspace/asset/tool controls.

---

## 3. Right sidebar

**What it is**: the minimap region (docked at the top), then `##UiRightSidebar` with a tab strip
(Inspector, World, Model, PM4) and a "Context" block.

**What is wrong**:
- **The minimap is a panel, not a map.** It had a control row (Map/Collapse, Dock/Undock, Hide), a
  zone label and a tile/coordinate readout. Removed in code on 2026-10-21; the plan must keep it
  removed.
- **The tab strip is a second navigation system.** It duplicates the bottom bar's destination
  concept inside one sidebar.
- **"Context" is prose.** It describes what the Inspector would show instead of showing it.
- **The sidebar is always present**, even in Explore, where it should be minimal or absent.

**Preset ownership**: Viewer (Inspector/World/Model), Editor (PM4 placement context), Archeology
(PM4 analysis), Explore (inspection only).

**Verdict**: `reshape` — one content surface per preset, no tab strip, no prose block.

**Operator correction 2026-10-21**: replace the permanent right data panel with a narrow icon bar;
Inspector/World/Model/PM4 actions open separate small closeable windows.

---

## 4. Minimap

**What it is**: `UiRegion.Minimap`, currently calculated relative to/docked at the top of the right sidebar, drawn by
`MinimapAndStatusService.DrawMinimapPanelContent()`.

**What is wrong**:
- It carried controls and text (fixed 2026-10-21).
- It is docked inside the right sidebar, so it disappears when the sidebar is hidden. The real
  client's minimap is independent chrome.
- In the operator's current screenshot, it competes with a huge right inspector and reads as a tiny
  rectangular map pane, not a prominent game minimap.
- It has no frame or border treatment that reads as a map.

**Preset ownership**: shared, independent of sidebars.

**Verdict**: `reshape` — a standalone, framed map with scroll-zoom, drag-pan and click-teleport, no
controls, no text.

---

## 5. Chat panel

**What it is**: `UiRegion.Chat`, a docked panel spanning the scene between the sidebars, with the
chat log on the left and a tools column on the right (added 2026-10-21).

**What is wrong**:
- The chat body still opens with a control row (General, Clear, Auto-Scroll, Hide) and a population
  control row before the log.
- The tools column is new and unproven; it needs a real inventory, not an ad-hoc list.
- Because chat width is computed as the viewport width minus both sidebar widths, the supplied
  current screenshot shows chat squeezed into a tiny center-bottom strip when persisted sidebars are
  wide. The chat must anchor independently at lower-left.
- The barker section is conditional on local barks, which is correct, but the condition is not
  visible to the operator.

**Preset ownership**: shared; Explore keeps it.

**Verdict**: `reshape` — chat log first, tools as a compact icon column, barker section only when
it has content.

---

## 6. Bottom bar

**What it is**: `UiRegion.BottomBar` — Viewer, Editor, Archaeology, Options, plus the FPS readout.

**What is wrong**:
- It is a text menu. The operator asked for "actual presets in those buttons of tools".
- It has three destinations, not the four the operator wants (Explore is missing).
- Options is a peer of the presets, which makes it look like a destination.

**Preset ownership**: the preset switcher itself.

**Verdict**: `reshape` — four preset buttons with icons and a one-line description, Options as a
separate affordance, FPS moved to the status bar.

**Operator correction 2026-10-21**: the current full-width destination row is not the requested
action bar. It must provide a compact, readable action strip with working tool shortcuts while
preserving preset navigation without taking over the scene. The Spellbook prototype was explicitly
rejected and must not remain in the UI.

---

## 7. Status bar

**What it is**: `UiRegion.StatusBar`, delegating to `MinimapAndStatusService.DrawStatusContent()`.

**What is wrong**:
- It prints a long path and a runtime summary on one line, so the useful part is clipped.
- It is the only place some state is reported (minimap loading, spawn status).

**Preset ownership**: shared.

**Verdict**: `reshape` — short status, with the long path in a tooltip.

---

## 8. Options panel

**What it is**: `UiRegion.Options`, anchored to the right sidebar, with Display / Overlays / Layers
/ Grids / Objects / Audio / Simulation groups.

**What is wrong**:
- It is a dump of every toggle in the program, regardless of preset.
- It contains editing toggles (placement, terrain) next to display toggles.
- It is the only home for some controls (minimap show/collapse/dock), which is correct, but the
  grouping does not say which preset owns what.

**Preset ownership**: split by group.

**Verdict**: `reshape` — group by preset, then by concern; hide groups the current preset does not
own.

---

## 9. Popups

**What it is**: `UiPopupRegistry` — Settings, About, Keyboard Shortcuts, Forget Base, UI Reference,
Map Spawns.

**What is wrong**:
- The popup host window covered the whole display and captured all input until `NoInputs` was added
  on 2026-10-21. That is why the camera died. The plan must keep the host input-transparent.
- Popups are the only way to reach Settings and the UI Reference, so they are mandatory chrome.

**Preset ownership**: shared.

**Verdict**: `keep` with the input fix; `reshape` the entry points so each preset can open the ones
it needs.

---

## 10. Editor pages

**What it is**: `WorkbenchNavigator.GetPageLabels(Editor)` — Placement & Objects, Terrain Tools,
Data I/O, Converters.

**What is wrong**:
- The pages are correct in name but the sidebar still shows Viewer sections (World Overview, File
  Browser) above them.
- Some editing actions still live in the Viewer sidebar (Scene Tools) rather than here.

**Preset ownership**: Editor.

**Verdict**: `move` — every editing action lands here; nothing editing remains in Viewer.

---

## 11. Archaeology pages

**What it is**: Weak Signal & Stratigraphy, UniqueId Timeline, Layers & Provenance, Playback &
Capture, Cartography.

**What is wrong**:
- Cartography contains export actions that write client data, which is editing, not archaeology.
- The sidebar repeats the Viewer sections.

**Preset ownership**: Archeology.

**Verdict**: `move` — exports move to Editor; Archeology keeps analysis and reconstruction.

---

## 12. Explore

**What it is**: does not exist.

**What is wrong**: the operator's museum preset is missing entirely. The taxi/route presentation
exists but is buried, and the closest thing to Explore is the Viewer with everything switched on.

**Preset ownership**: Explore.

**Verdict**: `build` — a museum preset whose default display is the taxi/route presentation, whose
only tools are inspection, and whose sidebar is minimal or absent.

---

## 13. Cross-cutting defects

| Defect | Where | Fix direction |
| --- | --- | --- |
| Text salad | every sidebar section, Options, Context | controls first; prose in tooltips and the UI reference |
| Duplicate homes | Open File, Settings, UI Reference, PM4, minimap controls | one authoritative home per control |
| Preset bleed | editing tools in Viewer/Archeology | preset ownership table enforced by the manager |
| Nested scroll regions | map list, file browser | one scroll region per sidebar |
| Unscaled style metrics | whole shell at 4K | scale `ItemSpacing`/`FramePadding`/`ScrollbarSize` with the font |
| Input capture | popup host | `NoInputs` on the host, always |
| No visibility model | whole shell | per-part show/hide with persistence |

---

## 14. What the plan must prove

1. A preset walk finds zero editing controls in Viewer or Archeology.
2. Explore opens on the taxi/route presentation.
3. Every shell part can be hidden and restored, and the choice persists.
4. No sidebar section opens with a paragraph.
5. Frame time with the UI shown is within 1 ms of the UI hidden at the same camera and scene.
6. The generated UI reference covers every visible control.
