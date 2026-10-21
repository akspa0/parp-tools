# Implementation Plan: UI Audit and Pre-Alpha-Style Refactor

**Spec**: [spec.md](spec.md) · **Audit**: [audit.md](audit.md) · **Branch**: `280-ui-rework`

**Status**: Plan authored 2026-10-21. No implementation started.

## 1. Context

The audit in [audit.md](audit.md) establishes the structural problem: the viewer has **five
navigation systems** for the same content (top menu, bottom bar, left navigator, right inspector,
workbench tab strips) plus **12 floating windows** that duplicate destination pages. The operator's
functional report adds the second half of the problem:

- Only the File menu mostly works.
- The Export menu is unreliable.
- Import does not work at all.
- "Open File..." asks the operator to type a filesystem path.

So this is not a cosmetic re-skin. It is a **navigation collapse plus a file-operation repair**.
The plan therefore has two tracks that must both land:

1. **Structure**: one destination model, one authoritative home per surface, no duplicate entry
   points.
2. **Function**: every remaining control does something real, and every file operation uses the
   shared picker and reports its result.

## 2. Architecture decisions

### AD-1 — One navigation model: bottom-bar destinations

The bottom bar becomes the only destination switcher: `Viewer | Editor | Archaeology | Options |
FPS/stats`. The top menu keeps only global actions (`File`, `View`, `Help`). The workbench tab
strips (`WorkbenchTab`, `WorkbenchNavigator`) are reduced to the destination's own page strip; the
legacy `Canonicalize` redirects are deleted once no caller depends on them.

**Rationale**: the operator asked for pre-alpha-style bottom-bar menu buttons. The current
`WorkbenchTab` enum already encodes the destinations but is wrapped in a second layer of
compatibility enums (`ModelBottomTab`, `WorldBottomTab`, `ToolsBottomTab`, `UtilitiesBottomTab`,
`TerrainBottomTab`, `Pm4BottomTab`, `ArcheologyBottomTab`). Those compatibility enums are the
nesting. They are removed, not aliased.

### AD-2 — Destination-scoped sidebars

The left navigator and right inspector render only the current destination's surfaces. A
destination declares its surfaces; the shell asks the destination for them. Cross-destination
shortcuts are explicit buttons, not duplicated panels.

**Rationale**: the audit found editor-only controls in Viewer and archaeology controls in Editor.
Scoping is the only way to make "Editor means editor" true.

### AD-3 — One authoritative home per data surface

Every data surface has exactly one home (Spec 223 FR-3). The audit's duplicate table is the
deletion list. A duplicate is replaced by a navigation shortcut, never by a second copy.

**Rationale**: the PM4 consolidation already proved the pattern works. This plan applies it to
archaeology, editor, and the floating windows.

### AD-4 — One file picker

All open/import/export path selection goes through `ImGuiPathPicker` (the shared picker already
used by most open actions). The bespoke text-path popup behind "Open File..." is deleted.

**Rationale**: the operator called the typed path "absurd". The shared picker already exists; the
defect is that one action bypasses it.

### AD-5 — Export/import contract

Every export action must:

1. Resolve an output path inside a project-managed output root (AGENTS §9.3).
2. Write the file.
3. Report the written path in the status line.
4. Report failure explicitly.

Every import action must read and apply a real file, or be removed. A button that opens a picker
and then does nothing is a defect, not a placeholder.

**Rationale**: the operator reports silent no-ops. The contract makes "does it work" testable.

### AD-6 — `SharedUiWidgets` is the only styling base

Every section header, body text, button row, and spacing value comes from `SharedUiWidgets`.
Local `ImGui.PushStyleColor`, bespoke `CollapsingHeader` headers, and per-panel font sizes are
removed. The theme supplies color; the widget set supplies structure.

**Rationale**: the audit found three different header conventions and hard-coded colors that
ignore the theme.

### AD-7 — No new god-class members

New behavior lives in owned service classes that receive state through constructors or parameters
(AGENTS §10). The refactor deletes and delegates; it does not add members to `ViewerApp` or
`WorldScene`.

### AD-8 — Delete, do not hide

A duplicate surface is deleted. It is not hidden behind a flag, a collapsed header, or a
"legacy" toggle. The audit's `duplicate` rows are the deletion list.

### AD-9 — PM4 matching moves offline

PM4 object matching, alignment, and reconciliation are batch jobs that patch ADTs, not
interactive viewer features. They move to the offline WoWMuseum patcher in
[Spec 281](../281-museum-adt-patcher/spec.md). The viewer keeps PM4 inspection only: Overlay,
Selection, Info, Outliner, and Object Groups. The `Pm4ObjectMatchStore` and the manual match
curation are deleted from the viewer.

**Rationale**: the operator's direction. The matching data (Museum ADTs plus original PM4s) is
expensive to assemble and belongs in a repeatable CLI, not a GPU-bound viewer session.

## 3. Target structure

```text
Top menu:  File | View | Help
Bottom bar: Viewer | Editor | Archaeology | Options | FPS/stats
Chat bar:   chat log | Auto-Scroll | Auto-Stream | Spawn BBs | Options ▾
```

### 3.1 Destinations

The full surface→home map is [information-architecture.md](information-architecture.md). It assigns
every visible surface exactly one home, including all 10 Utilities pages.

| Destination | Left navigator | Right inspector | Page strip |
|---|---|---|---|
| Viewer | Workspace, World Overview, File Browser, Scene Tools, Assets, Diagnostics, Chat | Inspector, World, Model, PM4 | none (Inspector tabs are the pages) |
| Editor | Editor workspace, Project Output, File Browser, Diagnostics, Chat | Editor tools | Placement & Objects, Terrain Tools, Data I/O, Converters |
| Archaeology | Archaeology workspace, Layer Projects, File Browser, Diagnostics, Chat | Archaeology tools | Weak Signal & Stratigraphy, UniqueId Timeline, Layers & Provenance, Playback & Capture, Cartography |
| Options | — (sub-menu anchored to the chat) | — | Display, Overlays, Layers, Grids, Objects, Audio, Simulation |

PM4 Analysis is not an Archaeology page. It is a Viewer surface reached by one explicit shortcut.

### 3.1.1 Utilities coverage

The `Utilities` tab has 10 pages. None survives as a Utilities page; each becomes a named section
in its logical home:

| Utilities page | New home |
|---|---|
| Minimap | Left sidebar > World Overview |
| Log | Left sidebar > Diagnostics |
| Perf | Left sidebar > Diagnostics |
| Render Quality | Options > Display |
| Taxi | Viewer > Scene Tools |
| Capture | Viewer > Scene Tools |
| Asset Catalog | Viewer > Assets |
| Runtime Stats | Left sidebar > Diagnostics |
| Lighting | Options > Display |
| Audio | Options > Audio |

`UtilitiesBottomTab` and the `Utilities` workbench tab are deleted.

### 3.1.2 Visual layout

The screen layout is specified in [layout.md](layout.md): old-WoW structure (minimap top-right,
chat bottom-left, world in the centre) with a 2026 surface treatment (flat surfaces, one brass
accent, one type scale, 8px grid, responsive collapse). The minimap moves from the left sidebar to
a floating top-right instrument; the left sidebar keeps the map list and map export.

### 3.2 File operations

| Action | Current | Target |
|---|---|---|
| Open File... | Bespoke text popup | `ImGuiPathPicker` |
| Open Game Folder (MPQ) | Shared picker | Keep |
| Open CASC Install | Shared picker | Keep |
| Open DAT Terrain Folder | Shared picker | Keep |
| Open Loose Map Folder | Shared picker | Keep |
| Open Loose PM4 / PD4 | Shared picker | Keep |
| Export GLB | Menu tree, unreliable | One action in the owning destination, writes + reports |
| Export Terrain (alpha/height/MCCV) | 5-level menu tree | One action per artifact in Terrain Tools, writes + reports |
| Import Terrain (alpha/height/MCCV) | 5-level menu tree, non-functional | One action per artifact in Terrain Tools, or removed |
| Export PM4 | Two homes | One home in the PM4 workbench, writes + reports |
| Export Synthesized Minimap | Tools menu | Cartography page, writes + reports |

## 4. File-level change map

### 4.1 Navigation and shell

| File | Change |
|---|---|
| `Workbench/WorkbenchNavigator.cs` | Delete compatibility enums; keep `WorkbenchTab` and the destination page labels |
| `Workbench/Services/MainMenu/MainMenuBarService.cs` | Reduce to File / View / Help; delete Tools > Panels and the export/import trees |
| `Workbench/Services/ViewerChrome/ViewerChromeService.cs` | Bottom bar becomes destination buttons + Options + FPS/stats; delete render toggles from the bar |
| `Workbench/Services/Dialogue/FakeChatWindowService.cs` | Keep the chat bar; host the Options sub-menu anchor |
| `Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs` | Destination-scoped inspector tabs; delete the legacy tab redirects |
| `Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs` | Destination-scoped left sidebar; rename WC3 to WC3 Beta |

### 4.2 Editor

| File | Change |
|---|---|
| `Workbench/Pages/EditorPlacementObjectsPage.cs` | Rename sections to their real owners; delete the one-button PM4 section |
| `Workbench/Pages/EditorDataIoPage.cs` | Delete the second PM4 export home; route exports through the owning destination |
| `Workbench/Pages/EditorTerrainToolsPage.cs` | Host the terrain export/import actions |
| `Workbench/Pages/EditorConvertersPage.cs` | Keep; route through `SharedUiWidgets` |
| `Workbench/Pages/EditorWorkbenchPages.cs` | Delete `MigrateLegacyEditorPageIndex` once persisted layouts are reset |

### 4.3 Archaeology

| File | Change |
|---|---|
| `Workbench/Services/Archaeology/ArchaeologyPanelService.Sidebars.cs` | Delete `DrawArcheologySubTabContent`, `DrawUniqueIdArchaeologyWindow`, `DrawArchaeologyQuickSection`; keep the 6-page dispatch |
| `Workbench/Services/Archaeology/ArchaeologyPanelService.PhaseLayers.cs` | Split the Cartography "Layers" page into named sections; keep the layer stack, DBC discovery, add controls, donor picker |
| `Workbench/Services/Archaeology/ArchaeologyPanelService.Editor.cs` | Keep reconciliation; route through `SharedUiWidgets` |
| `Workbench/Services/Archaeology/ArchaeologyPanelService.Host.cs` | Remove bridges for deleted members |

### 4.4 File operations

| File | Change |
|---|---|
| `Workbench/Services/MainMenu/MainMenuBarService.cs` | "Open File..." uses `ImGuiPathPicker`; delete the text popup |
| `Workbench/Services/EditorPanels/EditorPanelsService.cs` | Import actions read and apply, or are removed |
| `Workbench/Services/Pm4Workbench/Pm4WorkbenchService.Alignment.cs` | One export home; every export writes and reports; remove Correlation/Reconcile/Alignment tabs (Spec 281) |
| `Workbench/Services/Pm4Workbench/Pm4WorkbenchService.cs` | Remove the match-curation section from Selection (Spec 281) |
| `Pm4ObjectMatchStore.cs` | Delete (Spec 281) |
| `Workbench/Services/TerrainControls/TerrainControlsPanelService.Sidebars.cs` | Terrain export/import actions with the write-and-report contract |

### 4.5 Styling

| File | Change |
|---|---|
| `UI/SharedUiWidgets.cs` | Add any missing primitives (section header, body text, button row, empty state) |
| All sidebar/page files | Replace local colors, fonts, and headers with the shared primitives |

## 5. Phase roadmap

Each phase is independently buildable and verifiable. A phase is not complete until its
`tasks.md` receipt exists (AGENTS §9.2).

### P0 — Baseline and guardrails

- Record the current build/test baseline.
- Add a UI-string inventory test or script that lists every visible label, so deletions are
  measurable.
- Confirm the branch builds before any change.

### P1 — File operations repair (highest user value)

- Replace the "Open File..." text popup with the shared picker.
- Audit every export action: run it, confirm a file is written, add the status report.
- Audit every import action: run it, confirm it applies, or delete it.
- This phase is first because the operator cannot use the tool without it, and because it is
  independent of the navigation rework.

### P2 — Bottom bar and Options sub-menu

- Add the destination buttons and the Options button.
- Move the render/overlay/layer/transparency toggles into the Options sub-menu.
- Keep the FPS/stats readout.
- Delete the three bottom-bar popups.

### P3 — Top menu reduction

- Reduce to File / View / Help.
- Delete Tools > Panels and the export/import trees.
- Move each surviving action to its owning destination.

### P4 — Destination separation

- Make the left and right sidebars destination-aware.
- Delete cross-destination controls.
- Delete the compatibility enums and the legacy tab redirects.

### P5 — Archaeology dedupe

- Delete the legacy 5-tab strip, the floating UniqueId window, and the Quick launchers.
- Split the Cartography "Layers" page into named sections.
- Keep the 6-page destination strip as the only navigation.

### P6 — Editor section rename

- Rename "Tasks & Workspace", "3D Object Library", and "Imports & Exports" to their real owners.
- Delete the one-button "PM4 Placement Tools" section.
- Remove the second PM4 export home.

### P7 — Floating-window dedupe

- Delete the 12 Tools > Panels windows that duplicate destination pages.
- Keep only Log and Perf as global windows.

### P8 — Styling and naming

- Route every section through `SharedUiWidgets`.
- Remove local colors and bespoke headers.
- Rename WC3 data support to "WC3 Beta" in the UI and documentation.
- Final dead-control sweep.

### P9 — Visual layout

- Apply the design tokens from [layout.md](layout.md) §8 (colour, type, spacing).
- Move the minimap to the floating top-right instrument; add collapse, zone label, and
  coordinates; delete the Utilities Minimap page.
- Convert the chat to a resizable bottom-left dock; keep the population controls.
- Add sidebar collapse rails and persistent widths.
- Add the responsive breakpoints and default collapse.
- Optional: `Ctrl+K` command palette over destinations and actions.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Deleting a duplicate removes the only working path | The audit names the survivor before deletion; P1 proves the survivor works |
| Persisted layouts point at deleted pages | Reset or migrate the layout in P4; `MigrateLegacyEditorPageIndex` is deleted only after the reset |
| Export/import repair expands into format work | P1 is scoped to wiring and reporting, not to new format support; a broken format is reported, not fixed here |
| Destination scoping hides a control the operator needs | Cross-destination shortcuts are explicit buttons; the operator can name any missing shortcut |
| Styling pass changes behavior | Styling is last and mechanical; no logic changes in P8 |
| God-class growth | AD-7: new behavior in owned services; the refactor deletes and delegates |

## 7. Verification strategy

- **Build**: `dotnet build wow-viewer/WowViewer.slnx -c Debug` after every phase.
- **Tests**: `dotnet test wow-viewer/WowViewer.slnx -c Debug` after every phase.
- **Structural**: the audit's duplicate table is the checklist; each deleted row is a receipt line.
- **Functional**: P1 requires a real file written and a real file imported for each action; a
  build or unit test alone does not satisfy it (AGENTS §9.2).
- **Visual**: the operator confirms the bottom bar, Options sub-menu, and destination sidebars on
  a real client; the agent does not claim visual proof from compilation.
- **Naming**: a string search for `WC3` without `Beta` returns zero UI hits.

## 8. Out of scope

- New format support (PM4 decode, terrain formats, WC3 beyond Beta).
- Renderer performance work.
- The recorded but unfixed bugs: skybox support, population `humanmale` fallback, 3.3.5 creature
  textures. These are tracked in `memory-bank/activeContext.md` and are not part of this spec.
- A pixel-perfect pre-alpha art reproduction. The reference is layout and interaction only.
