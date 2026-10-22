# Implementation Plan: v0.7.0 UI Presets and the Explore Museum

**Spec**: [`spec.md`](spec.md) · **Visual analysis**: [`visual-analysis.md`](visual-analysis.md) ·
**Created**: 2026-10-21

## 1. Context

Spec 280 gave the viewer one UI manager and three destinations. It did not give it **presets**. The
same sidebar, the same Options dump and the same chat strip appear in every destination, editing
tools leak into Viewer and Archaeology, and the operator's museum preset does not exist. The
operator's verdict is that the result is a database browser with a viewport attached, not the early
WoW UI they asked for.

This plan turns the three destinations into **four presets with hard ownership boundaries**, adds
**Explore** as a museum, adds a **shell visibility model**, and removes the text salad. It is a
plan; no implementation is claimed here.

## 2. Operator intent check

| Operator statement | Plan response |
| --- | --- |
| "update the version of the tooling to v0.7.0" | P0 bumps `eng/Version.props` to `0.7.0`. |
| "taxi ui stuff should become the normal thing that is displayed ... in a new tab called 'Explore'" | Explore's default display is the taxi/route presentation (AD-3). |
| "disables most of the tooling except for inspection tools" | Explore owns inspection only; the preset ownership table (AD-1) is enforced by the manager. |
| "limited tools in the sidebars (if anything), the chat panel and tools next to it" | Explore's sidebar is hidden by default; chat + tools remain (AD-4). |
| "enable/disable parts of the ui" | Shell visibility model with persistence (AD-5). |
| "acts more like the screenshot ... of the early WoW ui" | Scene-first layout: minimap and chat are the only persistent chrome (AD-6). |
| "simplified and deduplicated ... wordy, text salad" | Prose moves to tooltips and the UI reference; one authoritative home per control (AD-7). |
| "doesn't slow the program down" | No per-frame work for hidden surfaces; no GPU rebuild on preset switch (AD-8). |
| "every iota of everything we requested, organized by the presets" | The surface inventory (§4) is exhaustive; nothing is deleted without a named new home. |
| "editing ... ends up in the Viewer or Archeology areas, which is wrong" | Editing actions move to Editor (AD-2). |
| "actual presets in those buttons of tools" | Bottom bar becomes four preset buttons with icons and a one-line description (AD-9). |
| "visual analysis of each part of the ui" | [`visual-analysis.md`](visual-analysis.md) covers all 12 surfaces plus cross-cutting defects. |

### Operator amendment — 2026-10-21

The operator clarified: "Condense our tooling into literal 'spells' in a 'spellbook', make the
viewer less aids and more game-ish, make use of the screen real-estate better." The UI shell therefore
gets a Spellbook entry point in place of the generic chat Tools list, and the minimap/chat use
compact footprints. The current text-button/tab implementation is only an initial code slice; it is
not visual acceptance of the requested game-like presentation. The operator-supplied reference
images and runtime walk remain the visual authority. Converter and 0.5.3 DBC requests remain tracked
outside this UI correction and are not claimed by its receipt.

### Operator amendment — screenshot-directed layout and transient preview — 2026-10-21

The operator reported that the running UI remained unusable and supplied a fresh screenshot showing
the oversized sidebars and undersized chat. The supplied early-WoW anatomy image is the presentation
target. Implementation must therefore satisfy these geometry/lifecycle invariants:

1. Default Viewer opens with compact left/right icon action bars; full content windows are opt-in and
   independently closeable.
2. The scene is full-width behind overlays. Sidebar widths cannot subtract from chat or minimap size.
3. The minimap is independently anchored at the upper-right, never hosted inside the right sidebar.
4. Chat is independently anchored at the lower-left with a bounded width and useful log area.
5. The bottom region is a compact action strip with working shortcuts to real viewer tools, not a
   row of tiny destination labels alone.
6. The WDL map-spawn preview draws only while its show flag is true; successful map load and explicit
   close/cancel must leave it closed.

The screenshot's character resources/HUD are not viewer functionality and will not be faked. Exact
visual acceptance remains an operator run against the supplied reference.

### Operator clarification — anchored action bars — 2026-10-21

The operator clarified that the side tool areas are compact anchored icon bars. Clicking an icon
opens a small, independently closeable window backed by that tool's existing controls, analogous to
the game's character/item windows. The bars remain visible; full-height text sidebars do not. This
supersedes the preceding idea of entirely hiding the edge controls.

### Operator correction — remove rejected Spellbook — 2026-10-21

The operator explicitly rejected the Spellbook prototype as unhelpful and directed that it be
omitted from the UI. Do not add a Spellbook window, button, or launcher. Tools are exposed directly
by the anchored icon bars and their closeable tool windows. The earlier Spellbook amendment is
retained only as history and is superseded by this correction.

### Integration disposition — 2026-10-22

The existing Navigator provider is split into separately hosted Workspace, Maps, File Browser,
Diagnostics and Project Output content bodies. Each remains a single authoritative implementation;
the icon-bar service does not replace map discovery, readers, camera/fog controls or runtime actions.
Maps and File Browser have their own icons; Project Output stays in the Editor window. Taxi binds to
taxi-only content, Capture to capture-only content, and Viewer Pages opens the already-owned Viewer
window rather than hosting another live instance of the same inspector. Options is a closeable tool
window too. Chat no longer gives a third of its width to an empty tools column, and population
configuration stays in its existing tool window. All icon labels use full-name tooltips, not two-letter
abbreviations. No real-client or visual acceptance follows from the build/tests.

## 3. Architecture decisions

### AD-1 — A preset is a declared surface set, not a filter

`ViewerUiManager` gains a `PresetDefinition` per preset: the sidebar sections it owns, the bottom-bar
pages it owns, the chat tool groups it owns, and the Options groups it owns. A surface not in the
current preset's definition is not drawn. This is a positive list, not a `switch` with a default
that leaks.

**Why**: the current `switch` in `NavigatorPanelService.DrawContent()` changes two sections and
leaves four shared, which is why presets do not feel scoped.

### AD-2 — Editing lives only in Editor

Every action that writes a file, mutates a placement, or edits terrain is owned by Editor. Viewer
and Archeology may *link* to it (`Open in Editor`), but may not host it.

**Why**: the operator named this as a correctness problem, not a preference.

### AD-3 — Explore's default display is the taxi/route presentation

Explore opens on the taxi/route presentation for the loaded map. If the map has no taxi data, it
opens on the museum asset browser. It never falls back to the Editor.

**Why**: the operator asked for the taxi UI to be "the normal thing that is displayed" in Explore.

### AD-4 — Explore's sidebar is hidden by default

Explore starts with both sidebars hidden. Inspection opens as a compact overlay or a single
inspection section, not as the full sidebar. Chat and its tools remain.

**Why**: "limited tools in the sidebars (if anything)" and "interactive museum".

### AD-5 — Shell visibility is a first-class, persisted model

`ShellVisibility` holds one flag per part: left sidebar, right sidebar, minimap, chat, status bar,
bottom bar, tooltips. `UiLayout` consumes it; `ViewerUiManager.Persistence` stores it. Every hidden
part is restorable from a visible control (Options > Shell, plus the keyboard shortcuts).

**Why**: the operator asked to enable/disable parts of the UI, and a hidden part with no way back is
a trap.

### AD-6 — Scene first

The default layout gives the scene every pixel that chrome does not need. The minimap and chat are
the only persistent chrome; sidebars start collapsed in Explore and are opt-in elsewhere.

**Why**: the early WoW reference is scene-first.

### AD-7 — One authoritative home per control

Every control has exactly one owning surface. A second surface may link to it, never re-implement
it. The generated UI reference records the owner.

**Why**: duplicate homes are the root of the "everything jammed together" complaint.

### AD-8 — No per-frame cost for hidden surfaces

Hidden surfaces are not drawn at all (not drawn-then-clipped). Preset switching swaps content
providers; it does not dispose or rebuild GPU resources. The UI reference is loaded once.

**Why**: "make it so it doesn't slow the program down".

### AD-9 — The bottom bar is a preset switcher

Four buttons: Viewer, Editor, Archeology, Explore. Each has an icon and a one-line description in
its tooltip. Options is a separate affordance, not a fifth preset. FPS moves to the status bar.

**Why**: "I expect actual presets in those buttons of tools".

### AD-10 — Prose is not a layout element

No surface opens with a paragraph. Explanatory text lives in tooltips and the generated UI
reference. A section's first row is a control or a labelled value.

**Why**: "wordy, text salad components".

### AD-11 — Style metrics scale with the font

`FontGlobalScale` alone produced overlapping lines at 4K. `ItemSpacing`, `ItemInnerSpacing`,
`FramePadding`, `CellPadding`, `ScrollbarSize`, `GrabMinSize`, `IndentSpacing` and the surface
`WindowPadding` scale with the same factor as the font.

**Why**: the screenshot's overlapping text is the most visible defect.

### AD-12 — The popup host is input-transparent

The full-display popup host window carries `NoInputs | NoFocusOnAppearing | NoNav`. Without it,
`WantCaptureMouse`/`WantCaptureKeyboard` are true everywhere and the camera dies.

**Why**: this was a real regression on 2026-10-21.

## 4. Surface inventory and preset ownership

This table is the contract. A surface appears in exactly one preset column; "shared" means it is
owned by the shell and available in every preset.

| Surface | Viewer | Editor | Archeology | Explore | Notes |
| --- | --- | --- | --- | --- | --- |
| Top bar (File/View/Help) | shared | shared | shared | hidden by default | hideable |
| Left sidebar — Workspace (open file, camera, fog) | yes | yes | yes | hidden | controls first |
| Left sidebar — Scene Tools | yes | no | no | no | inspection |
| Left sidebar — Assets | yes | no | no | no | inspection |
| Left sidebar — Editor Workspace | no | yes | no | no | editing |
| Left sidebar — Project Output | no | yes | no | no | editing |
| Left sidebar — Archaeology Workspace | no | no | yes | no | analysis |
| Left sidebar — Layer Projects | no | no | yes | no | analysis |
| Left sidebar — Diagnostics | shared | shared | shared | hidden | log/perf |
| Left sidebar — World Overview | yes | no | no | no | map discovery |
| Left sidebar — File Browser | shared | shared | shared | hidden | asset access |
| Right sidebar — Inspector | yes | no | no | overlay | inspection |
| Right sidebar — World | yes | no | no | no | inspection |
| Right sidebar — Model | yes | no | no | overlay | inspection |
| Right sidebar — PM4 | yes | yes | yes | overlay | inspection + placement context |
| Right sidebar — Context prose | delete | delete | delete | delete | replaced by Inspector |
| Minimap | shared | shared | shared | shared | standalone, no controls |
| Chat + tools | shared | shared | shared | shared | museum keeps chat |
| Bottom bar | shared | shared | shared | shared | preset switcher |
| Status bar | shared | shared | shared | shared | short status |
| Options — Display | shared | shared | shared | shared | |
| Options — Overlays | yes | yes | yes | yes | inspection toggles |
| Options — Layers | yes | yes | yes | yes | inspection toggles |
| Options — Grids | yes | yes | yes | no | editing aid |
| Options — Objects | yes | yes | yes | yes | inspection toggles |
| Options — Audio | shared | shared | shared | shared | |
| Options — Simulation | no | yes | no | no | editing |
| Options — Shell | shared | shared | shared | shared | visibility model |
| Editor pages (Placement & Objects, Terrain Tools, Data I/O, Converters) | no | yes | no | no | editing |
| Archaeology pages (Weak Signal, UniqueId, Layers, Playback, Cartography) | no | no | yes | no | analysis |
| Explore — taxi/route presentation | no | no | no | yes | default display |
| Explore — museum asset browser | no | no | no | yes | fallback |
| Explore — inspection overlay | no | no | no | yes | terrain/WMO/M2/PM4/liquid/spawn |
| Popups (Settings, About, Keyboard, UI Reference, Map Spawns) | shared | shared | shared | shared | input-transparent host |

**Moves required by this table**:

- Cartography export actions → Editor > Data I/O.
- Scene Tools editing actions → Editor > Terrain Tools / Placement & Objects.
- Options > Simulation → Editor only.
- Options > Grids → not in Explore.
- Right sidebar "Context" prose → deleted; Inspector shows the data.

## 5. Explore design

### 5.1 Default display

On entering Explore with a loaded map:

1. If the map has taxi nodes or routes, show the taxi/route presentation full-scene.
2. Otherwise, show the museum asset browser.
3. The chat dock and its tools remain visible.
4. Both sidebars are hidden.

### 5.2 Museum presentation

- The scene is the exhibit. Camera movement is free; no editing gizmos.
- Hovering an asset shows a compact inspection card (name, kind, source, key values).
- Clicking an asset opens the inspection overlay for that asset kind.
- A search field in the chat tools column finds an asset by name or path and flies the camera to it.

### 5.3 What Explore must not do

- No file writes.
- No placement mutation.
- No terrain edits.
- No converter dialogs.
- No Options groups that only exist for editing (Simulation, Grids).

## 6. Shell visibility model

```text
ShellVisibility
  LeftSidebar   bool
  RightSidebar  bool
  Minimap       bool
  Chat          bool
  StatusBar     bool
  BottomBar     bool
  Tooltips      bool
```

Rules:

- `UiLayout` reads the flags and gives freed space to `UiRegion.Scene`.
- `ViewerUiManager.Persistence` stores the flags in the versioned layout document (v4).
- Options > Shell exposes one checkbox per flag.
- Keyboard shortcuts remain: `[` left sidebar, `]` right sidebar, `M` minimap, `Tab` all chrome.
- A preset may set defaults on first entry but must not override an explicit operator choice.

## 7. Performance strategy

| Risk | Mitigation |
| --- | --- |
| Hidden surfaces still drawn | `DrawRegion` returns before `ImGui.Begin` when the region is not visible. |
| Preset switch rebuilds GPU resources | Preset switch only swaps content providers; renderers are owned by the app, not the preset. |
| UI reference parsed every frame | Loaded once into `UiControlRegistryService`. |
| Per-frame spawn scans for the barker section | `HasLocalBarkers` is computed when the applied spawn set changes, not per frame. |
| Style metric churn | Style metrics are applied only when the scale changes. |
| Text measurement in long lists | Lists use `ImGui` clippers; no `CalcTextSize` per item. |

Budget: UI shown vs UI hidden frame time within 1 ms at the same camera and scene, at 1080p and 4K.

## 8. Phase roadmap

### P0 — Version and guardrails

- Bump `eng/Version.props` to `0.7.0`.
- Add a preset-ownership test that fails when a surface is drawn outside its owning preset.
- Add a shell-visibility test that fails when a hidden part still consumes layout space.

### P1 — Preset model

- Introduce `PresetDefinition` and the four definitions from §4.
- Replace the destination `switch` in `NavigatorPanelService.DrawContent()` with the preset's
  sidebar section list.
- Replace `WorkbenchTab` with `Preset` (Viewer, Editor, Archeology, Explore) and migrate the
  persisted layout document to v4.

### P2 — Bottom bar as preset switcher

- Four preset buttons with icons and one-line tooltips.
- Options as a separate affordance.
- FPS moves to the status bar.

### P3 — Editing containment

- Move Cartography exports to Editor > Data I/O.
- Move Scene Tools editing actions to Editor.
- Restrict Options > Simulation and Options > Grids to Editor.
- Delete the right sidebar "Context" prose block.

### P4 — Explore

- Add the Explore preset.
- Default to the taxi/route presentation; fall back to the museum asset browser.
- Hide both sidebars by default.
- Add the inspection overlay and the asset search in the chat tools column.

### P5 — Shell visibility

- Add `ShellVisibility` and wire it through `UiLayout`.
- Add Options > Shell.
- Persist in layout v4.

### P6 — Text salad removal

- Remove every leading paragraph from every sidebar section and Options group.
- Move the text to tooltips and the generated UI reference.
- Regenerate the reference and pass the build check.

### P7 — Visual pass

- Apply the early-WoW proportions: scene first, minimap and chat as the only persistent chrome.
- Verify style metrics scale with the font at 1080p, 1440p and 4K.
- Verify the popup host is input-transparent.

### P8 — Verification

- Preset walk: zero editing controls in Viewer or Archeology.
- Explore opens on the taxi/route presentation.
- Shell visibility round-trip.
- Frame-time comparison.
- UI reference coverage.

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Preset ownership table drifts from code | The P0 test reads the table and fails on a mismatch. |
| Explore's taxi presentation is not ready to be a default | P4 starts with the existing presentation; if it is not usable, the museum asset browser is the documented fallback. |
| Moving exports breaks an operator workflow | Every move adds an `Open in Editor` link at the old location for one release. |
| Shell visibility hides a required control | Every hidden part is restorable from Options > Shell and a keyboard shortcut. |
| Performance regression from more surfaces | Hidden surfaces are not drawn; the frame-time gate is a release blocker. |

## 10. Verification strategy

| Gate | Command / method | Pass condition |
| --- | --- | --- |
| Build | `dotnet build wow-viewer/WowViewer.slnx -c Debug` | 0 errors |
| Preset ownership | new test in `WowViewer.Core.Tests` | 0 violations |
| Shell visibility | new test in `WowViewer.Core.Tests` | 0 violations |
| UI reference | `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | PASSED |
| Frame time | operator capture, UI shown vs hidden | ≤ 1 ms difference |
| Preset walk | operator walk of all four presets | 0 editing controls in Viewer/Archeology |
| Explore default | operator load of a taxi map | taxi/route presentation shown |

## 11. Out of scope

- New inspectors for asset kinds that have none.
- New taxi data generation.
- NPC population and game-object feature work (the operator named these as still needing work, but
  they are not this spec).
- A new renderer or a new theme system.
- The `Ctrl+K` command palette (still open from Spec 280).
