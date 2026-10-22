# Feature Specification: v0.7.0 UI Presets and the Explore Museum

**Feature Branch**: `283-v070-ui-presets-and-explore`

**Created**: 2026-10-21

**Status**: Draft (operator-originated)

**Input**: Operator, 2026-10-21: "let's update the version of the tooling to v0.7.0, since the ui is
a massive overhaul, and all the new features (npc population, game objects) need some serious work
still, but will be a welcome change to the viewer. The taxi ui stuff should become the normal thing
that is displayed instead of crummy text in the ui, in a new tab called 'Explore', which disables
most of the tooling except for inspection tools for all the assets and data in the viewer, with
limited tools in the sidebars (if anything), the chat panel and tools next to it, to keep the ui
clean and less deranged. We should be able to enable/disable parts of the ui, and make it so it acts
more like the screenshot I showed you of the early WoW ui. I don't think you fully grasp that I
wanted things simplified and deduplicated, as we have a lot of wordy, text salad components with
terrible ui choices in design. I need you to really work hard on the ui, make it so it doesn't slow
the program down, but still has every iota of everything we requested, organized by the presets of
Viewer/Editor/Archeology/Explore, leaving the Explore feature to act more like an interactive museum
than the database of databases -viewer that it has become. There are lots of functionality in the ui
that are for editing that ends up in the Viewer or Archeology areas, which is wrong. I expect actual
presets in those buttons of tools, not fucking everything jammed up my pisshole for no reason, like
it is right now. use speckit, we need to do visual analysis of each part of the ui and actually come
up with a real fucking plan instead of a the haphazard shitshow we built thusfar."

## Context

Spec 280 collapsed five navigation systems into three destinations and one UI manager. That was
structural progress, but the result is still a **database browser with a 3D viewport attached**:
every destination shows the same long sidebar of text, the same Options dump, and the same chat
strip, and editing tools leak into Viewer and Archaeology. The operator's reference is the early
WoW client UI: a small number of large, obvious surfaces, a minimap, a chat frame, and almost no
text.

This spec defines the v0.7.0 UI as **four presets** with hard boundaries, an **Explore** preset that
behaves like an interactive museum, and a **UI visibility model** so the operator can turn parts of
the shell off. It is a specification and plan, not an implementation.

### Operator correction — 2026-10-21

The operator rejected the current screenshot as unusable and directed that the second supplied
reference image govern the layout. The default viewer must be scene-first: compact independent
minimap at upper-right, compact chat at lower-left, tool actions represented by anchored icon bars,
and no full-height sidebars consuming most of the scene. The map-spawn
preview is transient: after loading the selected map, Cancel, or the window close control, it must
close and remain closed until explicitly opened again. This amendment supersedes the previous
sidebar-docked minimap and chat-between-sidebars arrangement. It does not authorize fabricating
character HUD state (health, mana, backpack, paper doll) that this viewer does not own.

The operator further clarified that the side tool areas are **anchored icon action bars**, not hidden
data sidebars and not rows of text buttons. Clicking an icon opens a small, independently closeable
window for that tool's existing controls, analogous to opening a character or item window in the
client. The scene stays visible behind these windows; detailed controls are opt-in and must remain
legible. The Spellbook prototype was explicitly rejected by the operator and is removed from the UI;
tool actions are launched directly from the icon bars.

**Operator naming correction, 2026-10-21:** do not use two-character abbreviations as control labels.
Use pictograms with full-name tooltips and readable window titles. Existing format names such as PM4
are names of supported data, not substitutes for tool labels.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Four presets, each with its own tools (Priority: P1)

The operator picks one of four presets from the bottom bar: **Viewer**, **Editor**, **Archeology**,
or **Explore**. Each preset shows only the tools that belong to it. Switching presets changes the
sidebars, the bottom bar pages, and the available actions; it never leaves a tool from another
preset on screen.

**Why this priority**: The operator's core complaint is that every button contains everything. A
preset that cannot be trusted to be scoped is not a preset.

**Independent Test**: Switch through all four presets and confirm that no editing action is
reachable from Viewer or Archeology, and no inspection-only action is the *only* way to do an
editing task in Editor.

**Acceptance Scenarios**:

1. **Given** the Viewer preset, **When** the operator opens every sidebar section and every bottom
   bar page, **Then** no control writes to a file, mutates a placement, or edits terrain.
2. **Given** the Editor preset, **When** the operator opens the sidebar, **Then** the editing tools
   (placement, terrain, data I/O, converters) are present and the inspection-only museum surfaces
   are not.
3. **Given** the Archeology preset, **When** the operator opens the sidebar, **Then** only analysis
   and reconstruction tools are present; no placement editing and no export-to-client actions.
4. **Given** any preset, **When** the operator switches to another preset, **Then** the previous
   preset's page selection is restored when they return.

### User Story 2 - Explore is an interactive museum (Priority: P1)

The operator selects **Explore** and the viewer becomes a museum: the taxi/route presentation is the
normal display, the world is presented for looking rather than editing, and the only tools are
inspection tools for assets and data. The sidebar is minimal or absent; the chat panel and its
tools remain.

**Why this priority**: The operator explicitly asked for Explore to stop being "the database of
databases" and become an interactive museum, with the taxi UI as the normal display.

**Independent Test**: Enter Explore, load a map, and confirm the taxi/route presentation is the
default view, that inspection works on every asset kind, and that no editing control is reachable.

**Acceptance Scenarios**:

1. **Given** the Explore preset, **When** a map loads, **Then** the taxi/route presentation is the
   default display, not a text list.
2. **Given** the Explore preset, **When** the operator inspects terrain, WMO, M2, PM4, liquid, or a
   spawn, **Then** the inspection surface opens and reports the asset's data.
3. **Given** the Explore preset, **When** the operator looks for an editing action, **Then** none is
   present in the sidebar, bottom bar, or chat tools.
4. **Given** the Explore preset, **When** the operator wants a specific asset, **Then** they can
   search or browse to it without leaving the museum presentation.

### User Story 3 - The shell can be simplified by the operator (Priority: P2)

The operator can turn individual shell parts on or off — sidebars, minimap, chat, status bar, bottom
bar, tooltips — and the layout remembers the choice. Turning a part off gives its space back to the
scene.

**Why this priority**: The operator asked to "enable/disable parts of the ui" and to make the shell
act like the early WoW UI, which is mostly scene.

**Independent Test**: Toggle each shell part off and on and confirm the scene grows and shrinks
accordingly and the choice survives a restart.

**Acceptance Scenarios**:

1. **Given** any preset, **When** the operator hides the left sidebar, **Then** the scene expands
   into the freed space and the sidebar can be restored from a visible control.
2. **Given** any preset, **When** the operator hides the chat, **Then** the chat dock disappears and
   the scene expands.
3. **Given** a saved visibility configuration, **When** the viewer restarts, **Then** the same parts
   are visible.

### User Story 4 - No text salad (Priority: P2)

Every surface leads with a control or a value, not a paragraph. Explanatory prose is available on
demand (tooltip or help), never permanently occupying a panel.

**Why this priority**: The operator called the current UI "wordy, text salad components with
terrible ui choices in design".

**Independent Test**: Walk every surface and confirm no panel is dominated by wrapped prose, and
that every removed paragraph is still reachable as a tooltip or help entry.

**Acceptance Scenarios**:

1. **Given** any sidebar section, **When** it is open, **Then** its first row is a control or a
   labelled value, not a sentence.
2. **Given** a control that previously had a paragraph under it, **When** the operator hovers it,
   **Then** the paragraph is available as a tooltip.
3. **Given** the generated UI reference, **When** the operator searches a control, **Then** the help
   text is present there.

### User Story 5 - The UI does not slow the viewer (Priority: P1)

The UI adds no measurable frame-time cost compared with the current build at the same scene and
resolution.

**Why this priority**: The operator asked for the UI to "not slow the program down".

**Independent Test**: Compare frame time with the UI hidden and shown at the same camera and scene,
and compare against the v0.6.1 baseline.

**Acceptance Scenarios**:

1. **Given** a loaded world, **When** the UI is shown, **Then** the frame-time difference against
   the UI-hidden frame is within the agreed budget.
2. **Given** a preset switch, **When** it happens, **Then** no frame stalls beyond the agreed budget.

### Edge Cases

- A preset with no map loaded: show the preset's empty state, not another preset's tools.
- Explore with no taxi data for the map: fall back to the museum's asset browser, not to the
  Editor.
- A shell part hidden by the operator but required by a preset: the preset must offer a visible way
  to restore it.
- A control that belongs to two presets: it gets one authoritative home and a link from the other,
  never a duplicate implementation.
- Very small windows: presets must degrade to a single sidebar and a collapsed bottom bar, not
  overlap.

## Requirements *(mandatory)*

### Presets

- **FR-001**: The shell MUST expose exactly four presets: Viewer, Editor, Archeology, Explore.
- **FR-002**: Each preset MUST declare the complete set of surfaces it owns (sidebar sections,
  bottom-bar pages, chat tools, Options groups) and MUST NOT render a surface it does not own.
- **FR-003**: Viewer MUST be inspection-only: no file writes, no placement mutation, no terrain
  edits.
- **FR-004**: Editor MUST own every editing action: placement, terrain, data I/O, converters.
- **FR-005**: Archeology MUST own analysis and reconstruction only: no placement editing, no
  client-export actions.
- **FR-006**: Explore MUST own the museum presentation and inspection only.
- **FR-007**: Switching presets MUST preserve each preset's page selection.

### Explore

- **FR-008**: Explore MUST present the taxi/route experience as its default display, not a text
  list.
- **FR-009**: Explore MUST provide inspection for terrain, WMO, M2, PM4, liquid, and spawns.
- **FR-010**: Explore MUST provide a way to find a specific asset without leaving the museum
  presentation.
- **FR-011**: Explore MUST NOT expose any editing action.

### Shell visibility

- **FR-012**: The operator MUST be able to show or hide each shell part: left sidebar, right
  sidebar, minimap, chat, status bar, bottom bar, tooltips.
- **FR-013**: Hiding a shell part MUST return its space to the scene.
- **FR-014**: Shell visibility MUST persist across restarts.
- **FR-015**: Every hidden part MUST be restorable from a visible control.

### Simplification

- **FR-016**: No surface may lead with a paragraph; prose MUST move to tooltips or the generated UI
  reference.
- **FR-017**: Every control MUST have exactly one authoritative home.
- **FR-018**: The generated UI reference MUST list every visible control with its preset, region,
  shortcut, and help text.

### Performance

- **FR-019**: The UI MUST NOT add per-frame work proportional to the number of controls that are
  not visible.
- **FR-020**: Preset switching MUST NOT rebuild GPU resources.

### Version

- **FR-021**: The viewer version MUST be `0.7.0` for this release.

## Success Criteria *(mandatory)*

- **SC-001**: An operator can name the preset that owns any given control without opening more than
  one preset.
- **SC-002**: Zero editing controls are reachable from Viewer or Archeology in a full walk of both
  presets.
- **SC-003**: Explore shows the taxi/route presentation on map load in 100% of tested maps that have
  taxi data.
- **SC-004**: Every shell part can be hidden and restored, and the choice survives a restart.
- **SC-005**: No sidebar section opens with a paragraph as its first row.
- **SC-006**: Frame time with the UI shown is within 1 ms of frame time with the UI hidden at the
  same camera and scene, at 1080p and 4K.
- **SC-007**: The generated UI reference covers 100% of visible controls.

## Key Entities

- **Preset**: Viewer, Editor, Archeology, Explore. Owns surfaces; has a persisted page selection.
- **Surface**: A sidebar section, bottom-bar page, chat tool group, Options group, or museum panel.
  Belongs to exactly one preset.
- **Shell part**: A chrome region the operator can show or hide.
- **Museum exhibit**: An inspectable asset or data record presented in Explore.

## Assumptions

- The existing `ViewerUiManager` remains the single shell owner; this spec changes what it renders,
  not who owns it.
- The taxi/route presentation already exists and can be promoted to the Explore default without a
  new renderer.
- Inspection surfaces already exist for the asset kinds listed; this spec routes them, it does not
  build new inspectors.
- The generated UI reference and its build check remain the enforcement mechanism for FR-018.
- "Early WoW UI" means: large scene, small number of obvious panels, minimap and chat as the only
  persistent chrome, no permanent prose.
