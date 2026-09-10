# Feature Specification: Map Save (Merged ADT / Alpha WDT) & New Map Creator (Spec 234)

**Feature Branch**: `234-map-save-new-map`

**Created**: 2026-09-09

**Status**: Draft — authored verbatim from operator directive; not planned

**Input**: Operator directive 2026-09-09 (verbatim intent): "the ability to save merged ADT's or
alphaWDT's does not exist in our Archeology functionality nor editor. We need that, as well as a
means of a 'New Map' creator in the Editor tab. We already have the wiring for all of that. We'd
love to see multi-map support to be possible, too, but that's a bigger feature."

## Context

The composition and writing building blocks already exist and must be composed behind UI surfaces,
not reinvented:

- **Writers (Spec 197)**: Alpha 0.5.3 WDT output ([AlphaWdtWriter](../../src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs))
  and LK v18 monolithic ADT output ([LkAdtWriter](../../src/core/WowViewer.Core.IO/Maps/LkAdtWriter.cs))
  are supported. Native Cataclysm/MoP split ADT is explicitly unavailable until a slot-aware writer
  exists (Spec 197 boundary — never a lossy best-effort).
- **Cartography composition (Spec 232)**: the layer-stack project (donor tiles, placement locks,
  rotation/mirror, Z transform, WDL edge-snap) already composes terrain, liquids, and placements in
  both the Alpha and Standard adapters. What it cannot do today is flatten that composed result into
  a real client-loadable map on disk.
- **Map generator (Spec 192)**: `terrain-generate-templated` CLI synthesizes walkable multi-tileset
  maps — the seed of a New Map creator.
- **Editor IA (Spec 231)**: the Editor tab's 4-page layout includes a Data I/O page — the natural
  home for save/export surfaces.

**Relationship to Spec 230**: Spec 230 (Reconstruction Editor) drafted a New Map Generator (US2)
and multi-era save targets (US3). This spec, from the operator's 2026-09-09 directive, is the
authoritative refinement of those two stories and adds the Archaeology-side save path. Spec 230
retains ownership of Rosetta-indexed object placement (US1); its US2/US3 are superseded by this
spec and should be marked as such in a dated amendment when 230 is next touched.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Save a merged (composed) map from Archaeology (Priority: P1)

An operator has built a cartography composition in Archaeology — one or more donor layers placed,
rotated, Z-shifted, and edge-snapped over a base map. They want to export that composed result as a
real map: either an Alpha 0.5.3 WDT (with its ADT tile set) or an LK v18 ADT set, loadable by the
matching game client. Today there is no save path for the merged result; the composition exists
only inside the viewer session/project file.

**Why this priority**: This is the payoff of the entire cartography pipeline — without it, merged
maps never leave the viewer. The operator named it first.

**Independent Test**: Can be fully tested by composing a small layer project, saving it to Alpha
WDT and to LK ADT, and loading each output back (viewer round-trip first; real-client load is the
operator's visual gate).

**Acceptance Scenarios**:

1. **Given** a composition with placed layers, **When** the operator chooses "Save Map" in
   Archaeology and selects the Alpha 0.5.3 WDT target, **Then** a complete Alpha WDT + ADT tile set
   is written to a project-managed output root, and the saved map reloads in the viewer with the
   composed terrain, liquids, and placements intact.
2. **Given** the same composition, **When** the operator selects the LK v18 ADT target, **Then** a
   complete LK WDT + split-per-tile ADT set is written and round-trips the same way.
3. **Given** a composition with an unsupported save target selected (e.g., Cataclysm/MoP split
   ADT), **When** the operator attempts to save, **Then** the save is refused by name with an
   explanation, and no partial output is written.
4. **Given** a composition whose donor content is only partially loaded (streamed-out tiles),
   **When** the operator saves, **Then** the save either resolves all referenced donor content or
   fails with a per-tile report — never silently writes holes.

### User Story 2 - Save the working map from the Editor tab (Priority: P1)

An operator working in the Editor tab (loaded map, staged placement edits, terrain patches) wants
the same save capability at hand in the Editor's Data I/O surface, without switching to Archaeology.
The Editor save writes the current map state through the same single write path as US1 — no forked
exporter.

**Why this priority**: The operator explicitly named the Editor as a place the save is missing; the
Editor is where day-to-day editing happens.

**Independent Test**: Can be fully tested by loading a map in the Editor, making a staged edit, and
saving to both targets from the Editor's Data I/O page.

**Acceptance Scenarios**:

1. **Given** a loaded map with staged edits in the Editor, **When** the operator saves from the
   Data I/O page to the Alpha WDT target, **Then** the output reflects the current map state and
   round-trips in the viewer.
2. **Given** the same session, **When** the operator saves to the LK ADT target, **Then** the
   output reflects the current map state and round-trips in the viewer.
3. **Given** any save from either surface, **When** the write completes, **Then** the save reports
   what was written (map name, target era, tile count, output root) and any tiles skipped, using
   the shared status surface — not a bespoke dialog.

### User Story 3 - New Map creator in the Editor tab (Priority: P2)

An operator wants to start a brand-new map from inside the Editor: name it, choose its basic shape
(blank terrain or a template-generated layout), and have it loaded for immediate editing and saving
(US1/US2 targets). The creator composes the existing generator tooling; it does not implement a new
terrain synthesizer.

**Why this priority**: It unlocks real editing on maps that don't exist yet, but it depends on the
save path (US1/US2) to be useful end-to-end.

**Independent Test**: Can be fully tested by creating a new map from the Editor, verifying it loads
as the active map, and saving it to both targets.

**Acceptance Scenarios**:

1. **Given** the Editor tab, **When** the operator opens the New Map creator, enters a unique map
   name, and chooses a blank map, **Then** a new empty map is created in a project-managed output
   root and becomes the active map in the viewer.
2. **Given** the New Map creator, **When** the operator chooses a template-generated layout and
   provides the generator parameters, **Then** the generated map is created, loaded, and immediately
   usable with editing and save (US1/US2).
3. **Given** a map name that already exists in the output root, **When** the operator attempts to
   create, **Then** creation is refused with a clear message and the existing map is untouched.

### Edge Cases

- What happens when the composition references donor tiles from a client root that is no longer
  configured? The save must fail with a named missing-source report, not write partial terrain.
- What happens when saving a map whose tile grid has holes (never-authored tiles)? Holes must be
  written as valid empty tiles for the target era, or refused with a report — decided at plan time
  and recorded in the receipt.
- What happens when the operator saves over an existing output directory? The save must require
  explicit confirmation before overwriting, and never merge old and new tile files.
- What happens when a placement references an asset path that does not exist in the target era's
  naming conventions? The save must report the unresolvable placements and either skip them
  (reported) or refuse — never write a dangling reference silently.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a "Save Map" action in the Archaeology cartography surface that
  writes the current composed map state to disk.
- **FR-002**: System MUST provide the same save action in the Editor tab's Data I/O surface.
- **FR-003**: Both save surfaces MUST write through one shared save pipeline (single write path;
  no per-surface exporter forks).
- **FR-004**: The save pipeline MUST support the Alpha 0.5.3 WDT target and the LK v18 ADT target,
  composing the existing writers.
- **FR-005**: The save pipeline MUST refuse unsupported target eras by name (Cataclysm/MoP split
  ADT remains unavailable until the slot-aware writer lands — Spec 197 boundary).
- **FR-006**: Every save MUST produce a validation receipt covering round-trip verification of the
  written output (per the Spec 221 harness discipline).
- **FR-007**: All save output MUST land inside the repository's project-managed output root or an
  operator-supplied path (AGENTS.md §9.3 write containment).
- **FR-008**: The Editor tab MUST provide a New Map creator that produces a new map (blank or
  template-generated) in a project-managed output root and loads it as the active map.
- **FR-009**: The New Map creator MUST compose the existing map generator tooling rather than
  implementing new synthesis.
- **FR-010**: Save and create actions MUST report failures with actionable, per-tile or per-item
  detail (missing source, unresolvable placement, unsupported era) rather than generic errors.
- **FR-011**: New UI surfaces MUST use the SharedUiWidgets primitives and register an inventory row
  in the same change (AGENTS.md §11), and MUST live in owned service classes (AGENTS.md §10 — no
  new `ViewerApp_*` partials, no new `WorldScene`/`ViewerApp` members).

### Key Entities *(include if feature involves data)*

- **Map Save Request**: the map identity (name, era), the chosen target (Alpha WDT / LK ADT), the
  output root, and the state to flatten (composed layers or current editor state).
- **Save Result**: per-tile write outcomes, skipped/reported items, output paths, and the receipt
  reference.
- **New Map Request**: unique map name, creation mode (blank / template-generated), generator
  parameters when applicable, and the output root.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An operator can take a composed multi-layer map from Archaeology to a client-loadable
  Alpha WDT and an LK ADT set without touching a CLI, in a single session.
- **SC-002**: Every save path round-trips: the written map reloads in the viewer with terrain,
  liquids, and placements matching the composed state (verified by receipt, witnessed by operator
  for visual acceptance).
- **SC-003**: An operator can create a new map from the Editor tab and save it to both targets
  without touching a CLI.
- **SC-004**: Zero silent failures: every refused or skipped item in a save appears in the save
  report and receipt.

## Out of Scope

- **Multi-map support** (loading/working with several maps simultaneously in one session): the
  operator explicitly deferred this as a bigger feature. This spec must not foreclose it — the save
  pipeline and New Map creator operate on named map identities, not a single implicit active map —
  but no multi-map UI or session model is built here. Multi-map gets its own spec when the operator
  calls for it.
- **Cataclysm/MoP split ADT output**: blocked on the slot-aware writer (Spec 197 boundary).
- **Terrain sculpting / hand-painting**: the operator's editing model is reconstruction from
  existing data (Spec 230 context).
- **Rosetta-indexed object placement**: remains owned by Spec 230 US1.

## Assumptions

- The existing Alpha WDT and LK ADT writers are correct for their supported eras and are used
  as-is; any writer bug is a separate spec with evidence (AGENTS.md §4 freeze rules).
- "Merged ADT" means the flattened result of a Spec 232 composition (or an edited map), not a new
  ADT merge algorithm.
- Real-client load acceptance (map actually boots in a game client) is operator-owned and never
  claimed from round-trip or unit tests alone (AGENTS.md §6).
- The New Map creator's template path reuses the Spec 192 generator's existing parameter surface.

## Dependencies

- Spec 197 (writers), Spec 232 (composition state to flatten), Spec 192 (generator), Spec 221
  (validation harness), Spec 231 (Editor Data I/O page as the UI home), Spec 227/228 (UI standard +
  owned service classes).