# Feature Specification: Viewer UI Consolidation Audit

**Feature Branch**: `223-ui-consolidation-audit`
**Created**: 2026-09-04
**Status**: Implementing Phase 6; interactive acceptance pending
**Input**: Operator directive — "audit all the sidebars, all the tab profiles, and make everything as concise as possible, so we have NO duplication of information panels. We currently have no less than 3 or 5 different ways to see details about the tile, and no way to easily change the doodadset for the WMO we have selected. It's all very murky waters that need to be audited carefully with a speckit plan at the top, to keep track of what is there, and then to allow us to come up with new epic UI overhauls that should not really require a lot of work, it's just a matter of cataloging all the things we have, and then deduplicating them into simple profiles for the viewer's intended use. The Editor tab is most under-organized, the Archeology tab is also meant to contain the Cartography features, but can exhibit Editor functionality in the way that we can save merged data from phase maps to new output files. If anything, we should merge Editor and Archeology features as a single profile first, then split off 'true' editor features to an Editor profile. The idea is to simplify the ui while retaining all the existing features and the Quick tab needs improving for every profile, to include the stuff we most want to fiddle with when we load into the viewer (Fog End is the first thing I set to 5000 or more, every time!)"

## User Scenarios & Testing *(mandatory)*

### US1 — Catalog the existing UI surface (audit-first)

An operator (or agent) opens the viewer and can consult a single authoritative inventory that lists
every tab profile, every sidebar section, every floating window, and every information panel —
what each shows, which spec owns it, and which other surfaces duplicate it. The catalog is the
source of truth for every later consolidation decision: nothing is removed, moved, or merged
without a catalog row saying where it went.

**Why this first**: the operator's core complaint is murkiness — features exist but cannot be
found, and the same information appears in several places. A catalog turns "3–5 ways to see tile
details" from a feeling into a list with owners.

**Priority**: P0 (blocking — every other story consumes the catalog).

**Acceptance criteria**:
- A UI surface inventory document exists, organized by tab profile (Quick, Viewer, Editor,
  Archaeology, Settings) and by surface type (left sidebar, right sidebar, floating window,
  status bar, top bar).
- Every information panel in the running viewer appears in the inventory with: name, location,
  owning spec (where one exists), the data it displays, and its duplicates.
- The inventory records at least the known duplicates: the multiple tile-detail surfaces (3–5
  distinct ways to see details about the selected tile) and the multiple selection/inspector
  surfaces.
- Every entry carries a disposition: keep-here, merge-into-X, move-to-Y, or retire (with the
  replacement named).

### US2 — One authoritative detail surface per object type

When the operator selects a tile, chunk, WMO, doodad, or model, exactly one inspector surface
shows that object's details. Other surfaces that used to repeat the same information either link
to the authoritative one or are removed. No information is shown in two places with two different
refresh behaviors.

**Why**: the operator reports 3–5 different ways to see tile details today, with no clear owner.
Duplicated panels drift apart, show stale data, and waste screen space.

**Acceptance criteria**:
- For each selectable object type (terrain tile, chunk, WMO, WMO doodad, M2/MDX model, PM4
  object), the inventory names exactly one authoritative detail surface.
- Selecting an object updates the authoritative surface; duplicate surfaces are removed or
  reduced to a pointer ("details in Inspector").
- Two different panels never disagree about the same selected object at the same time.

### US3 — WMO doodad-set switching on the selected WMO

When a WMO is selected, the operator can switch its active doodad set (MODS) directly from the
selection's detail surface, and the 3D view re-renders the WMO with the chosen set's doodads
without reloading the map.

**Why**: the operator explicitly calls this out as missing. The data and render path already
support doodad sets (MODS/MODD parsing exists); only the UI affordance is missing.

**Acceptance criteria**:
- Selecting a WMO exposes its available doodad sets (by name/index) and the currently active one.
- Choosing a different set updates the WMO's rendered doodads without a map reload.
- Switching sets does not clear the selection or the camera.

### US2a — Unified object Inspector (right sidebar, HUD-ready)

All object-inspection surfaces consolidate into a single **Inspector** tool in the right sidebar,
organized by object type — **ADT, MDX, M2, WMO, PM4, and WL\* liquid** — replacing the duplicated
detail surfaces (MCNK Explorer, ADT Chunk Investigation, hover overlays, Terrain Lab MCNK pages,
legacy InspectorTabs, PM4 detail pages). The Inspector is built HUD-ready: its content model is
structured so it can convert into the 3D object HUD (Spec 212) later or during this spec,
whichever makes sense.

**Why**: the operator's Q3 decision — one tool, split by type, no duplicated crap, and a path to
the still-MIA 3D object HUD.

**Acceptance criteria**:
- One Inspector surface in the right sidebar shows details for whatever is selected, with a
  section per object type (ADT/MDX/M2/WMO/PM4/WL*).
- Every former inspection surface's capabilities are present in the Inspector (verified against
  the inventory) before the old surfaces are removed.
- The Inspector's content is separable from its sidebar host so it can render as a HUD object
  without reimplementation.

### US4 — Editor and Archaeology merge first, then true-editor split

The Editor and Archaeology tab profiles are first merged into a single combined profile so every
feature is visible in one place; afterwards, features that are "true editor" work (authoring,
writing, saving transformed data) are split back into a dedicated Editor profile, leaving
Archaeology as the analysis/inspection home. Cartography (Spec 222) lives under Archaeology,
including its save-merged-data-to-output-files capability, which is the one piece of Editor
functionality Archaeology legitimately exhibits.

**Why**: the operator's stated sequencing — merge first so nothing is lost, then split
deliberately. Editor is described as the most under-organized tab.

**Acceptance criteria**:
- After the merge, every feature previously reachable from Editor or Archaeology is reachable
  from the merged profile; nothing is lost (verified against the inventory).
- After the split, the Editor profile contains only authoring/writing features; analysis and
  inspection features remain in Archaeology.
- Cartography appears under Archaeology with its save-merged-output capability present.
- The old separate Editor and Archaeology tabs no longer exist once the merged profile ships.

### US5 — Quick tab carries the high-frequency controls for every profile

Every tab profile's Quick tab exposes the controls an operator most often adjusts immediately
after loading a map — starting with Fog End (the operator sets it to 5000+ every session), plus
the other most-fiddled settings (time of day, camera speed, detail tile count, wireframe toggle).
The Quick tab is the same concept in every profile, populated per profile with that profile's
most-used controls.

**Why**: the operator names Fog End as the first thing changed every time; it must be on the
first surface they see, in every profile.

**Acceptance criteria**:
- Fog End is adjustable from the Quick tab in every profile, without opening Settings.
- Each profile's Quick tab includes that profile's top controls, derived from the audit (not
  guessed), and persists changes like the full panels do.
- No control exists ONLY in Quick: Quick is a mirror of authoritative controls, never a second
  implementation (FR-6).

### US6 — Reliable transport and a consistent four-tab workbench

The operator can ride a taxi or play a camera path while the world streams, inspect and pin
terrain chunks, and find the same four workbench tabs in every workspace profile.

**Acceptance criteria**:
- Active taxi route poses advance independently of route visibility, selection filters, and
  mount asset readiness. Stopping or switching transport releases the active ride override.
- Interactive camera playback starts without waiting for path warmup. Video/queued captures
  retain their readiness gates. Equivalent map/build formatting is accepted; genuinely different
  maps/builds remain protected, including changes during playback.
- The terrain context resolves pinned chunk, hovered chunk, camera chunk, then active-world
  overview. Pins cannot survive a change of terrain source or refer to stale cached chunk data.
- Terrain Inspector exposes area names, MCNK flags and overlays (including diagonal weak corners),
  holes, height range, textures/alpha, shadow/MCCV, liquids and tile placement counts, with frame,
  copy-texture-summary, copy-coordinates and clear-pin actions.
- Every profile shows **Quick / Inspector / Editor / Archaeology**. Inspector has **Context /
  Placements / LOD & Budget** subtabs. Quick stays selected across profile changes and settings
  continue to use their existing owners.
- Scene and Utilities have no top-level buttons. Every former route has a named, reachable
  replacement in the inventory, including keyboard/menu callers and legacy shell compatibility.
- Shared UI widgets provide section/help, wrapping actions, status, labeled values and compact
  sliders, and are used by the workbench. The user guide describes the resulting routes.

## Functional Requirements *(mandatory)*

- **FR-1 — Surface inventory first**: A complete UI surface inventory MUST be produced before any
  consolidation change lands. Every panel, sidebar section, tab, and floating window in the
  viewer shell MUST appear in it with location, owning spec, displayed data, and disposition
  (keep / merge / move / retire-with-replacement).
- **FR-2 — No orphaned features**: Every feature reachable in the UI today MUST remain reachable
  after consolidation. A surface may be removed only when its replacement is named in the
  inventory and reachable in the same or fewer interactions.
- **FR-3 — Single authoritative detail surface**: For each object type there MUST be exactly one
  authoritative detail surface; all other surfaces MUST NOT duplicate its content.
- **FR-4 — Doodad-set control**: The selected-WMO detail surface MUST expose doodad-set switching
  (US3).
- **FR-5 — Profile merge then split**: Editor and Archaeology MUST first be merged into one
  profile; the split into a true-Editor profile MAY follow only after the merged state is
  verified complete (US4 sequencing is normative).
- **FR-6 — Quick mirrors, never forks**: Quick-tab controls MUST invoke the same underlying
  actions as their full-panel equivalents; Quick MUST NOT contain a second implementation of any
  control (same rule as Spec 212 FR-032).
- **FR-7 — Floating windows are not a home**: New features MUST NOT place their primary surface
  in a floating window; floating windows that disappear cannot be recalled. Existing floating
  surfaces migrate into sidebar tabs where the audit marks them.
- **FR-8 — Cartography home**: Cartography (Spec 222) lives under Archaeology, with the
  synthesized-minimap tooling as a right-sidebar tab inside it (per Spec 222 T109/T109a), and the
  save-merged-data capability available there.
- **FR-9 — Audit is re-runnable**: The inventory MUST be maintainable — a future panel addition
  updates the inventory in the same change, so the catalog cannot silently rot.

## Key Entities

- **Tab profile**: A top-level workspace mode (Quick, Viewer, Editor, Archaeology, Settings —
  post-merge: Quick, Viewer, Workbench(merged), Settings) that determines which sidebar sections
  and tools are shown.
- **Surface**: Any UI region that displays information or accepts input: left sidebar section,
  right sidebar section, top tab, floating window, status bar, in-scene overlay.
- **Authoritative detail surface**: The single surface designated (in the inventory) to show an
  object type's full details.
- **UI surface inventory**: The catalog document; the spec's primary artifact and the input to
  every consolidation task.

## Assumptions

- The audit covers the ImGui shell in the viewer app (top tabs, both sidebars, floating windows,
  status bar). CLI tools and the defunct app are out of scope.
- **Operator decisions 2026-09-04 (binding)**: the merged profile keeps the name **Archaeology**;
  the legacy non-tab UI (`_useTabUi=false`) is **kept** until the 3D object HUD (Spec 212) is
  functional; the unified object Inspector is organized ADT/MDX/M2/WMO/PM4/WL* and must be
  HUD-ready (convertible to the Spec 212 3D object HUD).
- "Retaining all the existing features" means feature parity at the action level (every action
  remains reachable), not pixel-identical layout preservation.
- The known duplicate count ("3 or 5 ways to see tile details") is an operator estimate; the
  audit's job is to produce the exact list.
- Spec 222's Cartography directives (top-level tab, right sidebar) are amended by this spec's
  operator directive: Cartography lives under Archaeology rather than as its own top-level tab.
  Spec 222's tasks.md records the supersession.

## Success Criteria *(mandatory)*

- **SC-1**: The inventory lists every UI surface with an owner and a disposition; an operator
  walking the viewer against the inventory finds no unlisted panel.
- **SC-2**: For each object type, exactly one authoritative detail surface exists; selecting an
  object updates it and no duplicate panel shows the same data.
- **SC-3**: A selected WMO's doodad set can be changed in under 5 seconds without leaving the
  selection context (US3).
- **SC-4**: After the merge, a feature-by-feature walkthrough confirms every former Editor and
  Archaeology feature is reachable from the merged profile; the split afterwards moves only
  true-editor features.
- **SC-5**: Fog End (and the profile's other top controls) are adjustable from Quick in every
  profile, and changing them has the identical effect as changing them in the full panel (FR-6).
- **SC-6**: No consolidation change removed an action: a checklist pass over the inventory's
  "retire" entries shows each one's replacement reachable in equal or fewer interactions.

## Dependencies

- Spec 222 (Cartography) — its T109/T109a surfaces are folded into this spec's profile structure.
- Spec 212 (spatial UI shell) — FR-6 reuses its single-action rule; museum/HUD profiles are
  adjacent but not blocking.
- Spec 080/145 (prior UI consolidation passes) — historical context for what was already merged.
- Spec 197 (workspace profiles) — the profile system this audit reorganizes.

## Stakeholders

- Operator: primary user; defines "most-fiddled controls" and accepts each phase.
- Agents: maintain the inventory alongside any UI change (FR-9).
