# Feature Specification: Cartography — Multi-Map, Multi-Tile Composition Workbench

**Feature Branch**: `222-map-composition-workbench`

**Created**: 2026-09-04 (revised same day — operator accepted the workbench deal with an expanded scope)

**Status**: Draft v2 — awaiting operator sign-off

**Input**: Operator, 2026-09-04 (two messages):
1. "move the whole phase map thing somewhere in the Viewer's right sidebar, then expand it beyond
   just phase maps, allowing any map to be loaded in and dragged/aligned to the minimap as a layer
   on top of the base layer. I don't know why we keep going around in circles on this, but I keep
   explaining it and it never happens."
2. "make it so any tile can be loaded, not just any map, and it's a deal. That would cover the map
   copy/paste/rotate stuff too, bundle it all up into a single big thing to make it a cleaner
   replacement for the existing 'selection', chunk manipulator tool, and whatever else that can
   consolidate existing old right sidebar functionality into a single new 'Cartography' feature that
   obsesses over manipulating the maps and handling multiple layers of tile data from multiple maps
   (not just a single phase!)"

## Context

### What exists today (and why it must consolidate)

- **Spec 203** — multi-phase composition: left-sidebar "Phase Map Layers" panel, per-channel
  checkboxes, numeric tile offsets. Whole-map layers only.
- **Spec 219** — rotation/mirror/per-tile placement settings on those layers (core transform seam
  validated; workbench UI open). Also owns the retirement of Spec 195's UI.
- **Spec 195** — "Overhead Chunk Manipulator": a camera-centered 280px dark grid with no map
  backdrop, click-only despite claiming drag selection, camera-derived paste target, and a paste
  path that applies only a subset of its own exposed channels. Superseded as UI owner by 219 but
  still present.
- **Spec 208** — cross-map tile transplant design (channel reconciliation done, Phase 1 next).
- **Spec 137/135** — phased minimap overlay composition.
- **Right sidebar** — a growing pile of single-purpose panels (PM4 utilities, investigation
  toolboxes, phase layers) that each re-implement selection, listing, and transform fragments.

Three separate selection/transform systems (scene click-selection, chunk manipulator, phase layer
offsets) have drifted apart. The operator has now specified the destination three times: **one
Cartography feature in the right sidebar that obsesses over map manipulation — layers of tile data
from multiple maps, at whole-map AND single-tile granularity, visible and draggable on the minimap.**

### The motivating failure (measured 2026-09-04)

Adding `Shadowfang` over `Azeroth` (0.5.3) silently contributed nothing: the phase stack, adapter,
and WDT resolution all worked, but target tile (34,28) maps to source (34,28) at zero offset and
Shadowfang has no tile there. Numeric offsets with no spatial feedback fail silently. Diagnostics
added 2026-09-04 remain as the evidence trail; the interaction model is what this spec replaces.

## User Scenarios & Testing

### User Story 1 - Add any map OR any tile as a layer (Priority: P1)

The operator opens Cartography in the right sidebar and adds content at two granularities:
- **Whole-map layer**: every tile the donor map occupies becomes part of the layer.
- **Single-tile layer**: the operator picks one donor tile (from a tile grid of the donor map) and
  one target position on the base map. The rest of the base map is untouched.

Both are visible on the minimap immediately — a whole-map layer as a colored footprint, a
single-tile layer as a colored tile rect — even before terrain compositing.

**Independent Test**: Add a single donor tile from a map with no overlap with the base; confirm its
rect is visible on the minimap at its true coordinates and, when dropped onto a base tile, that base
tile renders the donor tile's content.

**Acceptance Scenarios**:

1. **Given** a loaded base map, **When** any map is added as a layer, **Then** its footprint is
   drawn on the minimap at its own coordinates.
2. **Given** a donor map's tile grid, **When** the operator picks one tile and a target, **Then**
   exactly that base tile composes the donor tile's terrain/placements per channel gates, and no
   other base tile changes.
3. **Given** a map that cannot be resolved, **When** added, **Then** the layer row states the
   failure inline.
4. **Given** a single-tile placement whose donor tile does not exist, **When** committed, **Then**
   the placement is rejected with a visible reason (never a silent no-op).

### User Story 2 - Drag layers and tiles on the minimap to align (Priority: P1)

Whole-map footprints and individual tile placements are draggable on the minimap. Dragging updates
offsets/placements live; the 3D view recomposes on release.

**Independent Test**: Drag a layer footprint N tiles; confirm 3D composition shifts by exactly N
tiles and the numeric fields reflect the drag. Drag a single-tile placement; confirm only that
placement moves.

**Acceptance Scenarios**:

1. **Given** a layer footprint, **When** dragged, **Then** the offset updates live and composition
   re-streams once on release.
2. **Given** a single-tile placement, **When** dragged, **Then** only that placement's target moves.
3. **Given** a drag off the 64×64 grid, **Then** the content clamps and the fields clamp to ±63.
4. **Given** multiple layers, **Then** each has its own color and is independently draggable; the
   selected one is highlighted.

### User Story 3 - Transform tools on any layer or tile selection (Priority: P1)

Rotate (90° presets, 45°, free-angle), mirror, and copy/paste operate on the current selection —
a whole layer, a tile set, or a single tile — reusing the validated Spec 219 transform seam and
Spec 208's channel reconciliation. This is the consolidation of the chunk manipulator's promised
functionality, done once, on the minimap canvas.

**Independent Test**: Select tiles from a donor layer, rotate the selection 90°, drop it on a target
region; confirm terrain, normals, holes, texturing, and placements all rotate as a unit and land
where predicted.

**Acceptance Scenarios**:

1. **Given** a selection (layer, tiles, or chunks), **When** a 90° rotation is applied, **Then** all
   channels rotate as one rigid whole about the selection's origin.
2. **Given** a copied selection, **When** pasted at a target, **Then** every channel the source
   carries is applied per the channel gates — the Spec 195 defect (partial paste) does not recur.
3. **Given** a 45° or free-angle transform, **Then** the viewer reports the approximation used
   (grid-snapped vs resampled) rather than silently resampling.
4. **Given** any transform, **Then** placements move and rotate with their terrain.

### User Story 4 - Per-layer channel and state controls (Priority: P2)

Each layer row keeps: enable toggle, per-channel checkboxes, presence gating, offsets, rotation/
mirror, per-tile placements, ordering, removal — relocated into Cartography's expandable rows, with
inline state badges (active / non-overlapping / unresolved).

**Acceptance Scenarios**:

1. **Given** a layer, **When** expanded, **Then** every control from today's Phase Map Layers panel
   is present and functional.
2. **Given** any edit, **Then** tiles re-stream exactly as `RefreshPhaseLayers` does today.
3. **Given** the old surfaces (left-sidebar phase panel, chunk manipulator UI), **When** Cartography
   ships its replacement, **Then** they are removed — one manipulation surface, not three.

### User Story 5 - Era-agnostic, multi-layer composition (Priority: P1)

Layers work on Alpha and Standard bases alike, and **multiple layers stack** — not a single phase —
with deterministic order (later wins shared channels), exactly as Spec 203 defined.

**Acceptance Scenarios**:

1. **Given** an Alpha base, **When** the add/drag/align flow runs, **Then** composition matches the
   channel gates.
2. **Given** a Standard base, **Then** the same workflow behaves identically.
3. **Given** three layers stacked, **Then** order is editable and the composed result is
   deterministic and reported.

### User Story 6 - Retire the old surfaces (Priority: P2)

When Cartography's replacement covers a surface's functionality, that surface is removed in the same
change: the left-sidebar Phase Map Layers panel, the chunk manipulator's parallel selection/clipboard/
paste state (Spec 195's UI, per 219's plan), and any right-sidebar panel whose function Cartography
absorbs.

**Acceptance Scenarios**:

1. **Given** the shipped Cartography, **Then** the left-sidebar Phase Map Layers panel no longer
   exists.
2. **Given** the chunk manipulator's capabilities, **Then** each is either present in Cartography or
   explicitly listed as deferred with an owner spec — nothing silently vanishes.
3. **Given** saved settings referencing layer stacks, **Then** they keep working (the layer model is
   extended, not forked).

## Functional Requirements

- **FR-1**: Cartography lives in the right sidebar and is the single surface for map/tile
  composition and manipulation.
- **FR-2**: Layers are added by map choice with no numeric input required to see the footprint.
- **FR-3**: Single-tile placements: pick any donor tile, drop on any target tile; multiple placements
  per layer; a placement claims its target over the whole-layer offset (Spec 219 semantics).
- **FR-4**: Footprints and tile placements are drawn on the minimap (per-layer color, selected
  highlight, non-overlapping badge) and are draggable; composition re-streams on drag release only.
- **FR-5**: Non-overlapping layers are visible and offer one-click alignment onto base occupied
  tiles (centroid or camera-tile target).
- **FR-6**: Unresolvable maps and missing donor tiles show inline errors; no silent no-ops.
- **FR-7**: Transform tools (rotate 90/45/free, mirror, copy/paste) operate on the current selection
  and apply ALL channels the source carries, gated by the layer's channel checkboxes.
- **FR-8**: Layer ordering, enable, removal, and per-layer controls behave as today.
- **FR-9**: Minimap interaction contract: click-without-drag on base terrain teleports; click on a
  footprint selects its layer; drag moves it. Teleport behavior is never broken.
- **FR-10**: Composition semantics (channel gates, presence gating, placement ownership, name-table
  remapping) are unchanged; Cartography changes WHERE and HOW content is manipulated, not what a
  channel means.
- **FR-11**: Old surfaces (left-sidebar phase panel; chunk manipulator UI once parity is met) are
  removed in the same change that lands their replacement.

## Success Criteria

- Adding and aligning a non-overlapping map — or a single donor tile — to a target takes under 30
  seconds with no manual offset arithmetic and no log reading.
- Every layer/placement state (unresolved, non-overlapping, missing tile, active, disabled) has a
  visible representation; zero silent no-op states remain.
- The same gesture workflow works on Alpha and Standard bases.
- Exactly one map-manipulation surface exists; the chunk manipulator and phase panel are gone.
- A tile selection can be copied, rotated 90°, and pasted with all channels intact (Spec 195's
  partial-paste defect cannot recur).

## Key Entities

- **Cartography Layer**: a donor map + enabled flag + channel gates + transform (offset, rotation,
  mirror) + per-tile placements + resolution state + footprint. Extends `PhaseLayerSettings`.
- **Tile Placement**: one donor tile → one target tile, with its own transform contribution.
- **Footprint**: the tile set a layer occupies, drawn on the minimap and draggable.
- **Selection**: the current layer/tile/chunk selection that transform tools operate on — the same
  selection object across minimap and 3D, replacing the chunk manipulator's parallel state.
- **Alignment**: a computed offset that moves a footprint onto base occupied tiles.

## Assumptions

- The minimap is the alignment canvas for this spec; Spec 219's full orthographic workbench (US6)
  remains a separate future feature.
- Cross-era layering (Alpha over Standard) is out of scope; layers resolve through the same data
  source as the base map.
- Destructive re-authoring (saving a transformed map) remains Spec 219's "Save Transformed Map"
  action; Cartography composes with it but does not re-implement it.
- The 2026-09-04 phase diagnostics stay as the fallback evidence trail.

## Dependencies

- Spec 203 (composition semantics), 219 (transform seam + workbench scope this consolidates),
  208 (cross-map tile transplant — channel reconciliation already done), 137/135 (minimap overlay),
  212 (right-sidebar surface), 195 (retired by this consolidation).

## Open Questions (settle before Phase 3, non-blocking for Phase 1)

1. **Q1 — Selection object model**: does Cartography adopt Spec 219's selection model directly, or
   wrap it? (Default: wrap — 219's seam is validated; wrapping avoids a third selection model.)
2. **Q2 — Chunk granularity in v1**: tiles first with chunk-level selection deferred to a fast
   follow, or chunk selection from the start? (Default: tiles first; chunks follow once tile drag
   is proven — the 195 lesson is breadth before correctness.)
3. **Q3 — 3D-side selection integration**: does the 3D click-selection (Spec 210/211) feed
   Cartography selections in v1? (Default: read-only highlight of the selected layer's tiles in v1;
   full 3D tile picking deferred.)