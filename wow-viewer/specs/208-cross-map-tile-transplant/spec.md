# Feature Specification: Cross-Map Tile Transplant (Pre-Alpha Restoration)

**Feature Branch**: `208-cross-map-tile-transplant`

**Created**: 2026-09-02

**Status**: Draft

**Input**: Operator, 2026-09-02: "we're using the phase maps feature to do non-phased map exploration
and restoration of data from dungeon maps that predate the 'overworld' main maps by about 2 years…
overlay the main world maps with the dungeon/raid maps that are older, and then get a cheap and free
restoration of earlier map data and object placements… instead of haphazard chunk copy/paste hell
that everyone else does with Noggit… We DO need to be able to do partial tile offsets, for later
content where they copy/pasted maps and rotated/mirrored the terrain… overlay only select tiles from
a 'phase map', or select specific tiles from one map to another… a 64x64 grid of checkboxes with each
tile's existing minimap as the guide."

## What this actually is

**This is not phasing.** The phase-layer machinery has been serving as a stand-in for a different
tool: **transplanting terrain and object placements from one map into another**, across maps that are
years apart in authoring time.

Instance maps — dungeons and raids — were built from copies of the overworld and then diverged. Many
of them preserve a **snapshot of the overworld's terrain from roughly two years earlier** than the
shipped overworld. Grafting those regions back onto the main maps recovers earlier world state that
exists nowhere else.

The alternative practice — manual chunk copy/paste in Noggit — is what this replaces. The operator's
objection is specific and it drives the requirements: it is **computationally expensive** and **not as
accurate as this codebase can be**, because it is done by eye instead of against the structured data
the harvester already extracts.

This is the primary tool of the **Pre-Alpha Restoration Project**, the standing goal behind the
project as a whole.

## Most of the engine already exists — do not rebuild it

**Spec 195 (Overhead Chunk Manipulator, marked complete) already delivers the transposition engine.**
`ChunkTranspositionOptions` carries `RotationDegrees` (0/90/180/270), `MirrorX`, `MirrorY`,
`RelativeHeights` + `HeightOffset` floor anchoring, `OverwriteDestination`, and per-attribute includes
(heights, textures, holes, liquid, vertex shading, M2 placements, WMO placements).
`ChunkTranspositionService` splits into `ExtractPayload` → `TransformPayload`, which is exactly the
extract/transform/apply shape a transplant needs. It operates on a unified **global chunk lattice**
`(Gx, Gy) ∈ [0,1023]²` where `Gx = Tx*16 + Cx`, so offsets are already expressed at **chunk
granularity — one sixteenth of a tile** — with no tile-boundary discontinuity.

Spec 203/207 contributes the other half: `PhaseDataChannel` and `PhaseSignalChannelMap`, the
channel model and its guard that every signal is classified.

**So this spec is deliberately small in engine terms.** What is missing is:

1. **Cross-map sourcing.** 195 transposes within one loaded map. A transplant reads from a *different*
   map than it writes to.
2. **Per-tile selection at map scale**, with the minimap as the guide.
3. **Provenance.** For an archival restoration, "where did this tile come from" is part of the
   output, not a nicety.

### One thing to reconcile, not duplicate

There are now **two** models for "which kinds of data to carry": `ChunkTranspositionOptions`'s
booleans (spec 195) and `PhaseDataChannel` (spec 203/207). Constitution II says one mechanism.
**This spec must map one onto the other, not introduce a third.** `PhaseDataChannel` is the better
carrier — it is a flag enum, it has a name-keyed signal map with a completeness guard, and it already
reaches the harvest signals the datastore stores.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Transplant selected tiles from one map onto another (Priority: P1)

The user picks a source map, picks which of its tiles to take, picks a target map, and applies the
selected tiles' terrain and placements onto the target.

**Why this priority**: It is the feature. Everything else refines it.

**Independent Test**: Transplant one known tile from an instance map onto a main map and confirm the
target shows the source's terrain at the target's coordinates, with the unselected channels untouched.

**Acceptance Scenarios**:

1. **Given** a source map and a target map, **When** the user selects source tiles and applies,
   **Then** only those tiles are affected on the target.
2. **Given** a channel is unticked, **When** the transplant applies, **Then** the target keeps its own
   data for that channel.
3. **Given** the source tile carries nothing for an enabled channel, **When** it applies, **Then** the
   target's data survives rather than being blanked (the presence gate from spec 203).
4. **Given** a transplant has been applied, **When** the user undoes it, **Then** the target returns
   to its previous state exactly.

---

### User Story 2 - A 64×64 tile picker guided by the minimap (Priority: P1)

A pop-out window shows the source map as a 64×64 grid, each cell drawn with that tile's existing
minimap, and each cell individually selectable.

**Why this priority**: Selecting tiles by number is unusable at map scale; the minimap is how a human
recognises the region worth taking. Without this the feature is technically present and practically
unusable.

**Independent Test**: Open the picker on a map with known minimap coverage, select a visually
identifiable region, and confirm the selection matches the tiles that were clicked.

**Acceptance Scenarios**:

1. **Given** a source map, **When** the picker opens, **Then** each existing tile shows its minimap
   and tiles absent from the map are visibly distinct from unselected ones.
2. **Given** the user drags across cells, **When** the drag ends, **Then** every crossed cell toggles
   as a group rather than one at a time.
3. **Given** a minimap tile is missing, **When** the grid renders, **Then** the cell is still
   selectable and marked as having no preview — a missing preview is not a missing tile.
4. **Given** a selection exists, **When** the user switches source map, **Then** the selection is not
   silently carried onto a different map's coordinates.

---

### User Story 3 - Rotation, mirroring and sub-tile offset (Priority: P2)

A transplant can be rotated by 90/180/270, mirrored, and placed at an offset finer than one tile.

**Why this priority**: Later content was produced by copying, rotating and mirroring existing terrain
to disguise its origin. Reversing that transform is what makes those regions recoverable at all. P2
only because US1 must work before a transform on it means anything.

**Independent Test**: Take a region known to be a rotated copy, apply the inverse rotation, and
confirm it aligns with the original within a stated tolerance.

**Acceptance Scenarios**:

1. **Given** a rotation of 90/180/270, **When** applied, **Then** heights, normals, textures, holes
   and placements are all rotated consistently — no channel is left in the original orientation.
2. **Given** a mirror, **When** applied, **Then** normals and placement rotations are mirrored too,
   not just vertex positions.
3. **Given** an offset that is not a whole number of tiles, **When** applied, **Then** the payload
   lands at chunk granularity and spans destination tiles correctly.

---

### User Story 4 - Every restored tile records where it came from (Priority: P2)

The result carries, per tile, which source map and source tile it came from, which channels were
taken, and what transform was applied.

**Why this priority**: This is a restoration project. A recovered tile whose origin is unrecorded is
an assertion, not evidence, and cannot be reviewed or reproduced later.

**Independent Test**: Apply several transplants from different sources and read back a per-tile
account of origin, channels and transform.

**Acceptance Scenarios**:

1. **Given** a transplant, **When** it is applied, **Then** the source map, source tile, channels and
   transform are recorded against the target tile.
2. **Given** a tile has been transplanted more than once, **When** provenance is read, **Then** the
   order of operations is recoverable, not just the last one.
3. **Given** provenance exists, **When** the result is exported, **Then** provenance is exported with
   it rather than being a UI-only artifact.

---

### Edge Cases

- A source tile that exists in the source map but has no counterpart tile in the target.
- Source and target maps with different texture tables — MCLY indices must be re-mapped, not copied.
- A rotation that is not a multiple of 90.
- Transplanting a tile onto itself (same map, same coordinates).
- A source map whose minimaps do not exist, so the picker has no preview (measured: 0.5.3 has no loose
  minimap directory, only `md5translate.txt`).
- Overlapping selections applied in sequence.
- Height discontinuity at the seam between a transplanted tile and its untouched neighbour.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A transplant MUST read from a source map that is not the target map.
- **FR-002**: The user MUST be able to select an arbitrary subset of the source map's tiles.
- **FR-003**: Channel selection MUST use `PhaseDataChannel`; `ChunkTranspositionOptions`'s booleans
  MUST be derived from it rather than maintained separately (Constitution II).
- **FR-004**: A channel the source does not carry MUST leave the target's data intact.
- **FR-005**: Transplants MUST support rotation by 90/180/270 and mirroring on either axis, applied
  consistently to every carried channel including normals and placement rotations.
- **FR-006**: Offsets MUST be expressible at **chunk granularity** (1/16 tile) using spec 195's global
  chunk lattice.
- **FR-007**: Every applied transplant MUST record source map, source tile, channels and transform
  against the target tile, and that record MUST survive export.
- **FR-008**: Transplants MUST be undoable through `EditorSession`.
- **FR-009**: The tile picker MUST render each source tile's minimap where one exists, and MUST
  distinguish "no tile" from "tile with no preview" from "unselected".
- **FR-010**: Texture layer indices MUST be re-mapped into the target's texture table; copying raw
  indices across maps is prohibited.
- **FR-011**: The result MUST be reviewable before it is written — a transplant is a proposal until
  applied.
- **FR-012**: No operation may silently drop a selected tile. A tile that could not be transplanted
  MUST be reported with the reason.

### Key Entities

- **Source map / target map**: the donor and the recipient. Distinct by definition.
- **Tile selection**: a set of source tile coordinates, expressed on the 64×64 grid.
- **Transplant payload**: spec 195's `ChunkTranspositionPayload`, extracted from the source.
- **Transform**: rotation, mirror, and a chunk-granular offset.
- **Provenance record**: source map, source tile, channels, transform, order.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A tile transplanted from an instance map onto a main map renders on the target with the
  source's terrain, verified by pixel comparison against the source tile rendered in isolation.
- **SC-002**: Unticked channels are provably untouched — the target's data for those channels is
  byte-identical before and after.
- **SC-003**: A 90/180/270 rotation applied and then inverted returns the region to within a stated
  tolerance of its original heights.
- **SC-004**: Every applied transplant has a provenance record; **zero** tiles are modified without
  one.
- **SC-005**: Undo restores the target byte-identically.
- **SC-006**: The picker renders a 64×64 grid for a full map without dropping frames below the
  interactive threshold, with previews loaded lazily.
- **SC-007**: A selected tile that cannot be transplanted is reported with a reason; zero silent
  skips.

## Out of Scope

- **Automatic discovery of which maps are worth transplanting.** This spec provides the tool; deciding
  that a given dungeon preserves older terrain is the operator's judgment, informed by specs 194/196.
- **Automatic detection of rotated/mirrored copies.** The operator supplies the transform. Detecting
  it is a candidate follow-on and is where `project_weak_tile_jigsaw`'s edge-agreement approach would
  apply.
- **Seam blending between transplanted and untouched tiles.** Spec 196 owns neighbour mesh fitting.
- **WLW/MCLQ liquid convergence** — spec 209.
- **Storage.** Composed output goes to the datastore via the existing ARRY handoff; Python owns the
  store (`feedback_python_owns_the_datastore`).

## Dependencies

- **Spec 195** — the transposition engine, global chunk lattice, rotate/mirror, undo/redo. This spec
  extends it to cross-map sourcing and must not fork it.
- **Spec 203/207** — `PhaseDataChannel` and `PhaseSignalChannelMap`, the channel model.
- **Spec 206** — the datastore handoff for exporting composed tiles.
- **Spec 196** — neighbour mesh fitting, for seams, if the operator wants them addressed.

## Assumptions

- **Instance maps preserve older overworld terrain.** This is the operator's premise and the reason
  for the tool. It is not re-litigated here; the tool makes it testable at scale.
- **Chunk granularity satisfies "partial tile offsets."** Spec 195's lattice gives 1/16-tile precision.
  If the operator means sub-*chunk* precision, FR-006 changes and the engine needs interpolation,
  which would be a materially larger piece of work.
- **Minimap previews are best-effort.** Measured: 0.5.3 has no loose minimap directory, so the picker
  must be usable without previews for some maps.
- **The target is written as a new artifact, not in place over client files.** Restoration output is
  reviewable and reversible.
