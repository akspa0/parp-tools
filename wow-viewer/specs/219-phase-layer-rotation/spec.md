# Feature Specification: Map Composition Selection & Transform Workbench

**Feature Branch**: `219-phase-layer-rotation`

**Created**: 2026-09-03

**Status**: Implementing — Phase 1 Core transform seam validated; workbench UI/runtime/export scope open

**Input**: Operator, 2026-09-03, on needing to rotate a phase map's whole tile set to fit the base
map: "I need to be able to rotate the whole set of tiles by 45 degrees, clockwise, from North to
North East, or in counter clockwise from North to West, as rotate tools for the tiles, including
all the objects if any apply. We need to be able to rotate the phase maps to fit the bill."

## Context

### What exists today

Spec 203's phase-layer composition supports **translation only**: a layer can be shifted in tile
space (`TileOffsetX`/`TileOffsetY`), and its placements are translated by the matching world
distance. There is no rotation of any kind. A phase map whose content was authored at a different
orientation than the base map cannot be aligned.

The separate Chunk Manipulator claims an overhead multi-tile selection workflow, but its current
surface is a small camera-centered dark grid rather than an orthographic whole-map view. It does
not show a minimap or heightmap, does not provide the claimed drag-box interaction, chooses paste
location from the camera tile rather than an explicit target, and its paste path does not apply all
of the channels exposed by its own controls. Maintaining this as a second selection and transform
system would preserve the defects and make phase composition drift from editing.

The Phase Map Layers panel has per-phase channel and tile-offset controls, but no base-map channel
controls and no phase rotation/mirror controls. Save/export exists elsewhere as disconnected
terrain/conversion actions rather than as an explicit save of the composition currently previewed.

### Why rotation, and why these angles

The operator aligns donor maps (instance/dungeon revisions of a zone, or cross-era restorations)
onto a base map. Some donors were authored rotated relative to the base — the content is the right
content at the wrong heading. Two rotations recur:

- **45° clockwise** (North → North-East): the half-step case.
- **90° counter-clockwise** (North → West): the quarter-turn case.

Both are named as first-class tools rather than buried in a free-angle field, because they are the
operations the operator actually reaches for. A free-angle input is still required for the cases in
between.

### The hard part is not the math

Rotating terrain is not a rigid transform of a mesh: each tile is a 16×16 grid of chunks, each
chunk a 9×9 lattice of height vertices with per-vertex normals, texture layers, alpha maps, hole
masks, and shadow maps, plus world-coordinate placements. A rotation must move **whole tiles** to
new grid positions (a 45° rotation does not land on the tile grid at all), rotate **within** tiles
where the angle permits, rotate **normals** with the terrain, rotate **placement positions and
rotations** together, and keep **liquid surfaces** consistent with the heights they sit on. What
happens to content that rotates off the tile grid, and whether the rotation is a live preview or a
destructive re-authoring, are the two decisions that shape everything else.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Quarter-turn rotation aligns a phase map (Priority: P1)

The operator picks a 90° rotation (clockwise or counter-clockwise) for a phase layer and sees the
layer's terrain, normals, and all placements rotated as one rigid whole, so a donor map authored a
quarter-turn off now aligns with the base map.

**Why this priority**: Quarter turns are lossless on a square tile grid — tiles map to tiles, chunk
lattices map to chunk lattices, and no resampling is required. This is the rotation that can be
made exact, so it is the one that must work first and perfectly.

**Independent Test**: Load a phased map with a 90° rotation applied to one layer and confirm the
layer's terrain features and objects appear rotated as a unit and land where the rotation predicts.

**Acceptance Scenarios**:

1. **Given** a phase layer with terrain and placements, **When** a 90° clockwise rotation is
   applied, **Then** every terrain feature and object appears rotated 90° clockwise about the
   layer's rotation origin, and the layer still composes over the base map per its channel
   settings.
2. **Given** the same layer, **When** a 90° counter-clockwise rotation is applied instead, **Then**
   the result is the mirror operation — applying one and then the other returns the unrotated
   arrangement.
3. **Given** a rotated layer, **When** the rotation is removed, **Then** the composed map matches
   the unrotated composition exactly.

---

### User Story 2 - 45° rotation is available and honest about its cost (Priority: P2)

The operator picks a 45° clockwise rotation (North → North-East) and the layer rotates; where the
45° result cannot land on the tile grid, the viewer says so rather than silently resampling.

**Why this priority**: The operator named 45° explicitly, but a 45° rotation of an axis-aligned
tile grid produces content that no longer aligns to the grid — tiles land between tiles. The
feature must deliver a usable 45° result and be explicit about what approximation it used, because
a silently resampled heightfield is indistinguishable from correct data until it is trusted.

**Independent Test**: Apply a 45° rotation to a phase layer and read the reported rotation mode;
confirm the visual result matches the reported mode (grid-snapped or free-rotated) and that
placements sit on their rotated terrain.

**Acceptance Scenarios**:

1. **Given** a phase layer, **When** a 45° clockwise rotation is applied, **Then** the layer's
   content appears rotated 45° clockwise about the rotation origin, and the viewer reports which
   approximation was used.
2. **Given** a 45°-rotated layer whose terrain was resampled or grid-snapped, **When** the operator
   inspects the layer, **Then** the applied rotation and its approximation are visible without
   opening logs.
3. **Given** a 45° rotation, **When** placements are rendered, **Then** each object's position and
   orientation are rotated by the same 45° about the same origin, so objects stay seated on the
   terrain they were placed on.

---

### User Story 3 - Free-angle rotation for fine alignment (Priority: P3)

The operator enters an arbitrary angle for a phase layer and the layer rotates by that angle about
a chosen origin, with the same placement/terrain consistency as the preset turns.

**Why this priority**: Named presets cover the common cases; the free angle covers everything else
and is the general form the presets are shortcuts for. It is deprioritised because the preset turns
deliver the operator's stated need on their own.

**Independent Test**: Enter an arbitrary rotation angle on a phase layer and confirm terrain and
placements rotate together by exactly that angle about the stated origin.

**Acceptance Scenarios**:

1. **Given** a phase layer and an arbitrary angle, **When** the rotation is applied, **Then** the
   composed map shows the layer rotated by that angle about the rotation origin.
2. **Given** two layers with different rotations, **When** both are active, **Then** each rotates
   independently and the composition order remains deterministic and reported.

---

### User Story 4 - Rotation composes with the existing layer controls (Priority: P2)

Rotation is one more control on the existing Phase Map Layers panel, composes with tile offsets and
channel selection, and the minimap reflects the rotated layer the same way it reflects offsets
today.

**Why this priority**: A rotation that only works in isolation is not usable — the operator aligns
a map with both a shift and a turn. This story is what makes rotation a *tool* rather than a demo.

**Independent Test**: Apply an offset and a rotation to the same layer and confirm the composed
result equals the rotation applied about the origin followed by (or composed with) the offset, in a
stated, deterministic order.

**Acceptance Scenarios**:

1. **Given** a layer with both an offset and a rotation, **When** the map loads, **Then** the
   result is deterministic and the applied transform order is stated in the UI or logs.
2. **Given** a rotated layer, **When** the minimap is viewed, **Then** the layer's tiles appear at
   their rotated positions, consistent with the 3D terrain.
3. **Given** a rotation applied to a layer whose channels are partially enabled, **When** composed,
   **Then** only the enabled channels are rotated and contributed, exactly as the channel gates
   behave for offsets today.

---

### User Story 5 - Per-tile placement: pick any tile, drop it on any tile (Priority: P1)

The operator picks any individual tile from the layer's 64×64 grid and drops it onto any individual
tile of the base map — one tile at a time, or many — without moving the rest of the layer. This is
the fine-grained counterpart to the whole-layer offset: where an offset moves every tile by the
same amount, per-tile placement lets the operator cherry-pick exactly which donor tile fills which
base tile.

**Why this priority**: Equal to the quarter-turn. Whole-layer offsets and rotations move
*everything*; the operator frequently needs only specific tiles from a donor map (a single keep, a
ruined district, one coastline) placed at specific spots. Without per-tile placement the only
option is authoring a whole second map, which is exactly the work this feature exists to avoid.

**Independent Test**: Map one donor tile to one base tile position, load the map, and confirm the
base tile shows the donor tile's terrain and placements at the target position, with all other
base tiles unchanged.

**Acceptance Scenarios**:

1. **Given** a phase layer and a chosen donor tile, **When** the operator drops it onto a target
   base tile, **Then** the target tile renders the donor tile's terrain, placements, and other
   channel content at the target position, and no other base tile changes.
2. **Given** several per-tile placements on one layer, **When** the map loads, **Then** each mapped
   tile lands at its own target and the layer's unmapped tiles behave exactly as the whole-layer
   offset rules dictate today.
3. **Given** a per-tile mapping, **When** it is removed, **Then** the composed map matches the
   composition without that mapping exactly.
4. **Given** a donor tile dropped onto a target that already has content, **When** composed,
   **Then** the per-channel gates decide what the dropped tile contributes, exactly as tile-level
   composition behaves for offsets today.
5. **Given** a per-tile placement combined with a layer offset or rotation, **When** composed,
   **Then** the result is deterministic and the applied order is the same stated order as
   US4's.

---

### User Story 6 - Select source and target content on an orthographic whole-map canvas (Priority: P1)

The operator opens one map-composition workbench and sees the entire map from directly overhead,
using either the available minimap imagery or a heightmap-derived backdrop. The operator switches
between tile, chunk, and terrain-cell selection grids, then clicks colored squares or paints,
drags, toggles, and box-selects regions. Every gesture snaps magnetically to the active grid and
clearly distinguishes source selection, target preview, conflicts, empty cells, and committed data.

**Why this priority**: Selecting the correct data is the first operation in every transform. A
small abstract grid with camera-derived paste targets makes the technically correct transform seam
unusable and error-prone.

**Independent Test**: Starting from the full-map view, select a non-rectangular set spanning several
ADT boundaries, change grid granularity, assign an explicit target, and verify the source and target
remain visually distinct and resolve to the coordinates displayed by the workbench.

**Acceptance Scenarios**:

1. **Given** a loaded map, **When** the workbench opens, **Then** all 64×64 tile positions fit in an
   orthographic overview and existing, missing, selected, targeted, conflicting, and transformed
   regions have distinct colors plus a legend.
2. **Given** tile granularity, **When** a tile square is clicked or checked, **Then** the full tile is
   selected; **given** chunk or cell granularity, the same action selects exactly that smaller unit.
3. **Given** an active grid, **When** the operator drags a box, paints, or lassos across boundaries,
   **Then** the selection snaps to that grid and can add, subtract, replace, or toggle units.
4. **Given** a source selection, **When** the operator chooses a destination on the canvas, **Then**
   a colored target preview appears before any composition or edit is committed.
5. **Given** no minimap imagery, **When** the canvas opens, **Then** a heightmap or occupancy
   backdrop remains usable and the absence of imagery is stated.

---

### User Story 7 - Use the same selection in the 3D renderer (Priority: P1)

The operator can activate a configurable terrain-selection mode in the 3D renderer and click,
paint, or drag over visible terrain to select tiles, chunks, or cells. The 3D overlay uses the same
selection and source/target colors as the orthographic canvas; edits made in either view appear in
the other immediately.

**Why this priority**: Fine terrain decisions often require seeing slopes, objects, liquids, and
seams in context. Noggit-style in-world chunk selection is the practical complement to the map-wide
orthographic overview.

**Independent Test**: Select chunks in 3D, confirm them on the orthographic canvas, subtract one on
the canvas, then confirm the exact same region updates in 3D without conversion or re-selection.

**Acceptance Scenarios**:

1. **Given** 3D selection mode, **When** the cursor hits terrain, **Then** the tile/chunk/cell under
   the cursor is highlighted before selection and the active granularity is visible.
2. **Given** an existing orthographic selection, **When** the operator enters 3D selection mode,
   **Then** the same units are outlined or tinted on terrain with no second selection state.
3. **Given** UI chrome or object-picking mode, **When** the operator clicks outside selectable
   terrain, **Then** the terrain selection does not change accidentally.
4. **Given** dense terrain, **When** the operator paints a selection in 3D, **Then** hover and drag
   feedback remain responsive and do not require synchronous loading of the whole map.

---

### User Story 8 - Control base and phase layers from one composition stack (Priority: P1)

The operator sees the base map as a fixed first layer and every phase map above it. The base exposes
the same terrain, texturing, liquid, doodad, WMO, and Area ID channel controls as phase layers.
Every phase card exposes named rotation and mirror controls directly: 90° CW/CCW, 45° CW/CCW,
Mirror H/V, free angle, rotation origin, and reset.

**Independent Test**: Disable base terrain and objects while leaving a phase enabled, rotate that
phase from its own card, and verify both canvas and 3D preview show phases-only content with the
reported transform.

**Acceptance Scenarios**:

1. **Given** the base layer, **When** a base channel is disabled, **Then** that channel is absent
   before phase layers compose; this is not only a final render-pass hide.
2. **Given** any phase card, **When** a named rotation, mirror, free angle, origin, or reset control
   is used, **Then** that phase updates independently in the canvas and 3D preview.
3. **Given** the stack, **When** Base Only, Phases Only, All, None, Terrain, or Objects preset is
   chosen, **Then** the resulting channel state is explicit and reversible.
4. **Given** a non-removable base layer, **When** all its channels are disabled, **Then** the map
   identity remains available and phases can still be inspected and transformed.

---

### User Story 9 - Replace the broken chunk manipulator with the shared workbench (Priority: P1)

The operator performs tile/chunk/cell copy, move, rotate, mirror, offset, target preview, and
undo/redo through the same workbench used by phase layers. The old Chunk Manipulator entry may
redirect to this page during transition, but it does not keep separate selection, clipboard,
transform, channel, or paste-target state.

**Independent Test**: Open the legacy Chunk Manipulator entry and the composition workbench,
perform a selection through each entry, and verify both operate on the same selection/operation
state and produce one undoable result with no duplicate controls.

---

### User Story 10 - Save the previewed composition to ADT or Alpha WDT output (Priority: P1)

The operator presses **Save Transformed Map** on the workbench page, previews the affected tiles,
output format, disabled/lossy channels, conflicts, and destination, then writes new output files.
The loaded source files are not overwritten by default.

**Independent Test**: Save one transformed composition as supported ADT output and one as Alpha WDT,
reload each output, and confirm the saved channel/transform state matches the preview. Requesting an
unsupported native format is refused before writing.

**Acceptance Scenarios**:

1. **Given** a dirty preview, **When** Save Transformed Map is pressed, **Then** a preflight lists
   target format, output location, affected tiles, channel omissions, mapping conflicts, and
   lossy/unsupported data before confirmation.
2. **Given** a supported ADT or Alpha WDT target, **When** save succeeds, **Then** output copies are
   written and the exact paths and validation result are reported.
3. **Given** an unsupported target such as native split MoP output, **When** save is requested,
   **Then** the action is disabled or refused with a specific explanation and no mislabeled fallback.
4. **Given** an existing output, **When** overwrite is requested, **Then** explicit confirmation is
   required; original client files are never silently modified.

---

### Edge Cases

- A rotation that moves a layer's content off the 64×64 tile grid entirely.
- A 45° rotation of a layer whose tiles are not centred on the rotation origin, so the rotated
  bounding region differs from the original.
- Placements whose rotated position falls outside any loaded tile.
- Rotating a layer that supplies tiles the base map does not have (phase-only tiles).
- Rotation combined with `OnlyTakeWhatThePhaseCarries` presence gating.
- 0.5.3 behaviour must not change for layers with no rotation set.
- Very large maps: rotation must not require loading every tile of the layer at once.
- A per-tile placement whose donor tile does not exist in the layer's WDT (empty source slot).
- Two per-tile placements targeting the same base tile — resolution must be deterministic and
  reported.
- A per-tile placement combined with a whole-layer offset, where the two mappings disagree about
  which donor tile fills a target.
- Per-tile placements whose placements (objects) straddle the donor tile's boundary.
- A selection spanning empty/unloaded tiles and existing tiles.
- Switching selection granularity while a mixed, non-rectangular selection exists.
- Selecting a terrain cell shared by neighboring chunk boundaries without duplicating the sample.
- Canvas and 3D selection gestures competing with camera navigation, object picking, or UI capture.
- Base channels disabled while phases supply only a subset of the missing channels.
- Saving a composition with unresolved target conflicts, off-map content, unsupported channels, or
  a target format that cannot represent the preview.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The operator MUST be able to rotate a phase layer by 90° clockwise and 90°
  counter-clockwise as first-class, named tools.
- **FR-002**: The operator MUST be able to rotate a phase layer by 45° clockwise and 45°
  counter-clockwise as first-class, named tools.
- **FR-003**: The operator MUST be able to enter an arbitrary rotation angle for a phase layer.
- **FR-004**: A rotation MUST rotate terrain heights, normals, texture layers/alpha maps, hole
  masks, shadow maps, vertex colours, and liquid surfaces together as one rigid transform of the
  layer's content.
- **FR-005**: A rotation MUST rotate placement positions AND placement orientations by the same
  angle about the same origin, so objects remain seated on the terrain they were placed on.
- **FR-006**: Every rotation MUST be defined about a stated rotation origin (default: the centre of
  the layer's occupied tile bounds), and the origin MUST be visible to the operator.
- **FR-007**: Rotation MUST compose deterministically with tile offsets; the applied order
  (rotate-then-shift or shift-then-rotate) MUST be fixed, stated, and reported.
- **FR-008**: Removing a rotation MUST reproduce the unrotated composition exactly.
- **FR-009**: Where a rotation cannot land content on the tile grid (e.g. 45°), the viewer MUST
  state which approximation it used (grid-snapped, resampled, or free-rotated) rather than
  silently choosing one.
- **FR-010**: Rotation MUST respect the per-channel gates: a channel the layer is not permitted to
  contribute is neither rotated nor composed.
- **FR-011**: 0.5.3 behaviour MUST NOT change for layers with no rotation set.
- **FR-012**: The rotation controls MUST be legible at every supported font scale, consistent with
  the offset controls' legibility requirement.
- **FR-013**: The operator MUST be able to map any individual tile of a phase layer's 64×64 grid
  onto any individual tile of the base map (per-tile placement), one tile at a time or many.
- **FR-014**: A per-tile placement MUST bring the donor tile's full channel content (terrain,
  placements, and the other gated channels) to the target tile, subject to the same per-channel
  gates that govern whole-layer composition.
- **FR-015**: Per-tile placements MUST compose deterministically with whole-layer offsets and
  rotations under the same stated transform order, and a target tile claimed by more than one
  mapping MUST resolve deterministically with the conflict reported.
- **FR-016**: Removing a per-tile placement MUST reproduce the composition without it exactly.
- **FR-017**: The per-tile placement UI MUST let the operator see which donor tile maps to which
  target tile at a glance, and MUST be legible at every supported font scale.
- **FR-018**: The operator MUST be able to mirror a phase layer's tile content horizontally and
  vertically as first-class, named tools, transforming terrain (heights, normals, textures, alpha
  maps, holes, shadows, vertex colours, liquid) and all placements (position mirrored, orientation
  adjusted for the handedness flip) as one rigid unit.
- **FR-019**: The transform math (rotate, mirror, offset) MUST be factored into a standalone Core
  transform seam that operates on tile content without knowing about the phase system, so later
  editing tooling can reuse it directly.
- **FR-020**: Whole-layer translation MUST support exact integer terrain-cell offsets in both axes,
  not only whole ADT tiles. One ADT is 16 chunks and 128 terrain cells per axis; chunk alignment is
  therefore every 8 cells and tile alignment every 128 cells. The UI MUST expose tile, chunk, and
  cell units over this single exact lattice so terrain pasted across unrelated source/target
  boundaries can be aligned precisely.
- **FR-021**: Cell-granular translation MUST move every enabled terrain channel and all placements
  together. It MUST re-slice content at target ADT/chunk boundaries without interpolating or
  resampling authored heights. Sub-cell/world-float offsets are out of scope unless a later,
  explicitly stated resampling mode is selected.
- **FR-022**: WDL macro-lattice magnetization MAY be used as an optional alignment aid for
  cell-granular translation. It MUST reuse Spec 196's `WdlLatticeMagnetizer`/WDL sampling contract,
  report the proposed snap and fit score, and require operator acceptance; it MUST NOT silently
  change the authoritative integer cell offset or duplicate the magnetization algorithm.
- **FR-023**: The workbench MUST present an orthographic overview of the entire 64×64 map with a
  selectable minimap backdrop, heightmap backdrop, and occupancy-only fallback.
- **FR-024**: Selection MUST support tile, chunk, and terrain-cell granularity with click/check,
  paint, rectangular drag, lasso, add, subtract, replace, and toggle gestures snapped to the active
  grid. Tile granularity MUST also be usable as simple colored squares or checkboxes.
- **FR-025**: Source selection, target preview, transformed preview, empty positions, conflicts, and
  committed content MUST use distinct visible states with a legend; color MUST NOT be the only cue.
- **FR-026**: The orthographic canvas and 3D terrain selector MUST use one shared selection state.
  A selection change in either surface MUST appear in the other without conversion or duplication.
- **FR-027**: The 3D selector MUST support configurable tile/chunk/cell hover and click/paint/drag
  selection while respecting camera input, object picking, UI capture, and explicit selection mode.
- **FR-028**: The base map MUST appear as a non-removable first layer with the same channel gates as
  phase maps. Disabling a base channel MUST remove it before phase composition rather than only
  hiding the final render pass.
- **FR-029**: Every phase-layer card MUST expose 90° CW/CCW, 45° CW/CCW, Mirror H/V, free-angle,
  rotation-origin, and reset controls. Each phase transforms independently and updates both previews.
- **FR-030**: The composition stack MUST provide Base Only, Phases Only, All, None, Terrain, and
  Objects presets whose resulting base/phase channel states are explicit and reversible.
- **FR-031**: The existing Chunk Manipulator MUST be retired as an independent implementation. Any
  compatibility entry point MUST open or delegate to this workbench and MUST NOT retain separate
  selection, clipboard, channel, transform, paste-target, or undo state.
- **FR-032**: The workbench MUST support copy, move, rotate, mirror, cell-offset, target preview,
  commit, undo, and redo for selected content through the same operation model used by phase
  composition; the explicit target MUST never be inferred solely from camera position.
- **FR-033**: The workbench MUST provide a visible **Save Transformed Map** action that preflights
  output format, destination, affected tiles, omitted/lossy channels, off-map content, and conflicts
  before writing the currently previewed composition.
- **FR-034**: Save MUST support output copies in each proven representable target format and MUST
  refuse unsupported targets before writing. It MUST NOT label a compatibility/down-converted file
  as a native format and MUST NOT overwrite source client files by default.
- **FR-035**: A successful save MUST report exact output paths and validation results; a failed or
  partially representable save MUST identify every refused tile/channel and leave prior outputs in
  a known state.

### Key Entities

- **Layer rotation**: The angle and origin attached to one phase layer; zero means unrotated.
- **Rotation origin**: The point about which the layer's content rotates; defaults to the centre of
  the layer's occupied tile bounds and is overridable.
- **Rotation approximation**: The declared strategy used when a rotation does not land on the tile
  grid (grid-snap, resample, or free).
- **Composed transform**: The single stated order in which a layer's rotation and tile offset are
  applied to its content.
- **Per-tile placement**: One donor-tile → target-tile mapping on a phase layer, independent of the
  layer's whole-layer offset; a layer may carry any number of them.
- **Tile mapping conflict**: Two mappings (or a mapping and the whole-layer offset) claiming the
  same target tile; resolved deterministically and reported.
- **Unified map selection**: One set of selected tile/chunk/cell units shared by the orthographic
  canvas, 3D terrain selector, phase placement workflow, and content editing operations.
- **Base layer**: The non-removable first composition layer whose individual channels may be enabled
  or disabled before phase layers apply.
- **Transform preview**: A non-destructive source-to-target projection shown before commit or save.
- **Save preflight**: The complete report of output representability, affected content, conflicts,
  and destination that must pass before output files are written.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A 90° rotation of a phase layer renders terrain, normals, and placements rotated as
  one rigid unit, confirmed by operator visual comparison against the donor map.
- **SC-002**: Applying a 90° clockwise rotation and then a 90° counter-clockwise rotation to the
  same layer reproduces the unrotated composition, verified by pixel comparison.
- **SC-003**: A 45° rotation produces a usable result whose approximation mode is reported in the
  UI or logs, not silently chosen.
- **SC-004**: An arbitrary-angle rotation keeps placements seated on the terrain they were placed
  on (object/terrain relative positions preserved to within visual tolerance), confirmed by
  operator inspection.
- **SC-005**: Rotation composes with tile offsets in a stated, deterministic order, and the
  composed result is identical across reloads.
- **SC-006**: A map with no rotation set renders pixel-identical to before this feature.
- **SC-007**: A single donor tile dropped onto a target base tile renders the donor tile's content
  at the target position with all other base tiles unchanged, confirmed by operator visual
  comparison.
- **SC-008**: A per-tile placement combined with a layer offset/rotation produces the same composed
  result across reloads, and a target-tile conflict is reported rather than silently resolved.
- **SC-009**: A horizontal mirror followed by a second horizontal mirror reproduces the unmirrored
  composition, and the same holds for vertical mirrors, verified by pixel comparison.
- **SC-010**: The Core transform seam (rotate/mirror/offset on tile content) is callable without
  any phase-system type in scope, confirmed by a Core unit test that exercises the seam directly.
- **SC-011**: Offsetting by 8 cells produces the same result as one chunk offset; offsetting by 128
  cells produces the same result as one ADT tile offset; inverse offsets round-trip the full set of
  enabled terrain channels and placement coordinates exactly.
- **SC-012**: With WDL snap enabled, the viewer reports the proposed cell offset and fit score;
  declining the proposal leaves the layer byte-identical to the unsnapped composition.
- **SC-013**: An operator can select a non-rectangular region spanning at least four ADT tiles from
  the orthographic overview and assign an explicit target in under 30 seconds without entering raw
  coordinates.
- **SC-014**: For a scripted set of 100 boundary clicks/drags at tile, chunk, and cell granularity,
  every selected unit equals the displayed snapped unit, with zero off-by-one boundary selections.
- **SC-015**: A selection created in 3D and edited in the orthographic canvas produces the same
  ordered selected-unit set on both surfaces after every operation.
- **SC-016**: Disabling all base channels and leaving one phase enabled produces phases-only
  composition in both previews; restoring the prior preset reproduces the prior composition.
- **SC-017**: Every phase-layer card can perform and reset all named rotations and mirrors without
  changing any other layer's transform state.
- **SC-018**: The legacy Chunk Manipulator compatibility entry and the workbench expose one shared
  selection and one undo history; repository audit finds no second active transform/paste pipeline.
- **SC-019**: Supported ADT and Alpha WDT saves reload with the same transformed channels and
  placement locations as previewed; unsupported native output is refused with zero files mislabeled.
- **SC-020**: Orthographic and 3D hover/drag selection remains interactive at the project's target
  frame rate for loaded terrain and does not require loading all map tiles synchronously.

## Assumptions

- **The transforms are general-purpose tile-content operations, not phase-specific tricks.** The
  operator's intent is a reusable transform library — rotate, mirror horizontally, mirror
  vertically — that takes the *entire contents* of a tile (terrain heights, normals, textures,
  holes, shadows, vertex colours, liquid, and every object placement) and transforms them as one
  rigid unit. The phase map system is the first consumer; later editing tooling (beyond the phase
  system) will call the same Core transform primitives directly. The plan must therefore factor
  the math into a standalone Core transform seam that does not know about phases.
- Rotation remains non-destructive while previewing, but this expanded workbench also owns an
  explicit, preflighted export of the preview through existing proven map-output capabilities.
- Mirroring (FR-018) is exact-grid like the 90° turns: heights mirror in place, normals mirror
  with the terrain (the mirrored axis component negates), texture alpha maps mirror, and
  placements mirror position with orientation adjusted for the flip (handedness flips).
- The default rotation origin (centre of the layer's occupied tile bounds) is a reasonable default
  because the operator's goal is fitting a donor onto a base map, and centre-of-content is where
  such alignment is usually judged.
- 90° rotations and mirrors are exact grid operations; 45° and arbitrary angles necessarily
  involve an approximation, and the feature's obligation is to state which one rather than to hide
  it.
- Whole-layer rotation controls apply to phase layers. Base-map channels can be disabled, and a
  selection from the base can be transformed as an explicit editing operation with preview,
  undo/redo, and save; the base layer itself is not reordered or removed.
- Per-tile placement is first previewed as viewer-side composition. It is written only when the
  operator explicitly invokes Save Transformed Map and accepts a passing preflight.
- A per-tile placement carries the donor tile's placements with it; objects whose bounds straddle
  the donor tile boundary are treated as belonging to the donor tile they were parsed from, and
  planning should confirm the boundary rule against real data.
- Captures and visual confirmation are operator-executed.
- The full-map canvas uses available minimaps by preference, height-derived imagery when minimaps
  are absent, and occupancy color when neither is available; selection is never blocked on artwork.
- The old Chunk Manipulator is migration input, not a second product to preserve. Proven coordinate,
  operation, and undo components may be reused after audit; its broken UI and incomplete paste path
  are retired.
- Save writes user-selected output copies. In-place source-client modification is not a default.
- Captures, real-client reloads, and visual confirmation are operator-executed.

## Dependencies

- Spec 203 (Multi-phase map composition): rotation extends the existing `PhaseLayerSettings` stack,
  channel gates, and composition order. This spec must not reorder 203's composition semantics.
- Spec 195 (Overhead chunk manipulator): superseded as an active UI/operation owner by this unified
  workbench; reusable coordinate and undo components are migration inputs only.
- Spec 196 (WDL lattice magnetization): owns WDL height sampling/magnetization; this spec may consume
  it as an optional fit signal and must not duplicate it.
- Spec 197 (map conversion targets): owns target-format representability and the native MoP split
  writer boundary used by save preflight.
- The tile-offset translation fix landed 2026-09-03 (placements translate by the full tile span) is
  the correctness baseline rotation builds on.
