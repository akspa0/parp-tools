# Feature Specification: Placement Authoring in the Viewport

**Feature Branch**: `175-placement-authoring`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**, especially
"the authoring code already exists and has no UI caller".
**Depends on**: [167](../167-editor-runtime-bridge/spec.md), [168](../168-editor-session-undo/spec.md),
[173](../173-asset-integrity-gate/spec.md).

## Scope

Full placement editing: select a placed doodad or WMO, move/rotate/scale it, add a new placement,
delete one, save the affected ADTs. Extends 167's translation-only proof into a real capability.

**Do not write an ADT writer.** `LkAdtWriter`, `AlphaWdtWriter`, and `AdtPlacementWriter` exist and are
correct — see the epic.

## User Story - Author placements in the viewport (Priority: P1)

The user edits placements directly in the scene — the one place where "which tile" and "which object"
are obvious — and saves working ADTs without leaving the viewer.

**Independent Test**: On a real map tile, move one object, rotate a second, add a third, delete a
fourth, save; reload the written ADT in a fresh session and confirm all four changes are present and
nothing else changed.

**Acceptance Scenarios**:

1. **Given** an object is selected, **When** the user edits position, rotation, or scale, **Then** the
   change is visible immediately and staged as an undoable operation.
2. **Given** a model or WMO is chosen, **When** placed in the world, **Then** a new placement entry is
   created with a unique ID that does not collide with any existing ID in the map.
3. **Given** a placement is selected, **When** deleted, **Then** it disappears from the scene and its
   entry is removed on save; undo restores both.
4. **Given** staged edits, **When** saved, **Then** the affected ADTs are written to the output
   directory and **unaffected chunks in those files are byte-identical to the source**.
5. **Given** an edit to an object whose source file is not writable, **When** saving, **Then** the
   reason is reported before any file is written.
6. **Given** placements edited across several tiles, **When** saved, **Then** every affected tile is
   written and the set of written files is reported.

### Edge Cases

- Placement ID space exhausted, or IDs reserved in the target era.
- Deleting a placement referenced by something else in the tile.
- Saving a tile while the viewer is actively streaming it.
- Undo of an edit after the affected tile unloaded.

## Requirements

### Functional Requirements

- **FR-001**: Select placed doodads and WMOs in the viewport and edit position, rotation, and scale,
  with changes visible immediately.
- **FR-002**: Add a new placement by choosing a model/WMO and a world position; delete an existing
  placement.
- **FR-003**: New placements receive IDs that do not collide with any existing ID in the map.
- **FR-004**: Every authoring action is an undoable Editor Operation under 168.
- **FR-005**: Saving writes only affected tiles; chunks not touched by an edit are byte-identical to
  the source.
- **FR-006**: All writes go through the existing core writers. **This spec adds zero ADT, WDT, or
  placement serializers.**
- **FR-007**: All output goes to the configured output directory. Never a game install, never an MPQ
  or CASC container (**Constitution VII**).
- **FR-008**: All scene reads and applied changes go through the 167 bridge; no direct renderer,
  scene, or app references.
- **FR-009**: The plugin declares supported build eras and is unavailable, with the reason stated, for
  maps outside them.

## Success Criteria

- **SC-001**: A user can move, rotate, add, and delete placements and save a working ADT without
  leaving the viewer or typing a command.
- **SC-002**: Saving a tile with one placement edited leaves every unedited chunk byte-identical,
  verified by chunk-level comparison.
- **SC-003**: The count of ADT/WDT/placement serializers in the repo is unchanged — this spec adds
  **zero**.
- **SC-004**: Every authoring action is undoable, verified across a mixed sequence of ≥15 operations.
- **SC-005**: No file inside any configured game install is created, modified, or deleted — verified
  by hashing before and after.
- **SC-006**: Written files load in at least one tool **outside this repo**, not only in the viewer
  that wrote them.

## Out of Scope

- Cross-tile / cross-era object transfer ([176](../176-object-transfer/spec.md)).
- Creating new tiles ([177](../177-adt-tile-creation/spec.md)).
- Terrain sculpting (heights, holes, texture painting).
- Modifying model/WMO assets themselves — objects are placed by reference.

## Assumptions

- The existing core writers are correct. A defect found here is fixed **in core**, so the CLI benefits
  equally — not worked around in the plugin.
- Placement unique IDs are allocated within the map being edited; global cross-map uniqueness is not
  attempted.
