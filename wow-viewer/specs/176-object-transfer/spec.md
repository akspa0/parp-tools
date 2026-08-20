# Feature Specification: Object Transfer Between Tiles and Eras

**Feature Branch**: `176-object-transfer`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [175](../175-placement-authoring/spec.md).

## Scope

Transfer selected placements from a source tile — a loaded ADT or an alpha WDT — into a target tile,
possibly in another map or another era's format. The fiddly part, and the reason this has no CLI
equivalent, is that the object-name tables (MMDX/MMID/MWMO/MWID) must be merged into the target and
IDs reconciled.

## User Story - Transfer objects between tiles and between eras (Priority: P1)

The user selects placed objects and moves them to another tile; referenced model/WMO names come along
so the target file resolves them.

**Independent Test**: Transfer a known set of objects from an alpha-WDT tile into a split-ADT tile in
another map; confirm the target renders them at the correct world positions and the written ADT loads
in an independent tool.

**Acceptance Scenarios**:

1. **Given** objects are selected in a source tile, **When** transferred, **Then** their placements
   appear in the target at the intended world positions.
2. **Given** transferred objects reference models absent from the target, **When** applied, **Then**
   the target's name tables gain the needed entries and **all index references are correct**.
3. **Given** transferred IDs collide with existing target IDs, **When** applied, **Then** new
   non-colliding IDs are assigned and **the remapping is reported**.
4. **Given** source and target are different eras, **When** applied, **Then** coordinates and rotations
   are converted correctly for the target era; **and if** the conversion cannot be performed
   faithfully, **Then** the transfer is **refused with the reason** — never approximated.
5. **Given** a transfer would place an object outside the target tile's bounds, **When** attempted,
   **Then** the user is warned with the offending objects named.
6. **Given** a transfer is applied, **When** undone, **Then** the target returns to its prior state
   **including its name tables**.

### Edge Cases

- A model name existing in the target under a different index.
- Source and target disagreeing on tile origin conventions.
- Objects selected across two tiles, transferred as one operation, where one target write fails.
- An alpha WDT whose per-tile data is present but whose object tables are empty.
- Undo of a transfer after the target file was modified by something else.

## Requirements

### Functional Requirements

- **FR-001**: Transfer selected placements from a source tile to a target tile, including across maps.
- **FR-002**: Merge referenced model/WMO name tables into the target and correct all index references.
- **FR-003**: Remap colliding IDs and report the remapping.
- **FR-004**: Cross-era transfer converts coordinates and rotations correctly for the target era, **or
  is refused with the reason**. Approximate transfer is prohibited.
- **FR-005**: Transfers spanning multiple target tiles apply as one operation — all targets written or
  none.
- **FR-006**: The transfer is a single undoable Editor Operation, including name-table changes.
- **FR-007**: Uses the existing core writers and converters; adds no serializer.
- **FR-008**: Declares supported eras and refuses transfers outside them.

## Success Criteria

- **SC-001**: Objects transferred between tiles render at the intended world positions, verified
  visually **and** by reading back the written placement values.
- **SC-002**: A cross-era transfer either round-trips correctly or is refused — **no transfer produces
  silently wrong coordinates**.
- **SC-003**: Name-table merges produce targets whose every index reference resolves, verified on ≥3
  transfers involving previously-absent models.
- **SC-004**: A multi-tile transfer with an induced write failure leaves **no** target modified.
- **SC-005**: Undo of a transfer restores the target byte-identically, name tables included.

## Out of Scope

- Creating the target tile ([177](../177-adt-tile-creation/spec.md)).
- Modifying the transferred assets themselves.

## Assumptions

- Cross-era conversion uses the existing converters. Where they cannot express a placement faithfully,
  refusing is the correct outcome — not a gap to paper over.
- ID remapping is reported rather than silent, because uniqueId is the world-layout chronology of
  record and a silent remap destroys that ordering information.
