# Feature Specification: DBC/DB2 Table Editing and Loose Save

**Feature Branch**: `171-dbc-table-editing`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [170](../170-dbc-table-browser/spec.md), [168](../168-editor-session-undo/spec.md).

## Scope

Cell editing, row add/delete, and saving a modified table as a **loose** `.dbc`/`.db2` in the output
directory. `DBCDStorage.Save(string)` already exists — this spec makes it reachable and safe, and does
not write a serializer.

## User Story - Edit rows and save a loose table (Priority: P2)

The user changes cell values, adds rows, and deletes rows, then saves the modified table to an output
directory. The game install is never written to.

**Independent Test**: Load a table, change a known value, add a row, delete a row, save, then reload
the saved file in a fresh session and confirm the changes persisted and the file parses. Separately,
round-trip an **unmodified** table and confirm the output is byte-identical to the input.

**Acceptance Scenarios**:

1. **Given** a cell is editable, **When** the user enters a value, **Then** it is validated against the
   column's declared type and range before acceptance; rejected values leave the cell unchanged with
   an explanation.
2. **Given** a table is modified, **When** viewed, **Then** modified cells and added/deleted rows are
   visually distinguished from unmodified ones.
3. **Given** a table is modified, **When** saved, **Then** a loose file is written to the configured
   output directory, its path is reported, and **no** file inside the game install is created,
   modified, or deleted.
4. **Given** a table is loaded and immediately saved with no edits, **When** compared to the source
   bytes, **Then** they are identical.
5. **Given** the user adds a row with no ID, **When** saving, **Then** an unused ID is proposed; **and
   given** a duplicate ID, **Then** the save is refused with the conflict named.
6. **Given** a table is modified, **When** the user closes it or the viewer without saving, **Then**
   they are warned and can cancel.

### Edge Cases

- Editing a table, switching clients, and returning — pending edits are preserved against the original
  client or discarded with an explicit warning; **never** silently applied to a different build.
- A DB2 storage variant DBCD's writer cannot round-trip.
- Output directory unwritable or disk full mid-save.
- Deleting the row a foreign key points at.

## Requirements

### Functional Requirements

- **FR-001**: Cell edits are type- and range-validated before acceptance.
- **FR-002**: Rows can be added and deleted; modified/added/deleted rows are visually distinguished.
- **FR-003**: Saving produces a loose `.dbc`/`.db2` in the output directory that reloads correctly.
- **FR-004**: An unmodified load-then-save is **byte-identical** to the source.
- **FR-005**: Adding a row proposes an unused ID; a duplicate or out-of-range ID blocks the save with
  the conflict named.
- **FR-006**: Never write into a game install. Never write an MPQ, CASC, or any Blizzard container
  (**Constitution VII**).
- **FR-007**: Edits are undoable Editor Operations under 168, at one-cell / one-row granularity.
- **FR-008**: Use the existing DBCD write path. **Do not write a second DBC/DB2 serializer.**
- **FR-009**: A failed or refused save leaves no partial file.

## Success Criteria

- **SC-001**: Load-then-save of an unmodified table is byte-identical, verified by hash on ≥20 tables
  spanning a pre-CASC and a modern-era build.
- **SC-002**: An edited table saved and reopened in a fresh session shows exactly the intended changes
  and no others.
- **SC-003**: No file inside any configured game install is created, modified, or deleted during any
  validation run — verified by hashing the install tree before and after.
- **SC-004**: The repo contains exactly one DBC/DB2 serializer after this spec.

## Out of Scope

- MPQ/CASC repacking, or any packaging that would make output client-loadable as an archive
  (prohibited by Constitution VII, not merely deferred).
- Bulk/scripted table transforms.

## Assumptions

- DBCD's existing `Save` is the write path. If it cannot round-trip a storage variant present in the
  validation clients, that variant is **documented as unsupported for save** — not worked around with
  a second writer.
