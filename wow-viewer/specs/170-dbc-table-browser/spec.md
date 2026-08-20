# Feature Specification: DBC/DB2 Table Browser

**Feature Branch**: `170-dbc-table-browser`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [166](../166-editor-plugin-host/spec.md).

## Scope

Read-only typed browsing of client tables: pick a table, see named and correctly typed columns,
search, sort, filter, and follow foreign keys. **The read path is already plumbed** — DBCD, 1,320
WoWDBDefs definitions, and `ArchiveReaderDbcProvider` all exist and are wired (see epic). This spec
builds the plugin and the grid, not a parser.

## User Story - Browse client tables with real types (Priority: P2)

A user opens the DBC editor, picks a table from the loaded client, and sees a grid with named,
correctly typed columns — not raw uint32 fields — and can click a foreign key to jump to the
referenced row.

**Independent Test**: Open a 0.5.3 client and a modern-era client's extracted tables; confirm column
names/types match the WoWDBDefs definition for that build, and that navigating a known foreign key
(e.g. an area's parent area) lands on the correct row.

**Acceptance Scenarios**:

1. **Given** a client is loaded, **When** the user opens the browser, **Then** the tables present in
   that client are listed; tables with no usable definition for that build appear as unavailable
   **with the reason stated**.
2. **Given** a table is open, **When** the grid renders, **Then** each column shows its definition
   name, and values render per their declared type (integer, float, string, localized string, array
   element) — never as raw words.
3. **Given** a table is open, **When** the user types a search term, **Then** rows filter across all
   displayed columns; **and when** a column header is clicked, **Then** rows sort by it, stably, in
   both directions.
4. **Given** a cell holds a foreign key with a declared relation, **When** activated, **Then** the
   referenced table opens at the referenced row, with a way back.
5. **Given** a table with 100,000+ rows, **When** open, **Then** scrolling and searching stay
   responsive, with no full-table materialization per frame.
6. **Given** the same table name in two clients of different builds, **When** each is opened, **Then**
   each resolves its own build-appropriate definition and the build used is displayed.

### Edge Cases

- A table present in the client with no definition for that build — unavailable with reason, **never**
  silently parsed as raw uint32.
- A definition whose field count or record size disagrees with the file — refuse to open, report both
  expected and actual layout, do **not** partially parse.
- A localized string column in a build whose locale count differs from the definition's assumption.
- A DB2 with sparse/offset-map storage or a non-inline ID column.
- No client loaded at all.

## Requirements

### Functional Requirements

- **FR-001**: List tables available in the loaded client and open any with a usable definition for
  that client's build.
- **FR-002**: Column names and types come from the vendored definitions, resolved against the client's
  build. The resolved build is displayed.
- **FR-003**: A layout mismatch between file and definition **refuses** the open and reports both
  expected and actual layout. Silent fallback to untyped parsing is prohibited.
- **FR-004**: Support search across displayed columns, sortable columns, and filtering, remaining
  responsive on the largest tables present in supported clients.
- **FR-005**: Declared foreign keys are navigable to the referenced row, with a way back.
- **FR-006**: Table bytes are obtained through the existing provider over the data-source boundary —
  no second byte path.
- **FR-007**: Reading logic lives in a shared library under `src/core/`.

## Success Criteria

- **SC-001**: A user who has never opened the browser can open a client, find a named table, and read
  correctly-labeled typed columns without consulting documentation.
- **SC-002**: Column names and types match the WoWDBDefs definition for the resolved build, verified
  on ≥20 tables across one pre-CASC and one modern-era build.
- **SC-003**: Search and scroll in the largest table present stay interactive with no perceptible
  stall.
- **SC-004**: Zero tables are opened with untyped/raw-word fallback.

## Out of Scope

- Editing, row add/delete, saving ([171](../171-dbc-table-editing/spec.md)).
- Consolidating the existing narrow DBC readers under `src/core/WowViewer.Core.IO/Dbc/` onto this
  layer — a later cleanup, not this spec.
- Relation inference. Only relations the definitions declare are navigable.

## Assumptions

- The vendored WoWDBDefs snapshot is the definition source; refreshing it is a separate operation.
- Modern-era validation may use extracted loose tables, since no CASC reader exists yet.
