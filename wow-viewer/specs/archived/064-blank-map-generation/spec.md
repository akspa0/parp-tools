# Feature Specification: Blank Map Generation + Relational ADT Understanding

**Feature Branch**: `064-blank-map-generation`

**Created**: 2026-06-16

**Status**: Draft

**Input**: We need to generate valid blank ADT/WDT files that load in the viewer and game engine, as a prerequisite for understanding every iota of the ADT and alphaWDT file formats. These files are relational databases (tables, indices, foreign keys) compressed into binary. We must treat them as such — using the same pattern we already use for DBC/DB2 files via DBCD + WoWDBDefs — and use Zarr as an intermediate datastore that respects this structure.

## Background

WoW's ADT, WDT, and PM4 file formats are highly optimized relational databases. They contain:
- **Tables**: MCNK chunks (terrain), MDDF/MODF (placements), MTEX/MMDX/MWMO (string tables), MCIN (chunk index), MH2O (liquid), MFBO (flight bounds)
- **Indices**: MCIN offsets point into MCNK, MCRF references into MDDF/MODF, MMID/MWID reference into MMDX/MWMO, MCLY references into MTEX
- **Foreign keys**: Alpha offsets reference into MCAL, texture layers reference MTEX entries, placement references cross-reference MDDF↔MMDX and MODF↔MWMO

The current approach (hand-coded binary readers/writers, opaque patchers) treats these as opaque blobs. It fails because it doesn't respect the relational structure. The right approach is to model these as tables the same way we already model DBC/DB2 files:

### The DBC Pattern (Already Working)

We already have a working pattern for this exact problem. DBC/DB2 files are literally SQL tables dumped to binary. Our codebase handles them correctly because we use:

1. **WoWDBDefs** (`wow-viewer/libs/wowdev/WoWDBDefs/definitions/*.dbd`) — schema definitions that are the equivalent of `CREATE TABLE` statements. Hundreds of `.dbd` files define column names, types, build-specific layouts, and foreign key relationships.
2. **DBCD** — a generic reader that takes a `.dbc` (binary data) + `.dbd` (schema) and produces a typed `IDBCDStorage` object you can query by column name.
3. **Typed accessors** — `AreaIdMapper`, `LightService`, `MapDiscoveryService`, `ReplaceableTextureResolver`, etc. use DBCD to read specific tables with known schemas.

This is **exactly the pattern we need for ADT/WDT/PM4**. The `.dbd` files are schema definitions. DBCD is the generic reader. The difference is that ADT/WDT/PM4 have **nested** tables (chunks within chunks) and **inter-file** foreign keys (WDT → ADT, _obj0.adt → PM4), while DBC tables are flat single-file tables. But the relational model is the same.

### The Right Approach: DBD-Style Schema Definitions for ADT/WDT

Instead of hand-coding binary offsets and sizes in C# (which is what killed the Pm4AdtWriter — it hallucinated byte offsets), we should define ADT/WDT schemas in a declarative format similar to `.dbd`:

```
# ADT schema (proto-.dbd style)
TABLE MHDR:
  col: flags          UINT32   offset 0x00
  col: ofsMcin        UINT32   offset 0x04   FK→MCIN
  col: nMtex          UINT32   offset 0x08   count(MTEX entries)
  col: ofsMtex        UINT32   offset 0x0C   FK→MTEX
  ...etc

TABLE MCNK (256 rows, PK=MCIN[i].offset):
  col: flags          UINT32   offset 0x00
  col: indexX         INT32    offset 0x04
  ...etc
  sub-tables: MCVT, MCNR, MCLY, MCRF, MCAL, MCSH, MCLQ, MCCV, MCLV
```

This is what Phase 2 of this spec produces — not just documentation, but a **machine-readable schema** that drives reading and writing the same way `.dbd` drives DBC reading.

The Zarr datastore (Phase 3) then becomes the natural runtime representation: one Zarr group per table, with columns as arrays, exactly like a column-oriented SQL database.

Before we can write proper ADTs from PM4 data, we must first prove we can generate **valid blank maps** — ADT + WDT + WDL files with flat white terrain, zero placements, and no textures — that the viewer and game engine can load without errors.

## User Scenarios & Testing

### User Story 1 - Generate a Blank LK ADT That Loads (Priority: P1)

A developer runs a CLI command to generate a blank 3.3.5 LK ADT file for a given tile coordinate. The output ADT loads in the viewer with flat white terrain (no textures, no placements, default heights), zero errors, and renders a visible terrain grid.

**Why this priority**: This is the baseline proof that our ADT writer understands the format. If we can't produce a blank ADT that loads, we can't produce any ADT that loads.

**Independent Test**: Run `map generate-blank-adt --tile-x 32 --tile-y 43 --output blank_32_43.adt`, then run `map inspect blank_32_43.adt` and verify valid structure. Open in viewer and confirm terrain renders.

**Acceptance Scenarios**:

1. **Given** a blank LK ADT for tile (32,43), **When** inspected with `map inspect`, **Then** the output shows valid MVER=18, MHDR with correct offsets, MCIN with 256 entries, 256 MCNK chunks with default heights, MDDF with 0 entries, MODF with 0 entries
2. **Given** a blank LK ADT loaded in the viewer, **When** the tile is rendered, **Then** a flat terrain grid is visible with no assertion errors, no missing chunk warnings, and no null reference exceptions
3. **Given** a blank LK ADT, **When** compared against a real ADT's chunk structure (not data), **Then** all mandatory chunks (MVER, MHDR, MCIN, MCNK×256) are present with correct offsets and sizes

---

### User Story 2 - Generate a Blank LK WDT + ADT Pair (Priority: P1)

A developer generates a WDT file that references the blank ADT from Story 1. The WDT+ADT pair loads in the viewer as a complete map tile.

**Why this priority**: The ADT alone isn't enough — the WDT tells the engine which tiles exist. Both must be valid.

**Independent Test**: Generate WDT for "blanktest" map with tile (32,43), generate corresponding ADT, load in viewer, confirm the full tile renders.

**Acceptance Scenarios**:

1. **Given** a blank LK WDT with tile (32,43) flagged, **When** inspected, **Then** MPHD flags and MAIN grid are correct
2. **Given** WDT+ADT pair for "blanktest" map, **When** loaded in viewer, **Then** tile (32,43) renders as flat terrain with no errors

---

### User Story 3 - Generate a Blank Alpha WDT That Loads (Priority: P2)

A developer generates a blank 0.5.3 Alpha-format WDT with flat terrain tiles. The viewer loads it without assertion errors.

**Why this priority**: Alpha format has different chunk structure (embedded tiles in WDT, monolithic format). Proving blank Alpha generation ensures our AlphaWdtWriter understanding is complete.

**Independent Test**: Use `AlphaWdtWriter.Build()` with blank `AlphaTileData` for all tiles, output WDT, load in viewer.

**Acceptance Scenarios**:

1. **Given** a blank Alpha WDT with flat terrain tiles, **When** loaded in viewer, **Then** tiles render without assertion errors
2. **Given** the blank Alpha WDT, **When** compared against a real Alpha WDT's chunk structure, **Then** MVER, MPHD, MAIN, and per-tile MHDR/MCIN/MCNK structure is present and valid

---

### User Story 4 - ADT Relational Schema Documentation (Priority: P2)

The developer has a single architecture document that describes the LK ADT and Alpha WDT formats as relational schemas — identifying every table, its columns, its primary key, foreign keys to other tables, and the binary encoding of each column. This document becomes the reference for all future ADT/WDT work.

**Why this priority**: Without this document, every writer is a hallucinated mess. With it, every future writer can be verified against the schema.

**Independent Test**: The document exists in `wow-viewer/docs/architecture/` and contains a complete table-by-table description of ADT chunks.

**Acceptance Scenarios**:

1. **Given** the schema document, **When** a developer reads it, **Then** they can identify every ADT chunk, its fields, its byte offsets, and its cross-references to other chunks
2. **Given** a real ADT file, **When** the schema is used to parse it, **Then** every byte in the file maps to a table/row/column in the schema

---

### User Story 5 - Zarr ADT Datastore Round-Trip (Priority: P3)

A developer can read an LK ADT into a Zarr-backed relational store (one Zarr group per table: MTEX, MMDX, MMID, MWMO, MWID, MDDF, MODF, MCNK×256, MH2O, MFBO), modify data in the store, and write it back out as a valid ADT that is byte-identical to the input (lossless round-trip) or intentionally modified (e.g., change a texture name, add a placement).

**Why this priority**: This is the long-term architecture goal. It proves the relational model is complete and the Zarr bridge works. But P1/P2 must land first — we need blank maps before we can round-trip real ones.

**Independent Test**: Read `development_32_43_obj0.adt` into Zarr store, write back out, diff bytes. Change a texture name in Zarr, write back out, confirm only the texture string table and its referent offsets changed.

**Acceptance Scenarios**:

1. **Given** a real LK ADT, **When** read into Zarr and written back, **Then** the output is byte-identical to the input (lossless round-trip)
2. **Given** a real LK ADT in Zarr store, **When** a texture name is changed and written back, **Then** only MTEX, MMID, MDDF/MODF name references, and MHDR offsets change; all MCNK data is preserved exactly

---

### Edge Cases

- What happens when a blank ADT has 0 texture layers? MTEX/MMID/MCAL chunks must still be valid (empty MTEX, empty MMID).
- What happens when MH2O has no liquid entries? The chunk must be absent or have a specific flag in MHDR masking it.
- What happens when MFBO has no flight bounds? The chunk must be absent or the MHDR flag must not reference it.
- How does MCIN handle 256 chunk offsets when chunks have variable sizes? Offsets must be computed after all chunk sizes are known.
- What are the exact MHDR flags for a minimal ADT? Must match what the viewer and game engine expect.

## Requirements

### Functional Requirements

- **FR-001**: System MUST generate a valid LK 3.3.5 monolithic ADT for any tile coordinate (x,y) with flat terrain, zero placements, zero textures, and zero liquid
- **FR-002**: System MUST generate a valid LK 3.3.5 WDT that flags specified tiles as existing
- **FR-003**: System MUST generate a valid WDL (world height low-res) for blank tiles with flat heights
- **FR-004**: System MUST validate generated files using `map inspect` and confirm zero errors
- **FR-005**: System MUST load generated files in the viewer and render flat terrain without assertion errors, null references, or missing chunk warnings
- **FR-006**: The blank ADT MUST contain correct MCIN offsets computed after all chunk sizes are finalized (no dangling offsets)
- **FR-007**: The blank ADT MUST contain 256 MCNK chunks with valid MCVT (145 floats), MCNR (448 bytes), and MCLY (0 layers)
- **FR-008**: All cross-reference offsets in MHDR, MCIN, and MCNK sub-chunk offsets MUST be correct relative to file start
- **FR-009**: System MUST document the ADT relational schema (tables, columns, keys, foreign keys, binary encodings) in `wow-viewer/docs/architecture/adt-relational-schema.md`
- **FR-010**: System MUST generate a valid 0.5.3 Alpha WDT with blank flat terrain tiles (subject to AlphaWdtWriter being unfrozen for this work, per Rule 10)
- **FR-011**: The `LkAdtWriter.Build()` method MUST produce byte-identical output when given the same `LkAdtData` twice (deterministic)
- **FR-012**: Any new blank-map generation MUST NOT modify `AlphaWdtWriter.cs` unless Rule 10 is explicitly reopened by the user

### Key Entities

- **LkAdtData**: Domain model for LK 3.3.5 ADT (already exists in `WowViewer.Core/Maps/LkAdtData.cs`). Extended with factory methods for blank generation.
- **BlankAdtFactory**: New static class that constructs `LkAdtData` with flat terrain defaults for any tile coordinate.
- **BlankWdtFactory**: New static class that constructs WDT data with tile existence flags.
- **ADT Relational Schema**: Architecture document describing every ADT chunk as a database table with columns, types, primary keys, and foreign keys — modeled after the `.dbd` schema pattern we already use for DBC/DB2 files.
- **DBCD Pattern**: Our existing infrastructure for DBC/DB2 (WoWDBDefs `.dbd` definitions + DBCD generic reader + typed accessors). The ADT/WDT schema work should follow this same pattern: declarative schema → generic reader/writer → typed accessors.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `map generate-blank-adt --tile-x 32 --tile-y 43 --output blank.adt` produces a file that `map inspect` validates with zero errors
- **SC-002**: The blank ADT loads in the viewer and renders a flat terrain grid (visible in the viewport)
- **SC-003**: A blank WDT+ADT pair loads in the viewer as a complete map tile
- **SC-004**: The ADT relational schema document describes every chunk (MVER, MHDR, MCIN, MCNK×256 sub-chunks, MTEX, MMDX, MMID, MWMO, MWID, MDDF, MODF, MH2O, MFBO) with byte offsets, field sizes, and cross-references
- **SC-005**: Round-trip test: read a real ADT → `LkAdtData` → write back → byte-identical output (Phase 2)
- **SC-006**: Zarr datastore round-trip: read ADT → Zarr → write → identical (Phase 3, stretch goal)

## Assumptions

- We already have `LkAdtWriter`, `LkAdtReader`, `LkAdtData`, `AlphaWdtWriter` (frozen), `LkWdtWriter`, and `WdlWriter` as working writer infrastructure
- The viewer can already load real LK ADT files; blank ADTs must conform to the same structural expectations
- `AlphaWdtWriter.cs` is frozen per Rule 10; blank Alpha WDT generation will use the existing writer without modification (construct `AlphaTileData` with blank fields)
- Zarr datastore work (Story 5) is a stretch goal and should not block P1/P2 delivery
- The relational schema documentation is a reference artifact — it informs future work but doesn't block P1/P2
- Existing `LkAdtWriter.Build()` already writes full ADT structure; the gap is constructing a valid `LkAdtData` with correct default values for a blank tile
- **The DBC/DB2 pattern (WoWDBDefs + DBCD) is the proven model for this work**. We have hundreds of `.dbd` schema files and a generic DBCD reader. ADT/WDT schemas should follow the same declarative approach: define the tables, columns, types, and foreign keys, then drive reading/writing from the schema rather than hand-coding binary offsets.

## Phasing

### Phase 1: Blank LK ADT + WDT + WDL Generation (P1/P2 Stories)

Prove we can generate valid files that load. Extend `LkAdtData` with blank factory, add `map generate-blank-adt` CLI command.

### Phase 2: ADT Relational Schema + Round-Trip

Document the format as relational tables. Prove lossless round-trip read→write on real ADT files.

### Phase 3: Zarr ADT Datastore (Stretch)

Zarr-backed relational store for ADT data. Read into Zarr, modify, write back out. This is the long-term architecture but depends on Phase 2 being complete.

## Out of Scope

- PM4 matching or restoration (that's spec 046)
- Modifying AlphaWdtWriter (frozen, per Rule 10)
- Alpha-era placement writing (deferred until blank LK proves the model)
- Training data or ML model integration
- Game engine playback validation (viewer-only validation is sufficient for P1)