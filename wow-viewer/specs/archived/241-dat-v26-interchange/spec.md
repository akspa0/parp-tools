# Feature Specification: DAT v26 as the Project Interchange Format

<!-- reconciliation-2026-09-23 -->
> **ARCHIVED 2026-09-23 — cold (historical reference, no live residue).** Operator-deferred; one-way export lives in 247. Successor: [Epic 248](../../248-epic-formats-and-conversion/spec.md). Status lines and checkboxes below are historical and were audited against the code ([audit](../reconciliation-2026-09-23/audit/batch-D2.md)); they are not implementation authority.

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6

**Created**: 2026-09-17

**Status**: Deferred (operator, 2026-09-17): not pursued. DAT v26 stays a format to read and study, not a save or interchange format. The capability table below is kept as measured facts.

**Depends on**: [Spec 237 DAT v26](../237-adt-v26-terrain/spec.md) (reader, writer, measured frames), [Spec 234 Map Save](../234-map-save-new-map/spec.md) (save targets)

**Input**: "if we save all our edits to this format, too, then we can just use it as an interchange format, since that's what it is for Blizzard … Might make the most sense for downstream dataset consumers in the harvest side of the tooling, too."

## Context

DAT files are the terrain project files the client's ADTs are built from. They store heights in inches (36× finer
than ADT yards), per-layer blend weights instead of sequential alpha, and object placements relative to the chunk
that holds them. As of commit f1138874 the project can read every byte of the 699-file v26 corpus, rewrite it
byte-identically, and build v26 tiles from client ADT data (`AdtAhdrTileBuilder`, viewer export).

That makes DAT v26 a candidate for the one format edits are saved in and datasets are exported from, with ADT/WDT
(Alpha, LK, split) as build targets generated from it.

### What v26 can and cannot carry today (MEASURED on the corpus)

> **v22/v23 correction (2026-09-20).** The table below was measured on the **v26** corpus only, when no
> real v22/v23 files existed. Real files now do, and two rows do **not** generalise: v22 carries real
> **area IDs** (`ACNK` +0x0C = 3519/3520) and **live shadows** (`ASHD` 289 of 767 non-zero), and v22
> **omits empty `ACNK`** (243-255, not 256). See
> [237 evidence](../237-adt-v26-terrain/evidence/first-v22-dat-render-2026-09-20.md). This spec stays
> **deferred**; one-way DAT->LK ADT export lives in
> [247](../247-dat-capture-and-adt-export/spec.md).

| Content | In v26? | Evidence |
|---|---|---|
| Heights, normals, vertex colours | Yes (`AVTX`, `ANRM`, `ACVT`) | round trip 700/700 |
| Texture layers + blend | Yes, up to 4 layers observed (`ALYR`/`AMAP` weights) | 99.93% predominant-map match |
| Doodad and WMO placements | Yes (`ACDO`), frame measured | 0.31 in median height fit |
| Liquids (MH2O/MCLQ) | **No chunk observed** | chunk inventory: no liquid chunk in 700 files |
| Holes | **No populated field** (`ACNK` +0x10 is 0 in all 179,200 chunks) | header survey |
| Area IDs | **No** (`ACNK` +0x0C is 0 everywhere) | header survey |
| Shadows | `ASHD` present but all zero | 15,360 chunks |
| Sound emitters, flight bounds, WMO doodad sets, MCMT materials, MTXP height texturing | **Not observed** | chunk inventory |
| More than 4 layers | Not observed (writer does not limit it) | 1.60.1 ADTs use up to 8 |

So saving edits only as v26 would lose liquids, holes and area IDs, plus everything in the last two rows. Any
interchange use needs a lossless answer for those fields before ADT output can be generated from DAT alone.

## User Scenarios & Testing

### User Story 1 — Save edits as DAT v26 (Priority: P1)

An editor session (terrain sculpt/paint, placements, cartography composition) saves the edited tiles as DAT v26
plus a sidecar for fields v26 lacks, and reopening that save reproduces the session exactly.

**Independent Test**: edit a tile, save as DAT, reload; heights, layers, placements, liquids, holes and area IDs
compare equal to the in-memory session (inches → yards rounding within 1/36 yd; weights ↔ alpha within 2/255).

### User Story 2 — Build client formats from DAT (Priority: P1)

A DAT save builds Alpha WDT, LK ADT and split ADT outputs through the existing Spec 234 writers, so DAT is the
source and ADT/WDT are targets.

**Independent Test**: ADT → DAT → ADT on real client tiles; the rebuilt ADT loads in the viewer and matches the
original within the stated rounding, and the Spec 234 client-load witness still passes.

### User Story 3 — Harvest datasets from DAT (Priority: P2)

Harvest reads DAT folders as a terrain source (inch heights, weights, chunk-relative placements) and emits the same
streams and ARRY blobs as for ADT, so downstream consumers get one representation for both real DAT data and
converted client data.

**Independent Test**: harvest the same tiles from ADT and from their DAT export; tensors agree within rounding.
The Python datastore remains the only Zarr/TensorStore writer.

## Requirements

- **FR-001**: The DAT save MUST be byte-compatible with v26 as measured (a real v26 reader sees a valid file).
- **FR-002**: Fields v26 cannot hold MUST be stored losslessly outside the v26 chunks (sidecar file, or clearly
  named extension chunks that a v26 reader skips); the choice is made in the plan and recorded with the reason.
- **FR-003**: Every conversion (ADT → DAT, DAT → ADT) MUST report what it rounded or dropped, per tile.
- **FR-004**: Unknown v26 fields keep the values observed in the corpus and are listed as such in the output.
- **FR-005**: Harvest MUST NOT implement Zarr/TensorStore IO in C#; DAT support is a new source for the existing streams.
- **FR-006**: DAT saves MUST be written under the project output tree, never over client data.

## Success Criteria

- **SC-001**: Save → reload of an edited session is identical within the stated rounding, including liquids, holes and area IDs.
- **SC-002**: ADT → DAT → ADT round trip on at least 25 real tiles from two eras, with a per-field difference report.
- **SC-003**: Harvest tensors from DAT and from the source ADT agree within rounding on the same tiles.

## Assumptions and open questions

- Blizzard's own tools may store liquids, holes and area data in files other than the v26 tiles (none were in the
  corpus); if such files appear, they replace the sidecar for those fields.
- v26 "one texture set per chunk ≤ 4 layers" is observed, not proven; 8-layer output is allowed but flagged.
