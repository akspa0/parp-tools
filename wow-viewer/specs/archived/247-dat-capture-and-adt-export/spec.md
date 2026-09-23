# Feature Specification: DAT Capture & LK ADT Export

<!-- reconciliation-2026-09-23 -->
> **ARCHIVED 2026-09-23 — open residue folded into an epic.** US3/US5 shipped; US1 AMAP, US2 capture, witnesses open. Successor: [Epic 248](../../248-epic-formats-and-conversion/spec.md). Status lines and checkboxes below are historical and were audited against the code ([audit](../reconciliation-2026-09-23/audit/batch-D2.md)); they are not implementation authority.

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6

**Created**: 2026-09-20

**Status**: **US3 DELIVERED** 2026-09-20 (`adt-ahdr export-lk`; receipt evidence/us3-dat-to-lk-adt-2026-09-20.md). **US5 CODE COMPLETE** 2026-09-20 (DAT folders as Cartography layers; receipt evidence/us5-dat-as-cartography-layer-2026-09-20.md), unwitnessed in the viewer. US1 (v22 AMAP codec) attempted and OPEN. US2 (capture) not started.

**Depends on**: [Spec 237 DAT v22/v23/v26](../237-adt-v26-terrain/spec.md) (reader, slicer, measured frames)

**Related**: [Spec 241 DAT v26 Interchange](../241-dat-v26-interchange/spec.md) is **deferred and stays
deferred** (operator, 2026-09-17: DAT is a format to read and study, not a save or interchange
format). This spec is deliberately the narrow, one-way case 241 did not cover: **read-only DAT files
in, artifacts out**. Nothing here makes DAT a save target, an edit format or a harvest source.

**Input** (operator, 2026-09-20): *"can we get DAT to have minimaps captured (not synthesized), and
can we build a converter for DAT to ADT? I'd like to save these results as something we can use, LK
ADT would be fine."*

## Context

Spec 237 established that all three DAT revisions load and render from real files, including the
first v22 ever seen — four Expansion01 tiles of Terokkar / Bone Wastes terrain
([evidence](../237-adt-v26-terrain/evidence/first-v22-dat-render-2026-09-20.md)). That data exists
only as loose developer files scattered through client MPQs under arbitrary extensions. It is
currently viewable and nothing more: there is no way to get a picture of it or a file any other tool
can open.

This spec turns viewing into keeping.

### Blocking defect

v22 `AMAP` alpha maps are an **unidentified encoding** (128–3474 bytes, never 4096;
[codec attempt](../237-adt-v26-terrain/evidence/v22-amap-codec-attempt-2026-09-20.md)). The viewer's
alpha guard requires a 4096-byte map on *every* layer, so v22 currently renders **layer 0 only**. Both
deliverables inherit that: a capture would photograph the wrong terrain and an exported ADT would
carry one layer where there should be up to four. The operator chose to resolve the codec **first**.

### What v22 carries that v26 does not (measured, corrects spec 241's table)

| Field | v26 | **v22** |
|---|---|---|
| `ACNK` +0x0C area id | 0 everywhere | **3519 (838 chunks), 3520 (159)** |
| `ACNK` +0x10 holes | 0 everywhere | 0 everywhere |
| `ASHD` shadows | present, all zero | **767 present, 289 non-zero**, 512 B = LK `MCSH` shape |
| `ACNK` count | exactly 256 | **243–255 (empties omitted)** |

Area IDs and shadows are therefore **carryable** for v22, not lost as 241's v26-derived table implied.

## User Scenarios & Testing

### User Story 1 — Read v22 texture blending (Priority: P1, blocking)

A user opens a v22 DAT folder and sees all its texture layers blended as authored, not just the base
layer.

**Independent Test**: for a chunk with 3–4 layers, the rendered surface differs from the layer-0-only
render, and the highest-alpha layer per 8×8 cell agrees with that chunk's `ACNK` +0x12 predominant
map.

### User Story 2 — Capture a picture of DAT terrain (Priority: P1)

A user with a DAT folder open captures top-down images of the loaded terrain — one per tile plus a
single stitched overview — and can view or share them outside the application.

**Independent Test**: capture a 4-tile folder; 4 per-tile images plus 1 overview appear in the project
output root; the overview places each tile at its correct grid position with no gaps or overlaps.

### User Story 3 — Export DAT tiles as LK ADT (Priority: P1)

A user exports an open DAT folder as Wrath-of-the-Lich-King map files and opens the result in existing
tooling that reads LK maps.

**Independent Test**: export the 4 Expansion01 tiles; the output loads in the viewer's own LK path
with terrain, texture layers and object placements present, and a written manifest states every field
that could not be carried.

### User Story 5 — Overlay DAT on the shipped map (Priority: P2)

A user adds a folder of DAT files as a Cartography layer over the map those tiles belong to, and
compares the project data against what shipped, using the layer's existing alignment controls.

Operator input (2026-09-20): *"it'd be nice if we could load DAT files as Cartography layers, since
they are effectively the real project files for existing tiles."*

**Independent Test**: add the Expansion01 DAT folder as a layer over a base map; its footprint draws
on the minimap at the donor's own tiles, and its terrain composes into the scene.

### User Story 4 — Know what was lost (Priority: P2)

A user reading the export manifest can tell exactly which source fields were carried, which were
dropped, and why, without reading code.

**Independent Test**: the manifest names every dropped field with a reason, and its claims match a
chunk-level diff of source against output.

## Requirements

### Functional — Phase 1: v22 alpha (blocking)

- **FR-001**: The system MUST decode v22 `AMAP` payloads to per-layer alpha, or record a dated,
  evidence-backed statement that the encoding remains unidentified after using the scoring oracle.
- **FR-002**: Candidate decodes MUST be scored against the `ACNK` +0x12 2-bit 8×8 predominant-layer
  map, which is confirmed present and self-consistent on v22 (0 violations across 997 chunks). Blind
  variant brute-forcing is explicitly out of scope; attempt 1 already exhausted 32 variants.
- **FR-003**: The terrain adapter MUST blend the layers that do have alpha maps rather than dropping
  all alpha when any layer lacks one. This MUST hold independently of FR-001.
- **FR-004**: v23 and v26 alpha behaviour MUST NOT change. A regression test MUST pin this.

### Functional — Phase 2: capture

- **FR-005**: The system MUST capture top-down images of the **loaded, rendered** DAT terrain. It MUST
  NOT compose images from layer metadata and textures (the existing synthesizer path is not the
  mechanism).
- **FR-006**: Output MUST be one image per loaded tile plus one stitched overview of the whole folder.
- **FR-007**: Output MUST be written under the project output root, never to an ad-hoc location.
- **FR-008**: Capture MUST work for all three revisions. For v22 before Phase 1 completes, output MUST
  be marked as layer-0-only rather than presented as faithful.
- **FR-009**: Tiles absent from the folder MUST appear as empty space in the overview, correctly
  positioned, not collapsed out.

### Functional — Phase 3: LK ADT export

- **FR-010**: The system MUST write LK v18 ADT files from an open DAT folder, plus the WDT needed to
  make them loadable.
- **FR-011**: Export MUST carry, where the source has them: heights, normals, texture layers and
  their blend, vertex colours, object placements, area IDs, holes, and shadows.
- **FR-012**: Export MUST address chunks by their `ACNK` index fields, never by ordinal position,
  because v22 omits empty chunks.
- **FR-013**: Export MUST write a manifest naming every field carried and every field dropped, with a
  reason for each drop.
- **FR-014**: The exported map MUST load in the application's existing LK path.
- **FR-015**: Heights MUST be converted from the source's inch units to the target's yard units, and
  the conversion MUST be stated in the manifest.

### Functional — Phase 4: DAT as a Cartography layer

- **FR-016**: A layer MUST be able to name a folder of DAT files as its donor, not only a map in the
  base adapter's own data source.
- **FR-017**: A DAT donor MUST compose through the existing layer pipeline, so tile offset, cell
  offset, rotation, mirrors, channel gating and locking all apply to it unchanged.
- **FR-018**: A DAT donor's footprint MUST draw on the Cartography overlay from its own occupied
  tiles.
- **FR-019**: A missing, empty or unreadable DAT folder MUST be a displayable state, never a crash.
- **FR-020**: Layers that name an ordinary map MUST behave exactly as before. A regression test MUST
  pin this.
- **FR-021**: Donor heights MUST be converted from inches to the base map's yards.

### Non-functional

- **NFR-001**: New behaviour lives in owned service classes. No new members on the two frozen
  god-classes (AGENTS.md §10).
- **NFR-002**: Every phase produces a receipt per AGENTS.md §9.2, including a criterion→evidence table
  against real files.
- **NFR-003**: No existing format reader or writer changes behaviour. Regression tests for the LK and
  Alpha paths MUST still pass.

## Success Criteria

- **SC-001**: A user can go from an unopened folder of DAT files to a viewable image of its terrain in
  under two minutes of interaction.
- **SC-002**: All four Expansion01 v22 tiles export and re-open, with zero chunks lost between source
  and output.
- **SC-003**: The exported map's terrain is visually indistinguishable from the source rendered in the
  viewer, judged side by side by the operator.
- **SC-004**: Every field the source carries is either present in the output or named in the manifest
  with a reason — no silent drops.
- **SC-005**: v22 chunks with multiple layers render with visibly different texture coverage than
  layer-0-only, agreeing with the predominant-layer map.
- **SC-006**: Capture and export both work unattended across a whole folder in one action.

## Key Entities

- **DAT folder**: a directory of AHDR-family terrain files of any revision, any filename, any
  extension; the read-only source.
- **Tile**: one terrain file, located by `ALOC` or by filename, holding a 16×16 chunk grid.
- **Chunk**: one cell of a tile, carrying heights, layers, alpha, shadows, objects and an area id.
- **Capture set**: per-tile images plus one stitched overview for a folder.
- **Export set**: LK map files plus a WDT plus a loss manifest.

## Assumptions

- The four Expansion01 v22 tiles and the IcecrownCitadel v23 tiles are the acceptance corpus; the
  699-file v26 corpus is the regression corpus.
- Heights, normals and vertex colours are interpreted the same way across revisions. This is assumed
  from v26 measurement and is **not separately proven for v22**.
- v22 `ASHD` (512 bytes) maps directly onto LK `MCSH` (512 bytes). Same size and shape; the mapping is
  assumed and must be verified before FR-011 claims shadows are carried.
- Area ID values 3519/3520 are real area table ids. Their **names are not confirmed** and must be
  checked against the area table before any output or document asserts a zone name.
- The operator owns all real-client visual proof and any heavy or long-running batch runs.

## Out of Scope

- Making DAT a save, edit or interchange format — that is spec 241 and stays deferred.
- ADT → DAT (the reverse direction already exists as an experimental writer in 237).
- Alpha 0.5.3 or split-ADT export targets; LK v18 only.
- Synthesized or reconstructed minimaps.
- Scanning client listfiles to discover more DAT files — proposed separately in the 237 evidence.
- Recovering liquids or any field no DAT sample has ever contained.

## Dependencies

- Spec 237 for the reader, the tile slicer and the measured coordinate frames.
- The existing LK ADT and WDT writers, the existing alpha weight→sequential conversion, and the
  existing offscreen capture plumbing. This spec adds no new format writer.

## Risks

- **FR-001 is open-ended research.** The encoding may not fall to the oracle-scored search. FR-003 is
  deliberately independent so Phases 2 and 3 are not hostage to it; if FR-001 stalls, the operator
  decides whether to proceed with v22 marked lossy.
- Area IDs, shadows and objects are each carried on evidence from a **four-file** v22 sample. Findings
  may not generalise to v22 files found later.
