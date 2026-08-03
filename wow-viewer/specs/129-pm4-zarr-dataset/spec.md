# Feature Specification: PM4 Zarr Dataset

**Feature Branch**: `129-pm4-zarr-dataset`

**Created**: 2026-08-03

**Status**: Draft

**Input**: User description: "we need to treat pm4's to a zarr dataset output, too, with properly decoded signals stored, since they are VERY complex files that are interlinked with bits and pieces and are loaded with detailed geometry that exceeds the resolution of the WMO data, by a great deal... you have to read the pm4's as both a whole map object and as a per-tile object, and likely also as a per-objectID thing, since it's just layers of an onion of data with nested coordinate systems to store data at different levels."

## Context

### Why a dataset at all

PM4 decoding is the oldest and deepest part of this codebase, and it works: 13 research analyzers,
a coordinate service, a fingerprint and surface matcher, and a 30-command CLI. What does not exist
is a **stored, queryable form**. Every question today means re-reading 616 files and re-deriving the
same signals, and every consumer re-implements the traversal — which is exactly how a reader
acquires a wrong assumption without noticing.

The terrain work already proved the value of the stored form: once tiles were a table with named
signals, questions that had been guesses became one-line queries, and four separate wrong beliefs
were caught by cross-checking columns that had never been put side by side.

### Nested coordinate systems are the central hazard

PM4 packs data extremely tightly using coordinate rotations and nested frames, and the nesting is
not uniform even within one file. Measured on `development_00_00.pm4` via `pm4 inspect`:

| chunk | axis 1 | axis 2 | axis 3 |
|---|---|---|---|
| MSVT | 168–501 | 31–450 | −12–133 |
| MSCN | 169–499 | 31–450 | −12–133 |
| MPRL | 31–364 | 5–40 | 168–499 |

MPRL's third axis is MSVT's first. Two chunks of the same file, different axis order. A reader that
assumes one convention silently produces garbage — as happened during this spec's own research,
where hand-parsing MPRL outside the coordinate service produced a confident and completely wrong
conclusion about data being stacked above and below the map.

**The store must therefore record the coordinate space of every spatial signal as data, not leave it
as an assumption in a reader.**

### Three levels, measured

Reading PM4s as one map object, as tiles, and as objects are all necessary, and the corpus proves
they disagree. From `pm4 cross-tile` over the 616-file development map:

- 309 non-empty files, **1229 distinct CK24 object keys**
- **266 CK24 values (21.6%) span two or more tiles**
- 204 of those also span multiple MSHD Field04 buckets
- CK24 = 0 spans 291 tiles and is a null sentinel, not an object
- Genuine cross-tile objects span 3 to 8 tiles

A tile-keyed store fragments one object in five. This is the measurement that decides the layout.

Tile 0_0 holds 16 CK24 groups / 4,110 surfaces while its neighbour 0_1 holds 2 groups / 230
surfaces — an asymmetry consistent with a tile carrying data that appears to belong elsewhere. The
analyzer explicitly cannot confirm what *should* be where, because the development-map ADTs are not
populated.

### Confidence is not uniform and must survive into the store

The decode rests on real-data observation, and `pm4 inspect` already publishes its own confidence:
MSLK TypeFlags "medium" with buckets 0x03 / 0x10 / 0x12 recorded as *"partial, not corpus-closed"*;
MSUR GroupKey and AttributeMask "low"; MSUR Height "medium" and behaving as a signed plane-distance
rather than a vertical height; MSLK GroupObjectId "low" and explicitly not a confirmed object
identity. A store that flattens these into equally-authoritative columns launders assumption into
fact.

### Resolution

PM4 geometry exceeds WMO resolution substantially and is **not distilled**, unlike terrain. It is
the highest-fidelity surviving record of placement geometry, which is why it is worth storing at
full fidelity rather than summarising.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Query decoded PM4 signals without re-reading the files (Priority: P1)

An analyst asks a question spanning the corpus — every object of a given type, every surface above a
plane distance, every tile with more than N groups — and answers it against the stored dataset
rather than by re-traversing 616 files.

**Why this priority**: This is the deliverable. Everything else is a property of it.

**Independent Test**: Reproduce a known `pm4 cross-tile` result (1229 distinct CK24, 266 cross-tile)
from the store alone, without reading a PM4.

**Acceptance Scenarios**:

1. **Given** a built dataset, **When** the analyst counts distinct object keys and those spanning
   multiple tiles, **Then** the answers match the CLI analyzer's on the same corpus.
2. **Given** a built dataset, **When** any stored signal is compared against the same signal read
   through the canonical decoder, **Then** they are identical.
3. **Given** a dataset, **When** the analyst inspects any spatial signal, **Then** its coordinate
   space is readable from the dataset itself.

---

### User Story 2 - Reach the same data at map, tile, and object level (Priority: P1)

The analyst moves between "everything in this map", "everything in this tile", and "everything
belonging to this object" without a different tool or a manual join for each, and an object spanning
several tiles stays one object.

**Why this priority**: Equal to US1 — a store that only answers one of the three levels forces the
other two back into ad-hoc traversal, which is the problem being solved. 21.6% of objects span
tiles, so this is not hypothetical.

**Independent Test**: Take a CK24 known to span 5 tiles; retrieve it as one object, then retrieve
each of its tiles, and confirm the parts reconcile.

**Acceptance Scenarios**:

1. **Given** an object spanning several tiles, **When** it is retrieved by object identity, **Then**
   all its parts are returned together with the tiles they came from.
2. **Given** a tile, **When** its contents are retrieved, **Then** objects only partly inside it are
   present and marked as extending beyond it.
3. **Given** the whole map, **When** totals are computed, **Then** an object spanning tiles is
   counted once, not once per tile.
4. **Given** the null object key, **When** the dataset is queried, **Then** it is identifiable as a
   sentinel and excludable, not counted as an object spanning 291 tiles.

---

### User Story 3 - Tell decoded fact from working assumption (Priority: P2)

An analyst using a field can see how well understood it is, and can list every low-confidence field
before publishing a conclusion that rests on one.

**Why this priority**: The decode is partly assumption by the author's own account. Without this the
dataset makes a guess indistinguishable from a certainty — but US1/US2 still deliver value first.

**Independent Test**: Query the dataset for all fields below a confidence level and confirm the list
matches what the inspector reports.

**Acceptance Scenarios**:

1. **Given** any stored field, **When** the analyst inspects it, **Then** its confidence and any
   caveat travel with it.
2. **Given** a field whose interpretation later changes, **When** the dataset is rebuilt, **Then**
   datasets built under different interpretations are distinguishable and never silently compared.
3. **Given** a field marked "not corpus-closed", **When** the analyst reads it, **Then** that is
   stated rather than implied by absence.

---

### Edge Cases

- Chunks present in some files and absent in others (MDSF is 21,472 bytes in 0_0 and 0 in 0_1/0_2).
- A file with a chunk repeated many times (0_0 carries 40+ MDBI/MDBF sub-chunks).
- Empty tiles: 307 of 616 files are empty and must be representable as empty, not missing.
- An object whose parts use different coordinate frames.
- Signals whose length varies per object (surfaces, links) alongside fixed-size per-tile counts.
- A cross-tile object where one tile's contribution is a single surface.
- Unknown chunks in a future client build.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The dataset MUST be built through the canonical PM4 decoder, never a reimplementation
  of chunk parsing or coordinate handling.
- **FR-002**: Every spatial signal MUST record its coordinate space, and the transforms between
  spaces MUST be part of the published dataset contract rather than reader knowledge.
- **FR-003**: The dataset MUST support retrieval at map, tile, and object level, where an object
  spanning tiles remains a single object with its tile membership recorded.
- **FR-004**: The dataset MUST preserve full geometry fidelity; no lossy summarisation of PM4
  geometry.
- **FR-005**: Every field MUST carry its confidence level and any caveat published by the decoder.
- **FR-006**: The dataset MUST record a version for the decode interpretation it was built under,
  and consumers MUST be able to detect a mismatch rather than silently compare across versions.
- **FR-007**: The dataset MUST distinguish a signal that is genuinely absent for a tile or object
  from one that was not extracted, and from a sentinel value such as the null object key.
- **FR-008**: The dataset MUST represent empty tiles explicitly.
- **FR-009**: Building the dataset MUST be reproducible: the same inputs and decoder version produce
  an identical dataset.
- **FR-010**: The dataset MUST record which chunks were present per file, including repeats, so
  absence is queryable rather than inferred.
- **FR-011**: The dataset MUST retain the object-key structure the decoder derives (the composite
  key and its type and object components) so grouping does not require re-deriving it.
- **FR-012**: The dataset MUST be verifiable against the existing analyzers: corpus-level counts
  reproduced from the dataset MUST match the analyzers on the same input.

### Key Entities

- **Map**: One PM4 corpus for one map, at one client build, under one decode version.
- **Tile**: One PM4 file's worth of content, including the empty case, with its per-tile counts and
  chunk presence.
- **Object**: A group identified by the decoder's composite key, its parts, the tiles it spans, and
  whether it is a sentinel rather than a real object.
- **Surface / link / position record**: The decoded per-object and per-tile geometry and linkage
  signals, each with its coordinate space and confidence.
- **Decode interpretation**: The versioned set of field meanings and confidences the dataset was
  built under.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Corpus statistics computed from the dataset match the existing analyzers exactly on
  the same input — including 1229 distinct object keys and 266 spanning multiple tiles.
- **SC-002**: Every spatial signal's coordinate space is discoverable from the dataset without
  consulting source code.
- **SC-003**: An object spanning multiple tiles is retrievable as one object, and totals count it
  once; verified against objects known to span 3, 5 and 8 tiles.
- **SC-004**: Every field carries a confidence level; the set of low-confidence fields is queryable
  and matches what the inspector reports.
- **SC-005**: Two builds from identical inputs and decoder version produce identical datasets.
- **SC-006**: A question that currently requires re-reading the corpus is answerable from the
  dataset alone.
- **SC-007**: Datasets built under different decode versions are detectably different and cannot be
  compared without the mismatch being surfaced.

## Assumptions

- **Object-primary layout.** Because 21.6% of objects span two or more tiles, a tile-keyed layout
  would fragment one object in five. The dataset is therefore organised so an object is the unit
  that stays whole, with tile membership recorded against it and a tile-level index provided for
  tile-scoped queries. This is the one structural decision worth revisiting first if it proves
  wrong in practice.
- The 616-file development map is the initial corpus; the design generalises to other maps.
- The existing C# PM4 stack is the decoder. This feature adds a stored form and does not re-decode.
- Field confidences and caveats come from the decoder's published terminology, not from new
  research; improving them is out of scope.
- The v50 zarr store conventions (a signal manifest, declared signals, recorded provenance) are the
  model to follow, adapted for variable-length per-object data.
- Out of scope: improving PM4 to asset matching (Spec 128), CASC support, and the terrain
  archaeology pipeline.
