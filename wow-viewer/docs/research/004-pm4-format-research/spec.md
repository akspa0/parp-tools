# Research Specification: PM4 Format — Full Audit and Unknowns Resolution

**Feature Branch**: `004-pm4-format-research`

**Created**: 2026-05-20

**Status**: Draft

**Input**: "PM4 may be a datastore format more than just a pathfinding overlay. We need a complete audit of every chunk — decoded vs. unknown — and a path to resolving MSHD and the remaining unknowns."

---

## 1. Problem Statement

The PM4 format has been reverse-engineered piecemeal across multiple codebases (MdxViewer, PM4Tool, Pm4Research, WoWRollback, wow-viewer) over several years. The current working hypothesis — that PM4 is a "server-side pathfinding supplement" — is a legacy label from the wowdev.wiki community. The actual data in PM4 files is substantially richer:

- **64.1% of all MSUR surfaces** belong to CK24 objects with type bytes identifying WMO (0x42/0x43) or M2 (0x40/0x41) collision geometry, not navigation mesh.
- **MSCN contains ~98% unique geometry** not present in MSVT — a separate collision hull layer.
- **MPRL positions** validate against `_obj0.adt` placement data — these are terrain-object intersection points.
- **CK24 objects cross tile boundaries** — 21.6% of CK24 values span multiple tiles.
- **Destructible building chunks** (MDBH/MDOS/MDSF) encode building state, not pathfinding.

The MSHD header chunk (32 bytes, 8x uint32) is completely opaque — fields 0x0C-0x1C are zero across the entire development corpus, and no correlation has been found between MSHD fields and any chunk counts, bounds, or metrics. This is a critical gap: if MSHD is a layout descriptor, we cannot trust our interpretation of the rest of the format.

**Goal**: Produce a single authoritative document that maps every PM4 chunk and field to a confidence level (verified / partial / unknown), identifies the remaining unknowns, and defines a research path to resolve them.

---

## 2. Research Objectives

### Primary

- **OBJ-1**: Audit every PM4 chunk field across all known chunk types, classify each as Verified (byte-layout + semantics proven), Partial (byte-layout known, semantics open), or Unknown (no decode).
- **OBJ-2**: Determine whether MSHD is a layout descriptor, a version gate, or dead padding. Identify whether fields 0x0C-0x1C are truly unused or encode information only populated in non-development clients.
- **OBJ-3**: Resolve the "datastore vs. pathfinding" question by mapping chunk relationships into a data-flow graph that reveals the actual information architecture.

### Secondary

- **OBJ-4**: Identify which unknown fields are high-impact (affecting object reconstruction, coordinate solving, or cross-tile linkage) vs. low-impact (diagnostic, redundant, or vestigial).
- **OBJ-5**: Define the minimum set of field resolutions needed to produce a self-contained PM4 specification that does not rely on external context (ADT, WDT, client memory).

---

## 3. Scope

### In Scope

All 16 known PM4 chunk types across the development corpus (616 tiles, 309 non-empty):

| Category | Chunks |
|----------|--------|
| Header | MVER, MSHD |
| Geometry | MSVT, MSVI, MSUR |
| Pathfinding | MSPV, MSPI |
| Links | MSLK |
| Scene | MSCN |
| Position Refs | MPRL, MPRR |
| Destructible | MDBH, MDBI, MDBF, MDOS, MDSF |

Plus: any unknown/undecoded trailing chunks found in specific files.

### Out of Scope

- Rendering pipeline changes in MdxViewer or WowViewer.App
- New PM4 writer implementations
- Changes to WowViewer.Core.PM4 reader code (this is a research/specification project)
- Multi-client-format PM4 variants (PD4, etc.) — unless directly relevant to resolving unknowns

---

## 4. Current State — Chunk Audit

### 4.1 MVER (Version)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — uint32, 4 bytes |
| Semantics | **Verified** — Version number (12304 in development corpus) |
| Unknowns | None |

### 4.2 MSHD (Mesh Header) — CRITICAL UNKNOWN

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — 32 bytes, 8x uint32 |
| Semantics | **Unknown** |

**Fields**:

| Offset | Field | Dev Corpus Value | Status |
|--------|-------|-----------------|--------|
| 0x00 | Field00 | Non-zero, varies | **Unknown** — correlates weakly with MSUR count but no exact match |
| 0x04 | Field04 | 1 (constant) | **Unknown** — possibly version or flag |
| 0x08 | Field08 | Non-zero, varies | **Unknown** — often equals Field00 |
| 0x0C | Field0C | 0 | **Unknown** — zero across all 616 files |
| 0x10 | Field10 | 0 | **Unknown** — zero across all 616 files |
| 0x14 | Field14 | 0 | **Unknown** — zero across all 616 files |
| 0x18 | Field18 | 0 | **Unknown** — zero across all 616 files |
| 0x1C | Field1C | 0 | **Unknown** — zero across all 616 files |

**Research questions**:
1. Are fields 0x0C-0x1C reserved for future use, or do they encode something only populated in non-development builds?
2. Does Field00/Field08 encode total surface count, total vertex count, or a memory layout hint?
3. Is Field04 always 1 across the entire WoW client corpus, or only in development?

### 4.3 MSVT (Mesh Vertices)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — Vector3 array, stride 12 bytes |
| Semantics | **Verified** — Tile-local mesh vertices (XYPlaneZUp, range 0..533.33) |
| Unknowns | None for basic decode. Coordinate ownership (tile-local vs world-space) is file-dependent and solved by heuristic. |

### 4.4 MSVI (Mesh Vertex Indices)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — uint32 array, stride 4 bytes |
| Semantics | **Verified** — Indices into MSVT |
| Unknowns | None |

### 4.5 MSUR (Mesh Surface)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — 32 bytes per entry |
| Semantics | **Partial** — core fields decoded, several open |

**Fields**:

| Offset | Field | Status | Notes |
|--------|-------|--------|-------|
| 0x00 | GroupKey | **Unknown** | "3=terrain, 18/19=WMO" per spec but not closed |
| 0x01 | IndexCount | **Verified** | Triangle fan/loop count (indices = IndexCount * 3) |
| 0x02 | AttributeMask | **Unknown** | Bit meanings open. Bit 7 = liquid? |
| 0x03 | Padding | **Verified** | Zero padding |
| 0x04-0x0F | Normal | **Verified** | True surface normal (validated on 518k surfaces) |
| 0x10 | Height | **Verified** | Signed plane-distance term |
| 0x14 | MsviFirstIndex | **Verified** | Start index into MSVI |
| 0x18 | MdosIndex | **Partial** | Index into MSCN; cross-reference with MDOS not fully validated |
| 0x1C | PackedParams | **Partial** | CK24, CK24Type, CK24ObjectId decoded; raw field meaning open |

### 4.6 MSPV (Path Vertices)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — Vector3 array, stride 12 bytes |
| Semantics | **Verified** — Navigation path vertices |
| Unknowns | None |

### 4.7 MSPI (Path Indices)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — uint32 array, stride 4 bytes |
| Semantics | **Verified** — Indices into MSPV |
| Unknowns | None |

### 4.8 MSLK (Mesh Link) — PARTIALLY DECODED

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — 20 bytes per entry |
| Semantics | **Partial** — several fields open |

**Fields**:

| Offset | Field | Status | Notes |
|--------|-------|--------|-------|
| 0x00 | TypeFlags | **Unknown** | "1=walkable, 2=walls" per spec; not validated |
| 0x01 | Subtype | **Unknown** | "Floor level?" — looks layer-like but not closed |
| 0x02-0x03 | Padding | **Verified** | |
| 0x04 | GroupObjectId | **Partial** | Low16 maps to CK24ObjectId with reuse; not globally unique |
| 0x08 | MspiFirstIndex | **Verified** | int24 first index into MSPI |
| 0x0B | MspiIndexCount | **Partial** | Ambiguity: "indices mode" vs "triangles mode" (count*3) |
| 0x0C | LinkId | **Verified** | Tile coordinate sentinel 0xFFFF_XXYY (100% decoded) |
| 0x10 | RefIndex | **Partial** | Primary: MSUR index. 4553/1273335 entries fail MSUR fit. Multi-domain. |
| 0x12 | SystemFlag | **Partial** | 0x8000 dominates — likely constant flag |

**Research questions**:
1. What does TypeFlags actually classify? Walkable vs. obstacle? Or something else?
2. Is Subtype a floor level, a layer index, or an object part identifier?
3. What is the RefIndex target when it doesn't map to MSUR? MSPI? MSVI? MSCN? MPRL?
4. Does MspiIndexCount ambiguity (indices vs triangles) indicate two distinct link types?

### 4.9 MSCN (Scene Nodes / Exterior Vertices)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — Vector3 array, stride 12 bytes |
| Semantics | **Partial** — coordinate space ownership open |

**Known**:
- ~98% unique geometry not in MSVT (collision hull vertices)
- Referenced by MSUR.MdosIndex
- Coordinate space is file-dependent (raw-world, swapped-XY, or tile-local)

**Unknown**:
- Authoritative coordinate space
- How to determine which coordinate convention applies to a given file
- Relationship to MSUR surfaces beyond index reference

### 4.10 MPRL (Position References)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — 24 bytes per entry |
| Semantics | **Partial** — core fields decoded, several open |

**Fields**:

| Offset | Field | Status | Notes |
|--------|-------|--------|-------|
| 0x00 | Unk00 | **Unknown** | |
| 0x02 | Unk02 | **Unknown** | Often -1 |
| 0x04 | Unk04 | **Verified** | Heading: packed angle (* 360/65536 degrees) |
| 0x06 | Unk06 | **Unknown** | Often 0x8000 — constant flag? |
| 0x08-0x13 | Position | **Verified** | ADT-placement-space position (validated against _obj0.adt) |
| 0x14 | Unk14 | **Unknown** | "Floor/level-like" — range -1..15 |
| 0x16 | Unk16 | **Partial** | 0 = normal, non-zero = terminator |

### 4.11 MPRR (Position Reference Graph)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — 4 bytes per entry (2x uint16) |
| Semantics | **Partial** |

**Fields**:

| Offset | Field | Status | Notes |
|--------|-------|--------|-------|
| 0x00 | Value1 | **Partial** | References MPRL or MSVT; 0xFFFF = sentinel |
| 0x02 | Value2 | **Unknown** | Secondary field — meaning open |

**Research questions**:
1. Is MPRR a linked list, a graph, or a flat array with sentinel-delimited groups?
2. What does Value2 encode — a weight, a type, a secondary index?

### 4.12 MDBH (Destructible Building Header)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — uint32 count |
| Semantics | **Partial** — only populated on development_00_00 |

### 4.13 MDBI (Destructible Building Indices)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — uint32 array |
| Semantics | **Unknown** — minimal understanding |

### 4.14 MDBF (Destructible Building Filename)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — null-terminated ASCII string |
| Semantics | **Partial** — filename reference, but to what? |

### 4.15 MDOS (Destructible Object States)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — 8 bytes (uint32 + uint32) |
| Semantics | **Partial** — buildingIndex + destructionState; 24 invalid MDBH refs in corpus |

### 4.16 MDSF (Destructible Surface-to-Object Mapping)

| Attribute | Status |
|-----------|--------|
| Byte layout | **Verified** — 8 bytes (uint32 + uint32) |
| Semantics | **Partial** — MsurIndex + MdosIndex mapping |

---

## 5. The "Datastore" Hypothesis

The user's hypothesis is that PM4 is a **datastore format** — a structured container for game-world collision, placement, and destructible-state data — rather than a pathfinding-specific overlay. Evidence:

1. **CK24 type classification** separates nav-mesh (0x00) from WMO (0x42/0x43) and M2 (0x40/0x41) collision. Pathfinding is 35.9% of surfaces; object collision is 64.1%.
2. **MSLK** links surfaces to position references and encodes tile coordinates — this is a scene-graph structure, not a nav-graph.
3. **MPRL** stores placement positions validated against ADT object data — this is world-building metadata.
4. **MSCN** stores collision hull geometry unique from the mesh — this is physics data.
5. **MDBH/MDOS/MDSF** encode destructible building state — this is gameplay data.
6. **Cross-tile CK24 objects** span multiple tiles — this is a global scene description, not local pathfinding.

**If PM4 is a datastore**, then:
- MSHD likely encodes a layout table (offsets, counts, or region boundaries) for the chunks that follow.
- Unknown fields in MSLK, MSUR, MPRL, MPRR likely encode metadata about data ownership, layering, or state.
- The format is designed for server-side scene queries (collision, placement, destruction) rather than client-side navigation.

---

## 6. User Stories

### User Story 1 — Complete Chunk Field Audit (Priority: P1)

As a format researcher, I need every field of every PM4 chunk classified as Verified, Partial, or Unknown with evidence, so that I can identify the exact resolution work remaining.

**Why this priority**: Without a complete audit, we cannot distinguish "we haven't looked" from "it's genuinely unknown." This is the foundation for all other research.

**Independent Test**: Audit document exists with every chunk/field mapped. Cross-reference against existing analyzers (Pm4ResearchAuditAnalyzer, Pm4ResearchUnknownsAnalyzer) to confirm no fields are missed.

**Acceptance Scenarios**:

1. **Given** the 16 known chunk types, **When** the audit is complete, **Then** every byte offset in every chunk has a classification (Verified/Partial/Unknown) with a one-line justification.
2. **Given** a field classified as Unknown, **When** the audit is reviewed, **Then** the field has a research question describing what evidence would resolve it.
3. **Given** the development corpus (616 tiles), **When** field value distributions are computed, **Then** the audit includes min/max/mode/null-rate for every Partial or Unknown field.

---

### User Story 2 — MSHD Resolution (Priority: P1)

As a format researcher, I need to determine whether MSHD fields 0x00 and 0x08 encode total surface/vertex counts or memory layout hints, and whether fields 0x0C-0x1C are reserved or dead, so that we can either decode the header or confirm it is not blocking format understanding.

**Why this priority**: MSHD is the only chunk that could全局 describe the rest of the format. If it encodes counts/offsets, our entire chunk-reading model may need adjustment. If it is dead padding, we can stop investigating it.

**Independent Test**: Run correlation analysis of MSHD.Field00 and MSHD.Field08 against MSUR.Count, MSVT.Count, MSPI.Count, MSLK.Count, MPRL.Count, MSCN.Count across the full development corpus. Test against non-development client PM4 files if available.

**Acceptance Scenarios**:

1. **Given** MSHD.Field00 across 616 development tiles, **When** correlation is computed against all chunk counts, **Then** either a statistically significant correlation is found (r > 0.95) or the field is classified as unknown/dead.
2. **Given** MSHD.Field0C-1C across 616 development tiles, **When** value distribution is computed, **Then** either non-zero values exist in non-development clients, or the fields are classified as reserved/unused.
3. **Given** any non-development PM4 files (e.g., retail client builds), **When** MSHD is inspected, **Then** the field values are recorded and compared to development values.

---

### User Story 3 — Datastore Architecture Document (Priority: P2)

As a format researcher, I need a data-flow graph showing how every PM4 chunk references every other chunk, so that I can evaluate the "datastore" hypothesis and identify the format's actual information architecture.

**Why this priority**: This is the intellectual payoff — moving from "here are chunks" to "here is the data model."

**Independent Test**: Document contains a directed graph (text or diagram) with nodes = chunks and edges = reference relationships (with field names). Cross-reference against Pm4ResearchUnknownsAnalyzer relationship edges.

**Acceptance Scenarios**:

1. **Given** all 16 chunk types, **When** the reference graph is built, **Then** every cross-chunk reference edge has a source field, target chunk, and confidence level.
2. **Given** the reference graph, **When** the "datastore" hypothesis is evaluated, **Then** the document states whether PM4 is best described as (a) a pathfinding supplement, (b) a collision/placement datastore, or (c) a hybrid — with evidence.
3. **Given** the reference graph, **When** orphaned chunks or fields are identified, **Then** each orphan has a research note about whether it is genuinely disconnected or just not yet understood.

---

### User Story 4 — High-Impact Unknowns Prioritization (Priority: P2)

As a format researcher, I need the unknowns ranked by impact on object reconstruction and coordinate solving, so that resolution effort is directed where it matters most.

**Why this priority**: Not all unknowns are equal. Resolving MSLK.TypeFlags might unlock layer semantics; resolving MPRL.Unk06 (a constant flag) is low value.

**Independent Test**: Unknowns list has an impact score (High/Medium/Low) with justification based on: (a) does it affect coordinate solving, (b) does it affect object grouping, (c) does it affect cross-tile linkage.

**Acceptance Scenarios**:

1. **Given** all Unknown/Partial fields, **When** impact is assessed, **Then** each field has a High/Medium/Low rating with a one-line justification.
2. **Given** High-impact unknowns, **When** the list is ordered, **Then** the top 5 fields are identified as the minimum resolution set for a self-contained spec.

---

### User Story 5 — Cross-Client Validation Plan (Priority: P3)

As a format researcher, I need a plan for validating PM4 field semantics against non-development client builds, so that we can distinguish development-only artifacts from format-wide truths.

**Why this priority**: The development corpus is a single build. Some MSHD fields may only be populated in retail. Some chunk semantics may differ across expansions.

**Independent Test**: Plan identifies which client builds to stage, what commands to run, and what field values to compare.

**Acceptance Scenarios**:

1. **Given** the staged client data at `output/tmp/wowarchive-clients/`, **When** PM4 files from different builds are inspected, **Then** the plan specifies which fields to compare and what constitutes a meaningful difference.
2. **Given** the plan, **When** it is reviewed, **Then** it identifies at least 3 client builds spanning different expansions for comparison.

---

## 7. Edge Cases

- What if MSHD.Field00/Field08 are actually hash values rather than counts?
- What if MSLK.RefIndex is context-dependent (points to different chunk types based on TypeFlags)?
- What if MPRR is not a graph but a flat array with a specific packing order?
- What if some PM4 files contain unknown chunk types not in the current 16?
- What if fields 0x0C-0x1C in MSHD are only populated in PD4 (the retail variant) and are genuinely dead in PM4?

---

## 8. Functional Requirements

- **FR-001**: Research document MUST classify every byte offset in every known PM4 chunk as Verified, Partial, or Unknown.
- **FR-002**: Research document MUST include value distributions (min/max/mode/null-rate) for all Unknown and Partial fields across the development corpus.
- **FR-003**: Research document MUST include a directed reference graph of all cross-chunk relationships.
- **FR-004**: Research document MUST rank unknowns by impact on object reconstruction, coordinate solving, and cross-tile linkage.
- **FR-005**: Research document MUST include a cross-client validation plan with at least 3 target builds.
- **FR-006**: MSHD analysis MUST include correlation of Field00/Field08 against all chunk counts across the full corpus.
- **FR-007**: MSHD analysis MUST include inspection of non-development PM4 files if available.
- **FR-008**: All claims MUST cite file paths, line numbers, and evidence from the codebase.

---

## 9. Key Entities

- **PM4 Chunk**: A typed data block within a .pm4 file. 16 known types. Each has a FourCC signature, a byte size, and a typed payload.
- **CK24**: A 24-bit object identity key extracted from MSUR.PackedParams. Groups surfaces into coherent objects. Type byte separates nav-mesh, WMO, and M2.
- **MSLK RefIndex**: A 16-bit reference from MSLK entries to other chunks. Primary target is MSUR; secondary targets unknown.
- **MPRL Position**: A 24-byte placement record with position, heading, and flags. Validates against ADT object placements.
- **MSHD Header**: A 32-byte header at the start of PM4. Completely opaque. Possible layout descriptor or dead padding.
- **MPRR Graph**: A 4-byte-per-entry graph structure chaining MPRL/MSVT references with sentinel delimiters.

---

## 10. Success Criteria

- **SC-001**: 100% of byte offsets in all 16 chunk types have a Verified/Partial/Unknown classification with evidence.
- **SC-002**: MSHD is either decoded (Field00/Field08 correlation found) or classified as "not blocking format understanding" with supporting evidence.
- **SC-003**: The reference graph contains all cross-chunk edges found in Pm4ResearchUnknownsAnalyzer plus any new edges discovered.
- **SC-004**: The top 5 high-impact unknowns are identified with concrete research steps to resolve each.
- **SC-005**: The datastore hypothesis is evaluated with evidence — not just stated as a possibility.

---

## 11. Assumptions

- The development corpus (616 tiles) is representative of the PM4 format structure.
- Non-development client PM4 files may have different field values but the same chunk structure.
- The 16 known chunk types cover all data in standard PM4 files (no hidden chunks).
- MSHD fields 0x0C-0x1C being zero in the development corpus is meaningful (not just uninitialized memory).
- The existing analyzer code (Pm4ResearchAnalyzer, Pm4ResearchUnknownsAnalyzer, etc.) is correct in its field-level decode.
- This research project produces documentation only — no code changes to wow-viewer or gillijimproject_refactor.

---

## 12. Source References

| Document | Path | Lines |
|----------|------|-------|
| PM4 Current Decoding Logic | `gillijimproject_refactor/documentation/pm4-current-decoding-logic-2026-03-20.md` | 731 |
| PM4 Raw Unknowns Map | `gillijimproject_refactor/documentation/pm4-raw-unknowns-map-2026-03-21.md` | 421 |
| PM4 Format Specification (WoWRollback) | `gillijimproject_refactor/WoWRollback/docs/-specifications-/PM4-Format-Specification.md` | 711 |
| PM4 Specification (next) | `gillijimproject_refactor/next/parpDocumentation/pm4-specification.md` | 202 |
| PM4 Chunk Types (gillijimproject) | `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Formats/PM4/Pm4ChunkTypes.cs` | 91 |
| PM4 Research Chunk Models | `wow-viewer/src/core/WowViewer.Core.PM4/Pm4ResearchChunkModels.cs` | — |
| PM4 Research Unknowns Analyzer | `wow-viewer/src/core/WowViewer.Core.PM4/Pm4ResearchUnknownsAnalyzer.cs` | — |
| PM4 Research MSHD Analyzer | `wow-viewer/src/core/WowViewer.Core.PM4/Pm4ResearchMshdAnalyzer.cs` | — |
| PM4 Research Integration Tests | `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4ResearchIntegrationTests.cs` | 1025 |
| Viewer PM4 Utilities | `gillijimproject_refactor/src/MdxViewer/ViewerApp_Pm4Utilities.cs` | 1380+ |
| PM4 Object Builder | `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Decoding/Pm4ObjectBuilder.cs` | 272 |
