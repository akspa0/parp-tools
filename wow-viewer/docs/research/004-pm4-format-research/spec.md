# Research Specification: PM4 Format — Full Audit and Unknowns Resolution

**Feature Branch**: `004-pm4-format-research`

**Created**: 2026-05-20

**Status**: Draft

**Input**: "PM4 is a compressed server-side pathfinding dataset from ~2010. MSCN contains surface centroids from the MSVT mesh. MSLK is a graph edge catalog. The peg/dowel points at tile boundaries are cross-tile pathfinding connection markers. Our naming has drifted from wowdev.wiki and our understanding of the data model has accumulated errors from early GPT-3.5 hallucinations. We need to correct the naming, verify the centroid hypothesis, and rebuild our understanding from first principles."

---

## 1. Problem Statement

The PM4 format has been reverse-engineered piecemeal across multiple codebases (MdxViewer, PM4Tool, Pm4Research, WoWRollback, wow-viewer) over several years. The current working hypothesis — that PM4 is a "server-side pathfinding supplement" — is a legacy label from the wowdev.wiki community. The actual data in PM4 files is substantially richer:

- **64.1% of all MSUR surfaces** belong to CK24 objects with type bytes identifying WMO (0x42/0x43) or M2 (0x40/0x41) collision geometry, not navigation mesh.
- **MSCN contains ~98% unique geometry** not present in MSVT — a separate collision hull layer.
- **MPRL positions** validate against `_obj0.adt` placement data — these are terrain-object intersection points.
- **CK24 objects cross tile boundaries** — 21.6% of CK24 values span multiple tiles.
- **Most cross-tile CK24 objects bridge multiple `MSHD.Field04` buckets** — `204/266` cross-tile CK24 values in the development corpus span 2+ distinct Field04 values, so Field04 is not the stitch key for most multi-tile WMO/M2 objects.
- **Destructible building chunks** (MDBH/MDOS/MDSF) encode building state, not pathfinding.

The MSHD header chunk (32 bytes, 8x uint32) is completely opaque — fields 0x0C-0x1C are zero across the entire development corpus, and no correlation has been found between MSHD fields and any chunk counts, bounds, or metrics.

Additionally, our codebase has accumulated naming errors and semantic misunderstandings from early work with GPT-3.5 hallucinations. Fields were given invented names that hardened into "truth" through repetition. The MSCN chunk was misidentified as "collision wall vertices" when it actually contains surface centroids. MSLK was interpreted as a scene-graph linkage when it is a pathfinding graph edge catalog.

### Project Context

This work is a continuation of a 2021 hobbyist effort to reconstruct the WoW development map from server-side data. The PM4 format is the **definitive object reference** — it tells us what objects exist and where they belong, so we can match PM4 data to real WMO/M2 objects and construct appropriate placement data for the development map, which is incomplete.

The 2021 group tried to fix the development map by hand and got close on some objects but not 100% correct. The user's PM4 decoding work surpassed what they achieved manually. Terrain reconstruction work has been validated by that community.

**The downstream goal**: Match PM4 collision/pathfinding objects to real WMO/M2 assets, then place them correctly on the development map. This eliminates the need for expensive model training to match objects to PM4 data — the format itself encodes the answer, if we can read it correctly.

**The ground truth**: Two screenshots from WoWEdit (Data 1.9.0) in "The WoW Diary" by John Staats show what the PM4 data represents:
1. **Outdoor view**: Terrain mesh (gray), wall/barrier edges (red), navigation nodes (blue markers) — the pathfinding collision mesh with graph nodes
2. **Interior view**: WMO dungeon with floor surfaces, walls, staircase (multi-level), and M2 doodads (candelabras, banner) — MPRL stores doodad placements as collision obstacles

These screenshots are the visual ground truth for what PM4 encodes. The challenge is matching PM4 collision objects (CK24 groups) to the real WMO/M2 models visible in the editor.

**Goal**: Rebuild our understanding of the PM4 format from first principles. Correct all naming drift from wowdev.wiki. Verify what MSCN, MSLK, and MPRL actually encode. Produce a single authoritative document that maps every chunk and field to a confidence level (verified / partial / unknown).

### The Object Splitting Problem

The core unsolved problem is **how to decompose CK24 groups into individual objects**. Today:

1. CK24 groups surfaces by a 24-bit key extracted from MSUR.PackedParams. The viewer treats this as a flat grouping key.
2. Within each CK24 group, the viewer applies `SplitSurfaceGroupByMslk()` — a union-find on MSLK.GroupObjectId — to decompose into sub-objects.
3. CK24=0 surfaces (35.9% of all surfaces) are handled separately: sub-grouped by `(GroupKey, AttributeMask)` then split by connectivity.
4. Cross-tile merging uses MSCN connector keys, NOT CK24 matching.

**What we don't know:**
- Whether CK24 itself encodes a hierarchy (the LSB-as-base-group hypothesis: 0x000000=group 0, 0x000001=group 1). Current evidence is inconclusive — Ck24ObjectId (low 16 bits) shows reuse across type bytes, arguing against clean hierarchy.
- What MSLK.TypeFlags and MSLK.Subtype actually classify — current real-data inspection now suggests `TypeFlags` carries per-surface family buckets (`0x03` = M2 top surfaces, `0x10` = interior WMO floors, `0x12` = exterior WMO solid surfaces), but the mapping is not corpus-closed yet and must stay distinct from `GroupObjectId`.
- How to properly split a single CK24 value into sub-objects when the same CK24 spans multiple tiles and contains hundreds of surfaces.
- Whether the low byte of MSUR.PackedParams (bits 0-7, currently discarded by the `>> 8` shift) carries meaningful data.

**What MSCN adds:**
- MSCN positions are consumed via `MSUR.MdosIndex` as connector keys for cross-tile object merging.
- MSCN contains ~98% unique geometry not in MSVT — "collision wall vertices" that represent object shapes.
- The active viewer only uses MSCN positions referenced by MdosIndex. Most MSCN data is unread.
- MSCN may be the key to understanding object containment and boundaries across tiles.

---

## 2. Research Objectives

### Primary

- **OBJ-1**: Audit every PM4 chunk field across all known chunk types, classify each as Verified (byte-layout + semantics proven), Partial (byte-layout known, semantics open), or Unknown (no decode).
- **OBJ-2**: Determine whether MSHD is a layout descriptor, a version gate, or dead padding. Identify whether fields 0x0C-0x1C are truly unused or encode information only populated in non-development clients.
- **OBJ-3**: Resolve the "datastore vs. pathfinding" question by mapping chunk relationships into a data-flow graph that reveals the actual information architecture.

### Secondary

- **OBJ-4**: Identify which unknown fields are high-impact (affecting object reconstruction, coordinate solving, or cross-tile linkage) vs. low-impact (diagnostic, redundant, or vestigial).
- **OBJ-5**: Define the minimum set of field resolutions needed to produce a self-contained PM4 specification that does not rely on external context (ADT, WDT, client memory).
- **OBJ-6**: Determine the correct CK24 decomposition strategy — whether CK24's 24 bits encode a hierarchy, whether MSLK.TypeFlags/Subtype provide the missing linking layer, or whether MSCN boundaries define the true object segmentation.
- **OBJ-7**: Map MSCN consumption — how much of the MSCN data is actually used today, what the unused portion contains, and whether the full MSCN data enables better object boundary detection across tiles.
- **OBJ-8**: Determine whether the low byte of MSUR.PackedParams (bits 0-7, currently discarded) carries meaningful data that could aid object splitting.
- **OBJ-9**: Audit all PM4 field names in the codebase against wowdev.wiki PM4/PD4 documentation, correct the `MdosIndex` → `MscnIndex` naming error, and document the mapping for every invented name.
- **OBJ-10**: Verify the MSCN centroid hypothesis — compute MSUR surface centroids and compare against MSCN points accessed via `MSUR._0x18`. If they match, confirm MSCN as the surface-centroid navigation node layer.
- **OBJ-11**: Identify MSCN "peg/dowel" points that fall outside tile bounds and correlate them with cross-tile CK24 objects, to map the cross-tile pathfinding connection network.
- **OBJ-12**: Reinterpret MSLK as a pathfinding graph edge catalog — test whether TypeFlags/classifies edge types (walkable, wall, ledge, etc.) and Subtype encodes edge properties (height level, direction, etc.).
- **OBJ-13**: Investigate MPRL's dual role — the user reports MPRL contains both terrain intersection points AND WMO doodad references. Determine whether MPRL serves dual purposes (pathfinding nodes + doodad placement) or whether the doodad data is actually a different chunk overlap.
- **OBJ-14**: Investigate the M2/WMO bleed problem in object splitting — adjacent M2 data sometimes gets included in WMO objects. Determine whether CK24 type bytes (0x40 vs 0x42/0x43) are insufficient for type separation, or whether the bleed comes from MSLK edge linking across type boundaries.
- **OBJ-15**: Investigate rare field values across all chunks — values that appear only once or twice in the corpus may be root group IDs or top-level scene identifiers, not noise. Cross-correlate rare values in MSHD, MSLK.Subtype, MSLK.TypeFlags, and MSUR.GroupKey to find the missing root grouping layer between CK24 and individual polygons.
- **OBJ-16**: Re-examine MSHD with the assumption that it encodes something that helps decode other chunks — test whether MSHD fields correlate with rare values in other chunks, or encode region/scene boundaries.

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
| 0x04 | Field04 | **227 distinct values; =1 only on empty tiles** | **Partial** — region-like scene bucket, but **not** packed tile `XX_YY`. Corpus proof: `0/502` files match `(TileX << 8) | TileY`, `0/502` match `(TileY << 8) | TileX`, and `73` distinct values are reused across multiple tiles. |
| 0x08 | Field08 | Non-zero, varies | **Unknown** — often equals Field00 |
| 0x0C | Field0C | 0 | **Unknown** — zero across all 616 files |
| 0x10 | Field10 | 0 | **Unknown** — zero across all 616 files |
| 0x14 | Field14 | 0 | **Unknown** — zero across all 616 files |
| 0x18 | Field18 | 0 | **Unknown** — zero across all 616 files |
| 0x1C | Field1C | 0 | **Unknown** — zero across all 616 files |

**Research questions**:
1. Are fields 0x0C-0x1C reserved for future use, or do they encode something only populated in non-development builds?
2. Does Field00/Field08 encode a version, checksum, or memory layout hint? Both favor 534 as dominant value but differ in ~72% of tiles.
3. **BREAKTHROUGH (updated 2026-06-03)**: Field04 is **not** packed tile `XX_YY`. The strongest byte-level coincidence is only `6/502` files, and `73` distinct Field04 values span multiple tiles. It still behaves like a reusable scene/group bucket, not a per-tile coordinate key.
4. What does Field04 actually encode when values like `3262` are reused across disconnected tile clusters (35_42-36_45 and 45_46-47_51)? Same scene archetype, nav-region family, or authoring bucket?

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
| 0x18 | _0x18 (MscnIndex) | **Verified** | Index into MSCN (scene nodes). Previously misnamed `MdosIndex`. Renamed 2026-05-20. |
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
| 0x00 | TypeFlags | **Partial** | Real-data inspection now suggests per-surface family buckets: `0x03` = M2 top surfaces, `0x10` = interior WMO floors, `0x12` = exterior WMO solid surfaces. Needs corpus-wide closure and offset-ownership recheck before treating as final. |
| 0x01 | Subtype | **Unknown** | "Floor level?" — looks layer-like but not closed |
| 0x02-0x03 | Padding | **Verified** | |
| 0x04 | GroupObjectId | **Partial** | Low16 maps to CK24ObjectId with reuse; not globally unique |
| 0x08 | MspiFirstIndex | **Verified** | int24 first index into MSPI |
| 0x0B | MspiIndexCount | **Partial** | Ambiguity: "indices mode" vs "triangles mode" (count*3) |
| 0x0C | LinkId | **Verified** | Tile coordinate sentinel 0xFFFF_XXYY (100% decoded) |
| 0x10 | RefIndex | **Partial** | Primary: MSUR index. 4553/1273335 entries fail MSUR fit. Multi-domain. |
| 0x12 | SystemFlag | **Partial** | 0x8000 dominates — likely constant flag |

**Research questions**:
1. Are the observed `TypeFlags` buckets (`0x03`, `0x10`, `0x12`) stable across the full corpus and other client builds, or are they only the first confirmed families?
2. Is Subtype a floor level, a layer index, or an object part identifier once TypeFlags is held fixed?
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

## 5. The Data Model — Rebuilt From First Principles

PM4 is a **compressed server-side pathfinding dataset** from ~2010. The format encodes a navigation graph in which:

- **MSHD.Field04** is a reusable scene/group bucket, not packed tile coordinates. It is still useful for grouping/coloring/debugging, but the corpus no longer supports treating it as a one-value-per-tile key or as the primary stitch key for most multi-tile WMO/M2 objects.
- **MSVT + MSVI + MSUR** define collision mesh surfaces (the polygons that define walkable and non-walkable areas)
- **MSCN** contains the centroid of each MSUR surface — these are the **navigation graph nodes**
- **MSLK** is the **edge catalog** — each entry links a surface (RefIndex → MSUR) to path geometry (MspiFirstIndex → MSPI → MSPV) with metadata (`TypeFlags`, `Subtype`) describing the surface or edge family. Current partial reads: `0x03` = M2 top surfaces, `0x10` = interior WMO floors, `0x12` = exterior WMO solid surfaces.
- **MPRL** provides world-space placement anchors for the navigation nodes
- **MPRR** chains MPRL entries into a graph structure
- **MDBH/MDOS/MDSF** encode destructible building state (dynamic obstacles in the pathfinding network)

The "peg" or "dowel" points at tile boundaries are cross-tile connection markers — MSCN centroids that exist in adjacent tile space, enabling the pathfinding graph to span multiple tiles.

**CK24 groups surfaces into objects.** Within each object, MSLK edges connect MSCN centroids into a sub-graph. The CK24 type byte (0x00 = terrain, 0x40 = M2, 0x42/0x43 = WMO) classifies the object type, and the object decomposition problem is really "how to identify independent pathfinding sub-graphs within a CK24 group." Current corpus proof says the cross-tile stitch more often survives across multiple Field04 buckets than within a single one, so the likely cross-region owner is CK24 plus connector/node evidence, not Field04.

**The format is a compressed dataset** — the user estimates it stores scene data at ~1/1000th the size of the uncompressed equivalent. Each chunk is a data layer optimized for its role, with its own coordinate convention.

### The MSLK Linkage Map

MSLK is the glue that connects everything. Every field has a role:

| Field | Role | Links To |
|-------|------|----------|
| TypeFlags | Per-surface or edge family classification | Current partial buckets: `0x03` M2 tops, `0x10` interior WMO floors, `0x12` exterior WMO solids |
| Subtype | Edge property / sequence position | Nothing — used for grouping/clustering |
| GroupObjectId | Sub-object membership | Surfaces via RefIndex (union-find partitioning) |
| MspiFirstIndex + MspiIndexCount | Path geometry window | MSPI → MSPV (navigation path vertices) |
| LinkId | Tile coordinate tag | Diagnostic — identifies source tile |
| RefIndex | **Dual-use**: surface index + position index | MSUR (surface partitioning) AND MPRL (position/heading) |
| SystemFlag | Constant flag (0x8000) | Nothing — format marker |

**The critical insight**: RefIndex is ALWAYS both a MSUR index AND a MPRL index. The code checks MSUR first for surface partitioning, then checks MPRL for position collection. Entries where RefIndex >= MSUR.Count are "mismatches" but may still validly index into MPRL.

### The CK24 Multi-Layer Hypothesis

The user recalls an experiment that subdivided objects down to every single polygon, creating 2 million objects. This suggests CK24 may encode multiple layers of keys — not just a flat object ID, but a hierarchy:

- **High byte (Ck24Type)**: Object type (0x00=terrain, 0x40=M2, 0x42/0x43=WMO)
- **Middle byte**: Group or category
- **Low byte**: Instance or sub-object
- **The discarded low byte of PackedParams (bits 0-7)**: Possibly another key layer

The 2-million-object explosion happened when the code treated every polygon as independent — meaning the sub-object splitting went too far. The correct decomposition is somewhere between "one CK24 = one object" and "one polygon = one object." `MSLK.GroupObjectId` is still the current sub-object partitioning mechanism, but it must not be conflated with the newly observed `TypeFlags` surface-family buckets.

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

**Why this priority**: MSHD is the only chunk that could comprehensively describe the rest of the format. If it encodes counts/offsets, our entire chunk-reading model may need adjustment. If it is dead padding, we can stop investigating it.

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

### User Story 6 — CK24 Object Decomposition Strategy (Priority: P1)

As a format researcher, I need to determine how to correctly decompose CK24 groups into individual objects — testing the LSB-hierarchy hypothesis, MSLK.TypeFlags/Subtype as linking layers, and MSCN boundary detection — so that we can reliably split a single CK24 value (which may contain hundreds of surfaces across multiple tiles) into coherent object instances.

**Why this priority**: This is the core unsolved problem. Without knowing how to split objects, we cannot reconstruct PM4 data into usable collision geometry, and we must rely on expensive model training to match objects to PM4 data.

**Independent Test**: For the reference tile (development_00_00), the top CK24 group (0x43A9AA, 896 surfaces) is decomposed into sub-objects using each strategy. The decomposition is validated against ADT _obj0.adt WMO/M2 placements.

**Acceptance Scenarios**:

1. **Given** the CK24 bit layout (`(PackedParams >> 8) & 0xFFFFFF`), **When** the LSB-hierarchy hypothesis is tested (low bits as base groups), **Then** either a clean hierarchical structure is found (r > 0.9 correlation between bit position and object identity) or the hypothesis is rejected with evidence.
2. **Given** MSLK.TypeFlags and MSLK.Subtype across the development corpus, **When** the values are correlated with object boundaries (ADT placements, CK24 group splits), **Then** the observed `TypeFlags` families (`0x03`, `0x10`, `0x12`) are either confirmed corpus-wide and extended, or the field ownership hypothesis is revised with evidence.
3. **Given** the unused low byte of MSUR.PackedParams (bits 0-7), **When** the byte is extracted and correlated with CK24 splits, **Then** either it carries meaningful grouping data or it is confirmed as padding/unused.
4. **Given** MSCN positions for a CK24 group, **When** the positions are used to define object boundaries (bounding boxes, containment tests), **Then** the MSCN-based boundaries are compared against MSLK-based and connectivity-based boundaries for accuracy.

---

### User Story 7 — MSCN Full Consumption Audit (Priority: P2)

As a format researcher, I need to understand what portion of MSCN data is consumed today versus raw, and whether the unconsumed data contains object boundary or containment information that would improve cross-tile object reconstruction.

**Why this priority**: The user reports that MSCN helps identify where objects exist across tiles. If we're only consuming a small fraction of MSCN data (the MdosIndex-referenced positions), the rest may hold the missing linking information.

**Independent Test**: For each MSCN vertex in the reference tile, determine whether it is (a) referenced by any MSUR.MdosIndex, (b) used as a connector key, (c) consumed by any other code path, or (d) completely unused.

**Acceptance Scenarios**:

1. **Given** all MSCN vertices in the reference tile (9,990 points), **When** MdosIndex references are tallied, **Then** the audit reports what percentage are referenced vs. unreferenced.
2. **Given** unreferenced MSCN vertices, **When** spatial clustering is computed, **Then** the clusters are compared against CK24 group boundaries to test whether unreferenced vertices define object containment.
3. **Given** the parpToolbox MscnRemapper (legacy), **When** its (Y,X,Z) swap and remap logic is analyzed, **Then** the audit documents what the swap achieves and whether the active viewer should adopt it.
4. **Given** MSCN coordinate space analysis results, **When** per-CK24 alignment modes are reviewed, **Then** the audit identifies whether a consistent coordinate convention exists or is file-dependent.

---

### User Story 8 — wowdev.wiki Naming Alignment (Priority: P1)

As a format researcher, I need every PM4 field in our codebase audited against the wowdev.wiki PM4/PD4 documentation, with invented names corrected and the mapping documented, so that new contributors can cross-reference our code against the canonical community documentation without encountering contradictions.

**Why this priority**: The naming drift has already caused one critical error (MSUR._0x18 named `MdosIndex` when it points to MSCN, not MDOS). Invented names that harden into "truth" through repetition are a hallucination vector that undermines research credibility.

**Independent Test**: Every field in `Pm4ResearchChunkModels.cs` has a comment mapping it to its wowdev.wiki equivalent. The `MdosIndex` → `MscnIndex` rename is applied across all code.

**Acceptance Scenarios**:

1. **Given** all 16 chunk types in `Pm4ResearchChunkModels.cs`, **When** the audit is complete, **Then** every field has either (a) a matching wowdev.wiki name, or (b) a `// wowdev.wiki: _0xNN` comment with the original offset name.
2. **Given** `MSUR._0x18` (currently `MdosIndex`), **When** the rename is applied, **Then** the field is renamed to `MscnIndex` (or `_0x18` with a comment) across all code, and no references to the old name remain.
3. **Given** `MSLK._0x10` (currently `RefIndex`, wiki calls it `msur_index`), **When** the mapping is documented, **Then** a comment explains why the rename was made (multi-domain RefIndex behavior).

---

### User Story 9 — Rare Value Analysis and Root Group Discovery (Priority: P1)

As a format researcher, I need to identify rare field values (appearing 1-2 times in the corpus) across MSHD, MSLK.Subtype, MSLK.TypeFlags, and MSUR.GroupKey, then cross-correlate them to find the missing root-level grouping layer between CK24 (too broad) and individual polygons (too granular).

**Why this priority**: The object decomposition problem is unsolved because we're missing a hierarchy level. CK24 groups hundreds of surfaces together. MSLK.GroupObjectId splits them into sub-objects. But there may be a root-level grouping above GroupObjectId that defines the top-level scene divisions. Rare values (appearing once or twice) are often identifiers, not noise.

**Independent Test**: For each target field, compute the per-value frequency across the development corpus. Identify values with frequency <= 5. Cross-correlate rare values across chunks to find co-occurrence patterns.

**Acceptance Scenarios**:

1. **Given** MSLK.Subtype values across 616 tiles, **When** frequency is computed, **Then** values appearing <= 5 times are identified and their MSLK entries are inspected for TypeFlags, LinkId, RefIndex, and GroupObjectId patterns.
2. **Given** MSLK.TypeFlags values across 616 tiles, **When** frequency is computed, **Then** rare TypeFlags values (<= 5 occurrences) are identified and compared against CK24 type bytes to test the type-byte-storage hypothesis.
3. **Given** MSUR.GroupKey values across 616 tiles, **When** frequency is computed, **Then** rare GroupKey values are identified and correlated with CK24 groups to test whether GroupKey encodes region or scene-level grouping.
4. **Given** MSHD.Field00 and MSHD.Field08 across 616 tiles, **When** value distribution is computed, **Then** rare values (appearing in < 5 tiles) are identified and checked for correlation with rare values in other chunks.
5. **Given** all rare values across all target fields, **When** co-occurrence is analyzed, **Then** any chunk pairs where rare values co-occur in the same tile are flagged as potential root-group candidates.

---

## 7. Edge Cases

- What if MSHD.Field00/Field08 are actually hash values rather than counts?
- What if MSLK.RefIndex is context-dependent (points to different chunk types based on TypeFlags)?
- What if MPRR is not a graph but a flat array with a specific packing order?
- What if some PM4 files contain unknown chunk types not in the current 16?
- What if fields 0x0C-0x1C in MSHD are only populated in PD4 (the retail variant) and are genuinely dead in PM4?
- What if the MSVT YXZ coordinate formula from wowdev.wiki (`worldPos.y = 17066.666 - position.y`) is the canonical transform and our axis-convention detection is over-engineered?
- What if `MSLK._0x04` ("An index somewhere" per wiki) is not a group/object ID but an index into a different chunk entirely?
- What if `MSLK._0x0c` ("Always 0xffffffff in version_48" per wiki) is a sentinel that we incorrectly decoded as tile coordinates?

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
- **FR-009**: Rare value analysis MUST compute per-value frequency for MSLK.Subtype, MSLK.TypeFlags, MSUR.GroupKey, MSHD.Field00, and MSHD.Field08 across the full development corpus, and flag values with frequency <= 5 as root-group candidates.
- **FR-010**: Rare value candidates MUST be cross-correlated across chunks to find co-occurrence patterns that suggest shared root-group identity.

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
| wowdev.wiki PM4 | `https://wowdev.wiki/PM4` | — |
| wowdev.wiki PD4 | `https://wowdev.wiki/PD4` | — |
| Naming Drift Analysis | `wow-viewer/docs/research/004-pm4-format-research/naming-drift-analysis.md` | — |
