# Feature Specification: PM4/PD4 generation from source geometry

**Feature Branch**: `184-pm4-generation-from-geometry`

**Created**: 2026-08-23

**Status**: Draft

**Input**: User description: "get pm4's built from real objects, or the method of how a pm4 object is generated from a wmo or m2, should be figured out, so we can build new pm4's that are accurate and proper"

## Context

Every PM4 effort in this repo so far has run in the **decode** direction: read a PM4, try to work out
which asset it came from. Specs 128/129/130 and the fingerprint and surface-correlation matchers all
sit on that side, and the asset-identity half of it is still unsolved (P@1 ≈ 1.3%).

This spec runs the **generation** direction instead: given the source geometry, produce the navmesh.
That inverts the research problem into a supervised one. A generator that can be compared
field-by-field against a real file turns every undecoded structure into a measurable residual,
because a mismatch localises to the exact rule that produced it. Correlational matching cannot do
that.

`Pm4Generator` already exists but was built to feed the fingerprint matcher, and its own comment says
so: *"we do not need to reproduce real PM4's exact merged polygons; we only need a comparable
triangle histogram."* Measured, its output is not a navmesh at all.

### Measured baseline — current generator versus real data

| structure | real data | `Pm4Generator` today |
|---|---|---|
| MSUR polygon size | **77% quads** (PD4 garrison: 4×2640, 5×385, 3×257, 6×151, 7×7) | always `IndexCount: 3` |
| MSLK adjacency | 1,273,335 records, exact partition of the stream | **1 record**, whole object |
| MSPV / MSPI wall geometry | 598,882 wall quads (47.03% of edges) | **empty** |
| MSCN | 1,342,410 points | **1 point** (the centroid) |
| MSUR `_0x18` | window start into MSLK | hardcoded `0` |

### Measured evidence this spec is built on

Produced 2026-08-23 by `pm4 msur-window` over the 616-file (309 non-empty) development corpus,
through `Pm4ResearchReader`:

- **`MSUR._0x18` + `MSUR.AttributeMask` is a running window that exactly partitions MSLK.**
  517,783 / 517,783 consecutive pairs chain (100.0000%), window lengths sum to 1,273,335 = the MSLK
  count exactly, zero windows out of range, 100% stream coverage.
- **The target is MSLK, not MSCN.** Against MSCN the identical windows overrun 6,240 times and cover
  93.52%. The documented "`_0x18` = index into MSCN" reading is eliminated.
- **Detector power was established before the claim.** Positive control
  (`MsviFirstIndex + IndexCount` → MSVI, known-true) fits 100%; negative controls reusing the same
  start field with a wrong length fit 22.25% and 8.46%.
- **`MSLK.RefIndex` names a NEIGHBOUR surface, not an owner.** The owner round trip scores
  18 / 1,273,335 (0.0014%); direct reciprocity scores **1,257,562 / 1,273,301 (98.76%)** with 18
  self-edges. MSLK is the undirected surface-adjacency graph.
- **47.03% of adjacency records carry a wall quad** — 598,882, which independently equals the
  "active path windows" count that the connective-geometry analyzer measured for MSPV/MSPI.

Composed, real PM4 is: *walkable polygons (MSUR), each owning a contiguous run of adjacency records
(MSLK) that name reciprocal neighbours, with a vertical quad (MSPV/MSPI) erected on the blocked
subset.* That is a navmesh. The current generator reproduces none of it.

### Why PD4 is the first target

PD4 is the per-WMO navmesh: object-local coordinates, no placement, no tile. The garrison PD4 holds
only `MVER, MCRC, MSPV, MSPI, MSCN, MSLK, MSVI, MSVT, MSUR` — exactly the object-level core, with its
MSVT bounds centred near the origin at (−29.83, −28.16, −2.10)..(29.79, 23.94, 30.79). PM4 is that
same core placed into tile space plus the tile-level chunks (MPRL, MPRR, MDOS, MDSF, MDBH, MDBI,
MDBF). Generating PD4 first isolates the object-generation question from placement and tile
machinery, both of which are separately solved or separately unsolved.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Score any navmesh file against real-data structure (Priority: P1)

A researcher points a conformance check at any PM4 or PD4 file — real or generated — and gets a
per-rule score describing how far it is from real Blizzard output: polygon-size distribution,
adjacency-window integrity, neighbour reciprocity, wall-edge fraction, and stream coverage.

**Why this priority**: It is the acceptance instrument for every other story. Without it, "accurate
and proper" has no definition and no generator can be said to have improved. It also has standalone
value: it scores the existing corpus and any future decode.

**Independent Test**: Run it over the 309-file real corpus and over the current generator's output.
Real files must score at or near the ceiling on every rule; the current generator must score at or
near zero on adjacency, walls, and polygon-size. A check that cannot separate those two populations
is not a measurement and must be reported as uninformative rather than as a pass.

**Acceptance Scenarios**:

1. **Given** a real PM4 tile, **When** the conformance check runs, **Then** the adjacency window
   partitions the link stream exactly, reciprocity is at least 98%, and no rule reports a violation
   that the corpus sweep does not already account for.
2. **Given** output from the current generator, **When** the same check runs, **Then** it reports
   zero adjacency records per surface, zero wall geometry, and a polygon-size distribution of 100%
   triangles, each as an explicit named failure rather than an absent result.
3. **Given** a file whose adjacency windows deliberately overrun the link stream, **When** the check
   runs, **Then** it reports the out-of-range count rather than silently clamping.

---

### User Story 2 - Assemble ground-truth source/output pairs (Priority: P1)

A researcher obtains the set of (source asset, real navmesh object) pairs the generator will be
measured against, with each pair's identity justified rather than assumed.

**Why this priority**: Generation cannot be validated without knowing which real object a given
source asset produced. Asset identity is explicitly unsolved in this repo, so the pairs must come
from cases where identity is *forced* by counting, not inferred by similarity — otherwise the
generator is scored against the wrong target and the whole lane produces confident nonsense.

**Independent Test**: Produce the pair list with a stated basis for each pair, and a count of tiles
rejected for ambiguity. Verifiable by re-deriving the counts from the corpus independently.

**Acceptance Scenarios**:

1. **Given** a development tile whose keyed object count equals its ADT world-model placement count
   and both equal one, **When** pairing runs, **Then** that pair is emitted as forced-identity.
2. **Given** a tile where those counts disagree, **When** pairing runs, **Then** the tile is excluded
   and counted as ambiguous rather than resolved by a similarity score.
3. **Given** the garrison PD4 whose source model is not present on disk, **When** pairing runs,
   **Then** it is reported as an unpaired reference file, not silently dropped.

---

### User Story 3 - Generate the walkable surface set (Priority: P2)

A researcher runs generation against a source model's collision geometry and gets a walkable surface
set whose polygon-size distribution resembles real output rather than raw triangles.

**Why this priority**: The surface set is the substrate every other structure indexes into. Adjacency
and wall geometry are defined between surfaces, so they cannot be built on one-triangle-per-surface
output. This is where the 77%-quad finding is discharged.

**Independent Test**: Generate from a paired source asset and compare the polygon-size histogram and
per-surface vertex-window integrity against its real counterpart.

**Acceptance Scenarios**:

1. **Given** a source model with collidable faces, **When** generation runs, **Then** coplanar
   adjacent faces are merged into single polygons and the resulting size histogram is reported
   against the paired real object's histogram.
2. **Given** the generated surface set, **When** the conformance check runs, **Then** the per-surface
   vertex windows partition the index stream exactly, with zero out-of-range windows.
3. **Given** a source model with no collidable faces, **When** generation runs, **Then** it emits an
   explicit empty result with a stated reason rather than a malformed file.

---

### User Story 4 - Generate adjacency and wall geometry (Priority: P2)

A researcher gets, for each generated surface, a contiguous run of adjacency records naming
reciprocal neighbours, with vertical wall geometry erected on the blocked subset.

**Why this priority**: This is the structure that makes the output a navmesh rather than a triangle
soup, and it is the part the new measurement newly makes buildable. It is separated from Story 3
because the surface set is independently testable and independently useful.

**Independent Test**: Run the conformance check on generated output: the adjacency windows must
partition the link stream exactly and reciprocity must be measured, not assumed.

**Acceptance Scenarios**:

1. **Given** a generated surface set, **When** adjacency generation runs, **Then** every surface owns
   a contiguous run of records, the runs partition the link stream exactly with zero overrun, and
   every non-boundary neighbour relation is reciprocal.
2. **Given** two adjacent surfaces separated by a height discontinuity, **When** generation runs,
   **Then** a vertical quad is erected on that connection and the connection is counted as blocked.
3. **Given** two coplanar adjacent surfaces, **When** generation runs, **Then** the connection is
   emitted as open passage with no wall geometry.
4. **Given** a surface on the outer boundary of the source model, **When** generation runs, **Then**
   its unmatched edges are reported under a stated boundary policy rather than producing a dangling
   neighbour reference.

---

### User Story 5 - Round-trip and inspect generated output (Priority: P3)

A researcher writes a generated object to a real file, reads it back through the existing reader, and
inspects it in the viewer alongside real navmesh data.

**Why this priority**: Structural conformance on in-memory data does not prove the bytes are
loadable. This closes the loop, but only matters once Stories 3 and 4 produce something worth
writing.

**Independent Test**: Write, re-read, and confirm the re-read object is structurally identical to the
in-memory one, then confirm it draws in the viewer.

**Acceptance Scenarios**:

1. **Given** a generated object, **When** it is written and read back, **Then** every chunk's entry
   count and every index window matches the in-memory original exactly.
2. **Given** a generated object placed in a tile, **When** the viewer loads it, **Then** it renders
   in the same space as real navmesh data for that tile.

---

### User Story 6 - Report what generation cannot yet reproduce (Priority: P3)

A researcher gets an explicit account of the structures the generator cannot produce, and what
evidence would be needed to produce them.

**Why this priority**: Several structures are genuinely undecoded — `MSCN` is now unowned, since the
window finding removed its only claimed index consumer. Recording that honestly is what stops the
next session from re-searching solved ground or shipping an invented rule as a decode.

**Independent Test**: The report names each unreproduced structure, its real-data population, and the
specific measurement that would settle it.

**Acceptance Scenarios**:

1. **Given** a completed generation run, **When** the gap report is produced, **Then** every chunk
   present in the paired real object but absent or stubbed in the generated one is listed with its
   real entry count.
2. **Given** a structure reproduced by a rule that is fitted rather than derived, **When** the report
   is produced, **Then** it is labelled as fitted and not counted as decoded.

---

### Edge Cases

- A source model whose collision geometry is non-manifold, so an edge is shared by more than two
  faces and "the neighbour across this edge" is not unique.
- A surface whose true neighbour lives in an adjacent tile — the likely explanation for the 1.24% of
  real edges that do not reciprocate, and untested.
- Degenerate faces (zero area, duplicate vertices) that must not become surfaces or adjacency
  records.
- LOD variants of one source model, which produce different navmeshes from the same asset name.
- A source model with collision flags set such that no face is collidable at all.
- Coplanar faces that are adjacent in space but belong to different source groups, where merging may
  or may not be correct.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The conformance check MUST evaluate a navmesh file against named structural rules and
  report each rule's result separately, never a single aggregate pass/fail.
- **FR-002**: The conformance check MUST report window integrity as exact counts — pairs tested,
  chained, out of range, and stream coverage — not as a fraction alone.
- **FR-003**: The conformance check MUST be demonstrated to separate real files from the current
  generator's output before any of its scores are used as acceptance evidence.
- **FR-004**: Ground-truth pairing MUST admit a pair only when identity is forced by counting, and
  MUST count and report every case rejected as ambiguous.
- **FR-005**: Ground-truth pairing MUST NOT use geometric similarity or fingerprint scoring to
  establish identity.
- **FR-006**: Generation MUST merge coplanar adjacent collidable faces into single polygons rather
  than emitting one polygon per source triangle.
- **FR-007**: Generation MUST emit, per surface, a contiguous run of adjacency records such that the
  runs partition the link stream exactly with no gaps and no overruns.
- **FR-008**: Generation MUST make neighbour references reciprocal for every connection between two
  generated surfaces.
- **FR-009**: Generation MUST erect vertical wall geometry only on connections classified as blocked,
  and MUST record the resulting blocked fraction for comparison against real data.
- **FR-010**: Generation MUST produce object-local output for the per-model form, with placement
  applied only when producing the tile form.
- **FR-011**: Generation MUST reuse the existing coordinate service for every space conversion and
  MUST NOT introduce a second coordinate convention.
- **FR-012**: Generation MUST NOT introduce a second reader or chunk parser for these formats.
- **FR-013**: Written output MUST re-read through the existing reader to a structurally identical
  object.
- **FR-014**: The system MUST report every structure present in a paired real object that generation
  does not reproduce, with its real entry count.
- **FR-015**: Any rule fitted to match observed data rather than derived from a stated mechanism MUST
  be labelled as fitted wherever its results are reported.
- **FR-016**: Support for the second source-model kind (animated models) MUST be scoped separately
  and MUST NOT block delivery of world-model support.

### Key Entities

- **Source collision geometry**: the collidable subset of a source model's faces, with the flags that
  decide collidability; the input to generation.
- **Walkable surface**: one merged planar polygon with a normal and plane distance, indexing a
  contiguous window of vertices.
- **Adjacency record**: one connection belonging to exactly one surface, naming a neighbouring
  surface and optionally carrying wall geometry.
- **Wall quad**: the vertical planar polygon standing on a blocked connection.
- **Conformance profile**: the set of structural rules and their real-data reference values.
- **Ground-truth pair**: a source asset and the real navmesh object it produced, with the basis on
  which that identity was established.
- **Gap report**: the enumerated structures generation does not reproduce, with real-data populations.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The conformance check scores all 309 non-empty real corpus files, and every structural
  rule it reports reproduces the corpus sweep's published figures exactly.
- **SC-002**: The conformance check separates populations: real files pass the adjacency, wall and
  polygon-size rules; the current generator's output fails all three, each as a named failure.
- **SC-003**: The ground-truth pair set is published with a per-pair basis and an ambiguity-rejection
  count, and contains at least one pair usable end to end.
- **SC-004**: For a paired source asset, generated polygon-size distribution is reported side by side
  with the real object's, and the share of non-triangle polygons rises from the current 0%.
- **SC-005**: Generated adjacency windows partition the link stream exactly — zero gaps, zero
  overruns, full coverage — matching the integrity real data shows.
- **SC-006**: Generated neighbour references reciprocate for at least 98% of connections between
  generated surfaces, the level real data shows.
- **SC-007**: Generated blocked-connection fraction is reported against real data's 47.03%.
- **SC-008**: A generated object survives write-then-read with identical entry counts and identical
  index windows for every chunk it emits.
- **SC-009**: The gap report accounts for 100% of chunks present in the paired real object, each
  either reproduced, stubbed with a stated reason, or named as undecoded.

## Assumptions

- The per-model form (PD4) is the first target and the tile form (PM4) builds on it, because the
  per-model form isolates object generation from placement and tile-level structures.
- Collidability is decided by the source model's existing face flags, as the current extraction
  already does; no new collidability model is invented here.
- The tile-level structures (placement lists, destructible payloads, the largest undecoded stream)
  are out of scope for generation in this spec and are addressed only by the gap report.
- The two reference PD4 files remain unpaired unless their source model becomes available; the lane
  does not block on obtaining it.
- Existing decode confidence levels travel with any claim reused here — in particular, surface
  grouping into whole objects is explicitly not a confirmed identity and is not treated as one.
- Real-client proof and any long corpus-wide sweep beyond read-only inspection are user-run.

## Out of Scope

- Solving navmesh-object-to-placed-asset identity in the general case; only forced-identity pairs are
  used.
- Generating the tile-level chunks.
- Decoding the largest undecoded stream (MPRR).
- Any change to the existing reader, coordinate service, or placement math.
- Writing these files into Blizzard container formats.
