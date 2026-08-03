# Feature Specification: PM4 Remaining Decode — Connective Geometry and Object Identity

**Feature Branch**: `130-pm4-remaining-decode`

**Created**: 2026-08-03

**Status**: Draft

**Input**: User description: "I found that I could dig down to every individual triangle, it's the whole entire original resolution of the data, like a negative mold that wraps around the surfaces that are walkable. It has other data, though, and I think we're still missing some of that connecting geometry or verts, that tie these objects together into fully negative versions of the original WMO and M2 data. I just don't know how to decode it, because it's likely that they encoded it inside of something else, as a whole other nested coordinate system... the viewer lets us click on individual objects, which is awesome, but it's not perfect, as it selects an individual surface, instead of the whole object itself."

## Context

### What is already solved

The walkable surface mesh is fully decoded and verified against the corpus. `pm4 unknowns` over the
616-file development map reports these edges with zero misses:

| relationship | fits | misses |
|---|---|---|
| MSUR.Msvi window → MSVI | 518,092 | **0** |
| MSVI → MSVT | 1,930,146 | **0** |
| MSLK.Mspi window → MSPI | 598,882 | **0** |
| MSPI → MSPV | 2,418,205 | **0** |
| MDSF.MsurIndex → MSUR | 2,684 | **0** |
| MDSF.MdosIndex → MDOS | 2,684 | **0** |

That is the "negative mold" down to individual triangles, at full original resolution.

### The two problems, which are one problem

**The viewer selects a surface instead of an object.** This is not a UI defect. The relationship
that would group surfaces into objects is measured at `MSLK.GroupObjectId → MPRL.Unk04` with
**65,819 fits against 1,206,977 misses** — roughly 5% resolved. The viewer selects surfaces because
surfaces are the largest unit the decode can currently justify. Fix the grouping and the selection
behaviour follows.

**The connective geometry is missing.** There is a strong candidate already in the data:
**MSPV/MSPI is a second geometry stream, larger than the surface mesh itself** — 2,418,205 index
fits versus MSVI→MSVT's 1,930,146. Its windows resolve perfectly, but what they *mean* does not:
`MSLK.MspiIndexCount` measures `indicesOnly=399,183`, `both=199,699`, `trianglesOnly=0`, and whether
a window is a polyline, a triangle run, or something else is undecided. A second vertex stream that
is bigger than the surface mesh and attaches to the same link records is exactly the shape of the
connective geometry that would close a surface set into a sealed negative volume.

### The nine open questions, as measured

| unknown | status | evidence |
|---|---|---|
| MSLK.RefIndex semantics | open | 4,553 entries do not fit MSUR; mismatch strongest in MSPI/MSVI/MSCN/MSLK, weak in MPRL (86 fits) |
| MSLK.MspiIndexCount interpretation | open | active=598,882 indicesOnly=399,183 trianglesOnly=0 both=199,699 |
| MSLK.TypeFlags / Subtype meaning | partial | 0x03=M2 tops, 0x10=interior WMO floors, 0x12=exterior WMO solids; 10 distinct TypeFlags, 19 Subtypes |
| MPRL.Unk14 / Unk16 | open | Unk14 range −1..15, Unk16 distinct=2 |
| MSHD header fields | open | Field00 distinct=155, Field04 distinct=227, Field08 distinct=152 |
| MPRR field semantics | open | Value1 fits MPRL=6,778,712 / MSVT=8,740,189; neither domain explains it; Value2 distinct=566 |
| Destructible payload integration | partial | MDBH/MDOS/MDSF populated on one tile; MDOS→MDBH fits=1 misses=24 |
| LinkId extended meaning | verified | sentinel tile links=1,273,335, zero=0, other=0 |
| Coordinate/frame ownership | open | needs PM4↔ADT/object correlation on trusted tiles |

MPRR deserves emphasis: it is the second-largest chunk in tile 0_0 at 327,744 bytes, and **neither**
candidate target domain explains it — 6.8M fits against 7.2M misses for MPRL, 8.7M against 5.2M for
MSVT. A stream that large with no resolved target is the single biggest undecoded surface in the
format.

### Why nesting is the working hypothesis

PM4 packs data tightly using coordinate rotations and nested frames, and the nesting is not uniform
even within a file. Measured on `development_00_00.pm4`: MSVT spans (168–501, 31–450, −12–133) while
MPRL spans (31–364, 5–40, 168–499) — MPRL's third axis is MSVT's first. If connective geometry is
encoded inside another structure under a different frame, that is consistent with how the rest of
the format already behaves.

### What must not be repeated

During research for the sibling specs, a chunk was hand-parsed outside the canonical decoder, its
axis order assumed rather than read, and a confident and entirely wrong conclusion followed. Any
decode claim here must be produced through the existing PM4 stack and validated against the corpus,
not asserted from a single file.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select a whole object in the viewer (Priority: P1)

A researcher clicks a walkable surface in the viewer and the entire object it belongs to is
selected, not the single surface under the cursor.

**Why this priority**: It is the user-visible symptom, it is what makes the viewer usable for
inspection, and it is the acceptance test for the grouping decode. If objects can be selected
correctly, the grouping is right.

**Independent Test**: Click surfaces belonging to objects of known extent and confirm the full
object highlights each time.

**Acceptance Scenarios**:

1. **Given** an object composed of many surfaces, **When** any one of its surfaces is clicked,
   **Then** every surface of that object is selected together.
2. **Given** an object spanning more than one tile, **When** one of its surfaces is clicked,
   **Then** the selection includes its parts in the other tiles.
3. **Given** a surface whose object membership cannot be determined, **When** it is clicked,
   **Then** it is selected alone and reported as ungrouped rather than being guessed into a group.
4. **Given** the same click twice, **When** the selection is compared, **Then** it is identical.

---

### User Story 2 - Establish what binds surfaces into objects (Priority: P1)

A researcher can state, with corpus evidence, which field or combination of fields groups surfaces
into whole objects, and how much of the corpus that explains.

**Why this priority**: Equal to US1 — it is the same problem stated as a decode question rather than
a UI one. Today's best candidate explains ~5% of the corpus.

**Independent Test**: Report a grouping rule's fit and miss counts across all 616 files and compare
against the current 65,819 / 1,206,977 baseline.

**Acceptance Scenarios**:

1. **Given** a proposed grouping rule, **When** it is evaluated corpus-wide, **Then** its fits and
   misses are reported per file and in total.
2. **Given** a rule that improves on the baseline, **When** it is published, **Then** the evidence
   that justifies it is recorded alongside the confidence it warrants.
3. **Given** surfaces that no rule groups, **When** results are reported, **Then** they are counted
   and characterised rather than absorbed silently.
4. **Given** an object grouping, **When** it is compared against a known WMO or M2 whose true extent
   is independently established, **Then** agreement or disagreement is reported.

---

### User Story 3 - Determine what the second geometry stream encodes (Priority: P2)

A researcher establishes what MSPV/MSPI windows represent, and whether they supply the connective
geometry that closes a surface set into a sealed negative volume.

**Why this priority**: It is the strongest lead for the missing connective geometry, and it is
larger than the mesh already decoded. It is P2 only because object grouping unblocks the viewer
first.

**Independent Test**: Reconstruct a known object using surfaces alone, then with the second stream
included, and compare the result against the real asset's shape.

**Acceptance Scenarios**:

1. **Given** the second geometry stream, **When** its window interpretation is evaluated corpus-wide,
   **Then** each candidate interpretation is reported with its fit and miss counts.
2. **Given** a decoded interpretation, **When** an object is reconstructed with and without the
   stream, **Then** the difference in the resulting volume is quantified.
3. **Given** the reconstruction, **When** it is compared against the corresponding real asset,
   **Then** whether the object is a sealed negative of that asset is reported as a measurement.

---

### User Story 4 - Resolve the largest undecoded stream (Priority: P3)

A researcher determines what MPRR references, or establishes with evidence that neither current
candidate domain is correct and narrows what remains.

**Why this priority**: It is the biggest single undecoded surface in the format, but nothing
currently depends on it, so it ranks below the work that unblocks the viewer and reconstruction.

**Independent Test**: Evaluate candidate target domains corpus-wide and report fits and misses for
each; a negative result that eliminates domains is a valid outcome.

**Acceptance Scenarios**:

1. **Given** a candidate target domain, **When** evaluated corpus-wide, **Then** fits and misses are
   reported and compared against the MPRL and MSVT baselines.
2. **Given** no candidate that explains the stream, **When** results are published, **Then** the
   eliminated domains are recorded so the search is not repeated.

---

### Edge Cases

- Fields whose meaning differs by TypeFlags family rather than being uniform.
- Objects spanning tiles, where grouping must reconcile parts from separate files.
- Tile 0_0, which is the only tile with a populated destructible payload and is explicitly noted as
  unrepresentative of the general mismatch population.
- 307 of 616 files are empty and cannot support any inference.
- A grouping rule that fits the corpus statistically but produces objects that are physically absurd.
- Relationships that resolve into more than one domain simultaneously, as MPRR appears to.
- A field that is genuinely padding or a compiler artefact with no semantic meaning.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: All decode work MUST be performed through the canonical PM4 stack; no reimplementation
  of chunk parsing or coordinate handling.
- **FR-002**: Every proposed field interpretation MUST be evaluated across the whole corpus and
  reported with fit and miss counts, never asserted from a single file or tile.
- **FR-003**: The system MUST report a per-surface object grouping, and MUST identify surfaces it
  cannot group rather than assigning them to a nearest guess.
- **FR-004**: Grouping quality MUST be measurable against the current baseline so improvement is
  demonstrated rather than claimed.
- **FR-005**: The viewer MUST select the whole object a clicked surface belongs to, and MUST fall
  back to selecting the single surface, visibly marked as ungrouped, when membership is undetermined.
- **FR-006**: Object selection MUST include parts of the object residing in other tiles.
- **FR-007**: Each interpretation MUST carry a confidence level and its supporting evidence, in the
  same terms the existing decoder already publishes.
- **FR-008**: Candidate interpretations that are eliminated MUST be recorded with the evidence that
  eliminated them, so the same search is not repeated.
- **FR-009**: The system MUST report, for the second geometry stream, what its windows encode, with
  corpus-wide evidence for each candidate interpretation.
- **FR-010**: Where a decoded interpretation enables reconstructing an object, the result MUST be
  comparable against the corresponding real asset, and the comparison reported as a measurement.
- **FR-011**: Findings MUST be published in a form the sibling dataset and matching work can consume
  without re-deriving them.
- **FR-012**: A field established to have no semantic meaning MUST be recordable as such, with
  evidence, rather than remaining permanently open.

### Key Entities

- **Field interpretation**: A proposed meaning for an undecoded field, its corpus-wide fit and miss
  counts, its confidence, and its supporting or eliminating evidence.
- **Grouping rule**: A proposed rule assigning surfaces to objects, with its corpus-wide accuracy
  and the population it fails to explain.
- **Object**: A set of surfaces the decode asserts belong to one original asset, its tile
  membership, and the confidence of that grouping.
- **Connective geometry**: Whatever the second stream encodes, once established — the data that
  closes a surface set into a sealed negative volume.
- **Reconstruction comparison**: An object rebuilt from decoded data measured against the real
  asset it should correspond to.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Surface-to-object grouping explains substantially more of the corpus than the current
  baseline of 65,819 fits against 1,206,977 misses, with the improvement measured on the same corpus.
- **SC-002**: Clicking any surface of a multi-surface object in the viewer selects the whole object,
  verified against objects of known extent, including at least one spanning multiple tiles.
- **SC-003**: Surfaces that cannot be grouped are reported and counted; none are silently assigned.
- **SC-004**: Each of the nine currently-open questions is either resolved with corpus evidence,
  narrowed with candidate domains eliminated, or documented as unresolvable with the reason.
- **SC-005**: The second geometry stream's window interpretation is established with corpus-wide
  fit and miss counts for every candidate considered.
- **SC-006**: At least one object is reconstructed and measured against its real asset, with the
  degree to which it forms a sealed negative reported numerically.
- **SC-007**: Every published interpretation carries confidence and evidence; none is stated as fact
  without both.

## Assumptions

- The nine open questions from the existing unknowns analyzer are the starting scope; new ones may
  emerge and are in scope if they block the same outcomes.
- The 616-file development map is the working corpus. Tile 0_0 is the richest but is explicitly
  unrepresentative, so findings must hold beyond it.
- MSPV/MSPI is the leading candidate for connective geometry on the evidence that it is larger than
  the decoded surface mesh and attaches to the same link records. Being a lead, not a conclusion, it
  may be eliminated.
- The nesting hypothesis — that missing data is encoded inside another structure under a different
  coordinate frame — is a working hypothesis, not an assumption to build on.
- A negative result is a valid outcome. Eliminating a candidate domain with evidence is progress and
  must be recorded as such.
- Out of scope: PM4 to asset matching (Spec 128), the PM4 zarr dataset (Spec 129), CASC support.
  This feature produces the decode those two consume.
