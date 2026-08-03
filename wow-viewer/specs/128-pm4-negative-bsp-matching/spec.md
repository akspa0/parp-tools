# Feature Specification: PM4 Negative-BSP Object Matching

**Feature Branch**: `128-pm4-negative-bsp-matching`

**Created**: 2026-08-03

**Status**: Draft

**Input**: User description: "We still need a better way to match pm4 objects to real game assets, for the 4.0.0-era development map restoration work, which is still part of this whole pipeline we're building." Plus the reframe: "PM4's provide more than the raw data that a WMO stores, for every object, despite not containing any of the geometry from the WMO — it contains the negative bsp data, ultimately, for every placed object on a map tile. We never thought about it that way until right now."

## Context

### The reframe

A PM4 file carries no object geometry, yet it records more per placed object than the object's own
model file does. What it holds is the **negative space**: the surfaces a player or NPC stands on,
walks along, and paths across. An object's PM4 footprint is the shape of the hole it makes in the
walkable world.

Matching today does not use that. It compares aggregate scalars — footprint area, minimum/maximum/
average plane distance, topology counts, anchor signals — which cannot separate two objects with
similar footprints and similar height ranges but different walkable structure. A round tower and a
square keep of the same base area and height are near-identical under the current signals and
completely different under their negative space.

### Why the fidelity argument matters

PM4 data is **not distilled**. Terrain heightfields are: the shipped ADT resolution is plausibly a
quarter or a sixteenth of the internal authoring resolution, and the undistilled state of the PM4s
is the basis for that estimate. That inverts the usual assumption — PM4 is the highest-fidelity
surviving record of object placement, so it should be treated as the reference that other sources
are checked against, not as a derived by-product of the models.

### What this unblocks

A 4.0.0-era development map whose object placements are otherwise lost. If PM4 segments can be
confidently matched to real assets, placements can be synthesized and the map restored. Match
quality is the whole gate.

### Historical context this work sits inside

Recorded because it shapes what the data is and must not be lost:

- The game began as **one world named Azeroth**, fitting in a container smaller than 64x64 tiles.
  When it outgrew that it was split into separate maps in separate map folders.
- The only surviving artifact of that 1999-era single-world build is the original **`world.def`**,
  which contains nothing but day/night light-cycle data for the world sun — alongside a hand-drawn
  blockout map showing **both continents in one frame**, carrying names that never shipped
  (Amberhorn Caverns, Jaedenar as a dungeon, Kobold Lair, Tauren-Newbie, Stonetalon Peak) with
  difficulty tiers already assigned, plus a handful of other images from the same period.
- The **64x64 = 4096 tiles per map** ceiling and **16x16 cells per chunk** addressing are hardcoded
  and unchanged through 2026. Blizzard raised visual density by scaling objects, players and cameras
  rather than the tile count.
- WoW began as a Warcraft 3 script-modded map and carried WC3 constraints through every rewrite;
  the 0.5.3 MDX format is an experimental newer WC3 MDX.
- WMO is Blizzard's answer to multi-nested Quake 3-style BSPs, scaled to 384 groups by the end of
  2003, where one WMO group is roughly one Quake 3 map. The dungeon generator used BSPs the same
  way, and the earliest WMOs were authored in Radiant.

The BSP lineage is the reason the negative-space reframe is likely correct: this data descends from
an engine family where the walkable surface set *was* the level representation.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Match a segment on the shape of its negative space (Priority: P1)

A restorer points the matcher at a development-map PM4 segment. Instead of scoring it on summary
statistics, the system compares the structure of the walkable surfaces the segment describes against
the same structure derived from candidate assets, and returns ranked matches with the evidence that
produced each score.

**Why this priority**: This is the reframe. Every other story depends on the match being right, and
today's aggregate-scalar scoring is the specific thing that is not good enough.

**Independent Test**: Take segments whose asset identity is already known, score them under both the
existing scalar signals and the structural signals, and compare correct-match rates on the same set.

**Acceptance Scenarios**:

1. **Given** two candidate assets with near-identical footprint area and height range but different
   walkable topology, **When** a segment of the first is scored, **Then** the correct asset ranks
   above the other, and the report names the structural difference that separated them.
2. **Given** any scored segment, **When** the restorer inspects the result, **Then** each candidate
   carries a score, a rank, and the per-signal contributions behind it.
3. **Given** a segment whose true asset is absent from the candidate set, **When** it is scored,
   **Then** it is reported unmatched rather than being forced onto the nearest candidate.
4. **Given** the same segment and candidate set scored twice, **When** results are compared,
   **Then** they are identical.

---

### User Story 2 - Know which matches can be trusted (Priority: P2)

A restorer needs to separate matches confident enough to synthesize a placement from those needing
review. The system distinguishes a confident match, an ambiguous one where several candidates are
statistically tied, and an unmatched segment — and says which it is and why.

**Why this priority**: A wrong placement silently corrupts a restored map, which is worse than a
missing one. Depends on US1 producing scores worth thresholding.

**Independent Test**: Score a set containing known-good matches, deliberate near-ties, and segments
with no valid candidate; confirm each lands in the right band.

**Acceptance Scenarios**:

1. **Given** a segment whose top candidates are within the ambiguity window, **When** it is scored,
   **Then** it is reported ambiguous with all tied candidates listed, not silently resolved.
2. **Given** a confident match, **When** a placement is synthesized, **Then** the placement records
   the match score and the signals that justified it.
3. **Given** a corpus of scored segments, **When** the restorer reviews them, **Then** they can be
   partitioned by confidence band without re-scoring.

---

### User Story 3 - Measure matching against a known answer key (Priority: P3)

Before trusting the matcher on a map whose placements are lost, the restorer runs it against a map
whose placements are known, and gets a correct/incorrect/ambiguous/missed breakdown.

**Why this priority**: Without this, "better matching" is an assertion. It is P3 only because US1
and US2 are usable manually on a small set first; at scale this becomes mandatory.

**Independent Test**: Run against a map with complete known placements and confirm the reported
accuracy matches a hand-audited sample.

**Acceptance Scenarios**:

1. **Given** a map with known placements, **When** the matcher runs, **Then** it reports how many
   segments matched correctly, incorrectly, ambiguously, and not at all.
2. **Given** two versions of the matching approach, **When** both run on the same map, **Then**
   their accuracy is directly comparable.
3. **Given** an accuracy report, **When** the restorer inspects a failure, **Then** they can reach
   that segment's full scoring evidence.

---

### Edge Cases

- A segment assembled from more than one object, or an object split across segments.
- Objects that are near-symmetric under rotation, where negative space cannot fix orientation.
- Assets appearing many times on one tile: matching must not collapse repeats into one placement.
- A segment whose walkable surfaces are partly cut by terrain or by another object.
- Assets that genuinely share negative space (a reskin with identical collision).
- Segments too small to carry structural signal, where scalar comparison is all that is available.
- Development-map assets that do not exist in the reference client at all.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST derive a structural description of a PM4 segment's walkable surfaces —
  their arrangement and connectivity — not only aggregate measurements of them.
- **FR-002**: The system MUST derive the equivalent structural description for candidate assets, so
  the two are compared on the same terms.
- **FR-003**: The system MUST rank candidate assets per segment, and record the per-signal
  contributions that produced each score.
- **FR-004**: The system MUST classify every result as confident, ambiguous, or unmatched, and MUST
  NOT resolve an ambiguous result silently.
- **FR-005**: The system MUST report an unmatched segment rather than forcing it onto the nearest
  candidate when no candidate is a real match.
- **FR-006**: Scoring MUST be deterministic: identical inputs produce identical output.
- **FR-007**: The system MUST retain the existing scalar signals alongside the structural ones, so
  the contribution of the new approach is measurable rather than assumed, and so segments too small
  to carry structural signal still score.
- **FR-008**: The system MUST support evaluation against a map with known placements, reporting
  correct, incorrect, ambiguous and missed counts.
- **FR-009**: The system MUST version its signal definitions, so results computed under different
  definitions are never silently compared.
- **FR-010**: Synthesized placements MUST carry the match score and supporting signals that
  justified them.
- **FR-011**: The system MUST preserve repeated instances of the same asset on a tile as distinct
  placements.
- **FR-012**: The system MUST record, per segment, which signals were unavailable, so a low score
  caused by missing input is distinguishable from a genuine mismatch.

### Key Entities

- **Segment**: One candidate object's worth of walkable surfaces extracted from a PM4 tile, with its
  position, orientation, and the surfaces themselves.
- **Structural description**: The arrangement and connectivity of a segment's walkable surfaces —
  the shape of the negative space — expressed so that a segment and an asset can be compared.
- **Asset reference**: The same structural description derived from a known game asset, plus the
  identity of that asset, under a recorded signal version.
- **Match result**: A segment, its ranked candidates, per-signal contributions, confidence band, and
  any unavailable signals.
- **Placement proposal**: A confident match turned into a concrete placement, carrying the score and
  evidence behind it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a map with known placements, correct-match rate improves measurably over the
  existing scalar-only scoring on the identical segment set and candidate set.
- **SC-002**: For pairs of assets with similar footprint area and height range but different
  walkable structure, the correct asset outranks the confusable one in at least 90% of cases.
- **SC-003**: Every match result carries per-signal evidence; no result is reported as a bare score.
- **SC-004**: Segments with no valid candidate are reported unmatched in 100% of cases, never
  assigned to a nearest candidate.
- **SC-005**: Repeated scoring of the same inputs produces identical results, verified across runs.
- **SC-006**: A restorer can go from a scored corpus to the evidence behind any single match without
  re-running the matcher.
- **SC-007**: Match quality is high enough that a restored development map's placements can be
  reviewed by exception — the confident band is trusted, and review effort concentrates on the
  ambiguous and unmatched bands.

## Assumptions

- Scope is PM4-to-asset matching and the placements it feeds. CASC support, client re-harvesting,
  and the terrain weak-signal archaeology pipeline are out of scope.
- The existing segment building, signal extraction, scoring, export and placement-synthesis
  components are improved rather than replaced; the reframe changes what is compared, not the shape
  of the pipeline.
- The 616 PM4 files under the development map are the working corpus.
- A reference client is available from which candidate asset descriptions can be derived.
- "Walkable surface structure" is deliberately left as a property to be designed during planning;
  the requirement is that structure is compared, not which specific formulation wins.
- Existing confidence thresholds are a starting point and expected to be re-derived once structural
  signals change the score distribution.
- Known MSLK type flags observed in real data (M2 top, interior floor, exterior solid) remain valid
  and continue to inform which surfaces belong to which kind of object.
