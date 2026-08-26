# Feature Specification: Rosetta Calibration Corpus for PM4 Object Identification

**Feature Branch**: `190-rosetta-calibration-corpus`

**Created**: 2026-08-26

**Status**: Draft

**Input**: User description: "We have access to every single object in the game. Instead of heuristic
matching that does not fully work on all PM4 tiles, generate map tiles that evenly place every game
object at a known position, decode that synthetic data back through our own pipeline to build a
complete labelled reference library, and automatically match real PM4 data against it. Also: we are
not generating ADTs for tiles that have no ADT — synthesize them so no tile is skipped."

> **Implementation checkpoint (2026-08-26).** US1 generation slice is source-proven and unit-tested:
> [`RosettaTilesetGenerator`](../../src/core/WowViewer.Core.IO/Maps/RosettaTilesetGenerator.cs)
> (deterministic designkit-style shelf layout, footprint-sized cells, occupied-tile skip) +
> [`RosettaTextPainter`](../../src/core/WowViewer.Core.IO/Maps/RosettaTextPainter.cs) (5x7 bitmap
> font rasterized into MCCV vertex colors, 127-neutral background) + `rosetta-generate` CLI command
> in `WowViewer.Tool.Inspect` enumerating assets via `NativeMpqService`, bounds via
> `MdxSummaryReader`/`WmoSummaryReader`, tiles built on `BlankAdtFactory` and written by
> `LkAdtWriter`. 5 focused tests pass (determinism, non-overlap, occupied-skip, LK round-trip,
> label sanitize); solution builds clean. Not yet done: reference-library builder (US2), lookup
> (US3), companion-ADT synthesis (US4), real-client enumeration run (user-owned).

**Consumed by**: [176](../176-object-transfer/spec.md) (reconciliation matching authority),
related evidence lanes [184](../184-pm4-generation-from-geometry/spec.md) and
[185](../185-pm4-pd4-format-documentation/spec.md). Tile-creation mechanics build on
[177](../177-adt-tile-creation/spec.md) prior art but this feature is an offline pipeline, not an
editor workflow.

## Problem

PM4 object identification today is per-tile heuristic scoring (`Pm4AssetMatchScorer`): each decoded
PM4 segment is ranked against whatever placements happen to exist in the paired ADT corpus, with a
score floor and ambiguity window. Consequences:

1. **Ground truth is scarce and uneven.** The current object library maps 904 PM4 objects to 243
   source assets — a fraction of what exists. Tiles whose companion ADT is missing or thin produce
   weak or empty candidate sets.
2. **Tiles without a companion `_obj0.adt` are skipped entirely** rather than processed, so those
   PM4 files contribute nothing.
3. **Scores are relative, not absolute.** A "best candidate" among three bad candidates still looks
   like a match. There is no complete, labelled reference to say what each asset actually looks like
   to the pipeline.

Meanwhile the configured client contains every placeable object in the game. That is a complete
labelled corpus waiting to be built — we control both the writer and the reader, so we can construct
data whose ground truth is perfect by construction.

## Solution Concept

Build a **Rosetta map**: a synthetic tileset in which every placeable game object is placed exactly
once at a deterministic grid position, with a manifest recording cell → asset identity. Write it with
our own placement authoring/writers, read it back with our own readers, and run the **same**
segmentation/signature pipeline used on real PM4 data over the result. Each asset's signature is then
known perfectly. Real PM4 objects are identified by looking up their signature in this complete
reference library — a deterministic comparison against total coverage, not a per-tile popularity
contest.

This is not an invented shape — it mirrors Blizzard's own level-design practice. Official development
files contain **designkit** maps: grids of objects laid out across multiple ADT tiles with each
object's name written below it in the world. No visible grid lines — the spacing is implicit — and
the layout freely crosses tile boundaries, because the underlying map is just a canvas to paint on.
Designers never need to know how ADTs work underneath. The Rosetta map is a **regenerated
designkit**: the same continuous labelled canvas — uniform invisible spacing, tile boundaries treated
as irrelevant — rebuilt by us so that instead of in-world name labels the identity lives in a
machine-readable manifest. Harvesting the shipped kit maps is explicitly not relied upon; we
regenerate the tiles ourselves so layout, labels, and coverage stay fully under pipeline control.

The pipeline must be **offline-only**: our writers emit the tiles, our readers decode them back, and
no real client ever loads the Rosetta map.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate the Rosetta tileset with perfect labels (Priority: P1)

The operator points the tool at a configured client root. The tool enumerates every placeable object
(model and world-model) the client offers, lays them out one-per-cell across as many synthetic tiles
as needed at fixed spacing, writes the tiles plus a manifest mapping every grid cell to its asset
identity, and verifies by reading the tiles back that every placement survived the round trip with
its identity intact.

**Why this priority**: Nothing else in the feature exists without the labelled corpus. It delivers
standalone value immediately: a complete inventory of placeable assets with known geometry.

**Independent Test**: Run generation against a configured client; read every emitted tile back;
confirm the number of recovered placements equals the number of manifest entries and each recovered
placement resolves to the manifest's asset for its cell.

**Acceptance Scenarios**:

1. **Given** a configured client root, **When** generation runs, **Then** every enumerable placeable
   object appears exactly once in the synthetic tileset or appears on an explicit exclusion report
   naming the reason it could not be placed.
2. **Given** the written tileset, **When** it is decoded with the project's own readers, **Then**
   100% of written placements are recovered and each maps back to its manifest entry.
3. **Given** two runs over the same inputs, **When** generation repeats, **Then** the outputs are
   byte-stable or differ only in recorded timestamps — layout is deterministic.
4. **Given** an object too large for one grid cell, **When** it is laid out, **Then** the layout
   accounts for its footprint so no two objects' footprints overlap, and the manifest records the
   cells it spans.

### User Story 2 - Build the labelled reference library (Priority: P1)

The decoded Rosetta tiles are pushed through the same object-segmentation and signature pipeline that
processes real PM4 data. The output is a reference library: one entry per source asset, holding that
asset's measured signature(s), keyed by the manifest's identity.

**Why this priority**: This is the piece that converts "synthetic map" into "ground truth". It shares
P1 because without it the corpus is inert.

**Independent Test**: Take any N assets from the library, re-run their Rosetta segments through the
matcher, and confirm each resolves to its own identity — the library self-test.

**Acceptance Scenarios**:

1. **Given** the decoded Rosetta tileset, **When** the standard segmentation pipeline runs, **Then**
   every produced segment carries the manifest identity of the cell it came from.
2. **Given** the segmented corpus, **When** the library is built, **Then** it covers every asset that
   produced at least one segment and reports assets that produced none.
3. **Given** the completed library, **When** the self-test runs (match Rosetta segments against the
   library), **Then** top-1 identification accuracy is effectively perfect (≥99%); any miss is
   reported as a library defect with the offending pair named.

### User Story 3 - Deterministic PM4 identification by lookup (Priority: P2)

A real PM4 file is decoded and segmented as today. Instead of scoring candidates from whatever the
paired ADT happens to contain, each segment's signature is looked up in the reference library. The
result is one of: identified (with the matched asset and the comparison evidence), ambiguous
(competing near-equal references, all named), or no-reference (nothing in the library resembles it —
which is itself valuable, flagging either an unenumerated asset or a decode defect).

**Why this priority**: This is the payoff, but it depends on Stories 1–2 and must be proven against
the real corpus before it becomes the authority.

**Independent Test**: Run lookup over the existing measured PM4 corpus; compare identified/matched
rates against the current scorer baseline (904 objects / 243 assets).

**Acceptance Scenarios**:

1. **Given** a real PM4 object whose true asset is in the library, **When** lookup runs, **Then** it
   identifies the asset or reports ambiguity with the true asset among the named competitors — never
   a confident wrong answer with the truth absent from the candidate list.
2. **Given** the full measured corpus, **When** lookup runs, **Then** the identified fraction meets
   or exceeds the current scorer's, and every result carries comparable evidence (which signals
   agreed, which disagreed).
3. **Given** a segment matching nothing in the library, **When** lookup completes, **Then** the
   result is an explicit no-reference status, not a low-score best guess.
4. **Given** both lookup and legacy scorer available, **When** they disagree on an object, **Then**
   the disagreement is surfaced in the report rather than silently resolved.

### User Story 4 - Synthesize companion ADTs for ADT-less PM4 tiles (Priority: P2)

For every PM4 tile that has no companion ADT, the pipeline synthesizes a minimal companion (empty or
flat terrain, correct era form, registered in the map's tile index) so downstream steps stop skipping
the tile. The synthesis report names every synthesized file and the tile it was created for.

**Why this priority**: It removes a whole class of silently-missing coverage and is independent of
the matching mechanism.

**Independent Test**: Enumerate PM4 tiles lacking companions before and after; after synthesis the
before-set is fully covered by the synthesis report and zero tiles are skipped for a missing
companion.

**Acceptance Scenarios**:

1. **Given** a PM4 tile with no companion ADT, **When** the pipeline runs, **Then** a valid companion
   is produced in the output directory in the correct era form and the tile proceeds through the
   normal pipeline.
2. **Given** a PM4 tile whose companion already exists, **When** the pipeline runs, **Then** the
   existing file is used and nothing is overwritten.
3. **Given** synthesis of a companion, **When** complete, **Then** the run report lists it with its
   source tile and content hash so synthesized data is always distinguishable from real data.

### Edge Cases

- More placeable objects than fit one tile → layout spans multiple tiles; manifest remains complete.
- The same underlying asset reachable under multiple names/paths → deduplicated by identity with all
  aliases recorded, or placed per-alias if the pipeline treats them as distinct; decision recorded in
  the manifest.
- Assets that fail to load or contain no usable geometry → exclusion report, never silent omission.
- Name-table capacity limits in the target era's format when a tile holds many distinct assets →
  layout spreads distinct names across tiles within format limits.
- A real PM4 object composed of multiple sub-objects spanning cells → segmentation must handle
  multi-cell footprints via the manifest's span records.
- Synthesized companions must never be mistaken for authentic data in later analysis → provenance is
  machine-readable and travels with the output.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST enumerate every placeable object available from the configured client
  root at runtime (never a hardcoded path), and record the enumeration (counts, sources, exclusions)
  in the run report.
- **FR-002**: The system MUST generate synthetic map tiles placing enumerated objects at
  deterministic grid positions with non-overlapping footprints, sized to each object.
- **FR-003**: The system MUST emit a machine-readable manifest mapping every grid cell (and
  multi-cell span) to its asset identity, plus the generation parameters needed to reproduce it.
- **FR-004**: Generation MUST use only existing project writers and readers for emission and
  round-trip verification; no new format serializer may be introduced.
- **FR-005**: The system MUST verify the write→read round trip for every generated placement and fail
  the run if any placement fails to recover with its identity intact.
- **FR-006**: The reference library MUST be built by running the same segmentation/signature pipeline
  used on real PM4 data over the decoded Rosetta tiles — never a separate simplified path.
- **FR-007**: Real PM4 identification MUST be performed as lookup against the reference library,
  returning identified / ambiguous / no-reference statuses with comparison evidence; a ranked-list
  score alone MUST NOT be presented as identification.
- **FR-008**: The legacy per-tile scorer MUST be demoted to a secondary signal used for tie-breaking
  and disagreement reporting; it MUST remain available for diagnostics.
- **FR-009**: The system MUST synthesize companion ADTs for PM4 tiles that lack one, in the correct
  era form, into the output directory, leaving existing companions untouched.
- **FR-010**: All synthesized files MUST be listed in a provenance report (source tile, parameters,
  content hash) distinguishing them from authentic data.
- **FR-011**: All outputs go to the configured output directory; no game install or Blizzard
  container is ever written.
- **FR-012**: Regeneration over unchanged inputs MUST be idempotent (stable layout and library
  content), so the library can be cached and version-checked rather than rebuilt blindly.
- **FR-013**: Synthetic layout MUST follow the designkit convention: one continuous canvas with
  uniform invisible spacing that crosses tile boundaries freely — no per-tile alignment or padding is
  introduced to respect file boundaries, because the manifest, not the terrain, carries identity.

### Key Entities

- **RosettaManifest**: The authoritative cell → asset mapping for a generated tileset, including
  layout parameters, multi-cell spans, aliases, and exclusions.
- **ReferenceLibrary**: The set of per-asset signatures derived from the decoded Rosetta tiles via the
  standard pipeline, keyed by manifest identity, versioned against pipeline and manifest versions.
- **IdentificationResult**: Per real-PM4-segment outcome — matched asset + evidence, ambiguous with
  named competitors, or no-reference — plus any disagreement with the legacy scorer.
- **CompanionSynthesisReport**: Record of every synthesized companion ADT: source tile, era form,
  content hash.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Round-trip recovery is 100%: every written Rosetta placement is read back and resolves
  to its manifest identity; any failure aborts the run.
- **SC-002**: Library self-test top-1 accuracy is ≥99% on Rosetta-derived segments; every miss is
  explained and tracked as a defect.
- **SC-003**: On the existing measured PM4 corpus, lookup identifies a strictly larger fraction of
  objects than the current baseline (904 objects mapped to 243 assets), with zero confident-wrong
  results where the true asset was absent from candidates.
- **SC-004**: After companion synthesis, zero PM4 tiles are skipped due to a missing companion ADT,
  and every synthesized file appears in the provenance report.
- **SC-005**: Reference-library coverage equals the enumeration minus reported exclusions — the
  operator can see, as a number, how much of the game's object space the matcher can possibly name.
- **SC-006**: Two consecutive generations over identical inputs produce identical manifests and
  libraries (modulo recorded timestamps).

## Assumptions

- **Offline-only** (operator decision): the Rosetta map is never required to load in a real client;
  project writers/readers are both halves of the round trip.
- Target era follows the project's active Alpha 0.5.3 focus; other eras are out of scope until the
  Alpha pipeline is proven.
- "Every object" means every placeable model/world-model the client enumeration exposes; non-placeable
  or unloadable entries land on the exclusion report rather than blocking generation.
- Existing placement-authoring and tile-writing seams (per Specs 175/177) are sufficient; if a gap is
  found it is raised as a bounded extension to the owning owner, not a new serializer here.
- The legacy scorer's saved choices and corpus signals remain readable evidence (Spec 176 decision 2);
  this feature changes which mechanism is *authoritative*, not the historical record.

## Out of Scope

- Loading the Rosetta map in a real client or harvesting client-side renders of it.
- Cross-era transfer mechanics (Spec 176 P1) beyond what lookup naturally enables.
- New ADT/WDT serializers; terrain synthesis beyond flat/minimal companion tiles.
- Editing the frozen Alpha WDT writer without an explicitly reopened decision.
