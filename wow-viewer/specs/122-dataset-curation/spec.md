# Feature Specification: Canonical Dataset Curation and Signal-Mismatch Bucketing

**Feature Branch**: `122-dataset-curation`

**Created**: 2026-07-30

**Status**: Draft

**Input**: User description: "Canonical C# dataset curation and signal-mismatch bucketing layer for
the wow-viewer terrain dataset pipeline. Our dataset curation logic (quality bucketing, mismatch
detection between signals, blank-tile detection, coverage sanity checks) has never survived from
one model generation to the next — it is reimplemented ad hoc in Python, per spec, scattered across
`v16_curation.py`, `mismatch_detector.py`, `spec111/lighting_buckets.py`, and several one-off
audit scripts. Hardwire ONE canonical curation implementation in C#, executed as part of or
immediately following the harvest pipeline, so curation state is durable and travels WITH the
dataset store. Curation MUST NOT be a filter that silently discards bad or mismatched tiles — it
must PARTITION every tile into labeled, fully-accessible buckets with reasons and severities
recorded, so bad/mismatched data stays available for deliberate use (hard-negative study,
mismatch-aware training, harvester debugging), not just quietly excluded."

## Problem Statement

Every model generation on this project (V16, V18, V22, V50, and the specs built on top of it —
109, 111, 115, 116, 117, 118, 121) has needed to answer the same question before training could be
trusted: *is this tile's signal set clean, and if not, clean how?* Each generation has answered it
by writing new, throwaway Python: `v16_curation.py` (difficulty buckets, blank-tile "what-plate"
detection, per-signal coverage helpers), `mismatch_detector.py` (height-vs-normal relief mismatch
scoring), `spec111/lighting_buckets.py` (lighting-bucket reconciliation from shading-match
provenance), and several one-off audit scripts (`build_v16_curation_manifest.py`,
`v50_audit_signal_coverage.py`, `v50_audit_artifacts.py`). None of this logic is shared across
generations; each spec either re-derives its own version, silently skips curation entirely, or
hand-ports fragments. The result is that "clean data" has been redefined, forgotten, and
re-discovered at least four times, and there is no single place a new spec can go to find out
which tiles are trustworthy and why.

A second, independently confirmed problem compounds the first: the curation logic that does exist
treats "bad" as something to filter out and forget. Only a filtered "good" view is readily
produced; the excluded rows and the reasons they were excluded are not durably kept alongside it.
This has already cost real signal — for example, `mismatch_detector.py` can score a
height-vs-normal mismatch, but nothing keeps that scored-and-rejected population queryable on its
own terms afterward. The project's own synthetic minimap renders are a concrete case in point: the
terrain shading/shadow pattern in a synthesized minimap tile does not match the authored client
minimap for that tile (a known, currently unquantified discrepancy), which makes synthetic tiles a
plausible source of systematic error if trained on uncritically — but today there is no durable,
queryable record of *how* mismatched any given synthetic tile is relative to its authored
counterpart, only ad hoc one-off comparisons.

## Governing Principle

Curation classifies; it does not delete. Every harvested tile receives a durable classification —
one or more quality buckets, zero or more mismatch findings with a reason and severity, and (where
applicable) a synthetic-vs-authored fidelity score — and that classification is written back
alongside the dataset store as a first-class, independently queryable artifact. A downstream
consumer MAY choose to train on the "clean" bucket only, but every other bucket (mismatched, blank,
pathological, not-evaluable) remains just as accessible, so the excluded population can be
inspected, measured, and deliberately used — never silently dropped.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One Canonical Classification Pass, Not a Version-Specific Reimplementation (Priority: P1)

A model-training operator starting a new spec needs to know, for a given dataset store, which
tiles are trustworthy and which are not, without writing new curation code or hunting through
5+ prior specs' Python for the closest-matching logic. They run one canonical curation pass over
the store and get back a durable, versioned classification record covering every tile: quality
bucket, mismatch findings (with reason and severity), and coverage statistics — consolidating what
`v16_curation.py`, `mismatch_detector.py`, and `spec111/lighting_buckets.py` each did separately
today.

**Why this priority**: This is the core ask — a single source of truth that survives past this
session, instead of a sixth reimplementation that the next spec will also have to rediscover.

**Independent Test**: Run the canonical curation pass against an existing dataset store; confirm
every tile row receives a bucket assignment and that tiles known (from prior sessions' findings)
to be blank, height-normal-mismatched, or low-confidence-lit are classified accordingly, matching
or improving on the scattered Python's own prior findings on the same data.

**Acceptance Scenarios**:

1. **Given** a dataset store with tiles of varying quality, **When** the curation pass runs,
   **Then** every tile receives exactly one primary quality bucket and zero or more mismatch
   findings, and the result is written as a durable artifact alongside the store (not printed to a
   console and discarded).
2. **Given** a tile previously flagged as a height-vs-normal mismatch by `mismatch_detector.py`,
   **When** the canonical pass classifies the same tile, **Then** it is flagged with an equivalent
   mismatch finding and severity, so no detection power is lost in the consolidation.
3. **Given** a tile that is genuinely clean on every checked dimension, **When** the pass runs,
   **Then** it is classified into a "clean" bucket with no mismatch findings, so clean tiles are
   not falsely flagged.
4. **Given** the curation pass is re-run on a store it already classified, **When** nothing about
   the store has changed, **Then** the classification is reproducible (same inputs, same outputs).

---

### User Story 2 - Every Bucket Stays Fully Accessible, Never Silently Dropped (Priority: P1)

A researcher wants to deliberately study the tiles curation flagged as bad or mismatched — for
example, to see whether a "hard" or "mismatched" bucket is worth training on anyway, or to debug
why the harvester produced a mismatched tile in the first place. They query the curation output
for exactly that bucket and get the full set of tiles, their reasons, and severities — the same
ease of access a "clean-only" consumer gets, not a special-case recovery operation.

**Why this priority**: This is the explicit, previously-missing capability the user identified as
the actual root problem: today only a filtered "good" view is convenient to get at, so the
excluded population is effectively invisible and never gets used or reasoned about, even though
studying it is exactly what would improve the next model generation.

**Independent Test**: Query the curation output for a non-clean bucket (e.g., "mismatched") and
confirm the full tile list, reasons, and severities are returned with no additional privileged
step compared to querying the "clean" bucket.

**Acceptance Scenarios**:

1. **Given** the curation classification for a store, **When** a consumer requests tiles in the
   "clean" bucket, **Then** they get those tiles; **When** the same consumer requests tiles in a
   "mismatched" or "pathological" bucket, **Then** they get those tiles with equal completeness —
   neither request is treated as a degraded or unsupported path.
2. **Given** a tile has multiple mismatch findings (e.g., both a height-normal mismatch and low
   alpha coverage), **When** the classification is queried, **Then** all findings are present on
   that tile's record, not just the first or most severe one.
3. **Given** a downstream trainer selects only the "clean" bucket, **When** the run record for
   that training run is produced, **Then** it states which bucket(s) were selected and how many
   tiles were excluded and why, so a "clean-only" choice remains visible and auditable rather than
   quietly erasing the rest of the dataset from the record.

---

### User Story 3 - Synthetic-vs-Authored Minimap Fidelity as an Exposed, Queryable Signal (Priority: P2)

A model-training operator suspects the project's synthesized minimap renders (used where an
authored client minimap is unavailable, or paired against one for comparison) are a source of
systematic error, because their terrain shading/shadow pattern does not visually match the real
authored minimap for the same tile. They want to know, per tile, how large that mismatch is,
so they can decide later whether to exclude synthetic tiles, weight them down, or use the mismatch
itself as a training signal — without this feature deciding that modeling question for them.

**Why this priority**: This is a concrete, previously-unmeasured instance of the general
mismatch-detection problem, called out directly because the project already has partial machinery
for comparing synthetic renders against authored minimaps (the existing shading/time-of-day match
scoring) that has never been repurposed to answer "how good is this synthetic tile," only "what
time of day does it look like."

**Independent Test**: Run the fidelity check on a set of tiles that have both a synthetic and an
authored minimap; confirm each such tile gets a recorded fidelity/mismatch magnitude, and that a
tile with a visibly implausible synthetic render (e.g., flat/blown-out shading with no correlation
to the authored image at any candidate) scores worse than a tile whose synthetic render tracks the
authored one closely.

**Acceptance Scenarios**:

1. **Given** a tile with both a synthesized minimap and an authored minimap, **When** the curation
   pass runs, **Then** a synthetic-fidelity finding is recorded on that tile with a magnitude and
   reason, independent of whether the tile is otherwise clean.
2. **Given** a tile with only a synthesized minimap (no authored minimap available), **When** the
   curation pass runs, **Then** the tile is explicitly marked as not evaluable for synthetic
   fidelity, distinct from being marked as fidelity-passing.
3. **Given** the recorded fidelity findings across a store, **When** a consumer queries them,
   **Then** the values are usable as a per-tile weighting or exclusion signal by later, separate
   model-side work — this feature exposes the measurement but does not implement a loss function
   or training behavior that consumes it.

---

### User Story 4 - Old Scattered Curation Scripts Stop Being the Source of Truth (Priority: P3)

A future spec's author, six months from now, needs curated data and — out of habit — starts
looking for the nearest prior spec's curation script to copy. They instead find that the scattered
scripts have a documented, working path back to the canonical classification (either by reading
its output directly or through a thin compatibility wrapper), so copying old logic is never the
easier option.

**Why this priority**: Consolidating the logic once does not, by itself, stop the next person (or
agent) from forking it again under time pressure. This story is the process fix that makes the P1
technical fix stick.

**Independent Test**: Confirm each of the currently scattered curation scripts either reads from
the canonical classification or is documented as retired/historical, with no remaining path that
silently reintroduces a second, divergent definition of "clean."

**Acceptance Scenarios**:

1. **Given** the canonical curation classification exists for a store, **When** a script that
   previously computed its own bucket/mismatch logic is invoked afterward, **Then** it either
   reads the canonical classification or is clearly marked historical/retired in its own output,
   never silently recomputing a competing answer.
2. **Given** a new spec needs curated data, **When** its author looks for how to get it,
   **Then** documentation points to the canonical classification as the only current answer.

---

### Edge Cases

- A tile lacks a signal a given mismatch check depends on (e.g., no decoded normals for the
  height-vs-normal check): the tile MUST be recorded as not-evaluable for that specific check,
  never silently treated as passing or silently treated as failing.
- A mismatch check is scoped to a specific client build (as the existing shading-match logic
  already is, to 0.5.3.3368) and the tile is from a different build: the tile MUST get an explicit
  "not evaluated — out of scope" reason for that check, distinct from "evaluated and clean."
  Non-scoped checks on the same tile still run normally.
- A tile qualifies for more than one quality bucket under a naive reading (e.g., it is both
  low-relief/"easy" and has a mismatch finding): the classification MUST retain both the bucket
  assignment and the mismatch finding rather than forcing a single label that hides one or the
  other.
- The curation pass is re-run after the underlying dataset store changes (re-harvested tiles,
  added signals): previously classified tiles MUST be distinguishable from newly (re-)classified
  ones so a partial rerun cannot be mistaken for a full one.
- A consumer requests a bucket that has zero tiles in it (e.g., no tiles are currently
  "pathological"): the query MUST return an empty, valid result, not an error, and not silently
  fall back to a different bucket.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide one canonical implementation of dataset curation — quality
  bucketing and signal-mismatch detection — that supersedes the logic currently scattered across
  `v16_curation.py`, `mismatch_detector.py`, `spec111/lighting_buckets.py`,
  `build_v16_curation_manifest.py`, `v50_audit_signal_coverage.py`, and `v50_audit_artifacts.py`.
- **FR-002**: The canonical curation implementation MUST live in the project's shared, durable
  library layer (not a per-spec or per-model-generation script), so it is available to every
  future spec without reimplementation.
- **FR-003**: Curation MUST run as part of, or immediately following, the harvest pipeline that
  produces a dataset store, so classification state is produced alongside the data it describes
  rather than as a separately-scheduled, easily-skipped step.
- **FR-004**: The system MUST classify every tile in a store into at least one quality bucket,
  covering at minimum: clean/good, blank/near-blank, low-signal-coverage, and pathological —
  equivalent in coverage to today's difficulty-bucket and blank-tile ("what-plate") logic.
- **FR-005**: The system MUST detect and record, at minimum, the mismatch categories already
  proven valuable in prior sessions: height-vs-normal relief mismatch (flat height, varied
  normals), non-finite/NaN signal values, and a signal's presence flag disagreeing with whether
  real data actually backs it.
- **FR-006**: The system MUST detect and record lighting/time-of-day match confidence per tile
  (matched / low-confidence / not-evaluated), consolidating the existing shading-match provenance
  logic into the same durable classification record as the other buckets and findings.
- **FR-007**: The system MUST detect and record a synthetic-vs-authored minimap fidelity finding
  for every tile that has both a synthesized and an authored minimap, expressed as a magnitude and
  reason, distinct from (but able to reuse the scoring machinery of) the existing time-of-day
  shading-match logic.
- **FR-008**: Curation output MUST NOT delete, hide, or omit any tile regardless of how poorly it
  scores; every tile MUST appear in the classification output with its bucket(s) and finding(s),
  including tiles classified as blank or pathological.
- **FR-009**: Every quality bucket and every mismatch category MUST be independently queryable —
  a consumer MUST be able to retrieve exactly the tiles in a given bucket or with a given finding
  with no more effort than retrieving the "clean" bucket.
- **FR-010**: A tile's classification record MUST support multiple simultaneous findings (a tile
  is not forced into a single label that hides co-occurring issues).
- **FR-011**: When a mismatch or fidelity check cannot run for a tile (missing dependent signal,
  out-of-scope build), the system MUST record an explicit not-evaluable/not-evaluated state for
  that specific check, distinguishable from both "checked and clean" and "checked and flagged."
- **FR-012**: The classification output MUST be versioned and reproducible: re-running curation
  against an unchanged store MUST produce the same bucket and finding assignments.
- **FR-013**: The classification output MUST record, for any consumer that selects a subset of
  buckets (e.g., "clean only"), enough information for that selection and its excluded counts and
  reasons to be reconstructed later — selection must be an auditable, reversible choice, not data
  loss.
- **FR-014**: The system MUST NOT modify, delete, or move the underlying harvested signals it
  classifies; curation is read-only with respect to the source dataset store.
- **FR-015**: Existing scattered curation scripts MUST each end up in one of two states: reading
  from the canonical classification, or explicitly documented as retired/historical — no script
  may continue silently computing a second, divergent definition of any bucket or mismatch this
  feature covers.
- **FR-016**: This feature MUST NOT define, select, or implement any model architecture, training
  loss, or loss-weighting behavior; measurements it produces (including the synthetic-fidelity
  finding) are exposed for later, separate model-side work to consume.

### Key Entities *(include if feature involves data)*

- **Tile Curation Record**: The per-tile classification result — identifies the tile (build, map,
  tile coordinates), its quality bucket(s), its mismatch findings (each with category, reason,
  severity, and evaluability state), and its synthetic-fidelity finding where applicable.
- **Quality Bucket**: A named, coarse classification (e.g., clean, blank, low-coverage,
  pathological) a tile is assigned to; a tile may belong to more than one bucket dimension at once
  (e.g., a relief-difficulty bucket and a coverage bucket are independent axes).
- **Mismatch Finding**: A specific detected problem on a tile — category (e.g.,
  height-normal-mismatch, non-finite-value, has-flag-mismatch, lighting-low-confidence,
  synthetic-fidelity-gap), severity, and a human-readable reason.
- **Curation Manifest**: The durable, versioned, store-level artifact collecting every tile's
  curation record, queryable by bucket or finding category without needing to re-run curation.
- **Selection Record**: An audit trail of which bucket(s)/finding(s) a downstream consumer (e.g.,
  a training run) selected from a curation manifest, and the excluded counts/reasons, so a
  filtered view remains traceable back to the full classification.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new model-training effort can obtain a full quality classification for an existing
  dataset store — every tile bucketed, every mismatch category checked — without writing any new
  curation logic, only invoking the canonical implementation.
- **SC-002**: Querying any non-clean bucket (mismatched, blank, pathological) returns the complete
  tile set for that bucket in the same way querying the clean bucket does — verified by comparing
  result completeness and query effort across at least three different buckets.
- **SC-003**: On a store re-classified after this feature ships, the set of tiles flagged as
  height-vs-normal mismatched matches (or is a documented, justified improvement over) what the
  legacy `mismatch_detector.py` flagged on the same store, with zero mismatched tiles becoming
  unrecoverable or unqueryable afterward.
- **SC-004**: Every tile with both a synthesized and authored minimap receives a synthetic-fidelity
  finding; tiles a human reviewer visually judges as poor synthetic matches score measurably worse
  than tiles judged as good matches, establishing the finding as a meaningful, not arbitrary,
  signal.
- **SC-005**: Six months (or one further model generation) after this feature ships, zero new
  per-spec curation scripts have been written from scratch to answer "is this tile clean" — every
  subsequent spec's curation questions are answered by querying the canonical classification.
- **SC-006**: No tile is ever silently absent from the curation manifest — a full-coverage check
  (every tile in the source store has exactly one corresponding curation record) passes on every
  run.

## Assumptions

- The harvest pipeline (C# readers for ADT/WDT/WDL/minimap signals) is complete and out of scope
  for changes; this feature classifies signals the harvester already produces, it does not add new
  raw-signal extraction.
- The existing shading/time-of-day match scoring (per-tile correlation between a synthesized
  candidate render and an authored minimap, currently scoped to the 0.5.3.3368 build) is available
  to reuse as the basis for the synthetic-fidelity finding in User Story 3, rather than needing an
  independent comparison method invented from scratch.
- "Durable, alongside the dataset store" means the classification is written as its own versioned
  artifact tied to the store's identity — the exact storage shape (a companion file, an embedded
  table, or another form) is an implementation decision for the planning phase, not fixed here.
- This feature defines and exposes buckets/findings; it does not mandate which bucket(s) any
  future trainer must select. "Clean-only by default, everything-else-by-request" is the assumed
  common case, not an enforced restriction — a consumer may request any bucket at any time.
- Model architecture, loss design, and whether/how the synthetic-fidelity signal or mismatch
  severities are ever used to weight or filter training samples are explicitly out of scope and
  deferred to future, separate specs — including the parallel external-research pass comparing our
  v50 signal catalog against other aerial/satellite-image-to-terrain projects, which informs but
  does not gate this feature.
- Consolidation (User Story 4) is a documentation-and-wrapper task, not a mass deletion of
  historical scripts; matching this repo's existing convention (e.g., the Spec 109 Phase 6
  shim pattern) of keeping old entry points as thin, non-diverging re-exports rather than removing
  them outright.
