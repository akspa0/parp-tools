# Feature Specification: Relational Terrain Layer Reconstruction

**Feature Branch**: `116-relational-terrain-layers`

**Created**: 2026-07-21

**Status**: Draft

**Input**: User description: "Relational terrain reconstruction — predict MCLY layer structure as table rows, not rasters."

## Motivation

Terrain reconstruction has been framed as continuous raster regression: minimap image in, per-vertex
height out. Four measurements taken against the v50.1 authored corpus say that framing is wrong at
the root, and they explain failures that tuning has not moved.

1. **The terrain is assembled from a library of reused pieces.** Matching 32×32 blocks of the first
   overlay layer across tiles, under all eight rotations and mirrors and excluding same-tile
   matches, finds **9.5% of blocks at 0.99 correlation or higher** with a block in a *different*
   tile (15.3% at 0.95+), against a median best-match of 0.662. At that dimensionality, 0.99 is a
   copy, not a resemblance. Because within-tile reuse was excluded, this is a floor. The target is a
   **discrete alphabet of reused pieces**, and an averaging regressor must return the mean of that
   alphabet — which is precisely the blur that has persisted through every model to date.

2. **A terrain tile is a serialized relational schema, not an image.** Texture, model, and
   world-object name lists are lookup tables. A layer's texture reference is a **foreign key** into
   that tile's own local texture table. Object placements are rows joined to those tables by index.
   Layer entries are **ordered rows**, where the slot is a row ordinal, not an image channel. The
   existing feature-label derivation already performs that foreign-key join. Current models neither
   enforce nor exploit any of this structure.

3. **Layer slots are not interchangeable and not frequency bands.** The base layer carries **no
   alpha map at all** — it is always opaque — so there are three detail layers over a base, not
   four. Successive detail layers are monotonically finer (low-band energy share falls 0.428 →
   0.364, high band rises 0.164 → 0.198), but this is a **gradient, not a partition**: every layer
   carries energy across all bands, so any design assigning one layer to one frequency band assumes
   a separation the data does not have. Per-pixel dominant layer is 76.35% base / 15.30% / 6.21% /
   2.13%, with an overlay present on **23.65% of pixels**.

4. **Two defects make current evaluation untrustworthy.** 99.6% of held-out tiles have a training
   tile as an immediate edge-neighbour and 42.4% are fully surrounded; because adjacent tiles share
   their edge vertices exactly, held-out scoring measures interpolation between memorised boundaries
   rather than generalisation. Separately, **no model has ever beaten the tile-mean baseline**
   (0.1387 versus 0.1723–0.1750 for every trained model), because 39% of height patches are
   effectively flat and 51 of 120 sampled tiles are more than 90% base-layer-only. Aggregate error
   is dominated by flat terrain that a constant predictor already solves.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Decide the head architecture from evidence (Priority: P1)

Before any model is designed, a practitioner determines whether a layer's **slot ordinal** is
recoverable from appearance, or whether only **texture family** is. The same texture may occupy
different slots in different chunks; only what the texture *looks like* is visible in a minimap. If
families map to slots consistently, slot-keyed prediction is viable; if not, prediction must key on
family and slot becomes a training-time grouping only.

**Why this priority**: This single measurement determines the output vocabulary of every model in
the feature. Building heads before answering it risks training against a target that is not
determined by the input — the failure mode that produced 0.17 road IoU.

**Independent Test**: Run the consistency measurement over the corpus and read the reported
family→slot distribution. Delivers a go/no-go architecture decision with no model trained.

**Acceptance Scenarios**:

1. **Given** the corpus of layer entries and their texture references, **When** family→slot
   consistency is measured, **Then** a per-family slot distribution and a single summary consistency
   score are reported.
2. **Given** the consistency score, **When** it falls below the decision threshold, **Then** the
   recommendation records that heads MUST key on family rather than slot, and that decision is
   carried into User Story 3.

---

### User Story 2 - Settle whether layer masks derive from terrain shape (Priority: P1)

A practitioner determines whether layer masks are **derived from the terrain surface** (elevation
and slope) or authored independently of it. The working hypothesis is that masks were distilled from
higher-resolution source artwork with hand fix-ups, which predicts a **bimodal** coupling across
tiles — strong where automated, weak where hand-edited.

**Why this priority**: The answer determines whether layer structure can be *derived* from predicted
geometry or must be predicted independently, which changes what models are needed at all. It is
cheap and it constrains User Story 3.

**Independent Test**: Fit a per-tile mapping from surface properties to layer coverage, report
explained variance per tile, and inspect the distribution for bimodality. No model trained.

**Acceptance Scenarios**:

1. **Given** per-tile elevation and slope alongside layer coverage, **When** a non-linear per-tile
   fit is evaluated, **Then** an explained-variance value is reported for every tile and layer.
2. **Given** those values, **When** their distribution is examined, **Then** the report states
   whether a distinct high-coupling population exists, and at what share of tiles.
3. **Given** a prior linear analysis found weak coupling with no bimodality, **When** the non-linear
   result disagrees, **Then** the report explicitly records that the linear test was underpowered
   for threshold relationships rather than treating the disagreement as noise.

---

### User Story 3 - Predict layer structure from the minimap alone (Priority: P2)

From a raw minimap image and nothing else, the system predicts the terrain's layer structure as
**rows** — which surface family covers each location, in what arrangement — rather than as
independent continuous masks. Predictions must respect the schema's constraints: a predicted texture
reference must be a legal entry in that tile's own table, and a layer must occupy a legal slot.

**Why this priority**: This is the feature's core deliverable, but it is only designable once User
Stories 1 and 2 have fixed the output vocabulary and established whether structure is derivable from
geometry.

**Independent Test**: Run prediction on held-out tiles and score per-class recall and IoU for each
structural class; run it on an out-of-distribution hand-painted image that has no ground truth
whatsoever and confirm it produces a legal, non-degenerate structure.

**Acceptance Scenarios**:

1. **Given** only a minimap image, **When** structure is predicted, **Then** every predicted
   reference is a legal entry for that tile and no constraint is violated.
2. **Given** a held-out tile, **When** predictions are scored, **Then** per-class recall and IoU are
   reported for every structural class, and aggregate accuracy is NOT used as the gate.
3. **Given** an arbitrary image with no client-derived ground truth, **When** prediction runs,
   **Then** it completes and emits an auditable record of what it produced.
4. **Given** the rarest structural class covers roughly 2% of locations, **When** results are
   gated, **Then** that class's own recall and IoU decide promotion.

---

### User Story 4 - Make evaluation trustworthy (Priority: P2)

A practitioner evaluates any model against a held-out set that does not touch the training set
spatially, and reads error **stratified by how much relief a region actually contains**, so that
performance is not hidden behind terrain a constant predictor already solves.

**Why this priority**: Without this, no result from User Story 3 or 5 can be believed. It is
separated from them because it delivers value on its own — it can re-score existing models
immediately.

**Independent Test**: Build the held-out set, verify no held-out tile touches a training tile, and
re-score an existing model to observe how much its reported quality changes.

**Acceptance Scenarios**:

1. **Given** the corpus, **When** the held-out set is constructed, **Then** zero held-out tiles
   share an edge or corner with a training tile, and the check is reported as a count.
2. **Given** any scored model, **When** results are reported, **Then** error appears separately for
   flat and relief-bearing regions, alongside the trivial-baseline error for each.
3. **Given** the held-out set changed, **When** results are compared to earlier runs, **Then** the
   report states that absolute comparison to prior runs is invalid and identifies which baseline
   must be re-run.
4. **Given** a model that does not beat the trivial baseline on relief-bearing regions, **When**
   promotion is considered, **Then** it is refused.

---

### User Story 5 - Feed predicted structure into geometry (Priority: P3)

Predicted layer structure is supplied to terrain height reconstruction, so that height prediction is
informed by what kind of surface each location is, and evaluated for whether it reduces height error
on relief-bearing regions.

**Why this priority**: The payoff, but it depends on every story above. Deferring it keeps the
earlier stories independently valuable even if this one does not pan out.

**Independent Test**: Train height reconstruction with and without predicted structure on the same
held-out set and compare relief-region error.

**Acceptance Scenarios**:

1. **Given** predicted structure, **When** height reconstruction consumes it, **Then** it consumes
   the *predicted* structure only, never ground-truth tables.
2. **Given** paired runs, **When** they are compared, **Then** relief-region error is reported for
   both against an identical held-out set.

---

### Edge Cases

- A tile whose base layer is the only layer present (more than 90% base-only in half of all sampled
  tiles) — structure prediction must not be scored as if overlays were expected.
- A tile at the corpus boundary with fewer neighbours available for held-out construction.
- An out-of-distribution image whose colours match no known surface family — the system must express
  low confidence rather than force a confident wrong assignment.
- A location where two layers have near-equal coverage, making the topmost ambiguous.
- Reused pieces appearing in both training and held-out sets, inflating apparent quality; measurable
  and must be reported even though it is smaller than the adjacency effect.
- A held-out set from which the rarest layer is entirely absent, making its class metric undefined
  rather than zero.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST measure how consistently each surface family occupies each layer slot
  across the corpus, and report both a per-family distribution and a summary score.
- **FR-002**: System MUST recommend a slot-keyed or family-keyed output vocabulary based on FR-001,
  and record that recommendation as a durable artifact consumed by later work.
- **FR-003**: System MUST fit a per-tile non-linear relationship from surface elevation and slope to
  layer coverage, and report explained variance per tile and per layer.
- **FR-004**: System MUST report whether the FR-003 distribution is bimodal and what share of tiles
  fall in any high-coupling population.
- **FR-005**: System MUST predict terrain layer structure from a minimap image alone, with no
  client-derived ground truth available at prediction time.
- **FR-006**: System MUST NOT consume ground-truth tables as a prediction input. Ground truth is
  admissible only as training-time supervision or loss shaping.
- **FR-007**: System MUST guarantee that predicted references are legal entries for the tile in
  question, and reject or repair any prediction that is not.
- **FR-008**: System MUST exclude the always-opaque base layer from any alpha stack, and MUST NOT
  combine layers by collapsing them into a single mask.
- **FR-009**: System MUST report per-class recall and IoU for every structural class, and MUST NOT
  use aggregate accuracy as a promotion gate.
- **FR-010**: System MUST construct a held-out set in which no held-out tile shares an edge or
  corner with a training tile, and MUST report the verified violation count as zero.
- **FR-011**: System MUST report error separately for flat and relief-bearing regions, alongside the
  trivial-baseline error for each region type.
- **FR-012**: System MUST refuse promotion to any model that does not beat the trivial baseline on
  relief-bearing regions.
- **FR-013**: System MUST measure and report how much reused-piece overlap exists between training
  and held-out sets.
- **FR-014**: Each model MUST be independently trained, checkpointed, and promoted. Shared weights
  and multi-task heads are prohibited.
- **FR-015**: Every training run MUST validate and print its plan without training by default, and
  MUST require explicit confirmation before consuming compute.
- **FR-016**: Every training run MUST record an identity binding tying its outputs to the exact
  inputs and configuration that produced them, including which held-out set was used.
- **FR-017**: System MUST state, wherever results are reported against a changed held-out set, that
  absolute comparison to prior results is invalid, and identify the baseline requiring re-run.
- **FR-018**: All training and heavy rebuilds MUST be executed by the user from documented commands
  with time and memory estimates. The assistant MUST NOT launch them.

### Key Entities

- **Surface Family**: A canonical category of terrain surface (base ground, path, water, structure,
  unknown) derived from a texture's identity. The visually-determined unit.
- **Layer Entry**: One ordered row of a chunk's layer table: a slot ordinal, a texture reference
  that is a foreign key into that tile's own texture list, and its coverage.
- **Texture Table**: The per-tile list of texture names that layer entries reference. Local to a
  tile; a reference is meaningless outside its own tile.
- **Coverage Map**: Per-location strength of a layer entry. Absent for the base layer, which is
  always fully opaque.
- **Dominant Structure**: Per-location resolution of which layer entry is visible, following paint
  order — the topmost entry whose coverage clears the threshold.
- **Reused Piece**: A region of coverage recurring elsewhere in the corpus under rotation or
  mirroring; evidence the corpus is assembled from a finite library.
- **Held-Out Set**: A spatially isolated group of tiles with no edge or corner contact with training
  tiles.
- **Relief Stratum**: A partition of locations by how much height variation they contain, used to
  report error where a trivial predictor cannot already succeed.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The family-versus-slot question is answered with a reported consistency score, and the
  resulting vocabulary decision is recorded before any model in this feature is trained.
- **SC-002**: The surface-shape-to-coverage relationship is reported per tile with an explicit
  bimodality finding, superseding the earlier inconclusive linear analysis.
- **SC-003**: Structure prediction from a minimap alone achieves per-class IoU above 0.60 for every
  class covering at least 5% of locations, and above 0.40 for the rarest class at roughly 2%.
- **SC-004**: 100% of predicted references are legal entries for their tile.
- **SC-005**: The held-out set has zero tiles sharing an edge or corner with a training tile,
  verified and reported as a count.
- **SC-006**: Error is reported separately for flat and relief-bearing regions in every evaluation,
  with the trivial-baseline figure shown alongside each.
- **SC-007**: At least one model beats the trivial baseline on relief-bearing regions — the first
  time any model in this project has done so on any stratum.
- **SC-008**: Reused-piece overlap between training and held-out sets is quantified and reported.
- **SC-009**: Structure prediction produces a legal, non-degenerate result on a hand-painted image
  with no client-derived ground truth, and the run is auditable afterwards.
- **SC-010**: Every reported result identifies which held-out set produced it, so no two
  incomparable numbers can be presented as a comparison.

## Assumptions

- The existing corpus contains the layer tables, coverage maps, and surface geometry needed; no new
  harvest pass is required. Measured facts in Motivation were taken from it directly.
- The existing derived surface-family labels remain valid and are reused rather than re-derived.
- Prediction at deployment has only a raw minimap image, possibly hand-painted, poorly upscaled, and
  unlike anything in the corpus. Degraded quality on such input is expected and acceptable.
- A minority of tiles carry internally inconsistent data. Perfect behaviour on every tile is not
  required; the user has explicitly accepted a small number of bad tiles.
- Rebuilding the held-out set invalidates absolute comparison with all prior results. A baseline
  re-run is an accepted cost of making evaluation trustworthy.
- Model scale stays comparable to existing small models in this project rather than growing;
  time-to-signal is preferred over exhaustively validating large architectures.
- Vision-language or agentic image interpretation is out of scope. It was tried and it failed.
- Height reconstruction consuming predicted structure (User Story 5) may show no benefit; the
  earlier stories are valuable regardless.
- Reported measurements are provisional when the measuring method could not have detected the effect
  being tested. A null result must be accompanied by evidence the test had power to find what it
  looked for.
