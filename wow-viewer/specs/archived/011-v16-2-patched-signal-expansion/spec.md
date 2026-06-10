# Feature Specification: V16.2 Patched Signal Expansion

**Feature Branch**: `011-v16-2-patched-signal-expansion`

**Created**: 2026-05-22

**Status**: Draft

**Input**: User description: "We need to include the new precise masks in this model spec as well, and patch the existing dataset with that data in the Zarr stores. Call it v16.2, since we've got more signals and better data now than we did in v16. Patch the existing dataset instead of rebuilding it, because it is a large compressed corpus and we only need to add and reindex the new signals so training stays fast."

## Problem Statement

The current V16 corpus is already a major success:

- it consolidates a large multi-build terrain corpus into compact per-build
  Zarr stores
- it compresses a large raw signal volume into a much smaller training surface
- it keeps training fast by avoiding millions of loose per-tile files

The problem is not that the existing corpus is wrong as a whole. The problem is
that the corpus now has newer, better signals than the original V16 contract
captured:

- more precise object-related masks
- renderer-truth visibility artifacts that separate terrain from objects better
- terrain-only rendered guidance surfaces such as `no_object_minimap`

Rebuilding the full corpus from raw client data would waste hours of work and
recompute already-correct signals just to add a smaller set of new ones.

At the same time, mutating the existing finalized V16 stores too early would
blur the proof boundary while the new renderer-truth surfaces are only capture-
validated on a subset of builds today.

`V16.2` names the next dataset-and-model contract that keeps the existing V16
stores as the foundation, stages the new signals into additive sidecar stores
first, reindexes metadata around that richer signal surface, and exposes the
upgraded signals to the next training lane without disturbing the current V16
base corpus.

## Goal

Promote the current terrain dataset contract from `V16` to `V16.2` by adding
new precise-mask and terrain-guidance signals through sidecar-first patch-and-
reindex workflows instead of full corpus regeneration or immediate mutation of
the existing per-build V16 Zarr stores.

The intent is to preserve the current storage efficiency and training speed
while upgrading the dataset quality and the model-facing signal surface.

## User Scenarios & Testing

### User Story 1 - Existing V16 Stores Gain Sidecar Signals Without Full Rebuild (Priority: P1)

A terrain researcher already has the full six-build corpus on disk and does not
want to spend hours regenerating terrain, alpha, liquid, and placement arrays
that are already correct. They stage only the newly available signals into a
separate sidecar store family and reindex the metadata so training can use the
upgraded corpus immediately while the original V16 stores stay intact.

**Why this priority**: The biggest practical value is avoiding unnecessary
corpus rebuilds while still capturing the better data.

**Independent Test**: A bounded patch run upgrades one build with a new
sidecar signal store and writes updated metadata without touching unrelated V16
base arrays.

**Acceptance Scenarios**:

1. **Given** an existing V16 build store, **When** the V16.2 sidecar workflow
  runs, **Then** it adds the new signals in a separate store family and updates
  metadata without requiring a full rebuild of existing arrays.
2. **Given** a build store that already contains valid base terrain signals,
  **When** the sidecar workflow completes, **Then** those base signals remain
  intact and only the new signal surfaces and their metadata are added.
3. **Given** the upgraded build pair, **When** training-ready validation runs,
  **Then** the corpus reports the upgraded signal coverage through the updated
  index and validation surfaces.

---

### User Story 2 - V16.2 Training Sees Precise Masks And Terrain-Only Guidance (Priority: P1)

A terrain researcher trains against the upgraded corpus and wants the model spec
to acknowledge that the dataset now includes more precise object-related masks
and terrain-only rendered guidance surfaces than original V16 did.

**Why this priority**: The dataset upgrade only matters if the model contract
can actually use the richer signals.

**Independent Test**: A bounded V16.2 training slice loads the new mask and
guidance signals from the patched stores and writes review artifacts proving the
signals are aligned and available.

**Acceptance Scenarios**:

1. **Given** a patched V16.2 tile, **When** a trainer loads it, **Then** the
   sample exposes the upgraded precise-mask signals and terrain-only guidance
   surfaces defined by the new contract.
2. **Given** a tile with renderer-derived guidance coverage, **When** training
   uses that sample, **Then** the trainer can consume the new terrain-only
   guidance without replacing the raw terrain target tensors.
3. **Given** a tile without the new optional guidance coverage, **When** the
   trainer loads it, **Then** the sample still remains usable through the
   defined fallback behavior.

---

### User Story 3 - Metadata Reindexing Keeps Training Fast And Predictable (Priority: P2)

A terrain researcher wants the upgraded dataset to preserve the current fast
training workflow. They do not want a pile of sidecar files or a second slower
dataset family. They want the same compact stores, but with richer metadata and
signals.

**Why this priority**: Storage efficiency and fast random access are already one
of the strongest properties of the current corpus. The upgrade should preserve
that win.

**Independent Test**: A patched store remains directly consumable by the live
dataset loader after metadata reindexing, with no extra preprocessing step.

**Acceptance Scenarios**:

1. **Given** a patched V16.2 build store, **When** the dataset loader opens it,
   **Then** the loader can resolve the new signal presence metadata from the
   reindexed store directly.
2. **Given** the upgraded six-build corpus, **When** training starts,
   **Then** the operator uses the same compact per-build store workflow rather
   than a new loose-file export path.
3. **Given** a patch-only upgrade run, **When** it completes, **Then** the
  result is one upgraded dataset contract with explicit base-plus-sidecar
  semantics rather than a destructive rewrite of the finalized V16 corpus.

### Edge Cases

- What happens when a store has partial new-signal coverage because a patch run
  was interrupted? The contract must define resumable or restart-safe behavior
  so metadata does not falsely report full coverage.
- What happens when different builds support different renderer-derived
  artifacts? The contract must support per-build and per-tile optional presence
  instead of assuming universal coverage.
- What happens when a new precise mask supersedes an older coarse mask? The
  contract must define whether the older surface is preserved, replaced, or kept
  only for compatibility.
- What happens when patching the new signals would grow the store beyond the
  acceptable storage budget? The upgrade must preserve the compact-corpus goal
  instead of turning the patch path into silent storage bloat.
- What happens when only some client builds have credible renderer-truth proof?
  The contract must separate capture-validated builds from unvalidated builds so
  later merge or promotion steps do not overstate proof.

## Requirements

### Functional Requirements

- **FR-001**: `V16.2` MUST name the upgraded terrain dataset and training
  contract that supersedes the original `V16` baseline when the active concern
  is richer patched signal coverage.
- **FR-002**: `V16.2` MUST preserve the existing per-build compact V16 store
  model as the canonical base corpus surface.
- **FR-003**: `V16.2` MUST support additive sidecar signal stores as the first
  upgrade surface instead of requiring a full corpus rebuild or immediate in-
  place mutation of already-correct base signals.
- **FR-004**: `V16.2` MUST include the upgraded precise-mask surfaces in the
  dataset contract, including improved object-related masks beyond the original
  V16 signal set.
- **FR-005**: `V16.2` MUST include renderer-derived terrain-only guidance
  surfaces in the contract where those surfaces are available, including
  `no_object_minimap`.
- **FR-006**: `V16.2` MUST define how renderer-truth visibility or precise-mask
  surfaces coexist with the older object-mask family so trainers know which
  surfaces are coarse compatibility signals versus richer upgraded signals.
- **FR-007**: The patch workflow MUST update the metadata index so the presence
  and provenance of the new signals are visible without a separate manual audit
  step, including whether a tile's upgraded signals live in the base store, the
  sidecar, or both.
- **FR-008**: The upgraded contract MUST support mixed per-tile coverage for the
  new signals so partially upgraded or build-specific surfaces do not make the
  corpus unusable.
- **FR-009**: The upgraded model-facing contract MUST let training lanes consume
  the new precise-mask and terrain-guidance signals without redefining raw
  terrain targets as rendered image truth.
- **FR-010**: `V16.2` MUST preserve the existing fast training workflow by
  keeping the compact base-plus-sidecar stores directly consumable by the
  dataset loader.
- **FR-011**: The first `V16.2` upgrade path MUST emphasize sidecar-first
  patch-and-reindex semantics rather than duplicating the entire corpus into a
  second full-sized dataset or mutating the finalized V16 base corpus by
  default.
- **FR-012**: The validation contract MUST expose review artifacts and coverage
  summaries that show whether the new signals were patched correctly and are
  spatially aligned with the existing terrain data.
- **FR-013**: Operator-facing docs MUST publish a bounded workflow for:
  locating or generating the new signal artifacts, patching them into existing
  stores, reindexing metadata, and validating the upgraded corpus.
- **FR-014**: Continuity docs MUST route future dataset and trainer follow-up
  work to `V16.2` when the question is about richer patched signal coverage and
  upgraded corpus semantics.
- **FR-015**: `V16.2` MUST track real renderer-truth validation coverage by
  build so the upgrade lane can require a six-build coverage matrix before any
  later merge of sidecar signals back into the canonical base corpus.
- **FR-016**: The operator workflow SHOULD expose a manifest-driven build and
  signal selection surface so users can choose builds, signal families, and
  model-facing outputs without hard-coding one fixed corpus recipe.

### Key Entities

- **V16.2 Base Store**: the existing per-build compact terrain store that
  preserves the original base V16 signals.
- **V16.2 Sidecar Store**: an additive per-build compact store that carries the
  upgraded signal surfaces without disturbing the finalized base V16 store.
- **Precise Mask Surface**: an upgraded object-related mask signal that carries
  more exact terrain/object separation than the original coarse mask family.
- **Terrain-Only Guidance Surface**: a rendered image signal such as
  `no_object_minimap` that exposes terrain appearance with world objects
  removed.
- **Patch-And-Reindex Workflow**: the bounded upgrade path that adds new
  signals to a sidecar store and refreshes the combined metadata contract.
- **Signal Coverage Metadata**: per-store or per-tile metadata that tells the
  loader and validator which upgraded signals are present.
- **Validation Coverage Matrix**: the per-build proof surface that records
  which client builds have real renderer-truth capture validation for the new
  signals.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A `V16.2` Speckit spec exists and clearly defines the patched
  signal-expansion contract.
- **SC-002**: The `V16.2` contract explicitly supports adding new precise-mask
  and terrain-guidance signals through sidecar-first upgrades without requiring
  a full rebuild of the already-correct corpus.
- **SC-003**: The upgraded metadata contract makes new-signal coverage visible
  after patching and reindexing.
- **SC-004**: The `V16.2` training contract remains based on compact per-build
  stores rather than a second loose-file export path.
- **SC-005**: The `V16.2` model spec acknowledges that the upgraded corpus now
  has more and better signals than original `V16`, including richer precise
  masks and terrain-only guidance data.
- **SC-006**: The `V16.2` handoff defines which builds have real renderer-truth
  validation already and which builds still need proof before any base-store
  merge is allowed.

## Assumptions

- The existing V16 stores already contain the base terrain signals worth
  preserving and should not be rebuilt without a concrete defect.
- The newer precise-mask and renderer-derived guidance surfaces are valuable
  enough to justify a contract upgrade instead of remaining side artifacts.
- Storage efficiency is a primary asset of the current corpus and must survive
  the upgrade.
- Sidecar-first patch-and-reindex is the right operator model because the
  corpus is large and expensive to regenerate end-to-end, and the finalized V16
  base stores should remain intact until the richer signals are validated across
  the intended client matrix.
- `V16.2` is a dataset-and-training contract upgrade, not a justification for
  discarding the compact-store architecture that already works.