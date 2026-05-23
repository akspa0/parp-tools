# Feature Specification: V16.1.1 Curated Normal Acceleration

**Feature Branch**: `007-v16-1-1-curated-normal-acceleration`

**Created**: 2026-05-21

**Status**: In Progress

**Input**: User direction to treat the next lane as `v16.1.1`, keep it V16/V16.1-derived, and focus on smarter curated normal training informed by recent surface-normal and hard-region mining research rather than restarting from a giant new model family.

## Problem Statement

V16.1 already split the V16 monolith into independent target-family trainers and
added a separate curation layer, terrain-aware normal masking, deformation-aware
loss steering, and bounded train-pool rotation.

That is not yet enough.

Current operator evidence still shows the normal lane spending too much effort on
easy broad terrain, low-information minimaps, and mixed-quality tiles that do
not teach the model much about deformation-heavy terrain structure. The next
useful gain is not a fresh architecture reset. The next useful gain is a
research-informed V16-derived refinement:

- smarter curation
- smarter epoch sampling
- smarter normal-loss weighting
- tighter failure isolation and evidence

V16.1.1 names that refinement lane explicitly.

## Goal

Keep the V16.1 codebase and one-target-per-trainer contract, but upgrade the
normal-first training lane so it learns more from deformation-rich, terrain-only
regions and wastes fewer updates on blank, flat, or semantically polluted tiles.

The V16.1.1 lane specifically targets:

```text
minimap_rgb_256 -> normal_xyz
```

with stronger curation and training intelligence built around the existing V16
Zarr truth surfaces.

## Scope

This spec is for the next normal-first V16-derived slice, not for the entire
future model family.

In scope:

- upgraded normal-oriented curation profile
- difficulty-bucketed curated manifests
- hard-region / hard-patch biased sampling
- optional uncertainty-guided normal loss
- stronger local geometry-consistency supervision
- operator-facing evidence and commands for small scouting runs

Out of scope:

- replacing V16.1 with a new giant foundation model
- re-merging targets into a multitask trunk
- switching away from the V16 Zarr dataset contract
- treating validation PNGs as the actual training truth
- rewriting the height, liquid, holes, or texcomp trainers in the same slice

## User Scenarios & Testing

### User Story 1 — Curated Difficulty Buckets Replace Flat Random Draw (Priority: P1)

A terrain researcher builds a normal-training manifest that separates easy,
medium, hard, and pathological tiles so short runs can over-index on useful
terrain structure instead of replaying mostly-open terrain.

**Why this priority**: Faster learning per epoch depends more on useful example
selection than on another round of blind training over the same weak pool.

**Independent Test**: Build a new normal-oriented manifest that records
difficulty buckets and per-tile scoring evidence. Verify the output contains
bucket counts and reviewable reasons.

**Acceptance Scenarios**:

1. **Given** a V16 Zarr store, **When** V16.1.1 curation runs, **Then** it can
   reject blank genesis tiles and low-signal mismatches before bucket
   assignment.
2. **Given** a tile whose WMO-driven loss gate wipes out most of the
  trainable terrain area, **When** V16.1.1 curation runs, **Then** that tile
  is rejected before it can dilute the normal-training pool.
3. **Given** a kept tile, **When** the manifest is written, **Then** it carries
   enough metadata to explain why it was scored easy, medium, hard, or
   pathological.
4. **Given** a short scouting run, **When** the trainer samples tiles,
   **Then** it can bias toward hard tiles without deleting easy examples from
   the available pool.

---

### User Story 2 — Normal Training Focuses On Hard Terrain Regions (Priority: P1)

A terrain researcher runs the normal trainer and knows the loss is concentrating
on deformations, coastlines, painted transitions, and terrain-only structure
instead of broad easy flats or object-polluted regions.

**Why this priority**: The core complaint in the current lane is not just raw
accuracy. It is wasted optimization budget.

**Independent Test**: A normal-only smoke run logs bucket usage and hard-region
weight statistics, and validation output shows the effective train mask /
detail-weight focus.

**Acceptance Scenarios**:

1. **Given** a deformation-rich tile, **When** normal loss is computed,
   **Then** terrain transitions carry more weight than bland flats.
2. **Given** object-heavy or liquid-heavy regions, **When** normal loss is
   computed, **Then** those regions remain downweighted by terrain-aware mask
   guidance.
3. **Given** a short run on a mixed 400-tile scouting pool, **When** training
   finishes, **Then** the run writes evidence showing which difficulty buckets
   were sampled and how strongly hard regions influenced optimization.

---

### User Story 3 — Uncertainty Helps The Trainer Spend Effort Wisely (Priority: P2)

A terrain researcher can optionally enable an uncertainty-guided normal lane so
ambiguous regions are handled explicitly rather than poisoning the whole loss
surface.

**Why this priority**: Some minimap-to-normal regions are genuinely ambiguous.
The trainer should not treat every pixel as equally trustworthy.

**Independent Test**: An uncertainty-enabled smoke run completes and writes
uncertainty metrics plus a review panel or tensor artifact for predicted
uncertainty.

**Acceptance Scenarios**:

1. **Given** the uncertainty option is enabled, **When** the model trains,
   **Then** the loss uses predicted uncertainty to attenuate or redistribute
   pressure on ambiguous pixels.
2. **Given** a validation tile, **When** outputs are written, **Then** the run
   includes uncertainty evidence separate from the normal RGB panel.
3. **Given** uncertainty hurts the lane, **When** it is disabled, **Then** the
   base V16.1.1 normal lane still works without changing the dataset contract.

## Requirements

### Functional Requirements

- **FR-001**: `V16.1.1` MUST name the next V16-derived normal-training upgrade
  lane and MUST be treated as an evolution of `V16.1`, not a separate
  architecture family.
- **FR-002**: `V16.1.1` MUST preserve the existing V16.1 one-target-per-trainer
  contract and MUST NOT reintroduce a multitask shared-weight normal/height/etc.
  trainer.
- **FR-003**: The primary implementation target for V16.1.1 MUST be the
  `train_v16_1_normal.py` lane plus its dataset/curation helpers.
- **FR-004**: The curation layer MUST support a new normal-oriented profile that
  scores tiles for deformation richness, target coverage, and minimap-vs-target
  usefulness rather than only rejecting obvious garbage.
- **FR-004A**: The curation layer MUST reject WMO-dominated tiles when the
  active object-loss gate leaves too little trainable terrain coverage for the
  normal lane to learn from usefully.
- **FR-005**: The curation manifest MUST support at least four difficulty
  buckets: `easy`, `medium`, `hard`, and `pathological`.
- **FR-006**: The normal trainer MUST be able to consume difficulty-bucketed
  manifests and bias epoch sampling toward harder tiles without excluding easier
  tiles entirely.
- **FR-007**: V16.1.1 MUST continue using terrain-only masking and object/liquid
  downweighting based on the raw supervision channels already present in V16.1.
- **FR-008**: The normal trainer MUST support hard-region weighting within a
  tile, not only hard-tile selection across tiles.
- **FR-009**: V16.1.1 MUST preserve the existing bounded-pool and per-epoch
  rotation seam so scouting runs can stay small and fast.
- **FR-010**: The initial operator-target scouting configuration MUST support a
  mixed-complexity curated pool around `400` train tiles and a smaller bounded
  validation pool.
- **FR-011**: The normal trainer SHOULD support an optional uncertainty-aware
  prediction/loss seam that can be enabled or disabled without changing the
  dataset format.
- **FR-012**: The normal trainer SHOULD support stronger local geometry
  consistency supervision so neighboring normal relationships matter, not only
  per-pixel vector agreement.
- **FR-013**: Validation artifacts MUST remain review aids only. Ground-truth
  training targets MUST continue to come from raw Zarr tensors, not exported
  image files.
- **FR-014**: The V16.1.1 operator docs MUST publish hand-runnable commands for:
  curated manifest build, small scouting run, and longer resumed run.
- **FR-015**: Continuity docs MUST explicitly route fresh chats to V16.1.1 as
  the next normal-lane implementation slice.
- **FR-016**: Validation preview artifacts MUST be best-gated, render multiple
  samples per output, and carry visible panel labels plus per-sample tile
  metadata so operator review is not worse than dataset validation.
- **FR-017**: The shared V16.1 normal trainer MUST support startup VRAM-aware
  batch-size autotuning from a candidate ladder using `--target-vram-gb`,
  record the chosen batch size as run evidence, and preserve a coherent
  steps-per-epoch budget when requested.

### Non-Goals

- Adopting a camera-centric monocular geometry foundation model as the new base
- Replacing the minimap input with a non-minimap primary input
- Building the height lane and normal lane jointly in one trainer
- Solving object segmentation in this slice
- Rewriting stitched inference in this slice

### Key Entities

- **V16.1.1 Normal Curation Profile**: the upgraded normal-oriented manifest
  scoring/filtering rule set
- **Difficulty Bucket**: one of `easy`, `medium`, `hard`, `pathological`
- **Hard-Region Weight Map**: per-pixel terrain emphasis inside a kept tile
- **Uncertainty Head**: optional normal-lane output used to weight ambiguous
  supervision
- **Scouting Pool Contract**: a bounded curated train/val configuration for fast
  experiments

## Success Criteria

### Measurable Outcomes

- **SC-001**: A new V16.1.1 spec pack exists and routes future work into the
  curated normal-acceleration lane.
- **SC-002**: A normal-oriented curation profile writes difficulty buckets and
  per-tile scoring evidence.
- **SC-003**: The normal trainer can read difficulty metadata and record bucket
  usage during a smoke run.
- **SC-004**: A bounded scouting run over a mixed `400`-tile train pool
  completes and writes bucket / hard-region evidence.
- **SC-005**: The operator docs publish the new curation and training commands.
- **SC-006**: If uncertainty is enabled, a smoke run writes uncertainty metrics
  and review artifacts without changing raw dataset truth handling.

## Assumptions

- The main short-term gain is better sample efficiency, not a giant model swap.
- The existing V16.1 raw supervision channels are strong enough to drive smarter
  normal curation and loss steering.
- Top-down minimap terrain reconstruction benefits from geometry-consistency
  ideas from modern normal-estimation work, but not necessarily from
  camera-centric foundation-model assumptions.
- Small bounded scouting runs are the right proving ground before another long
  full-corpus training attempt.

## Initial Implementation Direction

1. Create the V16.1.1 spec/plan/tasks pack and route fresh chats to it.
2. Upgrade curation from reject-only to score-and-bucket.
3. Add bucket-aware epoch sampling in the normal trainer.
4. Add stronger hard-region weighting inside the normal loss.
5. Add optional uncertainty-aware supervision if the simpler weighting lane
   proves insufficient.
6. Record the new operator commands only after focused proof.
