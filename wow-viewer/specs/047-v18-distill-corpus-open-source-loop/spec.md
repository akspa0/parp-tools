# Feature Specification: V18 Focused Two-Build Minimap-to-Terrain Loop

**Feature Branch**: `047-v18-distill-corpus-open-source-loop`

**Created**: 2026-06-04

**Status**: Draft — Scope Reset 2026-06-04

## Scope Reset (2026-06-04)

The active lane was cut back to the simplest useful training surface:

1. keep the corpus focused on `0_5_3_3368` and `3_3_5_12340` only,
2. train from a single `minimap_rgb` input image only,
3. predict `height_257` and `normal_xyz` with plain supervision,
4. stop depending on renderer-truth capture, object-mask gating, roof-mask
   gating, or extra loss contributions for active signoff.

Older ideas around synthesized inputs, teacher distillation, and an
open-source student model are deferred. They are not part of the active
implementation contract for this spec.

## Problem Statement

The V18 dataset namespace already exists and the V16.1 model family already
works. The real issue is that the current lane drifted into too many weak
signals and too much operational noise.

The repo needs one boring, trustworthy loop:

- `minimap_rgb -> height_257`
- `minimap_rgb -> normal_xyz`

That loop should run on the focused two-build corpus and produce outputs we can
actually use, without waiting on capture correctness or inventing more loss
surfaces than the task needs.

## Out of Scope

- Renderer-truth capture as training truth.
- Object-mask, roof-mask, or liquid-derived loss weighting for the active lane.
- New model architecture families.
- Expanding back out to the other four builds.
- Synthesized-data generation, teacher distillation, or open-source release.

## Goals

- One focused V18 corpus containing only `0_5_3_3368` and `3_3_5_12340`.
- One bounded height-training proof from `minimap_rgb` only.
- One bounded normal-training proof from `minimap_rgb` only.
- One reproducible command surface for rerunning both models on the focused
  stores.

## User Scenarios & Testing

### User Story 1 — Focus the corpus to two builds (Priority: P1)

A researcher can work only with `0_5_3_3368` and `3_3_5_12340` and does not
need the other four builds for this proof lane.

**Why this priority**: If the focused stores are not honest and reproducible,
the training loop is noise no matter how simple the model is.

**Independent Test**: Run the focused corpus path on the two staged builds and
validate both resulting stores.

**Acceptance Scenarios**:

1. **Given** staged clients for `0_5_3_3368` and `3_3_5_12340`, **When** the
   focused build runs, **Then** it produces one V18 Zarr store per build under
   `wow-viewer/output/datasets/v18/`.
2. **Given** those two stores, **When** validation runs, **Then** required
   training signals are present and coherent without requiring any of the other
   four builds.

---

### User Story 2 — Train height from one minimap input (Priority: P1)

A researcher can train a height model that consumes only `minimap_rgb` and
predicts `height_257`.

**Why this priority**: Height is the simplest core terrain target and gives the
basic proof that the focused corpus still trains cleanly.

**Independent Test**: Run a bounded height-training pass with plain L1
supervision and verify the trainer writes checkpoints and validation previews.

**Acceptance Scenarios**:

1. **Given** the focused two-build corpus, **When** the height trainer runs,
   **Then** its input contract is `minimap_rgb -> height_257`.
2. **Given** the active height lane, **When** loss is computed, **Then** it
   uses plain height supervision with no renderer-truth or object/roof/liquid
   weighting.
3. **Given** a bounded run, **When** it completes, **Then** the checkpoint and
   evidence land under `wow-viewer/models/v18/height/runs/<run-name>/`.

---

### User Story 3 — Train normals from one minimap input (Priority: P1)

A researcher can train a normal model that consumes only `minimap_rgb` and
predicts `normal_xyz`.

**Why this priority**: Normals are the other core terrain output we need for
useful downstream terrain reconstruction without reopening the old complicated
loss stack.

**Independent Test**: Run a bounded normal-training pass with masked cosine
loss and verify the trainer writes checkpoints and validation previews.

**Acceptance Scenarios**:

1. **Given** the focused two-build corpus, **When** the normal trainer runs,
   **Then** its input contract is `minimap_rgb -> normal_xyz`.
2. **Given** the active normal lane, **When** loss is computed, **Then** it
   uses plain masked cosine supervision from `normal_mask` with no extra
   object/roof/liquid weighting and no height-refiner dependency.
3. **Given** a bounded run, **When** it completes, **Then** the checkpoint and
   evidence land under `wow-viewer/models/v18/normal/runs/<run-name>/`.

### Edge Cases

- A tile with missing normals must still be indexed honestly and excluded only
  by the normal loss mask, not silently dropped.
- A tile with poor roof/object signals must not poison the active lane because
  those signals are not part of the basic training contract.
- A rerun with the same seed and same focused stores must reproduce the same
  train/val split and the same command/config evidence.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The focused-corpus workflow MUST treat `0_5_3_3368` and
  `3_3_5_12340` as the canonical build list for this spec.
- **FR-002**: The active height trainer MUST consume only `minimap_rgb` as
  model input.
- **FR-003**: The active height trainer MUST use plain supervision against
  `height_257` with no renderer/object/roof/liquid loss weighting.
- **FR-004**: The active normal trainer MUST consume only `minimap_rgb` as
  model input.
- **FR-005**: The active normal trainer MUST use masked cosine supervision
  against `normal_xyz` with `normal_mask` as the only active validity mask.
- **FR-006**: The active training lane MUST NOT require
  `object_visibility_mask`, `no_object_minimap`, or any renderer-capture
  artifact for signoff.
- **FR-007**: The focused build and both training runs MUST be reproducible
  from recorded commands, config, and seed.
- **FR-008**: All active scripts and outputs in this lane MUST remain under
  `wow-viewer/`.

### Key Entities

- **Focused Two-Build Corpus**: the V18 stores for `0_5_3_3368` and
  `3_3_5_12340`.
- **Height Model**: the existing V16.1 / V18 height trainer used with the
  input contract `minimap_rgb -> height_257`.
- **Normal Model**: the existing V16.1 / V18 normal trainer used with the
  input contract `minimap_rgb -> normal_xyz`.
- **Plain Height Supervision**: direct L1 loss against `height_257`.
- **Plain Normal Supervision**: masked cosine loss against `normal_xyz` using
  `normal_mask`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: One focused V18 build produces complete, valid stores for
  `0_5_3_3368` and `3_3_5_12340`.
- **SC-002**: One bounded height run completes on the focused corpus and writes
  checkpoint plus validation evidence under `models/v18/height/runs/`.
- **SC-003**: One bounded normal run completes on the focused corpus and writes
  checkpoint plus validation evidence under `models/v18/normal/runs/`.
- **SC-004**: Both active runs log the single-image input contract explicitly:
  `minimap_rgb -> height_257` and `minimap_rgb -> normal_xyz`.
- **SC-005**: Active signoff no longer depends on renderer truth, object-mask
  gating, roof-mask gating, or extra loss terms.

## Assumptions

- The existing V16.1 / V18 model family is good enough for the basic proof.
- The focused stores already contain the minimap, height, and normal arrays
  needed for this lane.
- The remaining roof/object signal mismatches may still exist in the stores,
  but they are not blockers for this minimap-only training contract.

## Deferred Historical Scope

The earlier spec draft also described:

- synthesized-input generation,
- teacher-on-synthetic distillation,
- an open-source student model and release artifact.

Those ideas are deferred. If they return, they should reopen this spec or move
into a follow-up spec instead of silently re-entering the active lane.

## Relationship to Existing Specs

- **Builds on**: `001-v18-dataset-spec` for the V18 corpus contract.
- **Reuses**: the existing V16.1 / V18 trainer surface.
- **Supersedes for active execution**: any older spec-047 wording that made
  renderer truth, object-mask gating, or distillation part of the required
  proof path.
