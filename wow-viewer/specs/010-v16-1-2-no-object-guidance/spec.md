# Feature Specification: V16.1.2 No-Object Guidance

**Feature Branch**: `010-v16-1-2-no-object-guidance`

**Created**: 2026-05-22

**Status**: Superseded by `011-v16-2-patched-signal-expansion`

**Input**: User description: "Build a new V16.1.2 normal-lane spec that uses renderer-derived `no_object_minimap` guidance as an additional supervisor guide, because it is a synthesized terrain-only minimap with terrain shadows and higher effective detail than the object-polluted baked minimap."

## Problem Statement

This narrower `V16.1.2` guidance-only spec has been superseded by the broader
`V16.2` patched-signal-expansion contract in
`wow-viewer/specs/011-v16-2-patched-signal-expansion/spec.md`, because the
active direction is no longer only auxiliary no-object guidance for the normal
lane. The active direction now also includes richer precise-mask signals and
patch-and-reindex upgrades to the existing compact corpus.

V16.1.1 improved the normal lane by curating harder terrain, rotating bounded
train pools, and weighting deformation-rich regions more aggressively.

That still leaves a core ambiguity in the input surface:

- the baked minimap includes visible world objects that are not terrain truth
- some terrain cues are partially obscured by buildings or doodads
- the normal trainer must infer terrain shape from an RGB image that mixes
  terrain appearance with non-terrain structure

The repo now has a renderer-truth validation path that can export a
`no_object_minimap` artifact alongside object-visibility evidence. That surface
is valuable because it removes world objects while preserving terrain shading
and terrain-adjacent visual cues that still matter for terrain reconstruction.

`V16.1.2` names the next bounded normal-lane upgrade that uses this
terrain-focused rendered surface as auxiliary supervision guidance while keeping
raw terrain tensors as the primary truth.

## Goal

Improve `minimap_rgb_256 -> normal_xyz` training by adding a terrain-focused
rendered guidance surface derived from `no_object_minimap`, so the model can
learn terrain structure from a less object-polluted image signal without
promoting rendered review artifacts into the authoritative target.

The immediate intent is not to replace the current minimap input or to restart
the model family. The intent is to help the normal lane disambiguate terrain
appearance from object clutter and use a better terrain-only visual guide where
that guide exists.

## User Scenarios & Testing

### User Story 1 - Terrain-Only Guidance Improves Normal Learning (Priority: P1)

A terrain researcher trains the V16.1 normal lane on the same six-build corpus,
but the trainer also consumes a terrain-focused `no_object_minimap` guidance
surface so the model can separate terrain shape from visible world objects.

**Why this priority**: The central complaint in the current normal lane is not
just insufficient epochs. It is that the input surface itself mixes terrain and
non-terrain information.

**Independent Test**: A bounded V16.1.2 smoke run completes on a curated pool
and produces evidence showing the guidance surface is loaded, aligned, and used
without changing the primary terrain target tensors.

**Acceptance Scenarios**:

1. **Given** a tile with renderer-derived `no_object_minimap` coverage,
   **When** the V16.1.2 trainer loads the sample, **Then** the tile includes the
   base minimap plus a terrain-focused guidance surface aligned to the same
   terrain footprint.
2. **Given** a tile with visible buildings or doodads in the baked minimap,
   **When** training uses the auxiliary guidance surface, **Then** the trainer
   can emphasize terrain-only interpretation without rewriting the ground-truth
   normal tensor.
3. **Given** a smoke comparison between V16.1.1 and V16.1.2 on the same bounded
   pool, **When** outputs are reviewed, **Then** the V16.1.2 run writes evidence
   that makes the guidance contribution inspectable instead of hidden.

---

### User Story 2 - Guidance Remains Auxiliary, Not False Truth (Priority: P1)

A terrain researcher wants the benefit of the `no_object_minimap` surface
without corrupting the training contract by treating rendered images as the raw
target truth.

**Why this priority**: The current V16/V16.1 contract is explicit that raw Zarr
tensors remain the supervised truth and rendered PNG-like surfaces stay
secondary. That rule should survive this upgrade.

**Independent Test**: The trainer can run with guidance enabled or disabled,
and both modes use the same raw terrain normal target tensor.

**Acceptance Scenarios**:

1. **Given** V16.1.2 guidance is enabled, **When** the normal loss is computed,
   **Then** normals remain supervised by raw terrain tensors rather than by the
   rendered `no_object_minimap` image.
2. **Given** a tile lacks `no_object_minimap` guidance, **When** training
   continues, **Then** the trainer falls back cleanly to the V16.1.1 behavior
   instead of dropping the tile outright.
3. **Given** the operator disables guidance, **When** a training run starts,
   **Then** the normal lane behaves like the V16.1.1 baseline with no required
   dataset migration beyond compatibility metadata.

---

### User Story 3 - Human Review Can See Whether Guidance Is Worth Keeping (Priority: P2)

A terrain researcher does not want another training change that only looks good
in scalar metrics. They want side-by-side evidence that shows what the
guidance surface looks like, where it was used, and whether it improved hard
terrain regions.

**Why this priority**: The recent object-mask work already showed that visual
inspection matters. A guidance upgrade without review artifacts is not credible.

**Independent Test**: A best-epoch validation review surface writes base
minimap, `no_object_minimap` guidance, train mask, and predicted-vs-target
normal panels for multiple samples in one artifact.

**Acceptance Scenarios**:

1. **Given** a new-best V16.1.2 checkpoint, **When** validation artifacts are
   written, **Then** the review panel includes both the baked minimap and the
   terrain-focused guidance image for the same tile.
2. **Given** a tile where object clutter previously hid terrain structure,
   **When** the researcher inspects the V16.1.2 review artifact, **Then** the
   guidance surface makes that terrain-visible difference obvious.
3. **Given** a later-build tile where terrain occlusion behavior matters,
   **When** validation artifacts are reviewed, **Then** the operator can tell
   whether the guidance surface preserved useful terrain shadows without
   inventing object edges as terrain truth.

### Edge Cases

- What happens when `no_object_minimap` exists for only a subset of tiles or
  builds? The trainer must preserve a fallback path so mixed-coverage corpora
  remain trainable.
- What happens when the guidance render is slightly misaligned or resolution-
  mismatched against the baked minimap? The training contract must define a
  single alignment/cropping policy and make misalignment reviewable.
- What happens when the terrain-only guidance removes useful context along
  structures such as bridges, overhangs, or city edges? The trainer must keep
  the baked minimap available so the guidance channel is supportive rather than
  blindly authoritative.
- What happens when the guidance image reinforces renderer artifacts or shadow
  quirks instead of geometry truth? The upgrade must expose review artifacts so
  the operator can reject the lane if the signal turns out to be misleading.

## Requirements

### Functional Requirements

- **FR-001**: `V16.1.2` MUST name the next bounded upgrade on top of the landed
  `V16.1.1` normal-training lane.
- **FR-002**: `V16.1.2` MUST preserve the existing one-target-per-trainer
  contract and MUST NOT reopen a multitask shared-weight terrain model.
- **FR-003**: The primary target surface for `V16.1.2` MUST remain
  `minimap_rgb_256 -> normal_xyz`.
- **FR-004**: `V16.1.2` MUST add a terrain-focused auxiliary guidance surface
  derived from renderer-produced `no_object_minimap` artifacts when those
  artifacts are available for a tile.
- **FR-005**: The auxiliary guidance surface MUST be treated as guidance only
  and MUST NOT replace raw terrain normal tensors as the authoritative
  supervision target.
- **FR-006**: The dataset/trainer seam MUST support mixed coverage so tiles that
  do not yet have `no_object_minimap` guidance can still participate in
  training through a defined fallback path.
- **FR-007**: `V16.1.2` MUST preserve the existing terrain-only masking and
  object/liquid downweighting from `V16.1.1`; the new guidance lane must extend
  that behavior rather than bypass it.
- **FR-008**: The training contract MUST define how `no_object_minimap`
  guidance aligns spatially with the baked minimap and terrain tensors,
  including any required resize or crop policy.
- **FR-009**: The normal trainer MUST be able to enable or disable the
  no-object guidance lane through an operator-visible run configuration.
- **FR-010**: Validation outputs for `V16.1.2` MUST expose the baked minimap,
  the `no_object_minimap` guidance image, and the effective terrain-focused
  weighting surface for the same review sample.
- **FR-011**: A bounded comparison workflow MUST exist for running a `V16.1.1`
  baseline and a `V16.1.2` guidance run against the same curated pool so the
  operator can inspect both metric deltas and image artifacts.
- **FR-012**: The first `V16.1.2` slice MUST focus on the normal lane only and
  MUST NOT widen into height, holes, liquids, or texcomp changes in the same
  step.
- **FR-013**: The dataset contract for `V16.1.2` SHOULD store the guidance
  surface in a form that preserves terrain shading detail sufficiently for
  training and inspection.
- **FR-014**: Operator-facing docs MUST publish a bounded command flow for:
  generating or locating the guidance surface, running a guidance-enabled smoke
  training pass, and reviewing the resulting validation artifacts.
- **FR-015**: Continuity docs MUST route future normal-lane follow-up work to
  `V16.1.2` when the active question is no-object terrain guidance rather than
  curation-only acceleration.

### Key Entities

- **No-Object Guidance Surface**: a terrain-focused rendered minimap image
  derived from the `no_object_minimap` artifact for a tile.
- **Guidance Coverage Flag**: per-tile metadata that tells the trainer whether
  a valid no-object guidance surface exists.
- **Alignment Policy**: the explicit rule that maps the guidance image onto the
  baked minimap and terrain tensor footprint.
- **Guidance Comparison Run**: a bounded pair of training runs that differ only
  by whether no-object guidance is enabled.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A `V16.1.2` Speckit spec exists and clearly defines no-object
  terrain guidance as the next bounded normal-lane upgrade.
- **SC-002**: A bounded smoke-ready training contract exists for using
  `no_object_minimap` as auxiliary guidance without replacing raw normal truth.
- **SC-003**: The first `V16.1.2` review artifacts make the baked minimap,
  guidance image, and prediction panels visible together for at least one
  best-epoch sample set.
- **SC-004**: The `V16.1.2` lane remains runnable on mixed-coverage corpora
  where some tiles have no guidance surface.
- **SC-005**: The operator can run a side-by-side `V16.1.1` versus `V16.1.2`
  comparison on the same bounded pool without inventing new dataset families or
  a new model family name.

## Assumptions

- The renderer-derived `no_object_minimap` surface contains terrain-useful
  structure that is harder to recover from the object-polluted baked minimap.
- Terrain shadows and terrain-only shading cues are useful for normal learning
  if they are treated as auxiliary guidance rather than as ground-truth output.
- Coverage will be partial at first, so mixed fallback behavior is necessary.
- Human image review is required before any claim that the guidance lane is an
  improvement worth keeping.
- `V16.1.2` is a bounded normal-lane upgrade, not a new terrain-model family.