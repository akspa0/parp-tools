# V16.1.1 Curated Normal Acceleration — 2026-05-21

## Purpose

`V16.1.1` is the next bounded lane on top of the landed `V16.1` trainer split.
It is not a new model family and not a foundation-model pivot. It is a
research-informed upgrade to the normal-first terrain lane.

The working premise is simple:

- V16.1 already split the monolith
- V16.1 already added curation, terrain-only masking, and bounded pool rotation
- the next gain should come from better training efficiency and better use of
  the existing V16 Zarr truth surfaces

## Why This Exists

Recent operator evidence shows the normal lane still wastes too much effort on:

- easy broad flats
- low-information minimaps
- target-aligned but low-value tiles
- ambiguous or semantically polluted regions

That means the next lever is not "train longer." The next lever is:

1. curate more intelligently
2. sample more intelligently
3. weight loss more intelligently

## Research Direction We Are Borrowing From

The current planning direction is informed by recent normal-estimation and
hard-region mining work, without committing the repo to a giant replacement
model:

- geometry-aware surface normal estimation
- uncertainty-guided supervision for ambiguous pixels
- hard-patch / hard-region mining so training spends more effort on useful
  structure

This should be applied as a bounded V16-derived upgrade, not as a broad
rearchitecture.

## V16.1.1 Execution Surface

Primary target:

```text
minimap_rgb_256 -> normal_xyz
```

Upgrade points:

- new normal-oriented curation profile with usefulness scoring
- difficulty buckets:
  - `easy`
  - `medium`
  - `hard`
  - `pathological`
- bucket-aware epoch sampling for small scouting pools
- stronger hard-region weighting inside each tile
- optional uncertainty-guided normal loss if needed

## First Proof Shape

Do not start with another long full-corpus run.

Start with:

- curated mixed-complexity train pool around `400` tiles
- bounded validation pool
- short bucket-aware normal scouting runs
- evidence that shows:
  - bucket counts
  - bucket usage by epoch
  - hard-region weighting stats
  - whether uncertainty improves or just adds noise

## Guardrails

- keep raw Zarr tensors as the supervised truth
- validation PNGs remain review-only
- keep object/liquid/invalid-terrain masking authoritative
- do not widen into a multitask trainer again
- do not restart the whole family under a new giant architecture name

## Fresh-Chat Routing

The fresh-chat implementation pack for this lane is:

- `wow-viewer/specs/007-v16-1-1-curated-normal-acceleration/spec.md`
- `wow-viewer/specs/007-v16-1-1-curated-normal-acceleration/plan.md`
- `wow-viewer/specs/007-v16-1-1-curated-normal-acceleration/tasks.md`
