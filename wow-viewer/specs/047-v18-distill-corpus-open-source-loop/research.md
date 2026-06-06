# Research: V18 Focused Two-Build Terrain Reconstruction System

## Decision 1: Keep only `0_5_3_3368` and `3_3_5_12340`

**Decision**: The active V18 lane uses only the pre-alpha anchor
`0_5_3_3368` and the 3.x-era anchor `3_3_5_12340`.

**Rationale**: These are the only eras the user cares about for the current
terrain reconstruction goal. Wider corpus scope increased duplication and
operational cost without solving the core generalization/correction problems.

**Alternatives considered**:
- Keep all six historical builds: rejected because it adds width, not clarity.
- Collapse to a single build: rejected because cross-era coverage is required.

## Decision 2: The V18 dataset is complete enough to train against now

**Decision**: Treat the focused V18 stores as the active training source.

**Rationale**: The user explicitly wants to stop waiting on more speculative
dataset work and start using the existing signals. The remaining blocker is not
dataset existence; it is operator workflow and focused curation/training.

**Alternatives considered**:
- Rebuild the whole dataset pipeline again first: rejected as more delay.
- Revert to V16-only stores: rejected because the V18 stores are the owner
  namespace now.

## Decision 3: Use curation-first filtering, not a large auxiliary loss stack

**Decision**: Keep liquids and terrain-validity signals in the curation and
validity story, but do not reopen the older object/roof/renderer loss bundle in
the active training lane.

**Rationale**: The biggest observed failures were more consistent with bad or
misleading tiles than with missing omission masks. Liquid-heavy hidden-terrain
tiles, erased-normal leftovers, and mismatched signals are curation problems
first.

**Alternatives considered**:
- Reintroduce broad object/liquid/roof loss weighting: rejected because it
  increases complexity before the curation-first hypothesis is exhausted.
- Ignore liquids completely: rejected because liquid-dominated hidden terrain is
  still a meaningful invalidity signal.

## Decision 4: Keep two independent terrain models

**Decision**: V18 uses two separate terrain models:

- `minimap_rgb -> height`
- `minimap_rgb -> normals`

**Rationale**: This aligns with the repo constitution, keeps checkpoints
debuggable, and matches the intended downstream use where both outputs are
consumed together during terrain reconstruction.

**Alternatives considered**:
- One multitask height+normal model: rejected because it violates the current
  modular-model rule and makes failure analysis harder.
- Height-only plus derived normals: rejected because explicit learned normals
  still carry useful local shape cues.

## Decision 5: Add focused V18 wrappers instead of forcing operators through older V16 naming

**Decision**: Create focused V18 operator wrappers for curation and training.

**Rationale**: The underlying curation/training code is usable, but the current
operator experience still relies on V16 naming and broad build defaults. That
creates avoidable human error and slows repeated iteration.

**Alternatives considered**:
- Keep using `build_v16_curation_manifest.py` and generic `train_v18.py`
  directly: rejected because the workflow remains harder to run correctly.
- Refactor the whole training stack before exposing focused commands: rejected
  because it delays useful progress.

## Decision 6: Quilt-level stitching is part of the final design, but not this implementation slice

**Decision**: The final V18 design explicitly includes a post-model terrain
quilt stage aimed at ADT reconstruction, but this slice implements the focused
curation/training surfaces first.

**Rationale**: The user wants stitched map reconstruction, not isolated tile
previews. That must be named now so the model outputs are aimed correctly.
However, implementing the stitch solver and ADT writeback in the same slice
would blur responsibilities and slow the immediate start of training.

**Alternatives considered**:
- Ignore quilt-level reconstruction in the design: rejected because it hides the
  actual product requirement.
- Implement stitch/ADT writeback immediately: rejected because the immediate
  blocker is still focused operator workflow and training execution.
