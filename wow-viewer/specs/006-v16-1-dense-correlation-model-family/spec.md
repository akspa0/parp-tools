# Feature Specification: V16.1 Dense Correlation Model Family

**Feature Branch**: `006-v16-1-dense-correlation-model-family`

**Created**: 2026-05-21

**Status**: In Progress

**Input**: User description: "rename this to v16.1, we're still basing it on v16 heavily, it just needs the sledgehammer approach to build a dense correlation network between minimaps and other data we have in the Zarr dataset, and then we link it all together to build the resulting output signals."

## Problem Statement

The current V16 terrain trainer uses one shared encoder-decoder backbone with
multiple output heads. It supervises height, normals, alpha, holes, liquid, and
MCLY simultaneously from the same minimap input.

Recent real-run behavior shows this shared model is brittle:

1. Validation improvement stalls for long stretches.
2. Some targets learn broad structure while others collapse to safe defaults.
3. Normal prediction is especially fragile and often acts as if no useful signal
   is present.
4. More training time is not reliably producing better convergence.

The repo's long-standing V14 philosophy already called for small independent
models with no shared weights. V16.1 formalizes the next step without claiming
to be a clean break from V16:

- still based on the V16 Zarr dataset and supervision contract
- split by dense minimap-to-signal correlation families instead of one fragile
  shared-head monolith
- linked together at the output assembly boundary to build the resulting terrain
  signals

## Goal

Keep the V16 dataset and signal contract, but replace the V16 monolithic
multitask trainer with a V16.1 dense-correlation family of linked models, each
centered on:

```text
minimap_rgb_256 -> one target family
```

The first-class V16.1 model surfaces are:

- minimap -> height
- minimap -> normals
- minimap -> holes
- minimap -> liquid presence / placement / type
- minimap -> MCLY/MCAL decomposition + recomposition

## Implemented So Far

The first implementation slice is now landed under `wow-viewer/data-harvester/`:

- dataset + shared target contract:
  - `src/harvester/v16_1_dataset.py`
- independent per-family model hosts:
  - `src/harvester/v16_1_models.py`
- shared non-weight trainer utility surface:
  - `scripts/train_v16_1_common.py`
- target-family entrypoints:
  - `scripts/train_v16_1_height.py`
  - `scripts/train_v16_1_normal.py`
  - `scripts/train_v16_1_holes.py`
  - `scripts/train_v16_1_liquid.py`
  - `scripts/train_v16_1_texcomp.py`
- stitched inference entrypoint:
  - `scripts/infer_v16_1.py`

Focused proof already exists for:

- 1-epoch CPU height smoke:
  - `wow-viewer/models/v16_1/height/runs/smoke_height_cpu/`
- stitched inference smoke from the height checkpoint into an output Zarr store:
  - `wow-viewer/output/datasets/v16_1_inference/smoke_infer_height/3_3_5_12340.pred.zarr`

Normal, holes, liquid, and texture-decomposition trainers are implemented but do
not yet have dedicated smoke-proof run roots.

## User Scenarios & Testing

### User Story 1 — Height Can Converge Without Multitask Interference (Priority: P1)

A terrain researcher trains only the V16.1 height model on the V16 dataset and
gets stable validation movement without unrelated heads competing for backbone
capacity.

**Why this priority**: Height is the main terrain geometry signal and must be
recovered before the rest of the stitched solution is worth trusting.

**Independent Test**: Run `train_v16_1_height.py` for a 1-epoch CPU smoke against
`3_3_5_12340`, then a bounded multi-build CUDA run. Verify a dedicated
checkpoint, dedicated validation images, and dedicated metrics JSON exist.

**Acceptance Scenarios**:

1. **Given** a finalized V16 Zarr store, **When** `train_v16_1_height.py` runs,
   **Then** it reads only minimap + height-related supervision and writes only
   height-focused outputs.
2. **Given** the height trainer checkpoint, **When** validation runs,
   **Then** best-checkpoint selection is based on height metrics only.
3. **Given** a future alpha or normals regression, **When** the height trainer
   is rerun, **Then** the height trainer code and checkpoint contract remain
   unchanged.

---

### User Story 2 — Normals Are Isolated From Height/Alpha/Liquid Tradeoffs (Priority: P1)

A terrain researcher trains only the V16.1 normals model and can inspect whether
the minimap-to-normal problem is truly learnable on its own, instead of being
hidden inside the V16 multitask compromise.

**Why this priority**: Recent V16 evidence suggests normals are one of the most
fragile outputs. The team needs direct signal on whether normals fail because of
task interference or because the target itself needs a different treatment.

**Independent Test**: Run `train_v16_1_normal.py` for a 1-epoch smoke against
`3_3_5_12340` and inspect normal GT/pred panels without any unrelated heads in
the run.

**Acceptance Scenarios**:

1. **Given** the normal trainer, **When** it trains, **Then** its loss is
   derived only from normal supervision plus its own masking rules.
2. **Given** a normal-only run, **When** validation images are written,
   **Then** the run exports minimap, normal GT, normal mask, and normal pred
   without height/alpha/liquid panels mixed in.
3. **Given** the model underperforms, **When** future changes are made,
   **Then** those changes can target the normal trainer alone without touching
   the height trainer.

---

### User Story 3 — Liquids Are Predicted As Presence Plus Type, Not Just A Soft Mask (Priority: P1)

A terrain researcher can train a dedicated V16.1 liquid model that learns where
liquids are, how they are placed, and what liquid type they represent from the
minimap image.

**Why this priority**: The current V16 liquid head only predicts a broad liquid
mask. That is not enough for correct minimap interpretation, liquid placement,
or type-aware downstream reconstruction.

**Independent Test**: A dedicated liquid trainer exists and writes separate
artifacts for liquid footprint and liquid type. It completes a 1-epoch CPU
smoke run against `3_3_5_12340`.

**Acceptance Scenarios**:

1. **Given** `train_v16_1_liquid.py`, **When** it runs, **Then** it writes liquid
   footprint metrics and liquid-type metrics without coupling to height or
   normals.
2. **Given** a tile with water, ocean, magma, or slime evidence, **When** the
   liquid model validates, **Then** the output includes both placement evidence
   and type classification evidence.
3. **Given** a future better liquid checkpoint, **When** stitched inference
   runs, **Then** only the liquid outputs need to be swapped.

---

### User Story 4 — Alpha Prediction Becomes A Dedicated Decomposition/Recomposition Family (Priority: P1)

A terrain researcher trains alpha as a dedicated MCLY/MCAL decomposition model
instead of a generic shared alpha head.

**Why this priority**: Alpha is not just "another mask." It is part of the
terrain texturing decomposition problem. The model must learn to infer
texture-layer identity and blend structure together, then prove that the result
can recompose back toward the observed minimap.

**Independent Test**: `train_v16_1_texcomp.py` completes a 1-epoch CPU smoke run
and writes MCLY predictions, MCAL predictions, and a recomposed minimap review
panel.

**Acceptance Scenarios**:

1. **Given** `train_v16_1_texcomp.py`, **When** it runs, **Then** it predicts
   MCLY texture IDs and MCAL alpha together as one texture decomposition family.
2. **Given** a validation tile, **When** the decomposition model validates,
   **Then** it writes a recomposed terrain-only image derived from its own
   predicted MCLY/MCAL outputs.
3. **Given** the model underperforms, **When** future changes are made,
   **Then** those changes stay isolated to the decomposition/recomposition
   trainer instead of touching height or liquid trainers.

4. **Given** the existing D1 tileset-decomposition work in the repo, **When**
   V16.1 texture decomposition is implemented, **Then** it reuses that prior work
   as the migration baseline instead of starting from a blank model design.

---

### User Story 5 — Stitched Inference Replaces Shared-Head Inference (Priority: P2)

A terrain researcher can point the pipeline at a set of per-target V16.1
checkpoints and produce a combined predicted terrain package.

**Why this priority**: The model split is only useful if inference can assemble
the pieces back into one terrain result.

**Independent Test**: A manifest or CLI surface accepts separate checkpoint
paths for height, normals, holes, liquid, and texture decomposition, then
writes a single combined output bundle.

**Acceptance Scenarios**:

1. **Given** a checkpoint manifest with only height and liquid models,
   **When** stitched inference runs, **Then** it emits height and liquid outputs
   while leaving other targets absent or explicitly degraded.
2. **Given** a full checkpoint manifest, **When** stitched inference runs,
   **Then** it writes one combined terrain prediction package with per-target
   provenance.
3. **Given** one target model is retrained, **When** inference is rerun,
   **Then** only that target's checkpoint needs to be swapped.

## Requirements

### Functional Requirements

- **FR-001**: V16.1 MUST be the name of the next terrain-model family and MUST
  be treated as a V16-derived evolution rather than a separate post-V16 lane.
- **FR-002**: V16.1 MUST define linked dense-correlation model/trainer surfaces
  for:
  `height`, `normal`, `holes`, `liquid`, and `texture decomposition`.
- **FR-003**: Each V16.1 trainer MUST consume `minimap_rgb_256` as its primary
  input and MUST supervise exactly one target family.
- **FR-004**: V16.1 trainers MUST NOT share trainable weights across target
  families.
- **FR-005**: Each V16.1 target family MUST have its own model module, training
  script, validation artifact contract, and checkpoint naming surface.
- **FR-006**: Each V16.1 trainer MUST be runnable independently against the
  existing V16 Zarr dataset contract. No new corpus format is allowed for the
  initial split.
- **FR-006A**: V16.1 MUST explicitly build dense minimap-to-signal correlation
  models using the richer supervision already present in the V16 Zarr dataset,
  then link those per-family outputs together into assembled terrain outputs.
- **FR-007**: The height trainer MUST own height-specific best-checkpoint
  selection and MUST NOT depend on normal/alpha/liquid metrics.
- **FR-008**: The normal trainer MUST own normal-specific masking and metrics
  and MUST NOT depend on height/liquid/texture-decomposition losses.
- **FR-009**: The holes trainer MUST own hole-specific metrics and MUST NOT be
  coupled into the height or liquid optimization surface.
- **FR-010**: The liquid trainer MUST predict both liquid footprint and liquid
  type, and MUST expose type-aware validation artifacts.
- **FR-011**: The liquid trainer MUST treat liquid type as a first-class output
  surface, not as a comment derived only after inference.
- **FR-011A**: The initial liquid-type contract MAY start as a coarse `16x16`
  class grid derived from `mcnk_flags_16`, with the first class set defined as:
  `0=none`, `1=water`, `2=ocean`, `3=magma`, `4=slime`.
- **FR-012**: The texture-decomposition trainer MUST jointly predict MCLY
  texture IDs and MCAL alpha in one dedicated family and MUST validate via
  recomposition back toward the minimap.
- **FR-012A**: The V16.1 texture-decomposition family MUST explicitly reuse and
  migrate the existing D1 tileset-decomposition work (`train_d1.py`,
  `D1UNet`, `D1Dataset`) where it remains useful, rather than re-inventing the
  model/training concept from scratch.
- **FR-012B**: That migration MUST move the decomposition trainer off the old
  shard-root/NPZ-only contract and onto the V16 Zarr dataset signals.
- **FR-012C**: The migrated decomposition trainer MUST consume the improved V16
  supervision surfaces such as `alpha_256`, `mcly_texture_ids`,
  `mcly_layer_mask`, and object-mask-derived loss gating.
- **FR-013**: V16.1 trainers MUST support shared object-mask-derived loss gating
  where appropriate so object-heavy pixels stay downweighted across target
  families that should not overfit baked objects.
- **FR-014**: Shared object-mask gating MUST reuse the existing V16 object-mask
  dataset signals (`object_filtered_mask`, and optionally `mddf_mask` /
  `modf_mask`) instead of inventing a separate mask format for each trainer.
- **FR-015**: V16.1 inference MUST stitch separate checkpoint outputs into one
  combined terrain prediction surface through an explicit manifest or CLI
  contract.
- **FR-016**: V16.1 validation outputs MUST make failure isolation obvious by
  showing only the target family relevant to the run plus the shared minimap
  input.
- **FR-017**: V16 MUST remain available as a baseline/reference path until at
  least the V16.1 height and normal trainers have smoke proof.

### Non-Goals

- Rebuilding the dataset format before the first V16.1 slice
- Solving object segmentation in the same first implementation slice
- Making one "universal" trainer that conditionally switches target types
- Reusing a shared multitask backbone under a different V16.1 name
- Removing object-mask loss gating from terrain-adjacent trainers

### Key Entities

- **V16.1 Height Model**: minimap -> `height_257`
- **V16.1 Normal Model**: minimap -> `normal_xyz` / `normal_mask` supervised
- **V16.1 Holes Model**: minimap -> `holes_16`
- **V16.1 Liquid Model**: minimap -> liquid footprint + liquid type
- **V16.1 Texture Decomposition Model**: minimap -> `mcly_texture_ids` +
  `mcly_layer_mask` / `alpha_256` + recomposed terrain-only view
- **Legacy D1 Baseline**: existing shard-based tileset decomposition work in
  `train_d1.py`, `harvester/d1_model.py`, and `harvester/dataset.py` that must
  be migrated, not ignored
- **Shared Object Loss Gating**: reusable training weights derived from
  `object_filtered_mask` and related object-mask signals
- **V16.1 Checkpoint Manifest**: a stitched-inference input that maps target
  families to checkpoint paths

## Success Criteria

### Measurable Outcomes

- **SC-001**: Dedicated scripts exist for `train_v16_1_height.py` and
  `train_v16_1_normal.py`; the height trainer already completes a 1-epoch CPU
  smoke run and the normal trainer still needs its own smoke proof.
- **SC-002**: Dedicated scripts exist for `holes`, `liquid`, and texture
  decomposition, even if some start as thin wrappers around shared training
  utilities.
- **SC-003**: Each target family writes checkpoints under its own run root
  without mixing metrics from unrelated targets.
- **SC-004**: The liquid trainer emits both footprint and liquid-type evidence.
- **SC-005**: The texture-decomposition trainer emits both decomposition outputs
  and a recomposed review image.
- **SC-006**: A stitched-inference contract exists that can combine multiple
  V16.1 checkpoint outputs into one prediction bundle, and the first partial
  height-only smoke write is proven.
- **SC-007**: Future validation review can answer "which target failed?"
  without reading a multitask overview panel.

## Assumptions

- The current V16 dataset is good enough to start V16.1 trainer separation.
- The main current bottleneck is architectural interference, not lack of raw
  supervision.
- Liquid type supervision may initially be coarser than footprint supervision
  and may rely on currently available dataset provenance such as
  `mcnk_flags_16`, with richer per-pixel typing layered later.
- The existing D1 decomposition work is good enough to serve as the starting
  point for V16.1 texture decomposition, but it needs migration to the V16 Zarr
  truth surfaces and current loss-gating contract.
- Some targets may later prove better as deterministic derivatives or residual
  refinements, but V16.1 starts by honoring the user's requested direct
  `minimap -> target-family` split while linking those families back together
  into final outputs.
- Shared utility code is acceptable; shared trainable weights are not.

## Initial Implementation Direction

1. Start with `height` as the first V16.1 production slice.
2. Land `normal` second so the current failure mode is tested directly.
3. Land `liquid` as a dedicated footprint + type family.
4. Land texture decomposition/recomposition as the dedicated `MCLY/MCAL`
   trainer.
5. Keep `holes` as an independent follow-up trainer that reuses helper code but
   not weights.
6. Only after the per-target trainers exist should stitched inference become
   the next integration step.
