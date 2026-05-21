# V16.1 Dense Correlation Model Family

## Purpose

V16.1 is the new terrain-model family name for the next V16-derived
architecture reset.

Its defining rule is simple:

```text
one minimap input
one target family
one model
one trainer
one checkpoint stream
```

V16.1 exists because the V16 shared-head terrain trainer has shown clear signs of
task interference during real training:

- long validation plateaus
- uneven head quality
- fragile normal prediction
- hard-to-interpret failure surfaces

V16.1 keeps the existing V16 Zarr corpus as the dataset contract and keeps V16
as the baseline, but rejects the V16 shared multitask model as the long-range
architecture owner.

## Core Rule

No shared trainable weights across target families.

Shared utility code is fine.
Shared dataset code is fine.
Shared run-layout helpers are fine.

But height, normals, holes, liquids, and texture decomposition must each have
their own model/trainer/checkpoint surface, and then be linked together to
build the resulting outputs.

## Target Families

The first V16.1 surfaces are:

- `minimap -> height`
- `minimap -> normal`
- `minimap -> holes`
- `minimap -> liquid footprint + liquid type`
- `minimap -> MCLY/MCAL decomposition + recomposition`

These are intentionally direct mappings because the immediate goal is to build a
dense minimap-to-signal correlation network without hiding failures inside one
shared model. Alpha is treated as a decomposition/recomposition problem rather
than a generic mask head.

## Existing Reuse

The texture-decomposition lane is not greenfield.

The repo already contains earlier minimap-to-tileset work:

- `data-harvester/scripts/train_d1.py`
- `data-harvester/src/harvester/d1_model.py`
- `data-harvester/src/harvester/dataset.py`

That earlier D1 work should be treated as the migration baseline for the V16.1
texture-decomposition family.

What changes in V16.1 is not the existence of the idea, but the contract around
it:

- move from old shard-root / NPZ assumptions to the V16 Zarr dataset
- use the better modern supervision surfaces already present in Zarr
- carry forward object-mask-derived loss gating
- emit recomposition proof as part of validation

## What V16.1 Replaces

V16.1 replaces further architecture investment in the V16 monolithic terrain
trainer.

V16 remains:

- a baseline/reference run surface
- a source of validation evidence
- a compatibility training path while V16.1 comes online

V16 does not remain the design owner for future terrain-model complexity.

## Initial Implementation Order

1. `v16_1_normal`
2. `v16_1_height`
3. `v16_1_liquid`
4. `v16_1_texcomp`
5. `v16_1_holes`
6. stitched inference from per-target checkpoints

This order is deliberate:

- normals are the current fragile target that most needs isolation
- the normal lane is the best first terrain signal for learning what the
  minimap is really telling us about terrain shape
- height can follow after those normal-lane findings sharpen the terrain-only
  supervision strategy
- liquids need type-aware interpretation, not just a broad mask
- alpha belongs inside a dedicated MCLY/MCAL decomposition family
- existing D1 work should be migrated into that family, not replaced blindly

## Inference Direction

V16.1 inference should assemble outputs from separate checkpoints through an
explicit manifest/CLI contract rather than a shared-head model.

That means:

- swapping a better height checkpoint should not require retraining normals
- swapping a better liquid checkpoint should not touch alpha
- swapping a better decomposition checkpoint should not touch height/liquids
- per-target regressions should be visible and reversible independently

## Implemented Surface

The first real V16.1 code slice is now landed in `wow-viewer/data-harvester/`.

Implemented modules:

- `src/harvester/v16_curation.py`
- `src/harvester/v16_1_dataset.py`
- `src/harvester/v16_1_models.py`
- `scripts/build_v16_curation_manifest.py`
- `scripts/train_v16_1_common.py`
- `scripts/train_v16_1_height.py`
- `scripts/train_v16_1_normal.py`
- `scripts/train_v16_1_holes.py`
- `scripts/train_v16_1_liquid.py`
- `scripts/train_v16_1_texcomp.py`
- `scripts/infer_v16_1.py`

What is real now:

- independent per-family model hosts with no shared trainable weights
- shared Zarr-backed dataset contract for V16.1 targets
- separate reusable curation layer between Zarr stores and trainers
- shared object-mask-derived weighting for the first trainer set
- trainer-side gradient accumulation for low-VRAM micro-batch training
- V16-style training/runtime seams ported into the shared trainer:
  - `torch.compile`
  - auto CUDA-friendly worker resolution
  - persistent workers
  - prefetch-factor controls
- stitched inference CLI that accepts per-target checkpoint paths and writes a
  V16.1 output Zarr store

Focused proof already exists for:

- normal-oriented curation manifest proof:
  - `wow-viewer/output/datasets/v16/curation/smoke_normal_curation_335/`
- normal-only curated 1-epoch CPU smoke:
  - `wow-viewer/models/v16_1/normal/runs/smoke_normal_curated_cpu/`
- normal-only 1-epoch CPU smoke:
  - `wow-viewer/models/v16_1/normal/runs/smoke_normal_cpu/`
- normal-only 1-epoch GPU compile smoke:
  - `wow-viewer/models/v16_1/normal/runs/smoke_normal_compile_gpu/`
- height-only 1-epoch CPU smoke:
  - `wow-viewer/models/v16_1/height/runs/smoke_height_cpu/`
- normal-checkpoint stitched inference smoke:
  - `wow-viewer/output/datasets/v16_1_inference/smoke_infer_normal/3_3_5_12340.pred.zarr`
- height-checkpoint stitched inference smoke:
  - `wow-viewer/output/datasets/v16_1_inference/smoke_infer_height/3_3_5_12340.pred.zarr`

The other target families are implemented but still need their own smoke-proof
run roots.

## Normal-Lane Loss Focus

The first V16.1 normal trainer is not treated as a naive dense supervision
problem.

Its current loss mask intentionally combines:

- `normal_mask`
- object-filter-derived terrain weighting
- `mddf_mask` / `modf_mask`
- `liquid_mask`

That means the normal trainer is explicitly pushed toward terrain-only signal
instead of over-learning object silhouettes or liquid-covered areas where the
minimap is less faithful to terrain normals.

The current normal objective is a blend of:

- angular alignment
- vector agreement
- normal-`z` stabilization

The shared V16.1 trainer now also supports:

- `--grad-accum-steps <N>`
- `--no-compile`
- `--num-workers -1`
- `--persistent-workers`
- `--prefetch-factor`

That makes `batch-size 1` or `2` usable on constrained VRAM while still
reaching a larger effective optimization batch.

This is a first-pass terrain-aware normal contract, not a final claim that the
best loss shape is solved.

## Curation Layer Rule

Blank, nonsense, or target-misaligned tiles should not be decided ad hoc inside
every trainer.

V16.1 now has a separate curation layer:

- input: V16 Zarr stores
- output: reusable tile manifests
- consumption: trainer-side `--curation-manifest`

The first profile is `normal_terrain_v1`. It explicitly checks:

- blank/low-signal minimaps
- normal coverage
- minimap-vs-normal edge agreement
- related low-signal reject cases before training

This is the intended pattern for all future model families:

1. build a target-aware curation manifest
2. inspect kept/rejected worst cases
3. train only on the curated tile set

## Initial Liquid-Type Contract

The first liquid-type label surface is intentionally coarse and uses the
existing V16 dataset truth that is already available at loader time.

Current class set:

- `0 = none`
- `1 = water`
- `2 = ocean`
- `3 = magma`
- `4 = slime`

Current source:

- coarse `16x16` labels derived from `mcnk_flags_16`

This is a first-pass supervision contract for minimap correlation and placement
behavior, not a claim that the final liquid-type surface is finished.

## Shared Loss Gating

Object masks stay important in V16.1.

The model split does not remove the need for reusable loss weighting. Instead,
V16.1 should carry forward shared object-mask-derived loss gates across all
appropriate trainers so baked object pixels do not distort terrain-adjacent
targets.

The main allowed shared signals are:

- `object_filtered_mask`
- `mddf_mask`
- `modf_mask`

These are shared training weights or masks, not shared trainable weights.

For the texture-decomposition family specifically, this means the migrated D1
successor should use the same level of loss gating quality as the V16 terrain
lane instead of reverting to weaker legacy masking assumptions.

## Boundary With Object Work

Object segmentation and asset attribution are still important, but they are not
part of the first V16.1 split-up slice.

The immediate V16.1 goal is:

- split the current terrain targets into independent trainers
- build dense minimap correlations into those target families using the existing
  V16 Zarr supervision
- keep object-mask loss gating available across those trainers
- stop hiding target-specific failures inside one shared optimizer
- link the per-family outputs back together into resulting terrain signals

Object-aware terrain cleaning can layer on top later.

## Current Truth

- dataset contract: V16 Zarr stores
- current monolith baseline: `train_v16.py` + `V16Model`
- next architecture lane: `wow-viewer/specs/006-v16-1-dense-correlation-model-family/`

## Near-Term Proof Requirement

V16.1 is not considered live just because the docs exist.

The first real proof threshold is:

1. height-only trainer smoke run
2. normal-only trainer smoke run
3. liquid footprint/type trainer smoke run
4. texture decomposition/recomposition trainer smoke run
5. holes trainer smoke run
6. separate checkpoints and separate validation artifacts for all of them
