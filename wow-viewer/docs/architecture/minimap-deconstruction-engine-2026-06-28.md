# Minimap Deconstruction Engine

**Date:** 2026-06-28

**Status:** Phases 1–6 code-complete; static-review fixes landed; real-data proofs pending; analytic-normal decision landed

**Spec owner:** `wow-viewer/specs/077-minimap-deconstruction-engine/`

## Purpose

Define the current small-model execution order for terrain reconstruction from minimap imagery.

This note replaces the earlier assumption that a single terrain model can learn directly from the fully baked minimap or that a large multi-model stack should be built all at once.

## Core Idea

The minimap is a baked composite image. The engine must unbake it into simpler parts:

1. explain the object layer,
2. suppress the object layer to reveal a terrain-focused prior,
3. predict terrain height from that prior,
4. optionally refine normals later,
5. optionally restore objects afterward.

## Mandatory Constraints

- No one-model-does-everything architecture
- No shared-weight terrain-plus-object mega model
- Reuse the existing C# harvester/capture/placement surfaces
- Height-only terrain proof before normals
- ADT-backed teacher generation for training, ADT-free minimap-only inference later

## Stage Order

### Stage A - Per-Object Capture Library

Input:

- staged client assets
- harvested placement name tables and observations

Output:

- per-object image capture
- per-object precise mask
- original asset-path metadata
- capture review state

Owner surfaces:

- `AdtPlacementReader`
- `TerrainTileTensorPack` placement tables
- existing validation-capture lanes

### Stage B - Teacher Object Suppression

Input:

- V18 raw minimap
- filtered precise/object-filtered teacher masks
- optional object-library lookups for review/debugging

Output:

- teacher object mask
- teacher object confidence
- processed minimap prior

This is still ADT-backed and used for supervised dataset generation.

### Stage C - Height-Only Terrain Model

Input:

- processed minimap prior

Output:

- `height_257`

This stage is intentionally tiny and single-purpose. It does not own normals, liquids, or object reconstruction.

### Stage D - Minimap-Only Object Explanation

Input:

- raw minimap
- object library
- optional PM4 context

Output:

- predicted object mask
- asset candidates
- XY and yaw
- processed minimap prior without ADT teacher data

This is the development-map deployment path.

### Stage E - Optional Normal Follow-On

Baseline:

- derive normals analytically from predicted height

Only if needed:

- add a separate normal-refinement model with its own dataset, trainer, checkpoint, and validation

## Implementation Status (2026-06-28)

- **Stage A** (per-object capture library): code-complete on the Python
  side. C# data contracts (`ObjectLibraryEntry`, `ObjectCaptureVariant` + 4
  enums + deterministic ID rules) under
  `wow-viewer/src/core/WowViewer.Core/Maps/`. xUnit tests in
  `wow-viewer/tests/WowViewer.Core.Tests/ObjectLibraryContractsTests.cs`.
  Python module `data-harvester/src/harvester/object_library.py`,
  enumerator `enumerate_object_capture_jobs.py`, builder
  `build_object_library.py`, reviewer `review_object_library.py`, end-to-end
  pytest `test_object_library_e2e.py`, quickstart. C# capture-lane
  extension (T010) deferred.
- Static-review correction: C# and Python object IDs now share the same
  SHA1-14-hex `objlib_` and SHA1-16-hex `objvar_` truncation rules, and
  variant payloads use spec-string capture modes plus single-precision `G9`
  pose formatting on both sides.
- **Stage B** (teacher object suppression): code-complete. Python module
  `data-harvester/src/harvester/teacher_prior.py`, CLI
  `build_teacher_prior_dataset.py`, reviewer
  `review_teacher_prior_dataset.py`, pytest tests. T021 (real-data proof
  on a staged-client-backed V18 store) pending. This stage currently uses
  aggregate V18 tile masks, not per-object capture-library masks. The
  default teacher priority is `object_precise_mask`, then
  `object_filtered_mask`, then `object_mask`; older generated
  teacher-prior stores may have used filtered-first and should be rebuilt.
  The reviewer can render a targeted `--tile-id` with source V18 masks
  beside the teacher mask to diagnose missing ships/buildings. The
  visibility audit `scripts/audit_teacher_prior_visibility.py` scores
  whether each aggregate object mask is visibly represented in the raw
  minimap, buckets rows as `visible`, `weak`, `tiny`, or `empty`, and
  writes a `kept_tiles.parquet` manifest so weak/mismatched object-mask
  rows can be excluded from height training.
- **Stage C** (height-only terrain model): code-complete with the V18 perf
  stack ported (AMP, torch.compile, gradient clipping, multi-scale L1,
  optional Sobel + normal-consistency losses, optional V18 normal-guidance
  loss derived from predicted height, labeled preview panels,
  DataLoader with workers + prefetch, optional VRAM autotune,
  deterministic seeding, throughput reporting). The trainer now uses
  epoch-based training with deterministic train/validation split,
  epoch-level validation, resume from epoch + step state, and
  `*_latest.pt` / `*_best.pt` checkpoints (`*_model.pt` remains a latest
  compatibility alias). `--steps` is only a smoke/resume cap; `--epochs`
  is the production training contract. Normal guidance does not add a
  normal head; the model still predicts only `height_257`. Dataset
  `data-harvester/src/harvester/height_only_prior_dataset.py`, training
  script `scripts/train_height_only_prior.py`, pytest tests. T029 (real-
  data smoke proof) pending. D4 flip/rotate augmentation is no longer the
  canonical route for baked minimap RGB because terrain shadows have a fixed
  world direction; `--augment` defaults to the shadow-safe identity-only
  policy, while `--augment-policy d4` remains available only as an explicit
  ablation. The albedo guidance channel is precomputed for real runs by
  `scripts/build_albedo_dataset.py`, which writes `albedo_rgb_256` sidecar
  stores under `output/datasets/albedo/<build>.zarr` from V18 `alpha_256`
   plus `mcly_texture_ids` / `mcly_layer_mask`. This is stable texture-ID
   pseudo-colour guidance, not decoded BLP albedo. The trainer consumes these
   via `--albedo-path` and shows an `albedo input` panel in final and per-epoch
   validation previews.

  The `cuda_albedo_shadow_safe` run plateaued by epoch 240 with weak broad
  shape and visible grid/noise artifacts, so it should not be resumed. The
  bounded next base-model run remains height-only but must start fresh with
  `--model-norm group --decoder-upsample nearest`: GroupNorm removes
  BatchNorm train/eval running-stat drift, and nearest decoder upsampling
  avoids the legacy bilinear decoder path implicated in the grid artifact.
  The legacy defaults remain `batch` + `bilinear` so old checkpoints keep their
  architecture contract.

  Cloud training handoff is owned by
  `data-harvester/scripts/package_spec077_runpod.py`. The packager writes a
  RunPod-ready bundle containing only Python training code and derived
  training artifacts: teacher-prior stores, slim V18 target stores
  (`height_257`, `object_filtered_mask`, `normal_xyz`, `normal_mask`), albedo
  sidecars, and the visibility-audit manifest. It explicitly excludes staged
  game clients and raw archive data. The recommended first cloud route is a
  RunPod PyTorch CUDA Pod plus persistent `/workspace` storage or a network
  volume; Flash/Serverless can be added later once the bundle is resident on a
  volume and the training command is stable.

  `data-harvester/scripts/setup_spec077_runpod.py` wraps the packager and the
  RunPod REST API for the cheap training target. It requests `NVIDIA RTX 4000 Ada
  Generation` only by default; alternative GPU fallbacks require explicit
  `--gpu-fallback` or `--gpu-types` opt-in. The default request is one GPU, 50GB minimum
  RAM, 8 vCPU, a 150GB network volume mounted at `/workspace`, and a 50GB
  container disk. Network volumes require a concrete datacenter, so the helper
  resolves a candidate before creating the volume/Pod and deletes unused newly
  created volumes when a no-capacity Pod attempt fails. It accepts
  `RUNPOD_API_KEY` for Pod/network-volume creation and writes a local setup
  manifest without storing the key. The default transfer path avoids separate
  RunPod S3 credentials: the Pod bootstrap runs `runpodctl receive <code>` and
  the local setup helper starts `runpodctl send <bundle.tar> --code <same-code>`
  when `runpodctl` is on `PATH`. `rsync` or RunPod's separate S3 credentials
  remain manual alternatives.

  If small-detail quality remains weak after the base broad shape stabilizes,
  the next model should be a separate MCLY-guided residual refinement lane, not
  a new head on the base height model. The bounded contract is: freeze the base height
  checkpoint, generate base predictions, derive low/medium/high detail masks
  from MCLY layer activity and `alpha_256` transition gradients, then train a
  small model that predicts one signal (`height_delta_257`) only inside the
  high/transition-detail mask. Composition is `height_refined = base_height + detail_mask * height_delta`;
  normals remain analytic from the refined height.
- **Stage D** (ADT-free object explanation): code-complete on the
  consumer side. Contracts `data-harvester/src/harvester/inference_object.py`
  (InferenceObjectHypothesis, AssetCandidate, RecoveredObjectPlacement,
  hypothesis_to_recovered, collect_hypotheses). Matcher
  `data-harvester/src/harvester/asset_matcher.py` (pHash + masked-
  correlation ranker, library thumbnail loader). ADT-free prior builder
  `scripts/build_adt_free_prior.py`. pytest tests. T034 (object-mask
  training lane) and T038 (dev-map proof) pending.
- **Stage E** (normal follow-on): analytic baseline code-complete in
  `data-harvester/src/harvester/height_to_normal.py`. **Decision (T042):
  analytic normals are sufficient for the MVP; no normal model is
  trained.** T043/T044 deferred. Static review fixed batched numpy/torch
  channel-axis normalization and angular-difference reduction.

## Operator Guide

- Full PowerShell-first commands live in
  `wow-viewer/specs/077-minimap-deconstruction-engine/user-guide.md`.
- The guide covers validation commands, object-library build/review,
  teacher-prior build/review, height-only CPU/CUDA smoke runs, ADT-free
  prior generation, expected outputs, and troubleshooting.

## Why This Route

- It matches the existing repo's one-model-one-signal philosophy.
- It makes the terrain problem smaller before trying to solve it.
- It preserves ADT-backed teacher supervision while allowing ADT-free deployment.
- It makes development-map reconstruction possible without pretending hidden ADT truth is available at inference time.

## Relationship to Earlier Docs

- `multi-model-terrain-reconstruction-2026-05-16.md` remains useful historical thinking, but its model breakdown is too large and too optimistic for the current route.
- `025-object-roof-mask-library-and-minimap-sieve` is historical context, not the active execution plan.
- `077-minimap-deconstruction-engine` is the active planning surface for this route.
