# Minimap Deconstruction Engine

**Date:** 2026-06-28

**Status:** Phases 1–6 code-complete; real-data proofs pending; analytic-normal decision landed

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
- **Stage B** (teacher object suppression): code-complete. Python module
  `data-harvester/src/harvester/teacher_prior.py`, CLI
  `build_teacher_prior_dataset.py`, reviewer
  `review_teacher_prior_dataset.py`, pytest tests. T021 (real-data proof
  on a staged-client-backed V18 store) pending.
- **Stage C** (height-only terrain model): code-complete with the V18 perf
  stack ported (AMP, torch.compile, gradient clipping, multi-scale L1,
  optional Sobel + normal-consistency losses, early stopping, resume,
  labeled preview panels, DataLoader with workers + prefetch, optional
  VRAM autotune, deterministic seeding, throughput reporting). Dataset
  `data-harvester/src/harvester/height_only_prior_dataset.py`, training
  script `scripts/train_height_only_prior.py`, pytest tests. T029 (real-
  data smoke proof) pending.
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
  trained.** T043/T044 deferred.

## Why This Route

- It matches the existing repo's one-model-one-signal philosophy.
- It makes the terrain problem smaller before trying to solve it.
- It preserves ADT-backed teacher supervision while allowing ADT-free deployment.
- It makes development-map reconstruction possible without pretending hidden ADT truth is available at inference time.

## Relationship to Earlier Docs

- `multi-model-terrain-reconstruction-2026-05-16.md` remains useful historical thinking, but its model breakdown is too large and too optimistic for the current route.
- `025-object-roof-mask-library-and-minimap-sieve` is historical context, not the active execution plan.
- `077-minimap-deconstruction-engine` is the active planning surface for this route.
