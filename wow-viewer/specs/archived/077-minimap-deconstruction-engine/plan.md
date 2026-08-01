# Implementation Plan: Minimap Deconstruction Engine

**Branch**: `077-minimap-deconstruction-engine` | **Date**: 2026-06-28 | **Spec**: `wow-viewer/specs/077-minimap-deconstruction-engine/spec.md`

**Input**: Feature specification from `/specs/077-minimap-deconstruction-engine/spec.md`

## Summary

Build a minimap deconstruction pipeline that treats the baked minimap image as a composite to be explained in stages instead of a single monolithic terrain signal. The pipeline starts with a reusable per-object capture library generated from the existing C# harvester/capture surfaces, then uses ADT-backed teacher signals to build object-suppressed minimap priors, then trains a tiny height-only terrain model on those priors, and only afterward adds ADT-free minimap-only object explanation and optional normal refinement.

This is a successor execution path to the older object-roof and large multi-model drafts. The new plan is intentionally smaller, phase-gated, and explicit about one model per signal.

## Technical Context

**Language/Version**:

- C# / .NET 10 for capture, harvesting, shared library ownership, and object-library generation
- Python 3.11+ / `uv` for Zarr dataset preparation, training, inference, and review tooling

**Primary Dependencies**:

- Existing `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`
- Existing `WowViewer.Tool.Harvest` and `WowViewer.Tool.ValidationCapture`
- PyTorch for model training
- Zarr v3 + Parquet for dataset storage and indexing

**Storage**:

- Staged client roots under `output/tmp/wowarchive-clients/`
- Zarr datasets under `wow-viewer/output/datasets/`
- Model runs under `wow-viewer/models/`
- Review artifacts under `wow-viewer/output/analysis/`

**Testing**:

- xUnit for C# shared library/tool tests
- `pytest`, `py_compile`, and bounded `uv run` smoke commands for Python

**Target Platform**:

- Windows workstation with staged WoW clients and OpenGL-capable capture path

**Project Type**:

- Shared-library plus CLI tools plus Python ML pipeline

**Performance Goals**:

- Phase 1 object-library build should be incremental and restartable
- Phase 3 height-only training should remain in the small-model VRAM regime already proven by single-head terrain runs
- No phase should require a giant joint model or multi-day blind training before feedback exists

**Constraints**:

- Reuse existing harvester/capture owners; do not build a parallel asset-pipeline stack
- Use staged clients only
- One phase at a time; normals and downstream restoration wait until height proof lands
- Keep models tiny and single-purpose

**Scale/Scope**:

- First proof scope is bounded maps/builds and development-map proof tiles
- Full-corpus broadening comes only after bounded proof artifacts are accepted

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Repo independence: pass. All planned code and docs live under `wow-viewer/`.
- Library-first: pass. Shared object-library and prior-generation ownership will live in core libraries or existing tool layers, not only in scripts.
- Real-data validation: pass. Every phase names staged-client or V18 proof outputs.
- Residual model chain: pass. The plan explicitly forbids a monolithic joint model and uses independent lanes.
- Streaming-first dataset pipeline: pass. Existing harvest contracts remain canonical; new Zarr outputs are downstream curated datasets, not a replacement for the streaming harvester.
- No untrusted client paths: pass. All validation paths remain under staged clients.

## Project Structure

### Documentation (this feature)

```text
wow-viewer/specs/077-minimap-deconstruction-engine/
├── spec.md
├── plan.md
├── tasks.md
├── research.md
└── data-model.md

wow-viewer/docs/architecture/
└── minimap-deconstruction-engine-2026-06-28.md
```

### Source Code (repository root)

```text
wow-viewer/src/core/
├── WowViewer.Core/
│   └── Maps/
├── WowViewer.Core.IO/
│   └── Maps/
└── WowViewer.Core.Runtime/

wow-viewer/tools/
├── harvest/WowViewer.Tool.Harvest/
└── validation-capture/WowViewer.Tool.ValidationCapture/

wow-viewer/data-harvester/
├── scripts/
├── src/harvester/
└── tests/
```

**Structure Decision**: Reuse `Core`/`Core.IO` for reusable object-library and prior-generation contracts, keep CLI entrypoints thin in `tools/`, and put dataset/training/inference wrappers in `data-harvester/`.

## Phase 0 - Direction Lock and Contract Audit

**Goal**: Freeze the execution order and document the exact existing surfaces to reuse so later phases do not drift back into giant-model or duplicate-pipeline work.

1. Create the new successor spec package and architecture note.
2. Audit the exact object-library inputs already available in `TerrainTileTensorPack`, placement arrays, name tables, filtered masks, and capture tooling.
3. Record the explicit non-goals: no monolithic model, no joint terrain/object training, no normals in the first terrain proof.
4. Decide the first bounded proof scope: builds, maps, and development-map tiles.
5. Document the exact phase gate for the height-only proof before normals or restoration.

**Validation**:

- Spec, plan, tasks, research, and data-model docs exist.
- Architecture note points implementers to the correct owner surfaces.

## Phase 1 - Per-Object Capture Library Contract

**Goal**: Define and implement the reusable asset-centric object library using existing `wow-viewer` capture/harvest seams.

1. Add shared records for `ObjectLibraryEntry` and `ObjectCaptureVariant` under `src/core/`.
2. Define the Zarr/Parquet storage contract for the object library.
3. Enumerate the capture source list from harvested placement name tables and placement observations.
4. Add one-object-at-a-time capture orchestration to the existing capture tool path.
5. Persist image artifacts, precise masks, and capture metadata into the object-library store.
6. Mark uncaptured and low-confidence entries explicitly instead of silently dropping them.
7. Add bounded xUnit coverage for schema, serialization, and capture-artifact writing.

**Validation**:

- A bounded object-library proof run produces reusable entries with image, mask, and metadata.
- Review rows show original asset path, normalized path, and asset type.

## Phase 2 - Teacher Deconstruction Prior Generation

**Goal**: Turn ADT-backed placement and filtered precise signals into a reusable processed-minimap-prior dataset.

1. Define the teacher-prior dataset contract and its arrays.
2. Implement a shared prior-construction policy using raw minimap plus teacher object mask/confidence.
3. Build a Python dataset-generation script that reads V18 Zarr and writes the teacher-prior dataset.
4. Prefer `object_precise_mask` for the first teacher pass, with explicit CLI override for filtered/object-mask ablations.
5. Emit side-by-side review artifacts for raw minimap, teacher mask, source masks, processed prior, and source metadata.
6. Audit mask/minimap visibility mismatches and write a second-stage `kept_tiles.parquet` curation manifest.
7. Keep the phase-1 prior channels explicit and minimal; document them.
8. Add tests for pass-through no-object tiles, object-heavy tiles, compact-row alignment, and visibility buckets.
9. Run a bounded proof on at least one building-heavy map.

**Validation**:

- The teacher-prior dataset exists with documented arrays.
- Review artifacts clearly show object suppression behavior and identify weak/mismatched mask rows before training.

## Phase 3 - Height-Only Terrain Reboot

**Goal**: Prove that a small height-only model converges better on the processed prior than on the raw baked minimap contract.

1. Audit the smallest viable existing terrain model lane to reuse for a height-only contract.
2. Implement a dedicated processed-prior dataset reader for height-only training.
3. Add or adapt a height-only trainer that predicts only `height_257`.
4. Preserve the authoritative height truth and filtered terrain-valid weighting path.
5. Train from the visibility-audited teacher-prior manifest so weak/mismatched object-mask rows are excluded.
6. Add optional normal guidance as an auxiliary loss by deriving normals from predicted height and comparing to V18 `normal_xyz`; do not add a normal output head.
7. Use validation loss as a training control signal for LR plateau scheduling and best-checkpoint selection; do not backpropagate validation data.
8. Optionally emphasize hard training pixels with detached absolute-error weighting on the current training batch; validation abs-error remains diagnostic only.
9. Add preview artifacts that show raw minimap, teacher mask, processed prior, predicted height, ground truth, error, and loss weight.
10. Run a bounded smoke training pass.

**Validation**:

- Smoke training completes and writes previews.
- The lane is clearly height-only and does not carry extra terrain heads.

## Phase 3b - Coarse-To-Fine Residual Height Chain

**Goal**: Replace the muddy direct-height plateau with two independently trained height models that share the same source signals but split broad shape from fine residual detail.

1. Add an H0 model that predicts only `height_coarse_65` from the processed-prior input contract, including optional albedo and density channels.
2. Add an H0 training script that downsamples authoritative `height_257` / `weight_257` to the coarse target and writes its own checkpoint/metrics.
3. Add an H1 model that predicts only `height_delta_257` from the same input contract plus frozen/upscaled H0 height.
4. Add an H1 training script that loads a frozen H0 checkpoint, computes `height_delta_257 = height_257 - upsample(H0)`, and optimizes the composed height without updating H0.
5. Use high-frequency residual-friendly losses for H1: residual/reconstruction Charbonnier or L1, optional gradient loss, and optional normal guidance derived from composed height. Do not add a normal head.
6. Emit previews that show prior, optional albedo/density, frozen coarse base, residual delta, composed height, truth, and error.
7. Add pytest coverage for model shapes, deterministic residual composition, and script-level smoke behavior where feasible.
8. Update RunPod packaging so the H0/H1 scripts and helper shell commands are included in cloud bundles.

**Validation**:

- H0 and H1 smoke runs complete independently and write separate checkpoints.
- H1 uses a frozen H0 checkpoint and predicts one residual signal only.
- The residual chain can be compared against the direct `cuda_albedo_group_nearest` plateau without changing archived datasets.

## Phase 4 - Minimap-Only Object Explanation

**Goal**: Add the smallest ADT-free runtime object explanation path needed for development-map deconstruction.

1. Define a minimap-only object mask output contract.
2. Define a small asset-candidate matching/classification contract keyed to the object library.
3. Define a lightweight pose output contract limited to XY and yaw.
4. Implement a bounded dataset for minimap-only object explanation using teacher artifacts from Phase 2.
5. Train or prototype the smallest usable object-mask lane.
6. Train or prototype the smallest usable asset-candidate lane.
7. Generate processed minimap priors without ADT placement input.

**Validation**:

- A bounded development-map proof emits predicted object masks and processed priors without ADT placements.
- Output records include asset candidates and confidence values.

## Phase 5 - Development Map Proof

**Goal**: Prove the full deconstruction path on a map where minimap and PM4 exist but full ADT supervision does not.

1. Pick the first development-map proof subset.
2. Run minimap-only object explanation on the subset.
3. Produce processed priors.
4. Run the height-only terrain model on those priors.
5. Save review packages showing raw minimap, predicted mask, processed prior, height output, and PM4 context.

**Validation**:

- The proof package is reviewable without needing hidden teacher ADT tensors.

## Phase 6 - Optional Normals Follow-On

**Goal**: Add normals only after the height-only lane is validated.

1. Derive analytic normals from predicted height as the baseline.
2. Decide whether a separate normal-refinement model is still necessary.
3. If needed, define a separate normal-refinement dataset and trainer contract.
4. Keep the normal lane independent from the height lane.

**Validation**:

- Either the analytic normals are sufficient, or a separate normal lane has its own bounded proof.

## Complexity Tracking

No constitution violations are planned. The feature explicitly reduces complexity versus earlier drafts by forbidding a shared-weight mega model and reusing existing harvester/capture owners.
