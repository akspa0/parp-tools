# V18 Distill Corpus and Open-Source Release Loop — 2026-06-04

## Purpose

This document is the architectural reference for the focused V18 distill
corpus and the open-source release loop. It summarizes the lane, points at
the active spec, and records the major decisions so future readers do not
have to reconstruct them from chat or commit messages.

The active spec is:

- `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md`
- `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/plan.md`
- `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/tasks.md`

If this document and the spec ever disagree, the spec wins; this document
is a compressed architectural summary, not the source of truth.

## Why This Lane Exists

The V18 dataset and V18 model namespaces already exist and the V16.1
model line is the implementation layer. The actual problem is that:

1. The harvested corpus is six builds wide; the current proof only needs
   two of them.
2. The renderer-truth object-mask proof is bounded to one tile per
   build. The remaining tiles still rely on approximate placement-derived
   masks.
3. There is no path from the trained main model to a distributable
   open-source model.

This lane solves those three problems without redesigning the V18
model.

## Core Decisions

### Decision 1: Trim to two builds, keep the rest

The focused corpus is `0_5_3_3368` and `3_3_5_12340` only. The other
four builds stay where they are; they are simply out of scope for this
lane. The two chosen builds are already proven anchors for the
renderer-truth object-mask lane in the active context.

This is a "less is more" decision. The full six-build harvest costs
wall time and storage; the proof does not need it. If the open-source
release loop needs more data later, the other four builds can be
revisited.

### Decision 2: Renderer-truth object-mask becomes a first-class V18 signal

Currently, the renderer-truth object-mask artifacts are a V16.2
sidecar, captured only on a single anchor tile per build. The new lane
promotes the capture batch to cover the full focused two-build corpus
and treats the resulting coverage as a first-class V18 signal in
`index.parquet` and `signal_validation.json`.

The capture lane is the existing `WowViewer.Tool.ValidationCapture
capture-batch` per spec 012 and spec 025 Phase 2. No new capture
backend is added.

### Decision 3: Existing V16.1 / V18 model is the teacher

The main model is the existing V16.1 / V18 normal lane. There is no new
architecture. The teacher is trained on the focused two-build corpus
with the renderer-truth object-mask signal consumed as a first-class
loss-weight input.

The teacher checkpoint lives in-repo under
`models/v18/normal/runs/<run-name>/`. It is not redistributed.

### Decision 4: Synthesized inputs are procedural, not learned

The synthesized-input generator is a procedural generator. It produces
`256x256x3` uint8 RGB minimap-like patches from heightfields, albedo
proxies, and low-frequency terrain-like structure. Every input is
content-addressed by hash and recorded in a manifest.

The procedural choice is deliberate. A learned generator could leak
proprietary texture patterns from the training data. A procedural
generator is provably asset-free.

### Decision 5: Distillation emits provenance, not just labels

The distillation pass applies the trained main model to every
synthesized input and produces a per-input label store with at least
normal, height, holes, liquid footprint, and per-pixel object-mask
predictions. Every label row is linked via a provenance manifest to
the exact synthesized input hash and the teacher checkpoint id.

The provenance is what makes the labeled synthesized corpus auditable.
Without it, the student trainer cannot prove its labels are
reproducible.

### Decision 6: Student model is small and open-source

The student model is intentionally small. The candidate architectures
are a small U-Net or a ConvNeXt-tiny backbone; the final choice is
recorded in the release artifact. The student trainer supports CPU and
CUDA with no architecture-specific code paths.

The release artifact is packaged under
`models/v18_student/release/<version>/` with:

- the model checkpoint
- the training script
- the architecture definition
- the license (MIT or Apache 2.0)
- a zero-proprietary-data-dependency statement
- a provenance manifest

Only the student and the labeled synthesized corpus are distributable
under a permissive license. The main model and the focused real-data
corpus stay in-repo under the Bring Your Own Data policy.

## Pipeline Summary

```text
staged 0.5.3 + 3.3.5 client roots
        │
        ▼
build_focused_two_build_corpus.py            (Phase A1)
   └─► wow-viewer/output/datasets/v18/0_5_3_3368.zarr
   └─► wow-viewer/output/datasets/v18/3_3_5_12340.zarr
        │
        ▼
run_focused_capture_batch.py                  (Phase A2)
   └─► renderer-truth object_visibility_mask + no_object_minimap
       for every accepted tile
        │
        ▼
train_v18.py / train_v16_1_normal.py         (Phase A3)
   └─► models/v18/normal/runs/<run-name>/     (teacher checkpoint)
        │
        ▼
synthesize_v18_inputs.py                     (Phase B1)
   └─► wow-viewer/output/datasets/synthesized/<run-name>/
        procedural 256x256x3 uint8 RGB inputs
        │
        ▼
distill_v18_to_synthesized.py                (Phase B2)
   └─► wow-viewer/output/datasets/distilled/<run-name>/
        labeled synthesized corpus with full provenance
        │
        ▼
train_v18_student.py                         (Phase B3)
   └─► models/v18_student/release/<version>/
        open-source student model + license + provenance
```

## Boundary With Other Specs

- **Builds on**:
  - `001-v18-dataset-spec` (V18 dataset build contract).
  - `012-real-validation-batch-extraction` (capture batch tooling).
  - `025-object-roof-mask-library-and-minimap-sieve` (object-roof
    signal as an auxiliary object-mask source where the wow-viewer
    capture does not have coverage).
- **Reuses without modification**:
  - V16.1 / V18 model and trainer surface (`v18_models.py`,
    `v16_1_dataset.py`, `train_v18.py`).
  - `WowViewer.Tool.Harvest` and `WowViewer.Tool.ValidationCapture`.
  - The existing V18 streaming build shape from
    `build_v18_dataset.py`.
- **Reroutes (Superseded)**:
  - `015-v16-1-2-height-derived-normal-refiner` — refiner failed.
  - `017-v16-1-4-combined-normal-height-model` — combined head not
    used in the V18 lane.
  - `022-v17-unified-normal-height-refiner` — V17 hybrid was folded
    into V16.1.3 and the V18 lane.
  - `023-v17-1-global-minimap-signal-reconstruction` — V17.1 contract
    is implemented in V16.1 + V18.

## Risks and Guardrails

- **Capture batch throughput**: per-tile validation capture is slow
  on the legacy MdxViewer lane. The wow-viewer capture-batch is
  faster but still bounded. The new lane must batch tiles per loaded
  world session, not one tile at a time, to keep wall time
  reasonable. This is the same throughput risk already documented
  for the existing capture lane.
- **Open-source scope drift**: the student release must not
  accidentally include any proprietary ground truth, any proprietary
  texture, or any proprietary metadata. The asset-audit report and
  the provenance manifest are the guardrails here. Any change that
  touches the synthesizer or the distillation pass must keep the
  asset-audit report green.
- **Architecture drift**: the V18 model line is the model owner.
  This lane is a focused corpus + open-source release loop, not a
  model redesign. If a future spec needs a new model, it must be
  its own spec with its own scope.
- **Build scope drift**: the focused corpus is two builds. The other
  four builds are out of scope. If a future spec needs more builds,
  it must justify the re-expansion explicitly.

## Open Follow-Up

- Bounded throughput proof for the full focused two-build capture
  batch (current proof is per-tile on the development workstation).
- Comparison of small U-Net vs. ConvNeXt-tiny for the student
  architecture; final choice recorded in the release artifact.
- Documentation updates in `wow-viewer/README.md` and
  `wow-viewer/data-harvester/README.md` describing the focused
  two-build path, the synthesize-distill-student loop, and the
  open-source student release section.
