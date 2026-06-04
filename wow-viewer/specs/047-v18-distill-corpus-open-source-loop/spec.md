# Feature Specification: V18 Distill Corpus and Open-Source Release Loop

**Feature Branch**: `047-v18-distill-corpus-open-source-loop`

**Created**: 2026-06-04

**Status**: Draft

**Input**: User direction: "we have models that work, from the v16.1 line, but we're in the process of fixing up our data harvesting, which is probably more than we need. Let's just pull data for 0.5.3 and 3.3.5 and we should have more than enough valid bits of data to use. We can even do our own object masks through the precise wow-viewer automated capture for synthesized data. Ideally, the results of building the main model can be applied to the synthesized data, which in turn, can be used to train an open-source model that doesn't use any original data for implementing the correlation."

## Problem Statement

The V18 dataset and V18 model namespaces already exist and the V16.1 model line
works. What's missing is a focused, shippable pipeline that:

1. trims the harvested corpus from six builds down to two (`0_5_3_3368` and
   `3_3_5_12340`) which are already proven on the renderer-truth object-mask
   lane,
2. uses the wow-viewer automated capture pipeline (spec 012 / spec 025 Phase 2)
   to produce precise object-mask signals on the full two-build corpus rather
   than the current single-tile proof per build,
3. keeps training on the existing V16.1 / V18 model line unchanged,
4. generates **synthesized** minimap-like inputs that contain no proprietary
   game assets,
5. applies the trained main model to those synthesized inputs to produce
   per-pixel labels,
6. trains a small open-source student model on the labeled synthesized data,
7. ships that student model under a permissive license with no proprietary
   data dependency.

The point of this spec is not to redesign anything that already works. The
point is to focus the harvesting and capture surface onto the minimum corpus
that proves the approach, then turn the trained main model into a teacher
for an open-source student so the result is distributable.

## Out of Scope

- New model architectures. The main model is the existing V16.1 family re-
  exported under the V18 namespace (`v18_models.py` → `v16_1_models.py`).
- Adding more builds beyond `0_5_3_3368` and `3_3_5_12340`. The other four
  builds stay where they are.
- Distributing the main model or any data derived from copyrighted game
  assets. The main model and the labeled real-data corpus stay in-repo and
  follow the Bring Your Own Data policy from the constitution.
- Replacing spec 001 (V18 dataset canonical contract), spec 024 (V18 canvas
  paste refinement), or spec 025 (object roof mask library). This spec sits
  beside them and pulls from them where useful.

## Goals

- One canonical path from harvested `0_5_3_3368` and `3_3_5_12340` data to
  precise renderer-truth object-mask coverage on every accepted tile.
- One bounded proof run that the existing V16.1 / V18 normal lane still
  converges on the trimmed two-build corpus.
- One reproducible synthesized-input generator that emits minimap-like
  patches with no proprietary textures, no proprietary metadata, and no
  derived-from-proprietary data baked in.
- One student model that consumes only the synthesized inputs plus the
  main-model's predicted labels, with a published architecture and
  permissive license.

## User Scenarios & Testing

### User Story 1 — Trim the corpus to 0.5.3 and 3.3.5 (Priority: P1)

A researcher can run the canonical V18 build against only the two target
builds, and get a publishable two-build corpus without the four other
builds being required.

**Why this priority**: The full six-build harvest is wider than the current
proof actually needs. Two builds with renderer-truth object-mask proof are
enough to validate the approach and to feed the open-source student
pipeline.

**Independent Test**: Build the V18 corpus using only `0_5_3_3368` and
`3_3_5_12340` and verify the resulting stores are complete, valid, and
ready for training without falling back to any of the other four builds.

**Acceptance Scenarios**:

1. **Given** the two staged client roots under
   `output/tmp/wowarchive-clients/0_5_3_3368/` and
   `output/tmp/wowarchive-clients/3_3_5_12340/`, **When** the focused build
   runs, **Then** it produces a V18 Zarr store for each of the two builds
   with all required signals present.
2. **Given** the two-build focused corpus, **When** validation runs,
   **Then** signal coverage and decoded-metadata coverage match the index
   exactly, and the other four builds are not required for the validation
   to pass.

---

### User Story 2 — Renderer-truth object masks for the full two-build corpus (Priority: P1)

A researcher can run wow-viewer automated capture so every accepted tile
in the focused two-build corpus has a renderer-truth object-mask
artifact, not just one anchor tile per build.

**Why this priority**: The current renderer-truth object-mask proof is
bounded to one tile per build (`Azeroth_30_48`). To train a main model
whose object-mask signals are "proper" rather than approximate, the
capture lane must cover the full focused corpus.

**Independent Test**: Run the wow-viewer automated capture batch (spec
012 / spec 025 Phase 2 / `WowViewer.Tool.ValidationCapture capture-batch`)
against the focused two-build index and verify that for every accepted
tile there is a renderer-truth object-mask artifact, with per-tile
evidence of capture completion.

**Acceptance Scenarios**:

1. **Given** the two-build focused index, **When** the automated capture
   batch runs, **Then** every accepted tile produces a renderer-truth
   object-mask artifact and a corresponding no-object minimap artifact.
2. **Given** a tile that fails capture, **When** the build merges
   results, **Then** the failure is recorded explicitly with status
   (`captured`, `failed`, `skipped`) and the tile is not silently treated
   as having a renderer-truth mask.
3. **Given** a successful capture batch, **When** the main V18 build
   runs, **Then** the build promotes renderer-truth object-mask coverage
   to a first-class V18 signal (alongside the existing coarse
   `object_mask` family) and reflects that coverage in the build
   validation report.

---

### User Story 3 — Main V18 model trains on the focused two-build corpus (Priority: P1)

A researcher can train the existing V16.1 / V18 normal model lane (no
architecture changes) on the focused two-build corpus, and the
normal-lane convergence behavior is still observable on the trimmed
corpus.

**Why this priority**: This is the proof that the existing model line is
viable on the smaller corpus. If this fails, there is no teacher for the
student.

**Independent Test**: Run a bounded normal-lane training pass on the
focused two-build corpus (curated, with normal-aware gating plus
renderer-truth object-mask loss weighting) and confirm convergence
behavior is at least as good as on the full six-build corpus.

**Acceptance Scenarios**:

1. **Given** the focused two-build corpus, **When** the V16.1 / V18
   normal trainer runs, **Then** it consumes the renderer-truth
   object-mask signal as a first-class loss-weight input, not as an
   approximate fallback.
2. **Given** a curated focused-corpus training pool, **When** the run
   finishes its bounded epoch budget, **Then** evidence files show
   normal validation improvement and a per-tile loss breakdown that
   distinguishes tiles with renderer-truth object-mask coverage from
   tiles without.
3. **Given** the trained main model checkpoint, **When** the run is
   saved, **Then** the checkpoint is stored under
   `models/v18/normal/runs/<run-name>/` and is the named teacher for
   the distillation lane.

---

### User Story 4 — Synthesized data generation (Priority: P1)

A researcher can run a script that generates synthesized minimap-like
patches. Those patches contain no proprietary game assets and no
proprietary metadata. They are designed to look terrain-like enough to
probe the main model's learned correlations, but every byte of input is
either procedurally generated or sourced from a permitted open-content
asset pool that is named in the script's evidence.

**Why this priority**: The student model can only be open-sourced if its
training inputs are open. If the synthesized inputs leak any
proprietary texture, label, or metadata, the student cannot be released
under a permissive license.

**Independent Test**: Generate a bounded synthesized dataset, hash every
generated input, and verify the asset audit report shows zero
proprietary sources.

**Acceptance Scenarios**:

1. **Given** a seed and a tile count, **When** the synthesizer runs,
   **Then** it emits the requested number of synthesized inputs with
   matching manifest, hash, and asset-audit report.
2. **Given** the generated inputs, **When** the asset-audit report is
   inspected, **Then** every input is marked as `procedural`,
   `public_domain`, or `permissive_license` with the specific license
   name, and no input is marked as derived from copyrighted game
   client files.
3. **Given** a synthesized input, **When** it is rendered side-by-side
   with a real minimap, **Then** the visual format matches
   `256x256x3` uint8 RGB and shares the same channel layout so the
   trained main model can consume it without code changes.

---

### User Story 5 — Distill the main model onto synthesized data (Priority: P1)

A researcher can apply the trained main model checkpoint to every
synthesized input and produce a per-tile labeled dataset. The labels are
the main model's predictions on the synthesized inputs, not the real
ground truth. The labeled synthesized dataset is the open-source
training corpus.

**Why this priority**: This is the bridge between the proprietary main
model and the open-source student. The student's training corpus must
contain no proprietary content, so it has to come from synthesized
inputs plus the main model's predictions on those inputs.

**Independent Test**: Run distillation on a bounded synthesized dataset
and verify the output is a per-input label store with the same shape
contract as the real supervision tensors, plus a provenance manifest
that ties every label back to the synthesized input hash and the main
model checkpoint.

**Acceptance Scenarios**:

1. **Given** a trained main model checkpoint and a synthesized input
   dataset, **When** distillation runs, **Then** it emits a per-input
   label store with at least normal, height, holes, liquid footprint,
   and per-pixel object-mask predictions.
2. **Given** the labeled synthesized dataset, **When** the provenance
   manifest is inspected, **Then** every label row is linked to the
   exact synthesized input hash and the main model checkpoint id.
3. **Given** a re-run of distillation with the same seed and inputs,
   **When** the second run finishes, **Then** the output labels are
   byte-identical to the first run (the synthesizer and the trained
   main model are both deterministic under seed control).

---

### User Story 6 — Open-source student model trained on labeled synthesized data (Priority: P1)

A researcher can train a small open-source student model on the labeled
synthesized dataset, with no proprietary data dependency. The resulting
model and its training pipeline can be released under a permissive
license.

**Why this priority**: This is the actual end goal — a model the
project can share with the open-source community without any proprietary
data or proprietary training checkpoint.

**Independent Test**: Train the student model on a bounded labeled
synthesized dataset and verify the training script, the model
architecture, and the evidence files contain zero references to
proprietary game client data or proprietary real-data labels.

**Acceptance Scenarios**:

1. **Given** the labeled synthesized dataset, **When** the student
   trainer runs, **Then** it consumes only synthesized inputs and
   main-model-predicted labels, with no access to real ground truth
   arrays from the V18 Zarr stores.
2. **Given** the trained student model, **When** the student is
   evaluated on a held-out slice of the labeled synthesized dataset,
   **Then** the evaluation metrics are recorded in a permissively-
   licensed evidence file.
3. **Given** the trained student model, **When** a release artifact is
   produced, **Then** it includes the model checkpoint, the training
   script, the architecture definition, the license, and a clear
   statement that the student has no proprietary data dependency.

### Edge Cases

- A tile that fails the wow-viewer automated capture must not be
  silently dropped from the focused corpus. The build must record
  capture status per tile and let the user decide whether to skip or
  re-run capture.
- A synthesized input that produces a degenerate label (e.g., all-zero
  normals, all-zero height) must be excluded from the student's
  training pool with explicit reason logged in the distillation
  evidence.
- The main model checkpoint may evolve over time. The distillation
  pipeline must record exactly which checkpoint version produced each
  label so reruns are auditable.
- The student model must be small enough and self-contained enough
  that shipping it does not require shipping the main model or any
  proprietary corpus slice.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The focused-corpus build MUST accept `--builds
  0_5_3_3368 3_3_5_12340` as the canonical build list and MUST NOT
  require any of the other four builds to produce a publishable
  dataset.
- **FR-002**: The wow-viewer automated capture batch MUST be runnable
  end-to-end against the focused two-build index, with per-tile status
  reporting and resumable capture.
- **FR-003**: The V18 build MUST promote renderer-truth object-mask
  coverage to a first-class V18 signal and reflect it in the
  validation report alongside the existing coarse object-mask family.
- **FR-004**: The V16.1 / V18 normal trainer MUST consume the
  renderer-truth object-mask signal as a loss-weight input on the
  focused corpus.
- **FR-005**: A bounded normal-lane training pass on the focused
  two-build corpus MUST produce evidence files matching the existing
  V16.1 / V18 evidence contract.
- **FR-006**: The synthesized-input generator MUST produce
  `256x256x3` uint8 RGB inputs with no proprietary content and MUST
  emit an asset-audit report for every generated input.
- **FR-007**: The synthesized-input generator MUST be deterministic
  given a fixed seed, with the same inputs produced across reruns.
- **FR-008**: The distillation pipeline MUST apply the trained main
  model to every synthesized input and produce a per-input label
  store with at least normal, height, holes, liquid footprint, and
  per-pixel object-mask predictions.
- **FR-009**: The distillation pipeline MUST record provenance: every
  label row is linked to the exact synthesized input hash and the
  main model checkpoint id.
- **FR-010**: The student trainer MUST consume only the labeled
  synthesized dataset and MUST NOT require any proprietary
  ground-truth arrays.
- **FR-011**: The student model release artifact MUST include the
  model checkpoint, the training script, the architecture definition,
  the license, and a clear statement of zero proprietary data
  dependency.
- **FR-012**: All scripts and pipelines in this lane MUST live under
  `wow-viewer/` and MUST NOT reference paths outside the repo except
  for game client paths on disk.
- **FR-013**: The student training script MUST avoid any
  CUDA-only-hardwired architecture. The training loop MUST keep
  backend seams open so the same trainer can run on CPU, CUDA, and
  any other runner that PyTorch supports in the future.
- **FR-014**: The full lane (focused build, capture, main training,
  synthesize, distill, student training) MUST be reproducible from a
  recorded config and seed.
- **FR-015**: The Bring Your Own Data policy MUST remain in force for
  the main model and the focused real-data corpus. Only the student
  model and the labeled synthesized corpus are distributable.

### Key Entities

- **Focused Two-Build Corpus**: a V18 dataset scoped to
  `0_5_3_3368` and `3_3_5_12340` only.
- **Renderer-Truth Object-Mask Signal**: a per-tile
  `object_visibility_mask` plus `no_object_minimap` pair produced by
  the wow-viewer automated capture pipeline, now treated as a
  first-class V18 signal rather than a V16.2 sidecar.
- **Main Model Checkpoint**: the trained V16.1 / V18 model that
  serves as the teacher for the open-source student.
- **Synthesized Input**: a procedurally generated `256x256x3` uint8
  RGB minimap-like patch with no proprietary content, plus its
  asset-audit entry.
- **Labeled Synthesized Dataset**: a per-input label store
  containing the main model's predicted normal, height, holes, liquid
  footprint, and per-pixel object-mask outputs, plus a provenance
  manifest linking every label to its synthesized input hash and
  teacher checkpoint id.
- **Open-Source Student Model**: a small model trained only on the
  labeled synthesized dataset, released under a permissive license
  with no proprietary data dependency.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: One canonical focused V18 build on `0_5_3_3368` and
  `3_3_5_12340` produces two complete, valid V18 stores with no
  post-build patch phase required.
- **SC-002**: The wow-viewer automated capture batch produces
  renderer-truth object-mask artifacts for at least 90% of the tiles
  accepted into the focused two-build corpus, with explicit per-tile
  status reporting for the remainder.
- **SC-003**: A bounded V16.1 / V18 normal-lane training pass on the
  focused corpus converges within the existing convergence envelope
  and writes evidence that the renderer-truth object-mask signal is
  consumed as a first-class loss-weight input.
- **SC-004**: A bounded synthesized-input generation run produces
  100% asset-audit-clean inputs and is byte-identical across reruns
  with the same seed.
- **SC-005**: Distillation on a bounded synthesized dataset produces a
  labeled synthesized corpus with full provenance linkage and
  byte-identical reruns.
- **SC-006**: The student model is trained on the labeled synthesized
  corpus with no access to proprietary real-data labels, and the
  release artifact (model + training script + architecture +
  license + provenance statement) is ready for open distribution.
- **SC-007**: The full lane (focused build → capture → main
  training → synthesize → distill → student training) is reproducible
  from a recorded config and seed.
- **SC-008**: The student training script runs on CPU and on CUDA
  without code changes, and the open-source release artifact names
  the supported backends.

## Assumptions

- The existing V16.1 / V18 model line is the model owner. No
  architecture change is needed for this lane.
- The wow-viewer automated capture pipeline (spec 012, spec 025
  Phase 2) is the renderer-truth object-mask owner. No new
  capture backend is needed.
- The two builds `0_5_3_3368` and `3_3_5_12340` are already staged
  under `output/tmp/wowarchive-clients/` and have been used as the
  bounded proof anchors for the renderer-truth object-mask lane
  in the active context.
- The trained main model checkpoint stays in-repo. It is not
  redistributed.
- The synthesized-input format (`256x256x3` uint8 RGB) is enough to
  exercise the main model's learned correlations for the student's
  purpose. The student does not need to reach the same accuracy as
  the main model; it needs to demonstrate that the distillation loop
  produces a usable open-source terrain model.
- The "permissive license" for the student release is MIT or Apache
  2.0, with the final choice recorded in the release artifact.

## Relationship to Existing Specs

- **Builds on**: `001-v18-dataset-spec` (V18 dataset build contract).
- **Consumes**: `024-v18-canvas-paste-refinement-layer` outputs as
  optional refinement inputs (not required for the focused lane).
- **Consumes**: `025-object-roof-mask-library-and-minimap-sieve`
  object-roof signal as an auxiliary object-mask source where the
  wow-viewer capture does not have coverage.
- **Consumes**: `012-real-validation-batch-extraction` capture
  pipeline as the renderer-truth object-mask owner.
- **Consumes**: existing V16.1 / V18 model and trainer surface (no
  changes to model architecture).
- **Routes to**: future student-model release under a permissive
  license; the main model and the focused corpus stay in-repo.
