# Implementation Plan: V18 Distill Corpus and Open-Source Release Loop

**Branch**: `047-v18-distill-corpus-open-source-loop` | **Date**: 2026-06-04 | **Spec**: [`wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md`](spec.md)

**Input**: Feature specification from [`wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md`](spec.md)

**Note**: The owner spec declares the lane "Two plans under one umbrella" in
the migration decision. The standard Speckit pattern is one `plan.md` per
spec, so this file documents the umbrella plan and clearly labels Plan A
(Distill Corpus) and Plan B (Open-Source Release Loop) as the two
sub-plans the user picked. Each sub-plan is split into bite-sized phases
that are independently validatable, with no more than 10 steps per phase
per the constitution.

## Summary

The current V18 dataset and V18 model line are working but the surface is
wider than the proof needs. This plan narrows the harvest to
`0_5_3_3368` and `3_3_5_12340`, expands the wow-viewer automated capture
batch to the full focused corpus (not just the current single-tile proof
per build), trains the existing V16.1 / V18 model on the trimmed
corpus, then uses that trained main model as a teacher: a procedural
synthesizer generates asset-free minimap-like inputs, the main model
labels them, and a small open-source student model is trained on those
labels. The main model and the focused real-data corpus stay in-repo
under the Bring Your Own Data policy. Only the student and the labeled
synthesized corpus are distributable under a permissive license.

This plan does not redesign the V18 model, does not introduce a new
architecture, and does not change the existing V16.1 / V18 trainer
surface. It focuses existing capabilities on a narrower corpus and wires
them into a new open-source release loop.

## Technical Context

- **Language/Version**: Python 3.11+ via `uv` for synthesizer, distillation,
  and student training. C# / .NET 10 reused as-is for the wow-viewer
  capture lane and the existing V16.1 / V18 trainer.
- **Primary Dependencies**: existing `WowViewer.Tool.Harvest`,
  `WowViewer.Tool.ValidationCapture`, `pyarrow`, `zarr`, `numpy`,
  `Pillow`, `torch`, existing `v18_models.py` / `v16_1_models.py` /
  `train_v18.py` surfaces.
- **Storage**: per-build V18 Zarr v3 stores under
  `wow-viewer/output/datasets/v18/`, plus per-tile capture ledger
  outputs under the existing capture batch root, plus a new
  synthesized-input dataset root under
  `wow-viewer/output/datasets/synthesized/`, plus a labeled distilled
  corpus root, plus a student model release root.
- **Testing**: real-data `uv run` builds, bounded normal-lane training
  on the focused corpus, deterministic rerun checks for the synthesizer
  and the distillation pass, license audit on the student release
  artifact.
- **Target Platform**: Windows 11 development workstation, staged client
  roots under `output/tmp/wowarchive-clients/`. No `H:\CLIENTS`.
- **Project Type**: data-harvester scripts plus a small new open-source
  student trainer script. No new C# / .NET projects are added in this
  lane; the existing capture and harvest tools are reused as-is.
- **Performance Goals**: full focused corpus capture batch in
  reasonable wall time on the existing development workstation; one
  bounded normal-lane training pass that fits inside the existing GPU
  budget; synthesizer and distillation must be deterministic under
  seed control.
- **Constraints**: no architecture changes, no new builds beyond the
  two target builds, no proprietary data in the open-source release
  artifact, no CUDA-only hardwiring in the student trainer.
- **Scale/Scope**: two builds, the focused corpus tile set, one
  bounded normal-lane training pass, one bounded synthesized-input
  generation, one bounded distillation pass, one bounded student
  training pass, one release artifact.

## Constitution Check

*GATE: Must pass before implementation. Re-check after each phase lands.*

- **Repo independence**: pass. All work stays under `wow-viewer/`.
- **Library-first**: pass. The synthesizer, distillation, and student
  trainer are scripts that consume the existing library surfaces
  (`WowViewer.Tool.Harvest`, `WowViewer.Tool.ValidationCapture`,
  `v18_models.py`, `v16_1_dataset.py`).
- **Real-data validation**: required for signoff. The focused build
  and the main-model training pass are real-data validated on the two
  staged builds under `output/tmp/wowarchive-clients/`. The
  synthesizer is validated on its own asset-audit report and
  deterministic-rerun evidence.
- **Residual model chain**: pass. The V18 model line is the
  per-signal V16.1 family. The student model is a separate, smaller
  model, not a shared-weight model; it trains on labeled synthesized
  data only.
- **Streaming-first dataset pipeline**: pass. The focused build
  reuses the existing V18 streaming build shape.
- **No untrusted client paths**: pass. Only staged client roots.
- **One phase at a time**: pass. Each sub-plan phase is independently
  validatable, and Plan A is finished and validated before Plan B
  starts.
- **Bring Your Own Data**: pass. The main model and the focused
  real-data corpus are not redistributed. Only the student and the
  labeled synthesized corpus are open-source release candidates.
- **Bite-sized plans**: pass. Each phase is 10 steps or fewer.
- **No new architectures**: pass. The V18 model line is the model
  owner. The student is a small new model, but it is the explicit
  release target, not a redesign of the main model.

## Project Structure

### Documentation (this feature)

```text
wow-viewer/specs/047-v18-distill-corpus-open-source-loop/
├── spec.md          # this spec
├── plan.md          # this file
└── tasks.md         # phase-by-phase task breakdown
```

### Source Code (repository root)

```text
wow-viewer/
├── data-harvester/
│   ├── scripts/
│   │   ├── build_v18_dataset.py             # existing; reused as the focused build entry
│   │   ├── build_focused_two_build_corpus.py # new; --builds 0_5_3_3368 3_3_5_12340 wrapper
│   │   ├── run_focused_capture_batch.py      # new; wow-viewer capture batch over focused index
│   │   ├── synthesize_v18_inputs.py          # new; procedural minimap-like input generator
│   │   ├── distill_v18_to_synthesized.py     # new; applies main model to synthesized inputs
│   │   └── train_v18_student.py              # new; small open-source student trainer
│   └── src/harvester/
│       ├── v18_dataset.py                    # existing; reused
│       ├── v18_synth_audit.py                # new; synthesized-input asset audit
│       ├── v18_distill_provenance.py         # new; per-label provenance manifest writer
│       └── v18_student_model.py              # new; student architecture
├── output/
│   └── datasets/
│       ├── v18/                              # focused two-build V18 stores
│       ├── synthesized/                      # asset-audit-clean synthesized inputs
│       └── distilled/                        # labeled synthesized corpus
└── models/
    ├── v18/normal/runs/                      # main model checkpoints (in-repo only)
    └── v18_student/runs/                     # student model checkpoints (release candidate)
```

**Structure decision**: This lane does not introduce a new C# project. The
new scripts are bounded to `wow-viewer/data-harvester/scripts/` and
`wow-viewer/data-harvester/src/harvester/`, with outputs in
`wow-viewer/output/datasets/` and `wow-viewer/models/`. The capture
batch reuses the existing `WowViewer.Tool.ValidationCapture capture-batch`
entrypoint from spec 012 / spec 025 Phase 2.

---

## Plan A — Distill Corpus

Plan A produces the teacher for the open-source release loop: a focused
two-build V18 corpus with renderer-truth object-mask coverage on every
accepted tile, and a trained main V18 model checkpoint on that corpus.

### Phase A1 — Focused two-build V18 build

Goal: a reproducible V18 build for `0_5_3_3368` and `3_3_5_12340` only,
without depending on the other four builds.

1. Add a focused-build wrapper script
   `build_focused_two_build_corpus.py` that calls the existing
   `build_v18_dataset.py` with `--builds 0_5_3_3368 3_3_5_12340` and
   validates the resulting two V18 stores.
2. Run the focused build on the two staged client roots under
   `output/tmp/wowarchive-clients/`.
3. Confirm `decoded_metadata.parquet`, `index.parquet`, and signal
   coverage all match the focused tile set with no leftovers from the
   other four builds.
4. Emit a focused-build evidence package: command, output roots,
   per-build signal coverage summary, and a hash of the index.
5. Document the focused-build operator path in the `wow-viewer/README.md`
   and `wow-viewer/data-harvester/README.md`.

Validation:

- The focused build produces two complete, valid V18 stores.
- Validation reports show decoded-metadata parity and signal coverage
  parity for the two builds.
- The other four builds are not required for the build to succeed.

### Phase A2 — Full-corpus renderer-truth object-mask capture

Goal: extend the renderer-truth object-mask coverage from the current
one-anchor-tile-per-build proof to every tile accepted into the
focused two-build corpus.

1. Generate a focused capture ledger from the focused-build index
   (per spec 012 / spec 025 Phase 2 ledger format).
2. Run `WowViewer.Tool.ValidationCapture capture-batch` against the
   focused ledger for both builds with batched tile execution.
3. Capture status is reported per tile (`captured`, `failed`,
   `skipped`) with deterministic rerun evidence.
4. Renderer-truth artifacts (`object_visibility_mask`,
   `no_object_minimap`) are written to the dataset root's
   `images/<tile>_*` paths.
5. A new V18 build step promotes renderer-truth object-mask coverage
   to a first-class V18 signal in `index.parquet` and
   `signal_validation.json`.

Validation:

- At least 90% of the focused-corpus tiles have renderer-truth
  object-mask artifacts.
- Tiles that fail capture are recorded explicitly with status, not
  silently treated as covered.
- The focused build re-run reflects the new coverage in its
  validation report.

### Phase A3 — Main V18 model training on the focused corpus

Goal: a bounded normal-lane training pass on the focused two-build
corpus, with renderer-truth object-mask signal consumed as a
first-class loss-weight input.

1. Reuse the existing V16.1 / V18 normal trainer
   (`train_v18.py` / `train_v16_1_normal.py`) without architecture
   changes.
2. Confirm the trainer reads the promoted renderer-truth
   object-mask signal from the focused V18 stores.
3. Run a bounded curated-pool training pass on the focused corpus
   (small scout pool plus bucket-aware sampling where applicable).
4. Save the best checkpoint to
   `models/v18/normal/runs/<run-name>/` as the named teacher
   checkpoint for Plan B.
5. Emit evidence: normal validation improvement, per-tile loss
   breakdown that distinguishes tiles with renderer-truth
   object-mask coverage from tiles without, and a frozen
   configuration snapshot.

Validation:

- The bounded pass converges within the existing convergence
  envelope.
- Evidence files match the existing V16.1 / V18 evidence contract.
- The teacher checkpoint is reproducible from the recorded config
  and seed.

---

## Plan B — Open-Source Release Loop

Plan B consumes the teacher from Plan A and produces an open-source
student model that has no proprietary data dependency.

### Phase B1 — Synthesized-input generation

Goal: a deterministic, asset-audit-clean synthesized-input
generator that emits `256x256x3` uint8 RGB minimap-like patches.

1. Add `synthesize_v18_inputs.py` with a seeded procedural
   generator (heightfield + albedo proxy + low-frequency
   terrain-like structure) that produces a fixed number of
   `256x256x3` uint8 RGB inputs.
2. Each input is content-addressed by hash and recorded in a
   manifest.
3. Each input is paired with an asset-audit entry: `procedural`,
   `public_domain`, or `permissive_license` with the specific
   license name. No input is marked as derived from copyrighted
   game client files.
4. Deterministic-rerun check: the same seed produces
   byte-identical inputs across reruns.
5. Format check: inputs match the real-minimap `256x256x3` uint8
   RGB contract so the trained main model can consume them
   without code changes.

Validation:

- 100% of generated inputs are asset-audit-clean.
- The same seed produces byte-identical inputs across reruns.
- The inputs render in the same format as real minimap tiles.

### Phase B2 — Distill the main model onto synthesized data

Goal: apply the trained main model to every synthesized input and
emit a labeled synthesized corpus with full provenance.

1. Add `distill_v18_to_synthesized.py` that consumes the
   teacher checkpoint from Plan A and the synthesized inputs
   from Phase B1.
2. The script emits a per-input label store with at least
   normal, height, holes, liquid footprint, and per-pixel
   object-mask predictions.
3. Every label row is linked via a provenance manifest to the
   exact synthesized input hash and the teacher checkpoint id.
4. Degenerate labels (e.g., all-zero normals, all-zero height)
   are excluded from the student's training pool with explicit
   reason logged in the distillation evidence.
5. Deterministic-rerun check: the same seed and inputs produce
   byte-identical labels across reruns.

Validation:

- The labeled synthesized corpus has full provenance linkage.
- Byte-identical reruns with the same seed and inputs.
- No label in the corpus references proprietary real-data
  ground truth.

### Phase B3 — Open-source student model training

Goal: a small open-source student model trained on the labeled
synthesized corpus, with a release-ready artifact.

1. Add `train_v18_student.py` and `v18_student_model.py` with
   a small student architecture (e.g., a small U-Net or
   ConvNeXt-tiny backbone; final architecture chosen in
   implementation and recorded in the release artifact).
2. The trainer consumes only the labeled synthesized corpus
   from Phase B2, with no access to V18 Zarr stores of real
   ground truth.
3. The trainer supports CPU and CUDA backends with no
   architecture-specific code paths.
4. Evaluation on a held-out slice of the labeled synthesized
   corpus is recorded in a permissively-licensed evidence
   file.
5. The release artifact is produced under
   `models/v18_student/release/<version>/` with: the model
   checkpoint, the training script, the architecture
   definition, the license (MIT or Apache 2.0), a clear
   statement of zero proprietary data dependency, and a
   provenance manifest.

Validation:

- The student trainer never reads proprietary real-data
  ground-truth arrays.
- The student trains successfully on CPU and on CUDA.
- The release artifact is self-contained and contains all
  required files plus the license and provenance statement.

---

## Complexity Tracking

No constitution violations are required.

The only deliberate tradeoffs are:

- The plan relies on the existing V16.1 / V18 model line rather
  than introducing a new architecture. This is intentional and
  matches the user's explicit "don't reinvent" direction.
- The plan narrows the harvest to two builds. The other four
  builds are not deleted; they are simply out of scope for this
  lane.
- The synthesizer is intentionally a procedural generator, not
  a learned generator. This keeps the synthesized inputs
  provably asset-free.
- The student model is intentionally a small new model, not a
  scaled-down copy of the main model. This is the explicit
  release target and is meant to be a separate, distributable
  artifact.
