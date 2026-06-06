# Tasks: V18 Focused Two-Build Terrain Reconstruction System

**Input**: Design documents from `/specs/047-v18-distill-corpus-open-source-loop/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md, contracts/

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no incomplete dependency)
- **[Story]**: Which user story this belongs to

---

## Phase 1: Design Closure

**Purpose**: turn the old draft into the final owner design before more script
changes land.

- [x] T001 [P] Rewrite [spec.md](/I:/parp/parp-tools/wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md) around the final V18 terrain reconstruction system: focused corpus, curation-first training, two independent terrain models, and quilt-level downstream reconstruction.
- [x] T002 [P] Rewrite [plan.md](/I:/parp/parp-tools/wow-viewer/specs/047-v18-distill-corpus-open-source-loop/plan.md) in Spec Kit form with final technical context and implementation phases.
- [x] T003 [P] Generate [research.md](/I:/parp/parp-tools/wow-viewer/specs/047-v18-distill-corpus-open-source-loop/research.md), [data-model.md](/I:/parp/parp-tools/wow-viewer/specs/047-v18-distill-corpus-open-source-loop/data-model.md), [quickstart.md](/I:/parp/parp-tools/wow-viewer/specs/047-v18-distill-corpus-open-source-loop/quickstart.md), and `contracts/`.

**Checkpoint**: `047` is the final V18 owner design, not another interim draft.

---

## Phase 2: Focused Corpus and Curation (Priority: P1)

**Goal**: make the two-build V18 corpus easy to curate and reuse.

**Independent Test**: generate a focused V18 curation manifest using only
`0_5_3_3368` and `3_3_5_12340`.

- [x] T010 [US1] Add [build_v18_curation_manifest.py](/I:/parp/parp-tools/wow-viewer/data-harvester/scripts/build_v18_curation_manifest.py) as a focused wrapper over the existing curation builder with V18 dataset defaults.
- [x] T011 [US1] Default that wrapper to `wow-viewer/output/datasets/v18/`, builds `0_5_3_3368` + `3_3_5_12340`, and a V18 curation output root under `wow-viewer/output/datasets/v18/curation/`.
- [x] T012 [US1] Document the focused curation command surface in [quickstart.md](/I:/parp/parp-tools/wow-viewer/specs/047-v18-distill-corpus-open-source-loop/quickstart.md) and [README.md](/I:/parp/parp-tools/wow-viewer/data-harvester/README.md).
- [x] T013 [US2] Reject focused rows whose surviving trainable terrain is too small, even when the wipeout comes from liquid-hidden terrain rather than only WMO loss gates.

**Checkpoint**: operators can generate a focused V18 manifest without falling
back to V16 paths or six-build commands.

---

## Phase 3: Focused Training Surface (Priority: P1)

**Goal**: make the V18 height and normal runs easy to launch against the
focused curated corpus.

**Independent Test**: launch focused `height` and `normal` training commands
through a V18-specific wrapper with no manual V16 dataset path wiring.

- [x] T020 [US2] Add [train_v18_focus.py](/I:/parp/parp-tools/wow-viewer/data-harvester/scripts/train_v18_focus.py) as a focused wrapper over [train_v18.py](/I:/parp/parp-tools/wow-viewer/data-harvester/scripts/train_v18.py).
- [x] T021 [US2] Default that wrapper to `wow-viewer/output/datasets/v18/`, builds `0_5_3_3368` + `3_3_5_12340`, and latest focused `kept_tiles.parquet` when `--curation-manifest` is omitted.
- [x] T022 [US2] Keep the actual model tasks independent: `height` and `normal` remain separate runs, separate checkpoints, and separate commands.
- [x] T023 [US2] Document the focused training command surface in [quickstart.md](/I:/parp/parp-tools/wow-viewer/specs/047-v18-distill-corpus-open-source-loop/quickstart.md) and [README.md](/I:/parp/parp-tools/wow-viewer/data-harvester/README.md).
- [x] T024 [US3] Mask focused `height` and `normal` losses to terrain-valid regions so liquid-hidden/object-hidden terrain no longer dominates optimization.
- [x] T025 [US3] Default focused training toward the observed 8 GB lane via startup batch autotune instead of the earlier smoke-budget operator defaults.
- [x] T026 [US3] Default focused two-build training to strict near-equal per-build sampling so skewed manifests cannot silently run full unbalanced epochs.
- [x] T027 [US3] Restore WMO roof/top-geometry participation in terrain-valid masking and focused preview evidence so active height/normal runs do not regress to basement-only object masking.

**Checkpoint**: the focused V18 training lane is easy to launch and hard to
misconfigure.

---

## Phase 4: Terrain Reconstruction Follow-Through (Priority: P2)

**Goal**: keep the model outputs aimed at stitched terrain reconstruction, not
isolated tile previews only.

**Independent Test**: define the quilt/inference contract and the artifacts it
must carry for later ADT writeback.

- [x] T030 [US5] Record the quilt-level terrain reconstruction contract in [data-model.md](/I:/parp/parp-tools/wow-viewer/specs/047-v18-distill-corpus-open-source-loop/data-model.md) and `contracts/`.
- [x] T031 [US5] Keep [spec.md](/I:/parp/parp-tools/wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md) explicit that post-model stitching owns cross-tile continuity.
- [x] T032 [US5] Leave actual ADT writeback implementation to a later slice instead of silently smuggling it into the training script work.

**Checkpoint**: the model lane stays honest about what is trained now and what
is reconstructed later.

---

## Phase N: Validation and Continuity

**Purpose**: close the loop on code, docs, and memory-bank continuity.

- [x] T040 [P] Validate new/updated Python surfaces with `uv run python -m py_compile`.
- [x] T041 [P] Sync [v18-distill-corpus-open-source-loop-2026-06-04.md](/I:/parp/parp-tools/wow-viewer/docs/architecture/v18-distill-corpus-open-source-loop-2026-06-04.md) to the final design contract.
- [x] T042 [P] Sync [activeContext.md](/I:/parp/parp-tools/gillijimproject_refactor/memory-bank/activeContext.md) and [progress.md](/I:/parp/parp-tools/gillijimproject_refactor/memory-bank/progress.md) after the wrapper scripts land.

**Checkpoint**: future sessions inherit the final focused V18 design and the
actual operator entrypoints.
