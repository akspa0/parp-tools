# Tasks: V18 Distill Corpus and Open-Source Release Loop

**Input**: Design documents from `/specs/047-v18-distill-corpus-open-source-loop/`

**Prerequisites**:
- `plan.md` (required)
- `spec.md` (required for user stories)

**Note**: The user stories in `spec.md` are organized into two plans
(Plan A — Distill Corpus, Plan B — Open-Source Release Loop). Tasks are
grouped by user story so each story is independently implementable and
testable. Plan A must finish and validate before Plan B starts. No more
than 10 tasks per phase per the constitution's bite-sized-plans rule.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- File paths reference real locations under `wow-viewer/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Wire the focused-build wrapper and the focused-capture
ledger into the existing V18 surface.

- [ ] T001 [P] Create `wow-viewer/data-harvester/scripts/build_focused_two_build_corpus.py` that wraps the existing `build_v18_dataset.py` with `--builds 0_5_3_3368 3_3_5_12340` and a focused-build evidence writer.
- [ ] T002 [P] Add `wow-viewer/data-harvester/src/harvester/v18_synth_audit.py` stub with the asset-audit entry dataclass and JSONL writer interface.
- [ ] T003 [P] Add `wow-viewer/data-harvester/src/harvester/v18_distill_provenance.py` stub with the per-label provenance manifest writer interface.
- [ ] T004 [P] Add `wow-viewer/output/datasets/synthesized/` and `wow-viewer/output/datasets/distilled/` placeholder directories plus a top-level `README.md` placeholder explaining the new dataset roots.
- [ ] T005 [P] Add `wow-viewer/models/v18_student/` placeholder directory plus a top-level `README.md` placeholder explaining the student release root.

**Checkpoint**: Setup done. The new script paths exist as stubs. Real implementation begins in the next phases.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Foundation that MUST be complete before any user story work begins.

- [ ] T006 Confirm the two staged client roots exist under `output/tmp/wowarchive-clients/0_5_3_3368/World of Warcraft/` and `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft/`. If either is missing, copy it from WoWArchive before continuing. Document the staged root paths in the focused-build evidence.
- [ ] T007 Confirm the existing V18 build pipeline (`WowViewer.Tool.Harvest` + `build_v18_dataset.py`) runs end-to-end on the focused two-build set in dry-run / preflight mode. If it does not, fix the focused-build wrapper before continuing.
- [ ] T008 Confirm the existing `WowViewer.Tool.ValidationCapture capture-batch` tool is buildable and discoverable from `build_v18_dataset.py capture-renderer-truth` (per spec 012 / spec 025 Phase 2). If it is not, fix the discovery seam before continuing.

**Checkpoint**: Foundation ready. The focused two-build set is staged, the V18 build pipeline runs in preflight, and the capture batch tool is discoverable. User story work can now begin in Plan A.

---

## Plan A — Distill Corpus

### Phase A1: User Story 1 — Focused two-build V18 build (Priority: P1) 🎯

**Goal**: One canonical V18 build for `0_5_3_3368` and `3_3_5_12340` only.

**Independent Test**: Run `build_focused_two_build_corpus.py` against the two staged roots and verify two complete, valid V18 stores with no post-build patch phase required.

#### Implementation for User Story 1

- [ ] T010 [US1] Implement `build_focused_two_build_corpus.py` body: invoke `build_v18_dataset.py build` with `--builds 0_5_3_3368 3_3_5_12340` and `--allow-zarr-write` and stream output to a focused-build evidence file.
- [ ] T011 [US1] Add focused-build evidence writer: emit `command.txt`, `output_roots.json`, per-build signal coverage summary, and a `selection_hash` of the focused index.
- [ ] T012 [US1] Run the focused build on the two staged client roots. Capture per-store validation reports and confirm `decoded_metadata.parquet` parity with `index.parquet` for both builds.
- [ ] T013 [US1] Add a focused-build operator section to `wow-viewer/README.md` and `wow-viewer/data-harvester/README.md` documenting the two-build canonical path.
- [ ] T014 [US1] Verify rerun: the same command on the same inputs produces byte-identical evidence hashes (modulo allowed build-state metadata).

**Checkpoint**: US1 done. The focused two-build V18 build is reproducible on the two staged client roots with no required post-build patch phase.

---

### Phase A2: User Story 2 — Full-corpus renderer-truth object-mask capture (Priority: P1)

**Goal**: Renderer-truth object-mask coverage on every accepted tile in the focused two-build corpus.

**Independent Test**: Run the focused capture batch and verify that for every accepted tile there is a renderer-truth object-mask artifact, with per-tile status reporting.

#### Implementation for User Story 2

- [ ] T015 [US2] Generate a focused capture ledger from the focused-build `index.parquet` (per spec 012 / spec 025 Phase 2 ledger format) and write it under the focused build evidence root.
- [ ] T016 [US2] Implement `run_focused_capture_batch.py` that invokes `WowViewer.Tool.ValidationCapture capture-batch` per build, with mode flags and resolution forwarded from the focused-build evidence.
- [ ] T017 [US2] Run the focused capture batch on both builds with batched tile execution. Record per-tile status (`captured`, `failed`, `skipped`) in a focused capture evidence file.
- [ ] T018 [US2] Add a V18 build step that promotes renderer-truth object-mask coverage to a first-class V18 signal in `index.parquet` and `signal_validation.json` (alongside the existing coarse `object_mask` family).
- [ ] T019 [US2] Re-run the focused build and confirm the validation report reflects the promoted renderer-truth object-mask coverage.
- [ ] T020 [US2] Verify coverage: at least 90% of focused-corpus tiles have renderer-truth object-mask artifacts. Tiles without coverage are recorded explicitly with status, not silently treated as covered.

**Checkpoint**: US2 done. The full focused two-build corpus has renderer-truth object-mask coverage as a first-class V18 signal.

---

### Phase A3: User Story 3 — Main V18 model training on the focused corpus (Priority: P1)

**Goal**: A bounded normal-lane training pass on the focused two-build corpus with renderer-truth object-mask signal consumed as a first-class loss-weight input.

**Independent Test**: Run a bounded normal-lane training pass and confirm convergence behavior plus evidence files matching the existing V16.1 / V18 evidence contract.

#### Implementation for User Story 3

- [ ] T021 [US3] Reuse the existing V16.1 / V18 normal trainer (`train_v18.py` / `train_v16_1_normal.py`) without architecture changes. Confirm the trainer reads the promoted renderer-truth object-mask signal from the focused V18 stores.
- [ ] T022 [US3] Run a bounded curated-pool training pass on the focused corpus (small scout pool, bucket-aware sampling where applicable) on the development GPU. Save the best checkpoint to `models/v18/normal/runs/v18_distill_focused_<run-name>/` as the named teacher for Plan B.
- [ ] T023 [US3] Emit evidence: normal validation improvement, per-tile loss breakdown distinguishing tiles with renderer-truth object-mask coverage from tiles without, and a frozen configuration snapshot.
- [ ] T024 [US3] Verify reproducibility: the same command, seed, and config reproduce the same teacher checkpoint hash and evidence.

**Checkpoint**: US3 done. The trained main V18 model is the named teacher for Plan B and is reproducible from a recorded config and seed.

---

## Plan B — Open-Source Release Loop

### Phase B1: User Story 4 — Synthesized-input generation (Priority: P1)

**Goal**: A deterministic, asset-audit-clean synthesized-input generator that emits `256x256x3` uint8 RGB minimap-like patches.

**Independent Test**: Run the synthesizer on a bounded seed and verify 100% of generated inputs are asset-audit-clean and the run is byte-identical across reruns.

#### Implementation for User Story 4

- [ ] T025 [US4] Implement `synthesize_v18_inputs.py` with a seeded procedural generator (heightfield + albedo proxy + low-frequency terrain-like structure) producing a fixed number of `256x256x3` uint8 RGB inputs.
- [ ] T026 [US4] Implement `v18_synth_audit.py` body: every generated input is paired with an asset-audit entry (`procedural`, `public_domain`, or `permissive_license` with the specific license name). No input is marked as derived from copyrighted game client files.
- [ ] T027 [US4] Add content-addressed manifest: every input is hashed and recorded in a JSONL manifest under `wow-viewer/output/datasets/synthesized/<run-name>/`.
- [ ] T028 [US4] Add deterministic-rerun check: same seed produces byte-identical inputs across reruns. Emit a `synth_determinism.json` evidence file.
- [ ] T029 [US4] Add format check: inputs match the real-minimap `256x256x3` uint8 RGB contract so the trained main model can consume them without code changes.

**Checkpoint**: US4 done. The synthesizer produces asset-audit-clean, deterministic, format-correct synthesized inputs.

---

### Phase B2: User Story 5 — Distill the main model onto synthesized data (Priority: P1)

**Goal**: Apply the trained main model to every synthesized input and emit a labeled synthesized corpus with full provenance.

**Independent Test**: Run distillation on a bounded synthesized dataset and verify the output is a per-input label store with full provenance linkage and byte-identical reruns.

#### Implementation for User Story 5

- [ ] T030 [US5] Implement `distill_v18_to_synthesized.py` that consumes the teacher checkpoint from Plan A and the synthesized inputs from Phase B1.
- [ ] T031 [US5] Emit a per-input label store with at least normal, height, holes, liquid footprint, and per-pixel object-mask predictions, written under `wow-viewer/output/datasets/distilled/<run-name>/`.
- [ ] T032 [US5] Implement `v18_distill_provenance.py` body: every label row is linked via a provenance manifest to the exact synthesized input hash and the teacher checkpoint id.
- [ ] T033 [US5] Add degenerate-label filter: all-zero normals, all-zero height, or otherwise unusable labels are excluded from the student's training pool with explicit reason logged in the distillation evidence.
- [ ] T034 [US5] Add deterministic-rerun check: same seed and inputs produce byte-identical labels across reruns. Emit a `distill_determinism.json` evidence file.

**Checkpoint**: US5 done. The labeled synthesized corpus has full provenance linkage and is reproducible from a recorded seed and teacher checkpoint.

---

### Phase B3: User Story 6 — Open-source student model training (Priority: P1)

**Goal**: A small open-source student model trained on the labeled synthesized corpus, with a release-ready artifact.

**Independent Test**: Train the student on a bounded labeled synthesized dataset and verify the release artifact is self-contained, permissively licensed, and never reads proprietary real-data ground truth.

#### Implementation for User Story 6

- [ ] T035 [US6] Implement `v18_student_model.py` with a small student architecture (small U-Net or ConvNeXt-tiny backbone; final choice recorded in the release artifact).
- [ ] T036 [US6] Implement `train_v18_student.py` that consumes only the labeled synthesized corpus from Phase B2, with no access to V18 Zarr stores of real ground truth.
- [ ] T037 [US6] Add backend support: trainer runs on CPU and on CUDA with no architecture-specific code paths. Add `--device cpu|cuda` selector.
- [ ] T038 [US6] Train the student on a bounded labeled synthesized dataset. Save the best checkpoint under `models/v18_student/runs/<run-name>/`.
- [ ] T039 [US6] Add evaluation: held-out slice of the labeled synthesized corpus is evaluated; metrics recorded in a permissively-licensed evidence file.
- [ ] T040 [US6] Produce the release artifact under `models/v18_student/release/<version>/`: model checkpoint, training script, architecture definition, license (MIT or Apache 2.0), zero-proprietary-data-dependency statement, and provenance manifest.

**Checkpoint**: US6 done. The student model is trained, evaluated, and packaged into a release-ready artifact that has no proprietary data dependency.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Wrap the lane and update continuity surfaces.

- [ ] T041 [P] Update `wow-viewer/README.md` and `wow-viewer/data-harvester/README.md` with the focused two-build path, the synthesize-distill-student loop, and the open-source student release section.
- [ ] T042 [P] Mark `wow-viewer/specs/010-v16-1-2-no-object-guidance/spec.md` Status as `Superseded by 011` (already done — verify) and add a `superseded_by` pointer.
- [ ] T043 [P] Mark `wow-viewer/specs/015-v16-1-2-height-derived-normal-refiner/spec.md` Status as `Superseded by 016 (refiner approach failed)` and add a `superseded_by` pointer.
- [ ] T044 [P] Mark `wow-viewer/specs/017-v16-1-4-combined-normal-height-model/spec.md` Status as `Draft — superseded by V18 distill corpus lane (spec 047)` and add a `superseded_by` pointer.
- [ ] T045 [P] Mark `wow-viewer/specs/022-v17-unified-normal-height-refiner/spec.md` Status as `Draft — superseded by V18 distill corpus lane (spec 047)` and add a `superseded_by` pointer.
- [ ] T046 [P] Mark `wow-viewer/specs/023-v17-1-global-minimap-signal-reconstruction/spec.md` Status as `Draft — superseded by V18 distill corpus lane (spec 047)` and add a `superseded_by` pointer.
- [ ] T047 [P] Update `gillijimproject_refactor/memory-bank/activeContext.md` to record spec 047 as the active V18 distill corpus + open-source release lane.
- [ ] T048 [P] Update `gillijimproject_refactor/memory-bank/progress.md` with the spec 047 entry under "New V18 planning lane" / "Validations" sections.
- [ ] T049 Add an architecture doc at `wow-viewer/docs/architecture/v18-distill-corpus-open-source-loop-2026-06-04.md` summarizing the lane, the focused two-build surface, and the synthesize-distill-student loop.

**Checkpoint**: Polish done. The lane is documented, the superseded drafts are marked, and the memory bank points future readers at the active owner.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — can start immediately.
- **Phase 2 (Foundational)**: Depends on Setup completion — BLOCKS all user stories.
- **Plan A user stories (US1 → US2 → US3)**: Must be done in order. US2 depends on US1's focused build. US3 depends on US2's capture coverage.
- **Plan B user stories (US4 → US5 → US6)**: Must be done in order. US5 depends on US4's synthesizer. US6 depends on US5's labeled corpus and US3's teacher checkpoint.
- **Phase N (Polish)**: Depends on all user stories being complete.

### Cross-Plan Dependencies

- **US3 depends on US2**: the teacher checkpoint can only be trained once the focused corpus has renderer-truth object-mask coverage.
- **US4 depends on US1 only**: the synthesizer does not need the teacher checkpoint.
- **US5 depends on US3 and US4**: the distillation pass needs both the teacher checkpoint and the synthesized inputs.
- **US6 depends on US5 only**: the student trainer only needs the labeled synthesized corpus.

### Within Each User Story

- Foundation stubs (T001–T005) must exist before user story code.
- Capture/status writing (T016–T017) before promotion (T018).
- Synthesizer (T025) before audit (T026), manifest (T027), determinism (T028), format check (T029).
- Distillation (T030) before labels (T031), provenance (T032), filter (T033), determinism (T034).
- Student model (T035) before trainer (T036), backend (T037), training (T038), evaluation (T039), release (T040).

### Parallel Opportunities

- T001, T002, T003, T004, T005 can run in parallel (different files).
- T042–T046 can run in parallel (different files, all status-header edits).
- T047, T048, T049 can run in parallel (different files, different layers of continuity).

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase A1 (US1): focused two-build V18 build.
4. **STOP and VALIDATE**: confirm the focused build produces two complete, valid V18 stores.
5. Demo / checkpoint before moving to US2.

### Incremental Delivery

1. Setup + Foundational → foundation ready.
2. US1 → focused two-build V18 corpus (MVP).
3. US2 → renderer-truth object-mask coverage on the focused corpus.
4. US3 → main V18 model teacher checkpoint.
5. US4 → synthesized-input generator.
6. US5 → labeled synthesized corpus via distillation.
7. US6 → open-source student model release artifact.
8. Phase N → documentation, memory bank, superseded markers.

### Why This Order

- The teacher must be trained before distillation. Distillation must run
  before student training. The student is the final release artifact.
- The synthesizer is independent of the teacher; it can be developed in
  parallel with the teacher-training slice if team capacity allows, but
  the spec orders them sequentially for clarity.

---

## Notes

- [P] tasks = different files, no dependencies.
- [Story] label maps each task to a specific user story for traceability.
- Each user story is independently completable and testable.
- Verify validation evidence before marking a phase complete.
- Commit after each task or logical group, with a concise "why" message.
- Stop at any checkpoint to validate the story independently.
- Avoid: vague tasks, same-file conflicts, cross-story dependencies that break independence.
