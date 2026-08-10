# Tasks: V7-Inspired Clean-Signal Terrain Reconstruction

## Phase 1: Setup

- [ ] T001 Record the v7 clean-signal feature identity and artifact paths in `wow-viewer/.specify/feature.json` and `wow-viewer/specs/139-v7-clean-signal-reconstruction/`.
- [x] T002 [P] Add the v7 clean-signal schema contract to `wow-viewer/specs/139-v7-clean-signal-reconstruction/contracts/v7-clean-signal.schema.md` and keep it aligned with the model/data code.
- [x] T003 [P] Add the clean-signal package exports in `wow-viewer/data-harvester/src/harvester/v60/__init__.py` without changing historical v50/spec103 imports.

## Phase 2: Foundational contracts

- [x] T004 Implement the four-channel observation validator in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_inputs.py`, including shape, range, finite-value, confidence-status, and forbidden-array checks.
- [x] T005 [P] Implement deterministic relative-height coarse/detail decomposition in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_targets.py` with a versioned low-pass kernel/cutoff and range-floor semantics.
- [x] T006 [P] Add corpus manifest and row-hash validation in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_corpus.py`, including source-group leakage and within-family/held-out-family split contracts.
- [x] T007 Add focused input/target/manifest fixtures in `wow-viewer/data-harvester/tests/v60/test_clean_signal_contract.py` for accepted, missing-confidence, textured-rejected, stale, malformed, and forbidden-signal rows.
- [x] T008 Add deterministic reproducibility tests in `wow-viewer/data-harvester/tests/v60/test_clean_signal_targets.py` for target decomposition and recomposition.
- [x] T009 Add the dry-run validator entrypoint in `wow-viewer/data-harvester/scripts/v60_validate_clean_signal_corpus.py` and fail closed on any contract violation.

## Phase 3: User Story 1 — Clean v7-style model (P1)

**Goal**: Reproduce the v7 coarse/detail structure with image-only clean signals.

**Independent test**: A tiny CPU fixture accepts exactly four input channels, emits coarse/detail/
height outputs, and completes forward/backward without reading forbidden arrays.

- [x] T010 [P] [US1] Define the shared encoder feature adapter and two-head output contract in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_model.py`.
- [x] T011 [US1] Adapt `pyramid_cnn` features to the v7-style coarse/detail decoder in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_model.py`.
- [x] T012 [P] [US1] Adapt `segformer_b0` features to the same v7-style coarse/detail decoder in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_model.py`.
- [x] T013 [P] [US1] Adapt `unet_lite_v2` as the low-capacity control under the same output contract in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_model.py`.
- [x] T014 [US1] Add forward/backward, parameter-count, output-range, and forbidden-input tests in `wow-viewer/data-harvester/tests/v60/test_clean_signal_model.py`.
- [x] T015 [US1] Add model identity serialization and checkpoint reconstruction tests in `wow-viewer/data-harvester/tests/v60/test_clean_signal_model.py`.

## Phase 4: User Story 2 — Synthetic clean observation guidance (P1)

**Goal**: Build exact, varied, reproducible training rows under the deployment-safe observation
contract.

**Independent test**: Two builds from the same synthesis configuration produce byte/hash-identical
rows and visual review identifies all required families and cross-tile continuity.

- [x] T016 [P] [US2] Implement clean observation packaging from the authoritative C# synthetic observation in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_inputs.py`, deriving only luma, gradients, and declared confidence.
- [x] T017 [US2] Implement the synthetic clean-signal corpus builder in `wow-viewer/data-harvester/scripts/v60_build_clean_signal_corpus.py`, preserving terrain, observation, albedo, confidence, and split provenance.
- [x] T018 [P] [US2] Add family/variant/cross-tile visual review in `wow-viewer/data-harvester/scripts/v60_visualize_clean_signal.py`.
- [x] T019 [US2] Add corpus validation and reproducibility tests in `wow-viewer/data-harvester/tests/v60/test_clean_signal_corpus.py`.
- [ ] T020 [US2] **USER RUNS** the synthetic corpus build and visual review after the dry-run command reports the expected source and output contracts.

## Phase 5: User Story 3 — Architecture and v7 guidance bakeoff (P1)

**Goal**: Determine whether v7's loss-side structural bias, rather than its leaked inputs, closes the
current baseline gap.

**Independent test**: A fixed split and model can run parity and structural loss profiles with
per-component and per-family metrics recorded under one report schema.

- [x] T021 [P] [US3] Implement parity and structural loss components in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_losses.py` for point, gradient, frequency, Laplacian, edge, transition, border, and LF/HF bands.
- [x] T022 [US3] Add zero/identity, smoothing-penalty, differentiability, and component-isolation tests in `wow-viewer/data-harvester/tests/v60/test_clean_signal_losses.py`.
- [x] T023 [US3] Implement the shared trainer, evaluator, best-checkpoint selection, and per-signal report in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_train.py`.
- [ ] T024 [US3] Add the PowerShell-ready dry-run/user-run CLI and fresh-output refusal in `wow-viewer/data-harvester/scripts/v60_train_clean_signal.py`.
- [x] T025 [US3] Add report tests proving identical splits across architectures/losses and independent family/bucket metrics in `wow-viewer/data-harvester/tests/v60/test_clean_signal_train.py`.
- [ ] T026 [US3] **USER RUNS** the within-family parity/structural matrix, then the complete-family gate for the best cells; Codex does not launch training.

## Phase 6: User Story 4 — Real albedo-normalized transfer (P2)

**Goal**: Prove that a synthetic-trained checkpoint can consume accepted arbitrary real minimaps
without WDL or target-derived inputs.

**Independent test**: The transfer command produces outputs and a zero-forbidden-read audit for
accepted rows and refuses rejected/quarantined rows.

- [ ] T027 [US4] Connect the versioned albedo-normalized observation artifact and confidence status in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_transfer.py`.
- [ ] T028 [US4] Implement image-only checkpoint loading and forbidden-signal auditing in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_transfer.py`.
- [ ] T029 [P] [US4] Implement transfer metrics, visual sheets, and hold/diagnose/expand decision output in `wow-viewer/data-harvester/src/harvester/v60/clean_signal_transfer.py`.
- [ ] T030 [US4] Add the PowerShell-ready transfer CLI in `wow-viewer/data-harvester/scripts/v60_transfer_clean_signal.py`.
- [ ] T031 [US4] Add accepted/rejected/quarantined and forbidden-read tests in `wow-viewer/data-harvester/tests/v60/test_clean_signal_transfer.py`.
- [ ] T032 [US4] **USER RUNS** the tiny explicit 0.x/1.x transfer after the synthetic promotion gate; no later-era expansion is included.

## Phase 7: Polish and cross-cutting validation

- [ ] T033 [P] Add schema/report validation fixtures in `wow-viewer/data-harvester/tests/v60/test_clean_signal_reports.py`.
- [ ] T034 [P] Add CLI help/dry-run smoke tests in `wow-viewer/data-harvester/tests/v60/test_clean_signal_cli.py`.
- [ ] T035 Run focused `ruff`, `py_compile`, and v60 pytest checks and record results in `wow-viewer/specs/139-v7-clean-signal-reconstruction/quickstart.md`.
- [ ] T036 Update `wow-viewer/memory-bank/activeContext.md`, `wow-viewer/memory-bank/progress.md`, and `wow-viewer/memory-bank/workstream-terrain-ml.md` with the implemented contract and promotion state.

## Dependencies

```text
T001-T009 foundational contracts
    -> T010-T015 clean model contract
    -> T016-T020 synthetic corpus
    -> T021-T026 architecture/loss bakeoff
    -> T027-T032 real transfer
    -> T033-T036 polish and continuity
```

The synthetic corpus and model adapter can proceed in parallel after T004-T009. Real transfer is
blocked until the structural bakeoff has a recorded promotion decision.

## Parallel execution opportunities

- T005, T006, and T007 can proceed in parallel after the input contract is written.
- T011, T012, and T013 can proceed in parallel once T010 fixes the shared decoder contract.
- T016, T018, and T019 can proceed in parallel once the corpus manifest fields are frozen.
- T021 and T022 can proceed in parallel after the loss names and tensor shapes are fixed.
- T027 and T029 can proceed in parallel after the transfer report schema is fixed.

## MVP strategy

1. Complete T004-T015 and prove the four-channel model contract on CPU.
2. Build/review the small clean synthetic corpus (T016-T020).
3. Run only the parity vs structural matrix on `pyramid_cnn` and the U-Net control (T021-T026).
4. Add SegFormer and real transfer only after the structural result is understood.
