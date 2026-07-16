# Tasks: Image-Only WDL Prior

- [x] T001 Document paired WDL, compact representative-corpus, and archive contracts in `specs/108-image-wdl-prior/`.
- [x] T002 [US1] Add RGB-only WDL predictor and exact target helpers in `data-harvester/src/harvester/spec103/`.
- [x] T003 [US1] Add trainer and archive-producing inference CLIs in `data-harvester/scripts/`.
- [x] T004 [US1] Add CPU tests for target mapping, RGB-only prediction, and checkpoint/archive contracts.
- [x] T005 [US2] Add explicit generated-prior loading to `scripts/infer_spec103_v7.py`.
- [x] T006 Validate CPU contracts and prepare the user-owned compact-corpus training command.
- [x] T007 [US1] Add a standalone minimap-PNG-to-lattice inference route with no paired-store input.
- [x] T008 [US1] Add a real-tile evaluator that separates RGB prediction from truth-only scoring.
- [ ] T009 [US1] Build `data-harvester/scripts/spec108_build_mixed_curriculum.py`: a 240-row mixed store (144 real 0.5.3 + 96 synthetic) with stable source groups and no duplicated lighting group across split.
- [ ] T010 [US3] Add fast real tile brush/paste descriptors in `data-harvester/src/harvester/` using one 16x16 cell signature pass; select irregular repeated motifs only, not regions or rectangular crops.
- [ ] T011 [US1] Add CPU tests proving map quotas, the <256 cap, mixed source provenance, group split isolation, and real motif selection to `data-harvester/tests/spec108/`.
- [ ] T012 [US1] User-run: build and inspect the mixed curriculum before any CUDA work.
- [ ] T013 [US1] User-run: train the WDL prior on the mixed store with a complete source-group holdout and inspect multi-row visual reports.
- [x] T014 [US2] Add `--generated-wdl-priors` to `data-harvester/scripts/train_spec103_v7.py`, with exact-store and selected-row gates.
- [ ] T015 [US2] User-run: generate mixed-corpus priors, train V8 with them, infer held-out mixed rows, and run `validate_spec103_labelfree.py`.
