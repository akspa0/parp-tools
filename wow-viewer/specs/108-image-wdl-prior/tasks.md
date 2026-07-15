# Tasks: Image-Only WDL Prior

- [x] T001 Document paired WDL, compact representative-corpus, and archive contracts in `specs/108-image-wdl-prior/`.
- [x] T002 [US1] Add RGB-only WDL predictor and exact target helpers in `data-harvester/src/harvester/spec103/`.
- [x] T003 [US1] Add trainer and archive-producing inference CLIs in `data-harvester/scripts/`.
- [x] T004 [US1] Add CPU tests for target mapping, RGB-only prediction, and checkpoint/archive contracts.
- [x] T005 [US2] Add explicit generated-prior loading to `scripts/infer_spec103_v7.py`.
- [x] T006 Validate CPU contracts and prepare the user-owned compact-corpus training command.
- [x] T007 [US1] Add a standalone minimap-PNG-to-lattice inference route with no paired-store input.
- [x] T008 [US1] Add a real-tile evaluator that separates RGB prediction from truth-only scoring.
- [ ] T009 User-run: train on selected representative patterns with a group holdout.
- [ ] T010 User-run: inspect one real-tile lattice report and repeat its exported PNG through standalone inference.
- [ ] T011 User-run: evaluate generated-prior V8 output through the label-free harness.
