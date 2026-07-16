# Tasks: Image-Only WDL Prior

- [x] T001 Document paired WDL, compact representative-corpus, and archive contracts in `specs/108-image-wdl-prior/`.
- [x] T002 [US1] Add RGB-only WDL predictor and exact target helpers in `data-harvester/src/harvester/spec103/`.
- [x] T003 [US1] Add trainer and archive-producing inference CLIs in `data-harvester/scripts/`.
- [x] T004 [US1] Add CPU tests for target mapping, RGB-only prediction, and checkpoint/archive contracts.
- [x] T005 [US2] Add explicit generated-prior loading to `scripts/infer_spec103_v7.py`.
- [x] T006 Validate CPU contracts and prepare the user-owned compact-corpus training command.
- [x] T007 [US1] Add a standalone minimap-PNG-to-lattice inference route with no paired-store input.
- [x] T008 [US1] Add a real-tile evaluator that separates RGB prediction from truth-only scoring.
- [ ] T009 [US1] User-run: train the WDL prior on `output/datasets/spec108/synthetic_varied_lighting_v1.zarr` with the complete `pattern=crater` holdout.
- [ ] T010 [US1] User-run: inspect row 192 synthetic WDL lattices and the paired-WDL OBJ visual review under `output/spec108_wdl_prior_synthetic_varied_crater_v2/`.
- [x] T011 [US2] Add `--generated-wdl-priors` to `data-harvester/scripts/train_spec103_v7.py`, with exact-store and selected-row gates.
- [ ] T012 [US2] User-run: generate `generated_wdl_all_rows.npz`, train V8 with it, infer held-out synthetic crater rows, and run `validate_spec103_labelfree.py`.
- [ ] T013 [US3] Deferred research: create full-map 0.5.3 irregular terrain-art motif evidence only for a later prefab-derived synthetic corpus; it does not block T009–T012.
