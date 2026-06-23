# Tasks: Scar Mask Segmentation Model

> Deprecated as primary task list (2026-06-23): Phase 1 exists as a diagnostic baseline. Phase 2/3 should not be implemented as the main path; use `076-full-map-fractal-brush-library` for corrected dataset and model-target planning.

**Input**: `wow-viewer/specs/075-scar-mask-segmentation/spec.md` and `plan.md`

## Phase 1: Dataset, Model, And Smoke Trainer

**Purpose**: Build the first useful single-output scar segmentation model.

- [x] T001 [US1] Create `wow-viewer/data-harvester/src/harvester/v21_scar_dataset.py` with `V21ScarMaskDataset` reading patched V18 Zarr `minimap_rgb` and deriving `scar_mask` from `alpha_256` layers `1,2,3` at threshold `0.05`.
- [x] T002 [US1] Create `wow-viewer/data-harvester/src/harvester/v21_scar_model.py` with `V21ScarMaskModel`, a single-output `(B,1,256,256)` logits model.
- [x] T003 [US1] Create `wow-viewer/data-harvester/scripts/train_v21_scar_mask.py` with CLI args for dataset dir, builds, batch size, epochs, `--max-steps`, `--val-max-steps`, threshold, layers, output dir, and device.
- [x] T004 [US1] Implement BCE-with-logits + soft Dice loss and metrics (`loss`, `bce`, `dice_loss`, `iou`, `f1`) in `train_v21_scar_mask.py`.
- [x] T005 [US1] Implement checkpoint, metrics JSON, and preview PNG output in `train_v21_scar_mask.py`.
- [x] T006 [P] [US1] Add `wow-viewer/data-harvester/src/harvester/test_v21_scar_mask.py` covering target construction, model output shape, and loss behavior.
- [x] T007 [US1] Run `uv run pytest src/harvester/test_v21_scar_mask.py` and fix failures.
- [x] T008 [US1] Run a real-data smoke command with `--max-steps 2 --val-max-steps 1` and verify checkpoint, metrics JSON, and preview PNG exist.
- [x] T009 [P] Add `wow-viewer/docs/architecture/v21-scar-mask-segmentation-2026-06-23.md` documenting model purpose, target definition, loss, validation command, and why exact-scar classification is out of scope.

**Checkpoint**: Phase 1 is complete when tests pass and the smoke run writes a model checkpoint and preview.

## Phase 2: Inference And Component Extraction

**Purpose**: Convert scar-mask predictions into component candidates for future scar-family retrieval.

- [ ] T010 [US2] Create `wow-viewer/data-harvester/scripts/infer_v21_scar_mask.py` that loads a checkpoint and writes probability masks.
- [ ] T011 [US2] Add connected-component extraction on thresholded predictions and write `predicted_scar_components.jsonl`.
- [ ] T012 [US2] Validate output component coordinates align to 074 catalog bbox coordinates.

## Phase 3: Follow-Up Retrieval Spec

**Purpose**: Define the next single-output model or retrieval path for scar-family assignment.

- [ ] T013 [US2] Write follow-up notes for scar-family retrieval using 074 `exact_patterns.jsonl` and `pattern_neighbors.jsonl`.

## Dependencies

- Phase 1 blocks Phase 2.
- Phase 2 blocks Phase 3.
- Do not train scar-family or layer-role models until Phase 1 is validated.
