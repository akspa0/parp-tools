# Tasks: Minimap OBB Object Detector & Metadata Sidecar Generator (Spec 120)

**Branch**: `120-minimap-placement-retrieval` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

## Phase 0 — OBB Label Contract & Dataset Builder

- [ ] **T001**: Implement `harvester/spec120/obb_contract.py` with world→pixel mapping functions, OBB target formatting `[class_id, cx, cy, w, h, angle]`, and sidecar metadata schema.
- [ ] **T002**: Implement `harvester/spec120/obb_dataset.py` to convert `placements.parquet` and `minimap_rgb_authored` into an OBB dataset.
- [ ] **T003**: Create `scripts/spec120_build_obb_dataset.py` dry-run-first CLI (`--write` required).
- [ ] **T004**: Add unit tests in `tests/spec120/test_obb_contract.py` and `test_obb_dataset.py` verifying world→pixel coordinate math and spatial split leakage check.

## Phase 1 — OBB Minimap Object Detector Trainer (US1)

- [ ] **T005**: Implement `harvester/spec120/obb_detector_model.py` containing `MinimapOBBDetector` network (conv/transformer encoder + OBB head predicting $cx, cy, w, h, \theta, conf, class$).
- [ ] **T006**: Implement `harvester/spec120/obb_detector_train.py` with OBB IoU loss, mAP@50 evaluation, and OneCycle LR schedule.
- [ ] **T007**: Create `scripts/spec120_train_obb_detector.py` dry-run-first CLI (`--confirm-run` required).
- [ ] **T008**: Add unit tests in `tests/spec120/test_obb_detector.py` verifying model parameter count, forward pass shape, loss calculation, and dry-run behavior.

## Phase 2 — Identity Retrieval & Metadata Sidecar Exporter (US2)

- [ ] **T009**: Implement `harvester/spec120/sidecar_exporter.py` to extract detected OBB crops, compute identity embeddings against `embeddings.parquet`, and serialize JSON/Parquet metadata sidecars.
- [ ] **T010**: Create `scripts/spec120_infer_sidecar.py` CLI that accepts a loose minimap PNG or tile array and outputs sidecar metadata.
- [ ] **T011**: Add unit tests in `tests/spec120/test_sidecar_exporter.py` verifying JSON/Parquet schema validation, continuous position/scale outputs, and loose PNG execution.

## Phase 3 — VLM Crop Annotator & Quality Audit (US3)

- [ ] **T012**: Implement `harvester/spec120/vlm_crop_annotator.py` operating on $64\times 64$ detected OBB crops for natural language metadata enrichment.
- [ ] **T013**: Add unit tests in `tests/spec120/test_vlm_annotator.py`.
