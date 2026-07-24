# Implementation Plan: Minimap OBB Object Detector & Metadata Sidecar Generator (Spec 120)

**Branch**: `120-minimap-placement-retrieval` | **Date**: 2026-07-23 | **Refactored**: 2026-07-24 | **Spec**: [spec.md](spec.md)

## Summary

This feature trains an Oriented Bounding Box (OBB) object detector on `minimap_rgb_authored` tiles (256×256) to extract continuous center position $(x, y)$, width $w$, height $h$ (scale), orientation angle $\theta$, confidence, and coarse class (`wmo` vs `mdx/m2`). It then crops the detected OBBs, matches them against the Object Library (`embeddings.parquet` / `assets.parquet`), and serializes position, scale, and identity data into a metadata sidecar file (`.json` or `.parquet`).

## Technical Context

- **Language/Environment**: Python 3.11+ managed by `uv` under `wow-viewer/data-harvester/`. No C#.
- **Primary Dependencies**: PyTorch, PyArrow (`pyarrow.parquet`), NumPy, Zarr v3. Optional `ultralytics` / `timm` / `transformers` for OBB backbone.
- **Storage**: Reads v50 map store (`minimap_rgb_authored`, `placements.parquet`) read-only; writes derived artifacts (checkpoints, sidecar files) to `<output-root>/spec120/`.
- **Testing**: `pytest` under `data-harvester/tests/spec120/`. Ruff clean.

## Phased Delivery

### Phase 0 — OBB Label Contract & Dataset Builder

1. `harvester/spec120/obb_contract.py`: World-to-pixel coordinate mapping, OBB target format `[class_id, cx, cy, w, h, angle]`, sidecar schema definition.
2. `scripts/spec120_build_obb_dataset.py` + `harvester/spec120/obb_dataset.py`: Reads `placements.parquet` and `minimap_rgb_authored`, builds OBB training dataset. Dry-run-first (`--write` required).
3. `tests/spec120/test_obb_contract.py` + `test_obb_dataset.py`: World→pixel unit tests, OBB shape validation, spatial split leakage check.

### Phase 1 — OBB Minimap Object Detector Trainer (US1)

4. `harvester/spec120/obb_detector_model.py`: Lightweight OBB object detection network (conv/transformer encoder + OBB head predicting $cx, cy, w, h, \theta, conf, class$). Constructable from `base` width alone.
5. `scripts/spec120_train_obb_detector.py` + `harvester/spec120/obb_detector_train.py`: Dry-run-first trainer with OBB IoU loss, mAP@50 evaluation, spatial split. Reuses `harvester.v50.lr_schedule`.
6. `tests/spec120/test_obb_detector.py`: Model parameter count, forward pass shape, loss calculation, dry-run plan check.

### Phase 2 — Identity Retrieval & Metadata Sidecar Exporter (US2)

7. `harvester/spec120/sidecar_exporter.py`: Given detected OBBs, crops patches, computes similarity against `embeddings.parquet`, and serializes sidecar metadata (`.json`/`.parquet`).
8. `scripts/spec120_infer_sidecar.py`: CLI that accepts a loose minimap PNG or tile array and outputs the metadata sidecar.
9. `tests/spec120/test_sidecar_exporter.py`: Tests JSON/Parquet schema validity, continuous position/scale outputs, loose PNG execution.

### Phase 3 — VLM Crop Annotator & Quality Audit (US3)

10. `harvester/spec120/vlm_crop_annotator.py`: Unsloth / HuggingFace LoRA wrapper operating on $64\times 64$ detected OBB crops to produce rich natural language sidecar annotations.
11. `tests/spec120/test_vlm_annotator.py`: Tests crop patch formatting and mock VLM inference.

## Validation Gates

| Gate | Where | Criterion | Who runs |
|------|-------|-----------|----------|
| Leakage check | Phase 0 | `verified_violation_count=0` | assistant (read-only) |
| SC-001 mAP@50 | Phase 1 | mAP@50 ≥ 0.65, center MAE < 2.0px | USER (`--confirm-run`) |
| SC-002 Scale Error | Phase 1 | Scale extraction error ≤ 15% | USER (from run record) |
| SC-003 Sidecar Schema | Phase 2 | Output JSON/Parquet schema valid | assistant (dry-run) |
