# Tasks: Terrain Brush Signature Classification

**Branch**: `132-terrain-brush-signature-classification` | **Date**: 2026-08-04 | **Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

## Phase Overview

| Phase | Story | Priority | Goal | Status |
|-------|-------|----------|------|--------|
| 1 | US1 | P1 | Three-tier classification (strong/normal/weak) | **In progress** |
| 2 | US2 | P2 | Nested weak signal detection | Pending |
| 3 | US3 | P2 | Brush-texture correlation | Pending |
| 4 | US4 | P2 | Cross-map fragment alignment | Pending |
| 5 | US5 | P2 | Pre-rescale boundary detection | Pending |
| 6 | US6 | P3 | Predictive model (texture from heightmap) | Pending |

## Dependency Graph

```
US1 ──> US2 ──> US3 ──> US4 ──> US5 ──> US6
```

US1 has no dependencies. US2 depends on US1's classification. US3 depends on US2's nested signals. US4/US5 can run in parallel after US3. US6 depends on all previous phases.

## Phase 1 — Three-tier classification (US1, P1)

**Story goal**: Every tile classified as strong, normal, or weak signal with published criteria.

**Independent test**: Load a map, run the three-tier classifier, and confirm tiles previously classified as "usable" now split into "strong" and "normal" with measurable criteria.

**Implementation**:

- [ ] T001 [P] Create `wow-viewer/data-harvester/src/harvester/v50/classify.py` — three-tier classification module with `SignalTier` enum, `compute_signal_tier()` function, and `STRONG_SIGNAL`/`NORMAL_SIGNAL`/`WEAK_SIGNAL` constants.
- [ ] T002 [P] Create `wow-viewer/data-harvester/scripts/v50_tile_classify.py` — CLI entry point that reads a V50 Zarr store, runs three-tier classification on every tile, and emits a CSV/JSON with per-tile tier, evidence, and counts.
- [ ] T003 [P] [US1] Update `wow-viewer/data-harvester/src/harvester/v50/tile_inventory.py` — add `signal_class` (strong/normal/weak) and `signal_class_evidence` fields to each inventory row, computed via `classify.py`. Keep existing `classification` field for backward compat.
- [ ] T004 [P] [US1] Update `wow-viewer/data-harvester/src/harvester/v50/tile_composite.py` — add green outline (`OUTLINE["normal_signal"]`) for normal-tier tiles, update legend text.
- [ ] T005 [P] [US1] Update `wow-viewer/data-harvester/scripts/v50_archaeology_from_npz.py` — emit three-tier classification in inventory rows and summary.
- [ ] T006 [P] [US1] Update `wow-viewer/data-harvester/scripts/build_v50_store_from_npz.py` — run `v50_tile_classify.py` as part of the archaeology pipeline.
- [ ] T007 [P] [US1] Create `wow-viewer/data-harvester/tests/v50/test_classify.py` — unit tests for `classify.py`: strong, normal, weak, edge cases, determinism.
- [ ] T008 [US1] Update `wow-viewer/data-harvester/scripts/v50_archaeology.py` — emit three-tier classification in summary (add `signal_class` to per-map counts).

**Checkpoint**: `dotest test` passes for classify module. Three-tier classification visible in inventory CSV.

## Phase 2 — Nested weak signal detection (US2, P2)

**Story goal**: Weak-signal tiles shown to contain multiple tiers of progressively weaker brush data.

**Independent test**: Select a weak-signal tile, run the nested-signal detector, and report how many distinct signal tiers exist and at what precision levels.

**Tasks**:

- [ ] T009 [P] Create `wow-viewer/data-harvester/src/harvester/v50/nested_signal.py` — nested signal detection module that quantizes height data at progressively coarser precision levels and counts surviving height levels.
- [ ] T010 [P] [US2] Create `wow-viewer/data-harvester/scripts/v50_nested_signal.py` — CLI entry point for nested signal detection.
- [ ] T011 [P] [US2] Create `wow-viewer/data-harvester/tests/v50/test_nested_signal.py` — unit tests for nested signal detection.
- [ ] T012 [US2] Validate on known weak-signal tiles from Expansion01.

**Checkpoint**: At least one weak-signal tile shown to contain multiple tiers.

## Phase 3 — Brush-texture correlation (US3, P2)

**Story goal**: Heightmap brush scars correlated with alpha-layer patterns, broken relationships identified.

**Independent test**: Select a tile, run the brush-scar correlator, and report the correlation score.

**Tasks**:

- [ ] T013 [P] Create `wow-viewer/data-harvester/src/harvester/v50/brush_correlate.py` — brush-texture correlation module with edge detection, ridge/valley finding, and correlation scoring.
- [ ] T014 [P] [US3] Create `wow-viewer/data-harvester/scripts/v50_brush_correlate.py` — CLI entry point for brush-texture correlation.
- [ ] T015 [P] [US3] Create `wow-viewer/data-harvester/tests/v50/test_brush_correlate.py` — unit tests.
- [ ] T016 [US3] Validate on DeadminesInstance vs Westfall.

**Checkpoint**: DeadminesInstance alpha masks shown to not match Westfall's current heightmap.

## Phase 4 — Cross-map fragment alignment (US4, P2)

**Story goal**: Copy-pasted terrain fragments detected across maps with rotation/mirror detection.

**Independent test**: Take a known DeadminesInstance tile, search Westfall, confirm rotation/mirror transform.

**Tasks**:

- [ ] T017 [P] Create `wow-viewer/data-harvester/src/harvester/v50/fragment_align.py` — fragment alignment module using phase correlation + template matching.
- [ ] T018 [P] [US4] Create `wow-viewer/data-harvester/scripts/v50_fragment_align.py` — CLI entry point.
- [ ] T019 [P] [US4] Create `wow-viewer/data-harvester/tests/v50/test_fragment_align.py` — unit tests.
- [ ] T020 [US4] Validate on DeadminesInstance alpha masks vs Westfall originals.

**Checkpoint**: DeadminesInstance fragment found in Westfall with correct rotation/mirror.

## Phase 5 — Pre-rescale boundary detection (US5, P2)

**Story goal**: 33.33% horizontal weak-signal roll detected marking the Nov 2001 rescale.

**Independent test**: Scan every tile in DeadminesInstance, report which tiles carry the 33.33% roll.

**Tasks**:

- [ ] T021 [P] Create `wow-viewer/data-harvester/src/harvester/v50/rescale_boundary.py` — rescale boundary detection module using horizontal signal discontinuity scanning.
- [ ] T022 [P] [US5] Create `wow-viewer/data-harvester/scripts/v50_rescale_boundary.py` — CLI entry point.
- [ ] T023 [P] [US5] Create `wow-viewer/data-harvester/tests/v50/test_rescale_boundary.py` — unit tests.
- [ ] T024 [US5] Build library of all pre-rescale tiles across all maps and builds.

**Checkpoint**: At least one DeadminesInstance tile confirmed to carry the 33.33% boundary.

## Phase 6 — Predictive model (US6, P3)

**Story goal**: Model that predicts alpha-layer patterns from heightmap shape.

**Independent test**: Train on intact tiles, test on broken-relationship tiles, measure prediction accuracy.

**Tasks**:

- [ ] T025 [P] Create `wow-viewer/data-harvester/src/harvester/v50/brush_model.py` — model architecture and training logic.
- [ ] T026 [P] [US6] Create `wow-viewer/data-harvester/scripts/v50_brush_model.py` — CLI entry point for training/inference.
- [ ] T027 [P] [US6] Create `wow-viewer/data-harvester/tests/v50/test_brush_model.py` — unit tests.
- [ ] T028 [US6] Train on tiles with intact brush-texture relationships.
- [ ] T029 [US6] Evaluate on tiles with broken relationships.

**Checkpoint**: Model predicts alpha patterns with >60% accuracy vs random baseline.

---

## Parallel execution opportunities

- T001–T002 (classify module + CLI) can be done in parallel with T003 (inventory update) — they touch different files.
- T004 (composite update) depends on T003 or T001 (needs the signal_class field name).
- T007 (tests) can be done in parallel with T003–T006.
- T008 (archaeology update) depends on T001/T002 (needs the classify CLI).
- Phases 2–6 each depend on the previous phase's classification output.

## MVP scope

Implement Phase 1 (US1) only. This is the P1 priority and the foundation for all subsequent phases. MVP is complete when:
1. `classify.py` module exists with `compute_signal_tier()` function
2. `v50_tile_classify.py` CLI runs against a V50 store and emits CSV/JSON
3. Inventory CSV includes `signal_class` column
4. Composite images show normal-tier tiles with green outline
5. Unit tests pass