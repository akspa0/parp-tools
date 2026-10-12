# Task Checklist: Spec 264

Receipt: [`evidence/receipt-spec264.md`](file:///I:/parp/parp-tools/wow-viewer/specs/264-development-ground-truth-shadow-height-and-object-placement/evidence/receipt-spec264.md)

## Phase 1: Authentic Development Ground Truth Extractor
- [x] **T001**: Implement `DevelopmentGroundTruthExtractor` in `wow-viewer/data-harvester/src/harvester/v60/development_ground_truth.py` to extract `height_257`, minimap RGB, and `_obj0` placements (MDDF/MODF) from authentic `original_development` ADTs. (Receipt: `receipt-spec264.md` §1.1)
- [x] **T002**: Create unit tests in `wow-viewer/data-harvester/tests/v60/test_development_ground_truth.py` verifying extraction on sculpted tiles (`development_16_33`, `development_16_38`, `development_15_34`). (Receipt: `receipt-spec264.md` §3.1)

## Phase 2: Calibrated Shadow-to-Height Model
- [x] **T003**: Implement `ShadowHeightCalibrator` in `wow-viewer/data-harvester/src/harvester/v60/shadow_height_calibrator.py` mapping bare terrain residual shadows to real world-space elevations in yards. (Receipt: `receipt-spec264.md` §1.2)
- [x] **T004**: Create unit tests in `wow-viewer/data-harvester/tests/v60/test_shadow_height_calibrator.py` verifying calibration accuracy (MAE $\le 12.0$ yds, $R^2 \ge 0.60$). (Receipt: `receipt-spec264.md` §3.1)

## Phase 3: Building Foundation Plateau Carver
- [x] **T005**: Implement `BuildingFoundationCarver` in `wow-viewer/data-harvester/src/harvester/v60/building_foundation_carver.py` to carve leveled foundation plateaus under building footprints. (Receipt: `receipt-spec264.md` §1.3)
- [x] **T006**: Create unit tests in `wow-viewer/data-harvester/tests/v60/test_building_foundation_carver.py` verifying foundation slope leveling and smooth perimeter transitions. (Receipt: `receipt-spec264.md` §3.1)

## Phase 4: Pipeline Integration & 3D Scene Materialization
- [x] **T007**: Update `v60_reconstruct_minimap.py` to integrate calibrated height recovery, foundation carving, and side-by-side ground truth mesh export. (Receipt: `receipt-spec264.md` §1.4, §3.2)
- [x] **T008**: Extend `mesh_exporter.py` to support exporting placed 3D object bounding boxes/markers alongside terrain meshes in `.glb` and `.obj`. (Receipt: `receipt-spec264.md` §1.4)

## Phase 5: Verification & Receipts
- [x] **T009**: Run end-to-end reconstruction on authentic development tiles (`development_16_33`, `development_16_38`, `development_15_34`, `development_16_35`, `development_20_38`, `development_26_34`), generating 3D `.glb` meshes and comparative visual diagnostics. (Receipt: `receipt-spec264.md` §3.2)
- [x] **T010**: Generate verification receipt in `wow-viewer/specs/264-development-ground-truth-shadow-height-and-object-placement/evidence/receipt-spec264.md` meeting all ACs, and update `STATUS.md` and `activeContext.md`. (Receipt: `receipt-spec264.md`)
