# Tasks — Spec 295 Visual PM4 Shape Library and Human-in-the-Loop Pairing

**Status:** Draft 2026-11-05. No task checked without a §9.2 receipt.
**Spec:** [`spec.md`](spec.md) · **Plan:** [`plan.md`](plan.md) · **Checklist:** [`checklists/requirements.md`](checklists/requirements.md)

## Phase 1 — Setup and measurement

- [x] T001 [P] Add a validation-tile measurement harness that reports the signed X/Y/Z offset between each proposed model footprint center and its hull footprint center in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeMatchAnalysisService.cs`. Receipt: [`evidence/phase0-1-core-2026-11-05.md`](evidence/phase0-1-core-2026-11-05.md).
- [ ] T002 [P] Record the measured -X offset and the validation tile/client fingerprint in `wow-viewer/specs/295-pm4-visual-shape-library/evidence/centroid-measurement-2026-11-05.md`.
- [x] T003 Define `Pm4CentroidCalibration` (signed offset, source note, reset, apply-once semantics) in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4CentroidCalibration.cs`. Receipt: [`evidence/phase0-1-core-2026-11-05.md`](evidence/phase0-1-core-2026-11-05.md).
- [x] T004 Add unit tests for calibration apply, idempotence, sign, and reset in `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeMatchAnalysisServiceTests.cs`. Receipt: [`evidence/phase0-1-core-2026-11-05.md`](evidence/phase0-1-core-2026-11-05.md).

## Phase 2 — Foundational core analysis

- [x] T005 Define `Pm4ShapeCandidate`, `Pm4ShapeMatchProposal`, and `Pm4ShapeMatchDecision` records in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeMatchModels.cs`. Receipt: [`evidence/phase0-1-core-2026-11-05.md`](evidence/phase0-1-core-2026-11-05.md).
- [x] T006 Implement `Pm4ShapeMatchAnalysisService` with ranked candidates, confidence, unmatched/low-confidence classification, and calibration application in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeMatchAnalysisService.cs`. Receipt: [`evidence/phase0-1-core-2026-11-05.md`](evidence/phase0-1-core-2026-11-05.md).
- [x] T007 Add conflict detection for multiple hulls claiming one object in `Pm4ShapeMatchAnalysisService`. Receipt: [`evidence/phase0-1-core-2026-11-05.md`](evidence/phase0-1-core-2026-11-05.md).
- [x] T008 Add focused tests for ranking, threshold behavior, conflict detection, and degenerate hulls in `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeMatchAnalysisServiceTests.cs`. Receipt: [`evidence/phase0-1-core-2026-11-05.md`](evidence/phase0-1-core-2026-11-05.md).
- [x] T009 Add a shape-library health check that marks missing/unreadable assets in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeLibraryHealth.cs`. Receipt: [`evidence/phase0-1-core-2026-11-05.md`](evidence/phase0-1-core-2026-11-05.md).

## Phase 3 — User Story 1: Visual shape library (P1)

**Goal:** the operator browses a visual catalog with previews, proportions, search, filter, and compare.
**Independent test:** open the catalog, filter to a known family, compare two entries, and confirm the previews match the assets.

- [ ] T010 [US1] Implement `Pm4ShapeLibraryVisualService` that builds catalog entries with previews, proportions, asset paths, and health flags in `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeLibraryVisualService.cs`.
- [ ] T011 [US1] Add the visual catalog panel with search, filter, selection retention, and side-by-side compare in `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeLibraryPanelService.cs`.
- [ ] T012 [US1] Register the new controls in `wow-viewer/eng/ui/ui-controls.json` and regenerate `wow-viewer/docs/WoWViewer/UI-REFERENCE.md`.
- [ ] T013 [US1] Add focused tests for catalog filtering, selection retention, and unhealthy-entry exclusion in `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeLibraryVisualServiceTests.cs`.

## Phase 4 — User Story 2: Automatic proposals with corrected centroids (P1)

**Goal:** analysis proposes ranked objects for each hull and previews the aligned model with the calibrated -X correction.
**Independent test:** run analysis on a validation tile and confirm the top proposal is the authored object and the footprint centers coincide within tolerance.

- [ ] T014 [US2] Wire `Pm4ShapeMatchAnalysisService` into the PM4 workbench analysis action in `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchWorkbenchService.cs`.
- [ ] T015 [US2] Add the 3D proposal preview (candidate aligned to hull, calibration visible/resettable) in `Pm4ShapeMatchWorkbenchService`.
- [ ] T016 [US2] Add the calibration readout and reset control to the PM4 workbench sidebar in `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs`.
- [ ] T017 [US2] Add tests for proposal ranking and calibration application on the validation fixture in `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeMatchProposalTests.cs`.

## Phase 5 — User Story 3: Accept, reject, or replace (P1)

**Goal:** the operator reviews proposals and commits decisions; replacements come from the visual library.
**Independent test:** reject a proposal, replace it with an alternate, save, reopen, and confirm the decision and audit history.

- [ ] T018 [US3] Implement `Pm4ShapeMatchReviewService` with proposal queue, accept/reject/replace, and audit history in `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchReviewService.cs`.
- [ ] T019 [US3] Add the review panel with accept/reject/replace and visual candidate picker in `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchReviewPanelService.cs`.
- [ ] T020 [US3] Persist decisions and audit history through the existing curation project in `wow-viewer/src/core/WowViewer.Core/Maps/ManualCurationProject.cs` and its file store.
- [ ] T021 [US3] Add save/reopen round-trip tests for decisions and audit history in `wow-viewer/tests/WowViewer.Core.Tests/Maps/ManualCurationProjectFileTests.cs`.
- [ ] T022 [US3] Enforce the no-auto-commit gate and the explicit bounded auto-accept mode in `Pm4ShapeMatchReviewService`.

## Phase 6 — User Story 4: Batch analysis and audit (P2)

**Goal:** run analysis over a map or tile range, watch progress, cancel safely, and export a reconciled summary.
**Independent test:** run a whole-map analysis, cancel midway, confirm accepted decisions remain, and export a summary whose counts reconcile.

- [ ] T023 [US4] Implement `Pm4ShapeMatchBatchService` with scope, progress, cancellation, and summary counts in `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchBatchService.cs`.
- [ ] T024 [US4] Add the batch panel with progress, cancel, and summary in `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchBatchPanelService.cs`.
- [ ] T025 [US4] Add summary export with hull, proposal, score, and decision provenance in `Pm4ShapeMatchBatchService`.
- [ ] T026 [US4] Add tests for cancellation preservation and summary reconciliation in `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeMatchBatchServiceTests.cs`.

## Phase 7 — Polish and verification

- [ ] T027 Update `wow-viewer/specs/STATUS.md`, `wow-viewer/memory-bank/activeContext.md`, and `wow-viewer/memory-bank/progress.md` with the implemented state and owed operator gates.
- [ ] T028 Run `dotnet build wow-viewer/WoWViewer.slnx -c Debug`, focused tests, and `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1`; record exits in the phase receipts.
- [ ] T029 Record operator visual/runtime proof gates (validation tile, whole-map run, save/reopen) as owed, never inferred from build/tests.

## Dependencies

- Phase 1 (T001–T004) blocks Phase 2 and Phase 4: the calibration contract must exist before proposals are previewed.
- Phase 2 (T005–T009) blocks Phases 3–6: the proposal model and analysis service are shared.
- Phase 3 (T010–T013) blocks the replace flow in Phase 5.
- Phase 4 (T014–T017) blocks Phase 5: the review queue consumes proposals.
- Phase 5 (T018–T022) blocks Phase 6: batch analysis reuses the review decision model.
- Phase 7 runs after the selected implementation phases.

## Parallel opportunities

- T001/T002 (measurement) can run in parallel with T003/T004 (calibration contract) once the validation tile is chosen.
- T010/T011 (visual catalog service/panel) can run in parallel with T014/T015 (proposal preview) after Phase 2.
- T023/T024 (batch service/panel) can run in parallel with T025 (export) after Phase 5.

## MVP scope

User Story 1 + User Story 2: a visual catalog and calibrated automatic proposals with a 3D preview. This is independently demonstrable and fixes the two operator-reported failures (text-only library and -X centroid bias) before the review queue is added.
