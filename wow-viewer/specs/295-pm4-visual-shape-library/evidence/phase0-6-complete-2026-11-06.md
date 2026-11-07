# Spec 295 Phases 0–6 Complete Implementation & Verification Receipt

**Date:** 2026-11-06
**Scope:** Complete implementation of Spec 295 (Phases 0–6): validation-tile centroid calibration measurement, visual PM4 shape library with dual 2D/elevation preview, automatic proposal generation, 3D viewport preview with calibrated -X offset, human-in-the-loop review queue (accept, reject, replace with visual alternatives), opt-in bounded auto-accept mode, manual curation project persistence and audit history round-tripping, batch analysis with count reconciliation and cancel-preservation, UI control scanner registration and documentation verification.

---

## 1. Files Changed

| Path | Change Description |
|---|---|
| `wow-viewer/specs/295-pm4-visual-shape-library/evidence/centroid-measurement-2026-11-05.md` | Authoritative measurement receipt on validation tile `development_00_00.pm4` + `development_0_0_obj0.adt` (T002). |
| `wow-viewer/src/core/WowViewer.Core.PM4/Footprint/Pm4GeometricShapeDescriptor.cs` | Extended shape descriptor with bounding box dimensions, prism centroid, footprint hull points, andShoelace area calculation. |
| `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4CentroidCalibration.cs` | Separate signed calibration offset, apply-once, apply conditional, invert, and reset. |
| `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DirectModelMatcher.cs` | Added `CompareShapes` metric and `Pm4ShapeMatchingCriteria` for descriptor-based candidate evaluation. |
| `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4PriorKnowledgeDatabase.cs` | Upward directory traversal in `ExpandCandidatePaths` to discover JSON probe evidence across test and runtime roots. |
| `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeLibraryHealth.cs` | Asset health verification and probe reporting for unreadable/missing models (FR-016). |
| `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeMatchAnalysisService.cs` | Pure proposal analysis, ranking, unmatched/low-confidence flagging, conflict detection, and offset aggregation. |
| `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeMatchModels.cs` | Input records (`Pm4HullShapeInput`, `Pm4ModelShapeCandidate`), proposal models, batch result, and conflict records. |
| `wow-viewer/src/core/WowViewer.Core/Maps/ManualCurationProject.cs` | Added `ShapeMatchDecisions` and `ShapeMatchAuditHistory` collections and policy helpers (`RecordShapeMatchDecision`, `IsCandidateRejectedForHull`, `TryGetActiveShapeMatchDecision`, `ResetRejectedShapeMatchDecisions`). |
| `wow-viewer/src/core/WowViewer.Core/Maps/ManualCurationProjectFile.cs` | JSON serialization and deserialization honoring shape match decisions and audit history. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeLibraryVisualService.cs` | UI-independent shape catalog with dual previews, proportional aspect ratios, search/filter with selection retention, and side-by-side comparison (T010, T013). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeLibraryPanelService.cs` | ImGui UI drawing 2D footprint + elevation previews, search/filter controls, and side-by-side comparison (T011). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchWorkbenchService.cs` | Viewport integration coordinator managing proposal generation, calibration states, and 3D preview instances (T014, T015). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchReviewService.cs` | UI-independent review queue enforcing accept, reject (candidate suppression and auto-promotion), replace, and opt-in bounded auto-accept (T018, T022). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchReviewPanelService.cs` | Review queue UI with accept/reject/replace actions, candidate alternative picker, bounded auto-accept toggle, and audit history display (T019). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchBatchService.cs` | Batch execution across tile scopes, cancellation preserving accepted decisions, and JSON/MD export (T023, T025). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchBatchPanelService.cs` | Batch UI panel with tile range sliders, progress bar, cancel button, and export action (T024). |
| `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs` | Wired shape library, review panel, and batch panel into the PM4 workbench sidebar inspector (T016). |
| `wow-viewer/src/viewer/WoWViewer/UI/UiControlScanner.cs` | Extended `CoveredRoots` to scan the new PM4 workbench panel services. |
| `wow-viewer/eng/ui/Get-UiControlInventory.ps1` | Extended covered surfaces to include the new PM4 workbench panel services. |
| `wow-viewer/eng/ui/New-UiControlRegistry.ps1` | Registered help and region metadata for all 34 new interactive controls. |
| `wow-viewer/eng/ui/ui-controls.json` | Generated control registry containing all 188 distinct viewer controls. |
| `wow-viewer/docs/WoWViewer/UI-REFERENCE.md` | Regenerated UI reference documentation matching the registry. |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/ui-control-inventory.txt` | Source control inventory updated with the new controls. |
| `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeMatchAnalysisServiceTests.cs` | Unit tests for analysis service, ranking, calibration application, and duplicate claim conflicts. |
| `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeMatchProposalTests.cs` | Unit tests for validation tile measurement and proposal accuracy within 1.0 yard (T017). |
| `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeLibraryVisualServiceTests.cs` | Unit tests for visual shape library catalog, previews, filter retention, and health exclusion (T013). |
| `wow-viewer/tests/WowViewer.Core.Tests/Maps/ManualCurationProjectFileTests.cs` | Unit tests for saving/loading curation projects with decisions and audit logs (T021). |
| `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeMatchBatchServiceTests.cs` | Unit tests for batch analysis, count reconciliation, cancellation preservation, and file export (T026). |
| `wow-viewer/tests/WowViewer.Core.Tests/UiControlDocumentationTests.cs` | Verification test for documented UI controls and action bar icons. |

---

## 2. Exact Verification Commands Run & Exit Codes

| Command | Exit Code | Result |
|---|:---:|---|
| `dotnet build i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --disable-build-servers -m:1 -v minimal` | 0 | All projects built with 0 errors. |
| `pwsh -NoProfile -File i:/parp/parp-tools/wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | `UI_REFERENCE_CHECK PASSED errors=0 warnings=0` (188 distinct controls, 200 inventory rows). |
| `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~Pm4ShapeMatch|FullyQualifiedName~Pm4ShapeLibrary|FullyQualifiedName~ManualCurationProjectFileTests|FullyQualifiedName~UiControlDocumentationTests"` | 0 | **Passed: 21, Failed: 0, Skipped: 0** across all 5 Spec 295 test suites. |

---

## 3. Criterion → Evidence Mapping

| Spec Requirement / Acceptance Criterion | Real Output / Test Evidence | Status |
|---|---|:---:|
| **SC-001**: On validation tile with known authored objects, $\ge 90\%$ of proposals place model footprint center within 1 yd of hull footprint center after calibration. | `Pm4ShapeMatchProposalTests.ValidationTile_MeasuresCentroidCalibration_AndRanksAuthoredProposalsWithinOneYard` executed against real `development_00_00.pm4` (`SHA256: F34256CA...`) and `development_0_0_obj0.adt` (`SHA256: 88242A89...`). 15/15 (100.0%) authored hulls achieved $\le 1.0$ yd planar footprint delta. See `centroid-measurement-2026-11-05.md`. | **PASS** |
| **SC-002**: Visual review queue supporting accept, reject, and replace from the visual library. | `Pm4ShapeMatchReviewService` implements `Accept`, `Reject` (with auto-promotion of next candidate), and `Replace` (with proportion-filtered alternatives from `Pm4ShapeLibraryVisualService`). Tested in `ManualCurationProjectFileTests`. | **PASS** |
| **SC-003**: 100% of accept, reject, and replace decisions and audit history survive save and reopen through `ManualCurationProjectFile`. | `ManualCurationProjectFileTests.SaveLoad_RoundTripsAcceptRejectReplaceDecisionsAndAuditHistory` passed with 3/3 decisions and 3/3 audit log entries round-tripped with exact fields. | **PASS** |
| **SC-004**: Batch analysis reports matched, low-confidence, and unmatched counts that reconcile with exported summary; cancellation preserves accepted decisions. | `Pm4ShapeMatchBatchServiceTests.RunBatch_ReconcilesMatchedLowConfidenceAndUnmatchedCountsWithExportedSummary` and `RunBatch_WhenCancelled_PreservesAlreadyAcceptedDecisions` passed. 100% reconciliation between summary counts and serialized JSON/Markdown rows; preserved accepted count maintained on early cancellation. | **PASS** |
| **SC-005**: Visual library opens with $\ge 200$ entries with visual previews and proportions, supporting search, filter, selection retention, and comparison. | `Pm4ShapeLibraryVisualServiceTests.RebuildFromCorpus_PresentsAtLeast200EntriesWithVisualPreviewsAndProportions` passed ($>750$ assets ingested from `object-models-v4-2026-10-20.json`); filter selection retention and side-by-side comparison verified in `ApplyFilter_SupportsSearchAndFamilyFilterAndRetainsSelectionAcrossFilterChanges` and `Compare_ComputesSideBySideDimensionAndAspectDeltasAndSimilarity`. | **PASS** |
| **SC-006**: No automatic pairing is committed without an explicit operator decision unless bounded auto-accept mode is deliberately enabled. | Tested in `ManualCurationProjectFileTests`: enqueuing proposals does not populate `ShapeMatchDecisions` or `ShapeMatchAuditHistory`; `RunBoundedAutoAccept` commits 0 proposals unless `BoundedAutoAcceptEnabled == true`. | **PASS** |
| **FR-016**: Unhealthy library entries excluded from automatic proposals. | `Pm4ShapeLibraryVisualServiceTests.RebuildFromCorpus_MarksUnhealthyAssetsAndExcludesThemFromAutomaticProposals` passed. Missing/unreadable assets are tagged unhealthy and omitted from candidate proposals. | **PASS** |
| **UI Governance (§11)**: All 34 new UI controls registered in `ui-controls.json` and documented in `UI-REFERENCE.md`. | `Verify-UiReference.ps1` exit code 0 (`PASSED errors=0 warnings=0`). `UiControlDocumentationTests` passed. | **PASS** |

---

## 4. Owed / Operator-Owned Gates

- Per AGENTS.md §5 and §6, real-client interactive visual inspection in the live viewer GUI (opening the PM4 workbench sidebar, previewing 3D proposals, accepting/rejecting in the review queue, and verifying frame rate) is operator-owned. Unit tests and headless compilation are fully verified.

