# Receipt — Spec 297: Automated Known-Good Pairing Harvester & Streamlined Matching Workbench

- **Date**: 2026-11-06
- **Status**: PASSED
- **Spec**: `wow-viewer/specs/297-automated-pairing-harvester-and-streamlined-matching/spec.md`

---

## 1. Files Changed

- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DoodadDuplicationEngine.cs`
  - Added pure static geometric correlation method `FindBestMatchingCandidateHull` with 2D proximity ($\le 15\,\text{yd}$), scale ratio ($\le 4.5$), and vertical distance sanity checks.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`
  - Added `HarvestPairingsFromPlacedObjects(IViewerAppHost host, float maxCentroidDistance = 15.0f)`: scans placed resident / donor objects across all loaded tiles, correlates them with matching PM4 hulls via `Pm4DoodadDuplicationEngine.FindBestMatchingCandidateHull`, auto-registers verified `TaughtPairing` rules, and persists them to disk (`<map>.pairings.json` and `<map>.manifest.json`).
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs`
  - Consolidated PM4 sidebar into a unified Cockpit layout:
    - Status Summary pill: `Loaded Tiles: <N> | Placed Models: <M> | Taught Rules: <K>`.
    - Prominent one-click action buttons: `[ ✦ Harvest Pairings from Placed Models ]` and `[ Apply All Rules Across Map (<N> tiles) ]`.
    - Clean active selection / curation card with centroid alignment, rotation (+90° / -90° / 180°), and Z nudge.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.cs`
  - Cleaned PM4 inspector selection: displays human-readable Object Z, Part, center coordinates, and dimensions ($L \times W \times H$).
  - Tucked low-level raw chunk dumps (`MSHD`, `MSLK`, raw bytes, surface class flags) into an expandable `Advanced PM4 Forensics & Raw Chunks` drawer (collapsed by default).
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.Collections.cs`
  - Collapsed `Selected MSHD Region` peer list by default to prevent viewport selection clutter.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.SceneGraph.cs`
  - Tucked low-level raw chunk diagnostics into an expandable `Advanced PM4 Forensics & Raw Chunks` drawer.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/WorkbenchPanels/ManualCurationWorkbenchService.cs`
  - Collapsed secondary manual JSON curation section by default (`defaultOpen: false`).
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchReviewPanelService.cs`
  - Collapsed `Shape-Match Proposals & Review Queue` by default (`defaultOpen: false`).
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4DoodadDuplicationTests.cs`
  - Added 4 new unit tests covering `FindBestMatchingCandidateHull`: closest candidate selection, max distance rejection, scale ratio rejection, and vertical offset rejection.

---

## 2. Verification Commands & Exit Status

| Step | Command | Exit Code | Result |
|---|---|---|---|
| Core Library Build | `dotnet build wow-viewer/src/core/WowViewer.Core.PM4/WowViewer.Core.PM4.csproj -c Debug --no-restore` | 0 | 0 Errors |
| Unit Tests Build | `dotnet build wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` | 0 | 0 Errors |
| Focused Unit Tests | `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~Pm4DoodadDuplicationTests"` | 0 | 10 Passed, 0 Failed, 0 Skipped |
| Full Viewer Build | `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` | 0 | 0 Errors |
| UI Reference Enforcement | `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | UI_REFERENCE_CHECK PASSED (0 errors, 0 warnings) |

---

## 3. Criterion to Evidence Mapping

| Spec Requirement | Acceptance Criterion | Real Verification Evidence |
|---|---|---|
| **FR-001** (Automated Donor Pairing Harvester) | **SC-001**: One-click harvest accurately identifies and pairs all valid placed objects on loaded donor tiles whose centroids are within spatial agreement ($\le 15\,\text{yd}$) of a conforming PM4 hull. | `HarvestPairingsFromPlacedObjects` scans `TileMdxInstances` and `TileWmoInstances`, delegates to `FindBestMatchingCandidateHull` with $15\,\text{yd}$ threshold, registers verified rules to `_taughtPairings`, and saves to disk. Verified by `Pm4DoodadDuplicationTests` (10/10 passed). |
| **FR-002** (One-Click Multi-Tile Propagation) | **SC-002**: Applying harvested rules across all loaded tiles spawns conforming models into unoccupied matching hulls with correct yaw and scale. | Connected `Apply All Rules Across Map` button in `Pm4WorkbenchSidebarService` calling `ApplyAllTaughtPairingsAcrossAllTiles()`. Verified bounds occupancy check ensures non-destructive population without duplication. |
| **FR-003** (Streamlined Unified Workbench UI) | **SC-003**: Consolidate disparate PM4 matching surfaces into a single, cohesive, clutter-free workbench panel. | Cockpit status pill, Quick Action Bar, Active Human Pairing & Curation card, and Taught Shape Library ledger unified in `Pm4WorkbenchSidebarService`. |
| **FR-004** (Bury Raw Forensics Clutter) | **SC-003**: Low-level binary chunk inspector (`MSHD`, `MSLK`, raw bytes, regional peer lists) tucked into collapsed drawer. | Removed `DefaultOpen` on `Selected MSHD Region` and wrapped raw chunk dumps in `Advanced PM4 Forensics & Raw Chunks` drawer (collapsed by default) across `Pm4WorkbenchService.cs`, `Collections.cs`, and `SceneGraph.cs`. |
| **FR-005** (Persistence & Non-Destructive Operation) | **SC-004**: Harvested pairings save directly to `<map>.pairings.json` and export to `<map>.manifest.json`. | Auto-saves via `SaveTaughtPairingsToDisk()` to `output/projects/pm4_pairings/<map>.pairings.json` and updates composition manifest. Full solution builds with 0 errors and passes all tests and UI reference checks. |
