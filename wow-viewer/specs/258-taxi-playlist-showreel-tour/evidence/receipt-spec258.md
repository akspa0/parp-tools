# Verification Receipt — Spec 258: Taxi Route Playlists & Dynamic Marketing Showreel Tour

**Date**: 2026-10-02  
**Branch**: `v0.6.0-dev`  
**Status**: COMPLETE (Code & Tests Verified; Awaiting Operator Smoke Test)

---

## 1. Files Changed

### Spec Documents
- `wow-viewer/specs/258-taxi-playlist-showreel-tour/spec.md`
- `wow-viewer/specs/258-taxi-playlist-showreel-tour/plan.md`
- `wow-viewer/specs/258-taxi-playlist-showreel-tour/tasks.md`
- `wow-viewer/specs/258-taxi-playlist-showreel-tour/evidence/receipt-spec258.md`
- `wow-viewer/specs/STATUS.md`
- `wow-viewer/memory-bank/activeContext.md`
- `wow-viewer/memory-bank/progress.md`

### Core & Runtime
- `wow-viewer/src/core/WowViewer.Core.Runtime/Marketing/ShowreelModels.cs`

### Viewer Architecture & UI
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/TaxiAreaPoi/TaxiPlaylistModels.cs`
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/TaxiAreaPoi/TaxiPlaylistService.cs`
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Recording/ShowreelOverlayService.cs`
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/TaxiAreaPoi/TaxiPanelService.CaptureAutomation.cs`
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/TaxiAreaPoi/TaxiPanelService.Sidebars.cs`
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/StartupAutomation/StartupAutomationService.cs`
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs`
- `wow-viewer/src/viewer/WoWViewer/ViewerApp_Host.cs`
- `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`

### Tests
- `wow-viewer/tests/WowViewer.Core.Tests/MarketingCapture/ShowreelModelTests.cs`
- `wow-viewer/tests/WowViewer.Core.Tests/MarketingCapture/TaxiPlaylistModelTests.cs`

---

## 2. Verification Commands & Exit Status

### Command 1: Solution Build
```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```
- **Exit Code**: 0
- **Errors**: 0

### Command 2: Focused Spec 258 Tests
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~Showreel|FullyQualifiedName~TaxiPlaylist"
```
- **Exit Code**: 0
- **Summary**: Passed: 19, Failed: 0, Skipped: 0.

### Command 3: Full Test Suite Regression Check
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug
```
- **Exit Code**: 1 (baseline environmental failures unchanged: 9)
- **Summary**: Passed: 1,639 (+19 new passing tests), Failed: 9 (exact pre-existing baseline), Skipped: 1. Zero regressions.

---

## 3. Acceptance Criteria & Evidence Ledger

| Criterion | Implementation & Evidence | Status |
|---|---|---|
| **AC-1**: Continuous Playlist Recording | `TaxiPlaylistService.cs`: Manages multi-route sequence; auto-advances segments on arrival without interrupting `RecordingCoordinatorService`; cleanly completes video on final destination. Tested in `TaxiPlaylistModelTests.cs`. | PASS |
| **AC-2**: Interactive Playlist Controls | `TaxiPanelService.Sidebars.cs`: Collapsible header "Flight Playlist & Marketing Showreel", segment list with reorder (^/v), remove (X), clear, loop, 4-hop auto-chaining, play, and record buttons. | PASS |
| **AC-3**: Dynamic Zone Banners | `ShowreelOverlayService.cs`: Large, bold centered zone & subzone banner in the center of the frame (`Y = 0.42 * (H - h)`) with double gold borders, drop-shadow, and area discovery tag. | PASS |
| **AC-4**: Live Telemetry HUD | `ShowreelOverlayService.cs`: Real-time camera position (X, Y, Z), compass heading, ADT tile coordinate `[32 - X/533.3, 32 - Y/533.3]`, MCNK chunk `[0..15, 0..15]`, and flight progress fraction. Offsets above active tour presentation beats. | PASS |
| **AC-5**: Proximity Discovery & Engine Badges | `ShowreelOverlayService.cs`: Proximity scan for nearest `TaxiNode` within 350 yards; rotating technical badges highlighting engine accomplishments. | PASS |
| **AC-6**: Clean Viewport Gating | `ShowreelOverlayService.cs`, `ShowreelModels.cs`: Overlays strictly disabled by default; only render when tour feature is active for video recording or explicitly previewed in viewport. | PASS |
| **AC-7**: Startup CLI Automation | `StartupAutomationService.cs`: `--record-taxi-playlist`, `--record-taxi-chain`, `--showreel-hud`, `--showreel-zone-banners`, `--showreel-telemetry`, `--showreel-landmarks`, `--showreel-engine-badges`. | PASS |
| **AC-8**: Architectural Governance | Adheres to AGENTS.md §4 (format readers untouched), §10 (no fields in god-classes; service classes under `Workbench/Services/`), §11 (UI standardized). | PASS |
