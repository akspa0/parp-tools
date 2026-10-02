# Receipt — Spec 257: Unified Video Recording & Automation System

**Date**: 2026-10-02  
**Status**: Completed  
**Branch**: `v0.6.0-dev`

---

## 1. Summary of Work

1. **Defect Fix (Crash on Stopping Taxi Route Recording)**:
   - Root cause: `TaxiPanelService.Sidebars.cs` invoked `StopVideoRecording()` which set `_activeVideoRecording = null`, immediately followed by dereferencing `_activeVideoRecording.OutputPath`, causing a fatal unhandled `NullReferenceException`.
   - Resolution: Switched all recording state queries to `RecordingCoordinatorService.IsRecording`, safe null-conditional `ActiveSession?.OutputPath`, and non-null `LastCompletedSession?.OutputPath`.

2. **Centralized Recording Architecture**:
   - Created `src/viewer/WoWViewer/Workbench/Services/Recording/RecordingModels.cs`: Defines `RecordingSourceKind` (Manual, CameraPath, TaxiRoute, Automation), `RecordingRequest`, `ActiveRecordingSession`, and immutable `RecordingSessionSummary`.
   - Created `src/viewer/WoWViewer/Workbench/Services/Recording/RecordingCoordinatorService.cs`: Authoritative coordinator managing FFmpeg process lifetime, raw RGBA video pipes, framerate pacing, UI chrome hiding/restoration, archaeology timeline playback synchronization, destination arrival auto-stopping, tour presentation advancing, and clean process teardown.
   - Wired `RecordingCoordinatorService` into `IViewerAppHost`, `ViewerApp_Host.cs`, and `ViewerApp.cs` (lifecycle loops `Update`, `Render`, `Dispose`).

3. **Taxi Flight Recording Enhancements**:
   - Added `TryGetTaxiRouteProgress` and `ResetTaxiRouteTravel` to `TaxiActorScene.cs`.
   - Implemented route arrival auto-stop: When a taxi flight completes its route, recording stops automatically and finalizes the video.
   - Added `BuiltinFeatureTourRecipes.CreateTaxiRouteOverview`: Automatic feature-tour callout recipe with origin/destination station names.
   - Added UI controls in Taxi Panel for destination auto-stop and route tour callouts.

4. **Camera Paths Integration**:
   - Refactored `CameraPathsService.cs` to submit unified `RecordingRequest` with attached tour recipes and warmup coordination.

5. **CLI Video Recording Automation**:
   - Added `--record-taxi-route <id>`, `--record-camera-path <name>`, `--record-duration <seconds>`, `--record-output <path>`, `--record-fps <fps>`, `--record-with-ui`, `--record-no-ui`, `--record-feature-tour`, and `--exit-after-record` options to `StartupAutomationService`.
   - Enables headless/unattended batch video generation.

---

## 2. Verification Commands & Output

| Command | Status | Details |
|---|---|---|
| `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` | Exit 0 | 0 errors, 597 warnings (only pre-existing library/format warnings). |
| `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~FeatureTourRecipeTests"` | Exit 0 | Passed: 7, Failed: 0. New taxi route tour recipe test verified. |
| `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~RecordingModelTests"` | Exit 0 | Passed: 3, Failed: 0. Recording models and requests verified. |
| `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` | Exit 1 (Baseline) | Passed: 1620, Failed: 9 (Known environmental failure set unchanged, 0 regressions). |

---

## 3. Criterion -> Evidence Mapping

| Acceptance Criterion | Evidence |
|---|---|
| 1. Crash Freedom on Stop Recording | `TaxiPanelService.Sidebars.cs` no longer dereferences `_activeVideoRecording`. Evaluates `_recordingCoordinator.ActiveSession` and `LastCompletedSession` safely. |
| 2. Unified Architecture | `RecordingCoordinatorService` handles all video encoding, frame pacing, and sessions across viewports, paths, and taxi routes. |
| 3. Source & Tour Support | `RecordingRequest` supports `Manual`, `CameraPath`, `TaxiRoute`, and `Automation`, with `BuiltinFeatureTourRecipes.CreateTaxiRouteOverview`. |
| 4. CLI Automation | `StartupAutomationService` parses `--record-taxi-route`, `--record-camera-path`, etc. and executes automated video recording. |
| 5. Code & God-Class Health | No new members in `WorldScene.cs` or `ViewerApp.cs`. New features live in owned `RecordingCoordinatorService`. File budget preserved. |
