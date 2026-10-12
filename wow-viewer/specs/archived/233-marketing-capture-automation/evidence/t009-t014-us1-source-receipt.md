# T009–T014 User Story 1 Source Receipt — 2026-09-08

## Files changed

- `src/core/WowViewer.Core.Runtime/Marketing/MarketingTourAttempt.cs`
- `src/viewer/WoWViewer/Capture/MarketingTourOverlayRenderer.cs`
- `src/viewer/WoWViewer/ViewerApp_CameraPaths.cs`
- `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs`
- `src/viewer/WoWViewer/ViewerApp.cs`
- `tests/WowViewer.Core.Tests/MarketingCapture/FeatureTourAttemptTests.cs`

## Test-first and verification record

| Command | Exit | Real output |
|---|---:|---|
| `dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~MarketingCapture" --no-restore` before attempt source | 1 | Expected compile failure: `MarketingTourAttempt` and `MarketingTourAttemptStartResult` did not exist. |
| `dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~MarketingCapture"` | 0 | `Passed: 12, Failed: 0, Skipped: 0`. |
| `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --no-restore` | 0 | `0 Error(s)`; 577 existing/project warnings. |

## Criterion-to-evidence mapping

| Criterion | Source evidence | Proof boundary |
|---|---|---|
| FR-001 recipe runs ordered callout beats | `MarketingTourAttempt` with transition/no-steady-allocation tests | Source behavior only. |
| FR-003 direct renderer capture | `Feature Tour + Video` composes existing `TryStartCurrentViewVideoRecording`; no second screen-capture route was added | Build proof, not recorded video proof. |
| FR-004 warm before recorded run | The existing `BeginCameraPathPreload` gate carries the attempt until ready | Build proof, not client-streaming proof. |
| FR-005 clean scene and scheduled callout | Existing chrome is hidden only during the tour; `MarketingTourOverlayRenderer` draws an active callout before the with-UI capture tap; prior chrome state is restored at recording terminal state | Build proof, not visual-timing proof. |
| SC-001 no manual option-bar interaction after start | Explicit **Feature Tour + Video** entry point and automatic beat state | Requires T015 operator witness. |

T015 remains open. This receipt does not claim a real video, UI appearance, smooth frame pacing, FPS, or encoder playback.

