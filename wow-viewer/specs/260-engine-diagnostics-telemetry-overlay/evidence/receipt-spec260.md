# Receipt: Spec 260 Live Engine Diagnostics, Pipeline Telemetry & Hitch Overlay System

**Date**: 2026-10-02  
**Epic**: Epic 251 (Viewer UX, Shell & Code Health) & Epic 249 (Renderer Performance, Lighting & Correctness)  
**Spec**: `specs/260-engine-diagnostics-telemetry-overlay/spec.md`  
**Status**: Implemented & Verified  

---

## 1. Files Changed

### Core Library Models & Mathematics
- `src/core/WowViewer.Core.Runtime/PromoVideo/ShowreelModels.cs`:
  - Updated `ShowreelOverlayConfig` with `ShowPerformanceTelemetry`, `ShowPipelineTelemetry`, `ShowHitchAlerts`, `HitchThresholdMs`, and `ExpandedDiagnostics`.
  - Added `HitchSeverity` enum (`None`, `Minor`, `Severe`, `Critical`).
  - Extended `ShowreelTelemetrySnapshot` with 20+ performance, geometry pipeline, and I/O diagnostic fields.
  - Added `ShowreelTelemetryMath.FormatCompactNumber`, `ShowreelTelemetryMath.FormatMemoryMb`, and `ShowreelTelemetryMath.ClassifyHitchSeverity`.

### Viewer HUD, Hitch Detection & Rendering
- `src/viewer/WoWViewer/Workbench/Services/Recording/ShowreelOverlayService.cs`:
  - Implemented rolling frame timing and hitch detection ($dt \ge \text{HitchThresholdMs}$).
  - Implemented `DetectDominantStage()` bottleneck attribution checking `LastRenderFrameStats` and `GcPauseMs`.
  - Implemented full pipeline metric aggregation in `CaptureTelemetrySnapshot()`.
  - Implemented dynamic frosted glass multi-tier HUD rendering in `DrawTelemetryHud()`.
  - Implemented flashing glowing crimson/amber hitch diagnostic alert callout badge.
  - Implemented `DrawExpandedDiagnostics` multi-stage timing side panel.

### UI & Automation Controls
- `src/viewer/WoWViewer/Workbench/Services/TaxiAreaPoi/TaxiPanelService.Sidebars.cs`:
  - Added checkboxes for Performance & Hitch Diagnostics, Pipeline & Raw Data Metrics, Visual Hitch Alert Badges, Expanded Stage Breakdown, and Hitch Spike Threshold slider.
- `src/viewer/WoWViewer/Workbench/Services/StartupAutomation/StartupAutomationService.cs`:
  - Added CLI startup flags: `--showreel-perf`, `--showreel-pipeline`, `--showreel-hitches`, `--showreel-expanded`, and `--showreel-hitch-threshold`.

### Test Suite
- `tests/WowViewer.Core.Tests/PromoVideo/ShowreelModelTests.cs`:
  - Verified default config toggles for performance and pipeline telemetry.
  - Verified snapshot instantiation with diagnostic and pipeline counters.
  - Verified `FormatCompactNumber` abbreviations (42, 999, 1K, 14.5K, 1.2M).
  - Verified `FormatMemoryMb` formatting (MB, GB).
  - Verified `ClassifyHitchSeverity` thresholds (`None`, `Minor`, `Severe`, `Critical`).

---

## 2. Verification Commands & Execution Status

### Command 1: Solution Build
```powershell
dotnet build WowViewer.slnx -c Debug
```
- **Exit Status**: 0
- **Errors**: 0

### Command 2: PromoVideo Test Suite
```powershell
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~PromoVideo"
```
- **Exit Status**: 0
- **Summary**: Passed: 49, Failed: 0, Total: 49 (Duration: 59 ms)

### Command 3: Ground Effects Test Suite
```powershell
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~GroundEffect"
```
- **Exit Status**: 0
- **Summary**: Passed: 7, Failed: 0, Total: 7 (Duration: 29 ms)

---

## 3. Criterion -> Evidence Mapping

| Acceptance Criterion | Implementation / Evidence | Status |
|---|---|---|
| **AC-001**: `ShowreelOverlayConfig` toggles for perf, pipeline, hitch alerts, threshold, and expanded layout | `ShowreelModels.cs`: `ShowPerformanceTelemetry`, `ShowPipelineTelemetry`, `ShowHitchAlerts`, `HitchThresholdMs` (33.3ms), `ExpandedDiagnostics` (false). Verified in `ShowreelOverlayConfig_DefaultsToDisabledForNormalViewing`. | **PASS** |
| **AC-002**: `ShowreelTelemetrySnapshot` encapsulates performance & pipeline metrics | `ShowreelModels.cs`: Snapshot carries `Fps`, `FrameTimeMs`, `IsHitch`, `RecentHitchCount`, `WorstFrameTimeMs`, `LastHitchStage`, `GcPauseMs`, `ManagedMemoryMb`, `ProcessWorkingSetMb`, `TotalDrawCalls`, `TerrainDrawCalls`, `M2InstancedCount`, `DetailDoodadCount`, `WmoBatchCount`, `LoadedTilesCount`, etc. Verified in `ShowreelTelemetrySnapshot_IncludesDiagnosticAndPipelineMetrics`. | **PASS** |
| **AC-003**: Rolling hitch detection and dominant stage attribution | `ShowreelOverlayService.cs`: `Update(double dt)` detects spikes against `Config.HitchThresholdMs`, calls `DetectDominantStage()` to inspect `LastRenderFrameStats` and GC pauses, and sets `_hitchAlertTimerSeconds = 2.5f`. | **PASS** |
| **AC-004**: Multi-tier frosted glass HUD layout with color coding & hitch badge | `ShowreelOverlayService.cs`: `DrawTelemetryHud` renders dark frosted glass card (`0xDC0D111A`) with dynamic line heights, color-coded FPS, glowing crimson/amber hitch alert callout, cyan pipeline metrics, and violet I/O metrics. | **PASS** |
| **AC-005**: Workbench UI & CLI automation controls | `TaxiPanelService.Sidebars.cs`: Checkboxes and threshold slider. `StartupAutomationService.cs`: Flags `--showreel-perf`, `--showreel-pipeline`, `--showreel-hitches`, `--showreel-expanded`, `--showreel-hitch-threshold`. | **PASS** |
| **AC-006**: Architectural boundaries and God-Class freeze (AGENTS.md §10) | Zero new members added to `WorldScene.cs` or `ViewerApp.cs`. All diagnostic state is encapsulated in `ShowreelOverlayService.cs` and `ShowreelModels.cs`. | **PASS** |
| **AC-007**: Unit tests validate snapshot generation and math helpers | `ShowreelModelTests.cs`: 49/49 passing tests including compact number formatting, memory formatting, and hitch severity classification. | **PASS** |
