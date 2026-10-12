# Tasks: Spec 260 Live Engine Diagnostics, Pipeline Telemetry & Hitch Overlay System

## Phase 1: Core Models & Formatting Helpers

- [x] T001: Update `ShowreelOverlayConfig` in `src/core/WowViewer.Core.Runtime/PromoVideo/ShowreelModels.cs` with `ShowPerformanceTelemetry`, `ShowPipelineTelemetry`, `ShowHitchAlerts`, `HitchThresholdMs`, and `ExpandedDiagnostics`.
- [x] T002: Extend `ShowreelTelemetrySnapshot` in `src/core/WowViewer.Core.Runtime/PromoVideo/ShowreelModels.cs` with performance metrics (`Fps`, `FrameTimeMs`, `IsHitch`, `RecentHitchCount`, `WorstFrameTimeMs`, `LastHitchStage`, `LastHitchDurationMs`, `GcPauseMs`, `ManagedMemoryMb`, `ProcessWorkingSetMb`) and rendering/pipeline metrics (`TotalDrawCalls`, `TerrainDrawCalls`, `TerrainChunksRendered`, `TerrainChunksCulled`, `WmoBatchCount`, `WmoInstanceCount`, `M2InstancedCount`, `M2UnbatchedCount`, `DetailDoodadCount`, `LiquidMeshCount`, `LoadedTilesCount`, `FileCacheHits`, `FileCacheCount`, `DeferredLoadsCount`, `DataSourceEra`).
- [x] T003: Add formatting and classification helpers to `ShowreelTelemetryMath` (`FormatCompactNumber`, `FormatMemoryMb`, `ClassifyHitchSeverity`).
- [x] T004: Update and expand unit tests in `tests/WowViewer.Core.Tests/PromoVideo/ShowreelModelTests.cs` to test new properties, snapshot record creation, and math helpers.

**Gate A: Core Models Verification**
- Build: `dotnet build tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` (PASS)
- Test: `dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~PromoVideo"` (PASS: 49/49)

---

## Phase 2: Telemetry Capture, Hitch Attribution & HUD Rendering

- [x] T005: Implement frame timing, rolling hitch detection, stage bottleneck attribution, and alert countdown in `ShowreelOverlayService.Update()`.
- [x] T006: Implement comprehensive pipeline telemetry collection in `ShowreelOverlayService.CaptureTelemetrySnapshot()`, querying `LastFrameRenderStats`, `TerrainRenderer`, `LiquidRenderer`, `GroundEffects`, `Assets`, and system memory.
- [x] T007: Implement enhanced multi-tier frosted glass HUD layout in `ShowreelOverlayService.DrawTelemetryHud()` rendering performance, memory, geometry pipeline, and I/O counters.
- [x] T008: Implement vivid glowing hitch diagnostic callout badge in `ShowreelOverlayService` indicating spike magnitude and responsible stage when $dt \ge \text{HitchThresholdMs}$.
- [x] T009: Implement expanded multi-column diagnostics layout mode for deep-dive technical captures.

**Gate B: Service Implementation & Solution Build**
- Build: `dotnet build WowViewer.slnx -c Debug` (PASS: 0 errors)
- Test: `dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` (PASS)

---

## Phase 3: UI Workbench Controls, CLI Automation & Verification

- [x] T010: Add toggle controls for performance stats, pipeline stats, hitch alerts, and threshold sliders in `TaxiPanelService.Sidebars.cs`.
- [x] T011: Add CLI startup flags (`--showreel-perf`, `--showreel-pipeline`, `--showreel-hitches`, `--showreel-expanded`, `--showreel-hitch-threshold`) to `StartupAutomationService.cs`.
- [x] T012: Verify full test suite passes and update `evidence/receipt-spec260.md`.
- [x] T013: Update `specs/STATUS.md`, `memory-bank/activeContext.md`, and `memory-bank/progress.md`.

**Gate C: Final Verification**
- Full test pass: `dotnet test WowViewer.slnx -c Debug`
