# Spec 260 Technical Plan: Live Engine Diagnostics, Pipeline Telemetry & Hitch Overlay System

## 1. Architecture & Component Placement

To satisfy AGENTS.md §4 (Core Library First) and AGENTS.md §10 (God-Class Freeze):
- All data models, snapshot contracts, and mathematical/formatting helpers live in `src/core/WowViewer.Core.Runtime/PromoVideo/ShowreelModels.cs`.
- All HUD rendering, hitch stage attribution, state tracking, and ImGui presentation logic live in `src/viewer/WoWViewer/Workbench/Services/Recording/ShowreelOverlayService.cs`.
- UI controls are added to `src/viewer/WoWViewer/Workbench/Services/TaxiAreaPoi/TaxiPanelService.Sidebars.cs`.
- CLI automation parameters are added to `src/viewer/WoWViewer/Workbench/Services/StartupAutomation/StartupAutomationService.cs`.
- Zero new members or state are introduced into `WorldScene.cs` or `ViewerApp.cs`.

```
                    ┌────────────────────────────┐
                    │  ShowreelTelemetrySnapshot │  (WowViewer.Core.Runtime)
                    └─────────────▲──────────────┘
                                  │ captures
┌─────────────────────────────────┴─────────────────────────────────┐
│                      ShowreelOverlayService                       │
│  - Hitch Detection & Stage Bottleneck Attribution                 │
│  - Pipeline Counters (Terrain, WMO, M2 Instancing, Ground Flora)   │
│  - Dynamic Frosted Glass HUD & Hitch Alert Banner Rendering       │
└────────▲───────────────────────▲──────────────────────▲───────────┘
         │                       │                      │
┌────────┴──────────────┐ ┌──────┴───────────────┐ ┌────┴────────────┐
│ TaxiPanelService UI   │ │ StartupAutomation    │ │ RecordingCoord. │
└───────────────────────┘ └──────────────────────┘ └─────────────────┘
```

## 2. Technical Design Details

### 2.1 Core Runtime Models (`ShowreelModels.cs`)

1. **`ShowreelOverlayConfig` Updates**:
   - `ShowPerformanceTelemetry`: bool (default true)
   - `ShowPipelineTelemetry`: bool (default true)
   - `ShowHitchAlerts`: bool (default true)
   - `HitchThresholdMs`: float (default 33.3f)
   - `ExpandedDiagnostics`: bool (default false)

2. **`ShowreelTelemetrySnapshot` Updates**:
   - Extended to include performance and pipeline telemetry fields:
     - Performance: `Fps`, `FrameTimeMs`, `IsHitch`, `RecentHitchCount`, `WorstFrameTimeMs`, `LastHitchStage`, `LastHitchDurationMs`, `GcPauseMs`, `ManagedMemoryMb`, `ProcessWorkingSetMb`.
     - Rendering Pipeline: `TotalDrawCalls`, `TerrainDrawCalls`, `TerrainChunksRendered`, `TerrainChunksCulled`, `WmoBatchCount`, `WmoInstanceCount`, `M2InstancedCount`, `M2UnbatchedCount`, `DetailDoodadCount`, `LiquidMeshCount`.
     - I/O & Formats: `LoadedTilesCount`, `FileCacheHits`, `FileCacheCount`, `DeferredLoadsCount`, `DataSourceEra`.

3. **`ShowreelTelemetryMath` Extensions**:
   - `FormatCompactNumber(int number)`: Formats numbers with `K`/`M` suffixes (e.g. `12450` -> `"12.5K"`).
   - `FormatMemoryMb(float mb)`: Formats megabytes cleanly (`"412 MB"` or `"1.2 GB"`).
   - `ClassifyHitchSeverity(float frameTimeMs, float thresholdMs)`: Classifies spike into `None`, `Minor`, `Severe`, `Critical`.

### 2.2 Telemetry Capture & Hitch Analysis (`ShowreelOverlayService.cs`)

1. **Hitch Detection Logic**:
   - In `Update(double dt)`:
     - Compute `frameTimeMs = (float)(dt * 1000.0)`.
     - If `frameTimeMs >= Config.HitchThresholdMs`:
       - Increment `_sessionHitchCount`.
       - Record `_lastHitchDurationMs = frameTimeMs`.
       - `_worstFrameTimeMs = Math.Max(_worstFrameTimeMs, frameTimeMs)`.
       - Query `_host.WorldScene?.LastFrameRenderStats` to identify the dominant stage with maximum `DurationMs`:
         - Iterate `DeferredAssetLoads`, `TaxiActorUpdate`, `Lighting`, `Sky`, `Wdl`, `Terrain`, `WmoVisibility`, `WmoSubmission`, `MdxAnimation`, `MdxVisibility`, `MdxOpaqueSubmission`, `Liquid`, `MdxTransparentSubmission`, `Overlay`.
         - If `stats.GcPauseMs` dominates, attribute to `GC Pause`.
         - Store stage name in `_lastHitchStage`.
       - Set `_hitchAlertTimerSeconds = 2.5f` to display the hitch banner on-screen.
     - Decrement `_hitchAlertTimerSeconds` on each update.

2. **Pipeline Telemetry Aggregation**:
   - In `CaptureTelemetrySnapshot()`:
     - Collect geometry numbers from `_host.WorldScene?.LastFrameRenderStats`.
     - Collect ground effects count from `_host.GroundEffects?.ActiveDoodadCount ?? 0`.
     - Collect terrain draw calls from `_host.WorldScene?.Terrain.TerrainRenderer.LastFrameDrawCalls ?? 0`.
     - Collect memory metrics via `GC.GetTotalMemory(false)` and `Environment.WorkingSet`.
     - Collect I/O stats from `_host.WorldScene?.Assets.GetReadStats()`.

3. **Rendering the Enhanced Telemetry HUD**:
   - Dynamically compute HUD card size based on enabled sections:
     - Compact HUD: 5 rows (POS/DIR, ZONE/MAP, PERF/MEM, PIPELINE, I/O).
     - Expanded Diagnostics: Multi-column breakdown detailing per-stage timings.
   - When a hitch occurs and `_hitchAlertTimerSeconds > 0`:
     - Render an amber/crimson glowing callout badge at the top-center of the telemetry card:
       `[!] HITCH: +XX.X ms (STAGE: <Stage>)`
     - Flash alpha based on remaining alert duration.

## 3. Phased Implementation Roadmap

- **Phase 1: Core Models & Formatting Helpers**
  - Update `ShowreelModels.cs` with new config properties, extended snapshot, and formatting helpers.
  - Add unit tests in `ShowreelModelTests.cs`.
- **Phase 2: Telemetry Capture, Hitch Detection & HUD Rendering**
  - Implement rolling frame timer, hitch attribution, and telemetry capture in `ShowreelOverlayService.cs`.
  - Implement enhanced multi-tier HUD rendering and hitch alert banner in `ShowreelOverlayService.cs`.
- **Phase 3: UI Controls, CLI Automation & Verification**
  - Add checkboxes in `TaxiPanelService.Sidebars.cs`.
  - Add CLI arguments in `StartupAutomationService.cs`.
  - Validate with `dotnet build` and `dotnet test`.
  - Author receipt in `evidence/receipt-spec260.md`.
