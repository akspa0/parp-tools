# Spec 260: Live Engine Diagnostics, Pipeline Telemetry & Hitch Overlay System

**Owner**: Epic 251 (Viewer UX, Shell & Code Health) & Epic 249 (Renderer Performance, Lighting & Correctness)  
**Origin**: Operator prompt (2026-10-02) to showcase under-the-hood engine parsing/decoding/rendering and provide real-time frame pacing diagnostics for video capture  
**Status**: Proposed / In Progress  

---

## 1. Executive Summary

The PromoVideo / Showreel system (Spec 257 & Spec 258) enables automated video captures and technical showreels for Patreon and public showcases. However, viewers cannot currently see the sheer scale of the engineering taking place under the hood: thousands of M2 doodads being instanced on the GPU, dynamic ground flora being synthesized across terrain and WMO surfaces, high-frequency ADT/MCNK chunk queries, asset caching, and multi-stage rendering passes.

Furthermore, when optimizing rendering performance or investigating frame drops, video captures lack embedded frame-pacing and stage-attribution diagnostics. 

This spec enhances the PromoVideo showreel overlay and recording system into a **live technical showcase and diagnostic telemetry HUD**:
1. **Raw Engine Pipeline Telemetry**: Real-time counters for rendered MCNK terrain chunks, WMO batch submissions, M2 GPU instancing ratios, dynamic ground effect (flora/clutter) counts, liquid meshes, and total draw calls.
2. **Data I/O & Asset Pipeline Telemetry**: Cache hit rates, loaded ADT tile counts, active memory footprints (managed GC heap and process working set), and deferred load queue telemetry.
3. **Automated Frame Hitch & Stutter Detection**: Real-time detection of frame time spikes ($dt \ge \text{HitchThresholdMs}$), recording the spike magnitude, attributing the bottleneck to the dominant render stage (`DeferredAssetLoads`, `Terrain`, `WmoSubmission`, `MdxVisibility`, `MdxAnimation`, `Liquid`, `GcPause`, etc.), and flashing an on-screen diagnostic callout badge during capture.
4. **Customizable Layouts & UI Controls**: Configurable via workbench sidebars, settings, and CLI automation flags (`--showreel-perf`, `--showreel-pipeline`, `--showreel-hitches`, `--showreel-expanded`), supporting compact HUD, expanded multi-stage telemetry, or cinematic minimal modes.

---

## 2. Requirements & Acceptance Criteria

### User Stories

- **US1: Showcase Under-The-Hood Architecture**: As a viewer of PromoVideo showreels and Patreon demonstrations, I see a sleek HUD displaying real-time metrics of raw data parsing and rendering (draw calls, active flora, instanced M2 doodads, ADT chunks, and memory).
- **US2: Video-Based Diagnostic & Hiccup Troubleshooting**: As a developer analyzing video captures to diagnose performance stutter, I can review the embedded HUD to pinpoint exact timestamps, spike durations, and stage attributions for any frame drop.
- **US3: Configurable Telemetry Modules**: As an operator setting up a flight playlist or camera path recording, I can independently toggle performance stats, pipeline metrics, hitch alerts, and layout density in the UI or via CLI flags.

### Acceptance Criteria

- **AC-001**: `ShowreelOverlayConfig` exposes `ShowPerformanceTelemetry`, `ShowPipelineTelemetry`, `ShowHitchAlerts`, `HitchThresholdMs`, and `ExpandedDiagnostics` toggles with clean defaults.
- **AC-002**: `ShowreelTelemetrySnapshot` encapsulates performance metrics (FPS, frame time, hitch count, peak frame time, dominant hitch stage, GC pause, memory) and pipeline metrics (draw calls, terrain chunks, WMO batches, M2 instancing, ground flora, liquid meshes, I/O cache stats).
- **AC-003**: `ShowreelOverlayService` computes rolling hitch detection, attributing spikes exceeding the threshold to the dominant render stage in `WorldRenderFrameStats`, and maintains an alert banner with cooldown.
- **AC-004**: The overlay HUD renders using frosted glass styling with color-coded status indicators (green for high FPS, gold for pipeline metrics, crimson for hitch callouts) and stacks neatly with tour presentation subtitles and banners.
- **AC-005**: Workbench UI (`TaxiPanelService.Sidebars.cs`) and CLI automation (`StartupAutomationService.cs`) expose controls for the new telemetry modules.
- **AC-006**: Architectural boundaries and AGENTS.md rules are respected: zero new members in `WorldScene.cs` or `ViewerApp.cs`; all state remains encapsulated in `ShowreelOverlayService` and `ShowreelModels.cs`.
- **AC-007**: Unit tests validate snapshot generation, hitch detection logic, threshold clamping, and formatting helpers.
