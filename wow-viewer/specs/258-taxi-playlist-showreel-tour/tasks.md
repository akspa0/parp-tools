# Tasks — Spec 258: Taxi Route Playlists & Dynamic Marketing Showreel Tour

Governance: AGENTS.md §9.2 receipts required for every checked task. Build/test output alone does not prove runtime or UI behavior.

---

## Phase 1 — Core Models & Showreel Telemetry Math

- [x] T001: Create `src/core/WowViewer.Core.Runtime/Marketing/ShowreelModels.cs` with `ShowreelTelemetrySnapshot`, `ShowreelOverlayConfig`, and telemetry formatting helpers (heading/compass, ADT tile coordinates, MCNK chunk coordinates).
- [x] T002: Unit tests for `ShowreelModels` covering compass calculation, ADT tile coordinate conversion, and telemetry snapshot formatting.

## Phase 2 — Taxi Playlist Service

- [x] T003: Create `src/viewer/WoWViewer/Workbench/Services/TaxiAreaPoi/TaxiPlaylistModels.cs` with `TaxiPlaylistItem` and `TaxiPlaylistState`.
- [x] T004: Create `src/viewer/WoWViewer/Workbench/Services/TaxiAreaPoi/TaxiPlaylistService.cs` managing playlist routes, auto-chaining path discovery, seamless multi-route transition, and multi-segment continuous recording.
- [x] T005: Expose `TaxiPlaylistService` through `IViewerAppHost` and `ViewerApp_Host.cs`.

## Phase 3 — Dynamic Showreel HUD & Overlay Renderer

- [x] T006: Create `src/viewer/WoWViewer/Workbench/Services/Recording/ShowreelOverlayService.cs` supporting live coordinates, compass orientation, ADT tile/chunk display, and flight progress tracking.
- [x] T007: Implement cinematic zone and subzone transition banner detection and timer animation in `ShowreelOverlayService`.
- [x] T008: Implement proximity landmark discovery callouts (approaching taxi flight masters, major structures) and engine technical badges in `ShowreelOverlayService`.
- [x] T009: Expose `ShowreelOverlayService` in `IViewerAppHost` and wire overlay rendering into `ViewerApp.cs` render pass.

## Phase 4 — UI & CLI Automation Integration

- [x] T010: Update `TaxiPanelService.Sidebars.cs` to add the "Playlist & Marketing Tour" section with route list, reorder controls, auto-chaining, and Showreel HUD toggles.
- [x] T011: Update `StartupAutomationService.cs` with CLI flags: `--record-taxi-playlist`, `--record-taxi-chain`, `--showreel-hud`, `--showreel-zone-banners`, `--showreel-telemetry`, `--showreel-landmarks`, and `--showreel-engine-badges`.
- [x] T012: Implement post-load execution of playlist playback and automated showreel recording in `StartupAutomationService.cs`.

## Phase 5 — Verification & Validation

- [x] T013: Unit tests for `TaxiPlaylistService` sequencing, chain generation, and `ShowreelOverlayService` logic in `WowViewer.Core.Tests`.
- [x] T014: Full solution build (`dotnet build`) and test execution (`dotnet test`).
- [x] T015: Write completion receipt under `specs/258-taxi-playlist-showreel-tour/evidence/receipt-spec258.md`, update `STATUS.md` and `activeContext.md`.
