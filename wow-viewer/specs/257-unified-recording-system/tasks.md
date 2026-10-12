# Tasks — Spec 257: Unified Video Recording & Automation System

Governance: AGENTS.md §9.2 receipts required for every checked task. Build/test output alone does not prove runtime or UI behavior.

---

## Phase 1 — Core Recording Architecture

- [x] T001: Create `src/viewer/WoWViewer/Workbench/Services/Recording/RecordingModels.cs` with `RecordingSourceKind`, `RecordingRequest`, `ActiveRecordingSession`, and `RecordingSessionSummary`.
- [x] T002: Create `src/viewer/WoWViewer/Workbench/Services/Recording/RecordingCoordinatorService.cs` encapsulating FFmpeg lifecycle, frame capture, rate pacing, auto-stop triggers, and tour integration.
- [x] T003: Expose `RecordingCoordinatorService` via `IViewerAppHost` and `ViewerApp_Host.cs`.

## Phase 2 — Taxi Node Video Recording Fix & Enhancement

- [x] T004: Fix the `NullReferenceException` in `TaxiPanelService.Sidebars.cs` by integrating `RecordingCoordinatorService` and safe session queries.
- [x] T005: Update `TaxiPanelService.CaptureAutomation.cs` to use `RecordingCoordinatorService.Start(...)` with options for auto-stop on destination arrival.
- [x] T006: Add destination arrival detection in `TaxiActorScene` / `TaxiPanelService` so route recording auto-stops cleanly when enabled.
- [x] T007: Add taxi route feature tour recipe generator in `BuiltinFeatureTourRecipes` with station name callouts.

## Phase 3 — Camera Path & Capture Automation Integration

- [x] T008: Refactor `CameraPathsService` to route recording requests through `RecordingCoordinatorService`.
- [x] T009: Delegate existing `CaptureAutomationService.Video.cs` recording methods to `RecordingCoordinatorService` to maintain backward compatibility.
- [x] T010: Ensure tour presentation overlays (`FeatureTourPresentation`) render correctly from `RecordingCoordinatorService.ActiveTourPresentation`.

## Phase 4 — Video Recording Startup Automation (CLI)

- [x] T011: Add `--record-taxi-route`, `--record-camera-path`, `--record-duration`, `--record-output`, `--record-fps`, `--record-with-ui`, `--record-no-ui`, `--record-feature-tour`, and `--exit-after-record` CLI parsing to `StartupAutomationService`.
- [x] T012: Implement post-load execution of the automated video recording request in `StartupAutomationService`.

## Phase 5 — Verification & Validation

- [x] T013: Unit tests for recording models, requests, taxi route tour recipes, and CLI option parsing.
- [x] T014: Full solution build (`dotnet build`) and test execution (`dotnet test`).
- [x] T015: Write completion receipt under `specs/257-unified-recording-system/evidence/receipt-spec257.md`.
