# Spec 257 — Technical Design & Plan

## 1. Architecture Overview

Spec 257 centralizes all video recording and automated recording pipelines into a single unified service: `RecordingCoordinatorService`.

```
                  ┌──────────────────────────────────────────────┐
                  │          RecordingCoordinatorService         │
                  │  - ActiveRecordingSession / Summary          │
                  │  - FFmpeg process & pipe lifetime            │
                  │  - Frame readback, rate pacing & encoding    │
                  │  - Tour presentation overlay synchronization │
                  │  - Auto-completion (Path / Taxi / Duration)  │
                  └──────────────▲────────▲────────▲─────────────┘
                                 │        │        │
         ┌───────────────────────┴─┐      │      ┌─┴───────────────────────┐
         │    CameraPathsService   │      │      │    TaxiPanelService     │
         │ - Playback + Warmup     │      │      │ - Ride camera attach    │
         │ - Path completion trigger│     │      │ - Destination arrival   │
         │ - Built-in tour recipes │      │      │ - Route tour metadata   │
         └─────────────────────────┘      │      └─────────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │     StartupAutomationService      │
                        │ - CLI video recording flags       │
                        │ - Headless / unattended runs      │
                        │ - Exit on completion              │
                        └───────────────────────────────────┘
```

---

## 2. Key Components & Responsibilities

### 2.1 `RecordingCoordinatorService` & `RecordingModels`
- **Location**: `src/viewer/WoWViewer/Workbench/Services/Recording/`
- **State**:
  - `ActiveRecordingSession? ActiveSession { get; }`
  - `RecordingSessionSummary? LastCompletedSession { get; }`
  - `bool IsRecording => ActiveSession != null;`
- **Session Tracking**:
  - Encapsulates encoder process, RGBA frame buffer, accumulator, dimensions, framerate, and tour presentation attempt.
  - Retains `LastCompletedSession` (status, output file path, duration, frame count, success/error) so the UI can safely display results even after the active session is cleared.
- **Auto-Stop Logic**:
  - Monitors camera path playback progress when recording camera paths.
  - Monitors taxi route progress and detects when the actor reaches destination waypoint (travel >= routeLength).
  - Enforces optional maximum duration limits.
  - Triggers graceful shutdown, process exit wait (with timeout kill safety), UI restoration, archaeology restoration, and optional window exit.

### 2.2 Crash Fix in `TaxiPanelService`
- **Bug Root Cause**: `TaxiPanelService.Sidebars.cs` called `StopVideoRecording()`, which immediately set `_activeVideoRecording = null`, followed by reading `_activeVideoRecording.OutputPath` on the very next statement without a null check.
- **Fix**:
  - Replace raw `_activeVideoRecording` dereference with queries to `_recordingCoordinator.IsRecording`, `_recordingCoordinator.ActiveSession?.OutputPath`, and `_recordingCoordinator.LastCompletedSession?.OutputPath`.
  - Provide a safe, consistent button state: "Start Route Video" vs "Stop Route Video".
  - Optionally auto-stop when taxi route finishes.

### 2.3 Feature Tour for Taxi Routes
- Provide built-in feature tour generation for taxi routes (`BuiltinFeatureTourRecipes.CreateTaxiRouteOverview(...)`), displaying departure station name, destination station name, flight route ID, and scenic callouts.

### 2.4 Command-Line Startup Automation
- Support CLI options:
  - `--record-camera-path <path_name>`
  - `--record-taxi-route <route_id>`
  - `--record-duration <seconds>`
  - `--record-output <file_or_dir>`
  - `--record-fps <fps>`
  - `--record-with-ui` / `--record-no-ui`
  - `--record-feature-tour`
  - `--exit-after-record`

---

## 3. Phase Roadmap

- **Phase 1**: Core recording models and `RecordingCoordinatorService`.
- **Phase 2**: Taxi recording integration & fix for the crash bug.
- **Phase 3**: Camera path recording integration & Feature Tour alignment.
- **Phase 4**: Startup CLI automation for video recording.
- **Phase 5**: Verification sweep, tests, and documentation.
