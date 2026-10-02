# Spec 257 — Unified Video Recording & Automation System

**Created**: 2026-10-02 · **Branch**: `v0.6.0-dev` · **Status**: Specified & In Progress
**Parent**: Epic 251 (Viewer UX, Shell & Code Health) / Epic 252 (World Simulation, Audio & Interaction). Governance: AGENTS.md §4, §9, §10, §11.

---

## 1. Problem Statement & Background

Video recording across the viewer was historically fragmented and fragile:
1. **Crash Bug on Stopping Taxi Route Recording**: In `TaxiPanelService.Sidebars.cs`, stopping route recording calls `StopVideoRecording()`, which sets `_activeVideoRecording` to null. On the immediately following line, `Path.GetFileName(_activeVideoRecording.OutputPath)` dereferences `_activeVideoRecording`, causing an unhandled `NullReferenceException` that terminates the viewer application.
2. **Scattered Recording Logic**: Video recording mechanics (FFmpeg process management, frame capture, viewport slicing, rate pacing, UI restoration, and tour presentation) were intermingled inside `CaptureAutomationService.Video.cs`, while `CameraPathsService`, `TaxiPanelService`, and `ViewerApp` each manually poked at `_activeVideoRecording` and duplicated lifecycle coordination.
3. **No Automated Video Recording**: While static screenshot captures can be automated via CLI flags (`--capture-shot`), there is no mechanism to automate video recordings of camera paths, taxi flights, or viewport sessions from the command line or scripts.
4. **Taxi Flight Recording Lacks Completion & Tour Integration**: Taxi flights loop endlessly by default. Recording a taxi flight had no auto-stop upon arriving at the destination node, nor any integration with Feature Tour callouts or node metadata (departure/arrival station names).

---

## 2. User Stories

- **US1 (P1 - Defect Fix)**: As a user recording a taxi route flight, when I stop recording (or the flight completes), the recording finishes cleanly and the video file is saved without crashing the viewer.
- **US2 (P1 - Centralization)**: As a developer and user, all video recording flows (viewport capture, camera paths, taxi routes, and feature tours) run through a single authoritative, cohesive recording coordinator that manages encoding, rate pacing, tour presentation sync, UI chrome restoration, and lifecycle state.
- **US3 (P2 - Taxi Route Polish)**: As a user, recording a taxi route can automatically complete and finalize the video when the flight reaches its destination node, with optional feature tour callout overlays showing route information.
- **US4 (P2 - Automation)**: As an operator or automated test runner, I can trigger video recordings via CLI options (`--record-camera-path`, `--record-taxi-route`, `--record-duration`, `--record-output`, `--record-fps`, `--record-with-ui`, `--record-no-ui`, `--record-feature-tour`, `--exit-after-record`) to produce unattended high-quality MP4/MOV captures.

---

## 3. Acceptance Criteria

1. **Crash Freedom**: Stopping video recording from the Taxi Node panel (or any other panel) never throws a `NullReferenceException`. UI safely queries recording state via non-null summaries.
2. **Unified Architecture**: A dedicated `RecordingCoordinatorService` (or unified recording subsystem) owns the video recording lifecycle, session state, FFmpeg pipe, frame capture, rate timing, and UI/tour coordination.
3. **Source Support**:
   - `FreeView / Manual`: Direct viewport recording with start/stop controls.
   - `CameraPath`: Coordinated playback, path warmup sync, tour overlays, and auto-stop on path finish.
   - `TaxiRoute`: Attached ride camera, optional auto-stop on route completion, and optional route tour overlay.
4. **CLI Automation**: Startup options allow automated recording of camera paths and taxi routes with configurable duration, resolution, output path, and auto-exit upon completion.
5. **No Regressions**: `dotnet build` succeeds with 0 errors; existing test suites pass; code conforms to AGENTS.md §10 (god-class freeze) and §11 (UI standardization).

---

## 4. Architectural Boundaries & Constraints

- No new members in `WorldScene.cs` or `ViewerApp.cs` (AGENTS.md §10). Features live in dedicated services under `Workbench/Services/`.
- File budget: All modified or new files remain well below 2,000 lines.
- Format readers remain untouched (AGENTS.md §4).
