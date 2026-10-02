# Spec 258 — Taxi Route Playlists & Dynamic Marketing Showreel Tour

**Created**: 2026-10-02 · **Branch**: `v0.6.0-dev` · **Status**: Draft / In Progress
**Parent**: Epic 251 (Viewer UX, Shell & Code Health) / Epic 252 (World Simulation, Audio & Interaction). Governance: AGENTS.md §4, §9, §10, §11.

---

## 1. Problem Statement & Background

Creating promotional multimedia and Patreon showreels from `wow-viewer` currently requires extensive manual recording and post-processing in external video editing software:
1. **Single-Route Flight Limit**: The viewer can only ride and record one taxi route at a time. To capture a multi-stop journey across a continent (e.g. Stormwind -> Sentinel Hill -> Darkshire -> Lakeshire -> Ironforge), the user must manually wait for arrival, find the next route, start recording again, and splice individual clips in an external video editor.
2. **Missing Dynamic Contextual Data Callouts**: While flying along taxi routes or following camera paths, the viewer possesses rich world knowledge (AreaTable zone/subzone names, ADT tile & MCNK chunk coordinates, live 3D position & compass heading, nearby taxi flight nodes, WMO structures, and engine capabilities), but none of this is overlaid dynamically on the recorded footage.
3. **Patreon Showcase Need**: High-quality promotional captures require broadcast-style lower-thirds, zone entry banners, flight telemetry HUDs, and landmark discovery popups rendered directly on the fly, eliminating the need for video editing applications.

---

## 2. User Stories

- **US1 (P1 - Taxi Route Playlists)**: As a user or content creator, I can assemble a playlist of multiple taxi routes (or auto-generate a connected tour), fly through them sequentially without interruption, and capture the entire multi-route journey as a single continuous high-definition video recording.
- **US2 (P1 - Dynamic Zone & Telemetry HUD)**: As a viewer or content creator, during tour playback and video recording, the viewer displays a sleek, broadcast-style telemetry overlay showing live world coordinates (X, Y, Z), compass heading, ADT tile / chunk indices, and flight progress.
- **US3 (P1 - Cinematic Zone Transition Banners)**: When the camera crosses into a new zone or subzone (resolved live via `AreaTableService`), a cinematic title card (e.g., "WESTFALL" / "Sentinel Hill") appears smoothly on screen and fades out, highlighting the authentic WoW geography.
- **US4 (P2 - Landmark & Proximity Discovery Callouts)**: As the camera passes within proximity of interesting world entities (such as flight masters/taxi nodes, major WMO structures, or key areas), dynamic discovery callouts appear to showcase what data the viewer is detecting in real-time.
- **US5 (P2 - Engine Showreel Badges)**: Optional showreel callouts highlight technical engine features (e.g., "64-bit ADT Hole Punching", "Native M2 GPU Skinning & Instancing", "Alpha 0.5.3 Multi-Era DBC Engine") to demonstrate the tool's engineering achievements for Patreon supporters.
- **US6 (P2 - CLI Automation)**: As an operator, I can automate playlist playback and showreel video recording via command line (`--record-taxi-playlist`, `--record-taxi-chain`, `--showreel-hud`, `--exit-after-record`).

---

## 3. Acceptance Criteria

1. **Continuous Playlist Recording**:
   - Starting a video recording of a taxi playlist records continuously across all route segments into a single output video file.
   - Reaching a segment destination automatically transitions to the next segment in the playlist without terminating the video recording.
   - Reaching the end of the final segment automatically concludes and saves the video recording cleanly.
2. **Interactive Playlist Controls in UI**:
   - Taxi panel allows adding routes to a playlist, removing routes, reordering (move up/down), clearing, and one-click "Chain Connected Route" generation.
   - Play and Record buttons for the entire playlist.
3. **Dynamic Showreel Overlay**:
   - **Zone Banners**: Auto-detects zone and subzone changes from `AreaContextService` and renders a clean, non-intrusive cinematic banner.
   - **Live Telemetry**: Real-time display of camera position (X, Y, Z), compass heading / pitch, current map name/ID, and ADT tile indices.
   - **Proximity Callouts**: Detects approaching taxi nodes within 350 yards and displays node name, ID, and distance.
   - **Toggles**: User can toggle telemetry, zone banners, and landmark callouts independently in the UI and via CLI.
4. **Broadcast Quality & Zero Post-Processing**:
   - The showreel overlay renders into the captured video frame when recording with UI enabled (or in clean HUD mode with main chrome hidden), producing finished promotional clips with no video editor needed.
5. **No Regressions & Strict Governance**:
   - Full solution builds with 0 errors. All new unit tests pass with zero regression in baseline test suites.
   - Follows AGENTS.md §10 (no new fields in `WorldScene.cs` or `ViewerApp.cs`; implemented via owned service classes) and §11 (UI standardization).

---

## 4. Architectural Boundaries & Constraints

- All playlist management lives in `Workbench/Services/TaxiAreaPoi/TaxiPlaylistService.cs` (or sub-service).
- All showreel telemetry and overlay rendering lives in `Workbench/Services/Recording/ShowreelOverlayService.cs` or `Capture/MarketingTourOverlayRenderer.cs`.
- Does not modify working format readers (AGENTS.md §4).
- File size budget: All files remain under 2,000 lines (AGENTS.md §10).
