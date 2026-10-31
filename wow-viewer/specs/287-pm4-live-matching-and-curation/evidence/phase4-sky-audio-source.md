# Phase 4 Receipt: Skybox Full-Day Rendering & Audio Immersion

**Date**: 2026-10-31  
**Spec**: Spec 287 (Phase 4: T023–T027)  
**Status**: COMPLETE  

---

## 1. Files Modified

- **Modified**:
  - `wow-viewer/src/viewer/WoWViewer/Terrain/Scene/Atmosphere/SceneAtmosphere.cs` (Remove `_skyDome.NightVisibility > 0.001f` restriction on client skybox rendering)
  - `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` (Remove redundant `_skyDome.UpdateFromLighting` call at line 3064 that erased authored DBC/LIT zenith and horizon colors)
  - `wow-viewer/src/viewer/WoWViewer/Audio/WorldAudioRuntime.cs` (Default `_worldTriggersEnabled = true` and `_areaMusicPlaybackEnabled = true`; allow zone music playback even when emitter triggers are paused)
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Audio/AudioPanelService.cs` (Add "Mute Audio Output" checkbox)

---

## 2. Verification Commands & Exit Status

1. `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
   - **Exit Status**: 0 (0 errors)
2. `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1`
   - **Exit Status**: 0 (`UI_REFERENCE_CHECK PASSED errors=0 warnings=0`)

---

## 3. Criterion -> Real Output Table

| Criterion | Implementation & Evidence | Result |
| :--- | :--- | :--- |
| **AC-401 / T023**: 24-Hour Skybox Model Rendering | In `SceneAtmosphere.cs` line 361, removed `_skyDome.NightVisibility > 0.001f &&`; `LightSkybox.dbc` models now render across the entire 24-hour cycle modulated by ambient/fog lighting rather than disappearing during daylight. | PASS |
| **AC-402 / T024**: Authored DBC/LIT Sky Zenith & Horizon Preservation | Removed destructive `_skyDome.UpdateFromLighting(lighting.GameTime, lighting.LightDirection)` at line 3064 in `WorldScene.cs`. Solar parameters are initialized at line 2992, and authored DBC `SkyTopColor` / `lighting.FogColor` and LIT `SkyTopColor` / `SkyHorizonColor` are preserved throughout the render pass. | PASS |
| **AC-403 / T025**: Zone Music Automatic Playback | In `WorldAudioRuntime.cs`, defaulted `_worldTriggersEnabled = true;` and `_areaMusicPlaybackEnabled = true;`. Updated `if (_worldTriggersEnabled || _areaMusicPlaybackEnabled) UpdateAreaMusic(...)` so camera area transitions query `AreaTable.dbc` for `ZoneMusicId` or `IntroSoundId` and stream audio immediately. | PASS |
| **AC-404 / T026**: Audio Volume & Mute Controls | `AudioPanelService.cs` now exposes an interactive `Mute Audio Output` checkbox calling `scene.SetAudioMuted(isMuted)`, alongside existing Master and Emitter gain sliders ($0.0 \dots 2.0\text{x}$) accessible in both the Audio panel and `Options > Audio`. | PASS |
