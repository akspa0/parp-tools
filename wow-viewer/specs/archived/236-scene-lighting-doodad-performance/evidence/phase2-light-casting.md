# Spec 236 Phase 2 Evidence — Terrain Light-Casting Slice + Ambient/Sun Contract

Date: 2026-09-18

## Scope

This source-only slice extends the Phase 2 multi-surface light casting beyond the WMO shell
consumer landed on 2026-09-16:

- **T010**: `SceneLightManager` now also carries the frame outdoor ambient/sun representation
  (`SceneAmbientLight`), satisfying the manager-contract half of FR-004. `WorldScene` publishes it
  from the active `TerrainLighting` profile each frame.
- **T012**: both terrain shaders (the legacy per-chunk program and the batched tile program) accept
  up to eight nearby local point lights and accumulate bounded per-fragment diffuse, satisfying
  FR-006 at source level.
- **T013 (terrain consumer)**: `WorldScene` now passes the shared `SceneLightManager` into the
  terrain render pass, and `TerrainRenderer.UploadLocalLights` selects the nearest lights that
  actually reach each chunk/tile bounding volume.

The doodad/model *external*-light consumer (FR-007) remains open: doodad MDX/M2 shaders still
evaluate only their own `LITE` data, not external scene lights. That is the next bounded Phase 2
slice. Operator runtime visual proof for torches/braziers spilling onto terrain and WMO geometry
(Gate 2) remains operator-owned.

## Files Changed

| Path | Change |
|---|---|
| `src/viewer/WoWViewer/Rendering/SceneLightManager.cs` | Added `SceneAmbientLight` record, `Ambient`, `SetAmbient(...)` (finite-guarded), and made `Clear()` reset ambient alongside point lights. |
| `src/viewer/WoWViewer/Terrain/WorldScene.cs` | `RebuildSceneLights` publishes the active `TerrainLighting` direction/color/ambient; terrain render call now forwards `_sceneLightManager`. |
| `src/viewer/WoWViewer/Terrain/TerrainRenderer.cs` | Added `LocalLightUniforms` resolution for both shader programs, `UploadLocalLights(...)` with `QueryAffecting` by chunk/tile AABB, per-chunk/per-tile upload, `vWorldNormal` varying, and a bounded 8-light diffuse loop in both fragment shaders. |
| `src/viewer/WoWViewer/Terrain/TerrainManager.cs` | `Render(view, proj, cameraPos, frustum, sceneLights)` forwards the light manager to `TerrainRenderer`. |

## Verification Commands

| Command | Exit | Output Evidence |
|---|---:|---|
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --nologo -v q` | 0 | `0 Error(s)`; `327 Warning(s)` (pre-existing analyzer/NU/CS0649 warnings, unchanged). |
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TerrainLighting\|FullyQualifiedName~WorldObjectPassCoordinator"` | 0 | `Passed! - Failed: 0, Passed: 21, Skipped: 0, Total: 21`. |

## Acceptance Criterion Evidence

| Criterion | Evidence | Status |
|---|---|---|
| FR-004: spatial manager aggregates light sources (position, color, intensity, radius) and the outdoor ambient/sun representation. | `SceneLightManager` carries `SceneLight` point lights plus the new `SceneAmbientLight` (direction, light color, ambient color); `WorldScene.RebuildSceneLights` calls `SetAmbient(baseLighting.LightDirection, baseLighting.LightColor, baseLighting.AmbientColor)` from the active `TerrainLighting`. | Source complete for the ambient/sun contract. Runtime proof operator-owned. |
| FR-006: terrain shaders evaluate nearby local lights affecting each chunk. | `TerrainRenderer` resolves `uLocalLightCount`/`uLocalLightPos[8]`/`uLocalLightColor[8]`/`uLocalLightIntensity[8]`/`uLocalLightStart[8]`/`uLocalLightEnd[8]` for both programs; both fragment shaders add a bounded 8-light point-diffuse term (`+ localLight`) to base lighting; `UploadLocalLights` queries `SceneLightManager.QueryAffecting(chunk/tile bounds)` per draw. | Source complete; runtime visual proof operator-owned. |
| FR-005: WMO shaders evaluate up to eight active local lights. | Unchanged from the 2026-09-16 WMO slice (`phase2-wmo-light-casting-slice.md`). | Source complete; runtime visual proof operator-owned. |
| FR-007: doodad shaders evaluate nearby scene lights affecting their bounding volume. | Not touched in this slice — doodad MDX/M2 shaders still evaluate only their own `LITE` data. | **Open** (next Phase 2 slice). |

## Proof Boundary

Build success proves the new source compiles, including both GLSL shader programs (they compile at
runtime, not build time — no runtime compile witness is claimed here). The focused test run proves
adjacent terrain-lighting math and world object-pass infrastructure are unchanged. **No runtime
visual, FPS, real-client, terrain-illumination, or doodad-consumer proof is claimed.** Gate 2 stays
open until the operator witnesses torch/brazier light spilling onto terrain and WMO geometry.