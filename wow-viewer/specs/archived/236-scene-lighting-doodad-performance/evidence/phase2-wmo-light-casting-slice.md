# Spec 236 Phase 2 Evidence — WMO Emitted-Light Casting Slice

Date: 2026-09-16

## Scope

This source-only slice implements the first bounded Phase 2 consumer: WMO shell geometry now evaluates up to eight nearby scene-emitted point lights. The slice also introduces the shared scene-light data path used to collect WMO `MOLT`, MDX `LITE`, M2 animated lights, and WMO-internal doodad lights.

Terrain and doodad shader consumers are intentionally left open for later Phase 2 work. Runtime visual proof for torches/braziers casting on WMO geometry and terrain remains operator-owned.

## Files Changed

| Path | Change |
|---|---|
| `src/viewer/WoWViewer/Rendering/SceneLight.cs` | Added world-space point-light record with position/color/intensity/attenuation metadata. |
| `src/viewer/WoWViewer/Rendering/SceneLightManager.cs` | Added active-light collection and nearest-light selection capped to eight shader lights. |
| `src/viewer/WoWViewer/Rendering/ISceneLightEmitter.cs` | Added opt-in emitter contract so renderer interfaces stay stable. |
| `src/viewer/WoWViewer/Rendering/ModelRenderer.cs` | Exposes MDX omni `LITE` lights as scene lights using existing bounded/clamped MDX light semantics. |
| `src/viewer/WoWViewer/Rendering/M2Renderer.cs` | For legacy-backed M2, delegates to MDX emitter; for native runtime M2, emits visible animated omni lights from the last evaluated animation state. |
| `src/viewer/WoWViewer/Rendering/WmoRenderer.cs` | Emits WMO `MOLT` lights and WMO-internal doodad lights; WMO shell shader uploads/evaluates up to eight local point lights. |
| `src/viewer/WoWViewer/Rendering/IGpuInstancedWmoRenderer.cs` | Added scene-light-aware WMO instanced batch overload. |
| `src/viewer/WoWViewer/Terrain/WorldScene.cs` | Owns a `SceneLightManager`, rebuilds scene lights from current visible placements, feeds WMO shell draws, and disables WMO shell instancing while active scene lights require per-placement light selection. |
| `specs/archived/236-scene-lighting-doodad-performance/tasks.md` | Checked T011 only; recorded open Phase 2 boundaries. |

## Verification Commands

| Command | Exit | Output Evidence |
|---|---:|---|
| `dotnet build wow-viewer/WowViewer.slnx -c Debug` | 0 | `0 Error(s)`; latest run produced `566 Warning(s)` from existing warnings. |
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~WorldObjectPassCoordinator"` | 0 | `Passed! - Failed: 0, Passed: 11, Skipped: 0, Total: 11`. |
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~WorldFramePassCoordinator\|FullyQualifiedName~WorldObjectPassCoordinator"` | 1 | Existing `WorldFramePassCoordinatorTests` expected the optional WDL pass by default; this failure is outside the emitted-light source path. |
| `dotnet test wow-viewer/WowViewer.slnx -c Debug --no-build` | 1 | Existing `LkToAlphaRoundTripTests.AlphaToLk_FlagContract_AllowsAlphaRoundTripThroughLkBytes` failure (`Alpha round-trip drift 0.9333 exceeds one byte step`); not caused by this lighting slice. |

## Acceptance Criterion Evidence

| Criterion | Evidence | Status |
|---|---|---|
| FR-004: Spatial `SceneLightManager` aggregates active light sources with position, color, intensity, radius. | `SceneLightManager` collects `SceneLight` records and selects nearest active lights by target AABB. `WorldScene.RebuildSceneLights` collects WMO `MOLT`, MDX `LITE`, M2 lights, and WMO-internal doodad emitters. | Partial: outdoor ambient/sun representation remains open in T010. |
| FR-005: WMO shaders evaluate up to eight active local lights affecting each group/cluster. | `WmoRenderer` shader declares `uLocalLightCount`, `uLocalLightPos[8]`, `uLocalLightColor[8]`, `uLocalLightIntensity[8]`, `uLocalLightStart[8]`, and `uLocalLightEnd[8]`; `UploadLocalLights` queries up to eight lights and the fragment shader accumulates bounded point-light diffuse. | Source complete for WMO shell slice; runtime visual proof remains operator-owned. |
| FR-006: Terrain shaders evaluate nearby local lights affecting each chunk. | Not touched in this slice. | Open (T012). |
| FR-007: Doodad shaders evaluate nearby scene lights affecting their bounding volume. | Doodads can now emit into the scene-light manager, but doodad/model shaders do not yet consume external scene lights. | Open (T013). |

## Proof Boundary

Build success proves the source compiles. Unit checks prove adjacent world pass infrastructure still passes the focused object-pass test set. No runtime visual, FPS, real-client, terrain illumination, or doodad-consumer proof is claimed here.
