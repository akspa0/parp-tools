# Spec 236 Phase 2 Evidence — Doodad/Model External Scene-Light Consumer (FR-007)

Date: 2026-09-18

## Scope

This source-only slice extends the Phase 2 multi-surface light casting to doodad/model surfaces:
world doodads and WMO-internal doodads now evaluate nearby *external* scene lights, not only their
own `LITE` data.

- Added an optional `SceneLightManager?` to `IModelRenderer.BeginBatch` / `RenderWithTransform`
  (additive; `null` preserves the prior behavior exactly).
- `MdxRenderer` (in `ModelRenderer.cs`) now consumes the shared scene-light set when a manager is
  supplied. The manager already contains this model's own emitted omni lights, so a doodad is lit by
  its neighbours and itself without double-counting the emitter. Ambient-type MDX lights still feed
  `uLocalAmbientColor` independently.
- `M2Renderer` forwards the manager to its legacy-backed renderer.
- `WmoRenderer` threads its `sceneLights` into the opaque/transparent WMO-internal doodad draws.
- `WorldScene` passes `_sceneLightManager` into the doodad unbatched, state-hoisted, and transparent
  passes (and the WMO doodad-batch fallback).

### Deliberate remaining boundary (why T013 stays open)

The **GPU-instanced opaque doodad batch** path (`IGpuInstancedModelRenderer.BeginGpuInstanceBatch`)
remains base-lit. An instanced batch aggregates many placements of one model into a single draw, and
its shader carries one light set — it cannot represent per-placement nearby lights. Disabling
instancing globally whenever any scene light exists would regress doodad throughput, which Spec 236
US3 explicitly exists to protect. Per-placement routing (instance when no light reaches the
placement, unbatched + lights otherwise) is the correct next step and is shared with Spec 242.
Native (non-legacy) `M2Renderer` sections likewise do not yet consume external lights.

## Files Changed

| Path | Change |
|---|---|
| `src/viewer/WoWViewer/Rendering/IModelRenderer.cs` | Optional `SceneLightManager? sceneLights = null` on `BeginBatch` and `RenderWithTransform`. |
| `src/viewer/WoWViewer/Rendering/ModelRenderer.cs` | `UploadMdxLights(modelMatrix, sceneLights)` selects nearest manager lights (omni, type 0) by transformed model AABB when a manager is present; always computes `uLocalAmbientColor` from own ambient lights; added `_batchSceneLights` + `TransformAabb`. |
| `src/viewer/WoWViewer/Rendering/M2Renderer.cs` | Forwards `sceneLights` to the legacy-backed renderer. |
| `src/viewer/WoWViewer/Rendering/M2CameraPathRenderer.cs` | Signature-only (camera overlays are unlit lines). |
| `src/viewer/WoWViewer/Rendering/WmoRenderer.cs` | Threads `sceneLights` into `RenderOpaqueDoodads` and the transparent WMO-internal doodad draws. |
| `src/viewer/WoWViewer/Terrain/WorldScene.cs` | Passes `_sceneLightManager` to the WMO doodad unbatched/batch-fallback, opaque MDX unbatched/state-hoisted, and transparent MDX passes. |

## Verification Commands

| Command | Exit | Output Evidence |
|---|---:|---|
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --nologo -v q -clp:ErrorsOnly` | 0 | `Build succeeded. 0 Error(s)` (276 pre-existing warnings). |
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TerrainLighting\|FullyQualifiedName~WorldObjectPassCoordinator\|FullyQualifiedName~M2Runtime"` | 0 | `Passed! - Failed: 0, Passed: 47, Skipped: 0, Total: 47`. |

## Acceptance Criterion Evidence

| Criterion | Evidence | Status |
|---|---|---|
| FR-007: doodad shaders evaluate nearby scene lights affecting their bounding volume. | `WorldScene` passes `_sceneLightManager` into the unbatched, state-hoisted, and transparent doodad passes plus the WMO doodad-batch fallback; `MdxRenderer.UploadMdxLights` queries `SceneLightManager.QueryAffecting(model AABB)` and uploads up to eight nearby external lights (omni) into the existing per-fragment local-light slots; `WmoRenderer` threads its `sceneLights` into WMO-internal doodad draws. | **Partial**: instanced-opaque doodad batches and native (non-legacy) M2 do not yet consume external lights. |
| No regression when no manager is supplied. | The new parameter is optional and defaults to `null`; with `null` `UploadMdxLights` takes the unchanged own-LITE path, so standalone model viewing and editor use are identical to before. | Source complete. |

## Proof Boundary

Build success proves the source compiles. The focused test run proves adjacent terrain-lighting math,
world object-pass infrastructure, and M2 runtime tests are unchanged. **No runtime visual, FPS,
real-client, terrain-illumination, or doodad-illumination proof is claimed.** Gate 2 and T013 remain
open pending the instanced/native-M2 completion and the operator's runtime visual witness.