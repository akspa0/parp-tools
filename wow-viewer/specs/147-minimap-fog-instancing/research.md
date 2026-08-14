# Research: Minimap, Fog-Bounded Residency, and Doodad Instancing

## Current owners inspected

| Concern | Current owner | Evidence | Planning implication |
|---|---|---|---|
| Fullscreen minimap | `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs` `DrawFullscreenMinimap` and `DrawInteractiveMinimapSurface` | Fullscreen and docked views call the shared helper with `MinimapTeleportMode.Armed`; the helper uses an ImGui `InvisibleButton`, tracks drag state, and classifies clicks on mouse release | Extract the gesture decision from ImGui timing/window state so drag and three-click behavior are deterministic and shared |
| Fullscreen ownership | `src/viewer/WoWViewer/ViewerApp.cs` around the shell pass and post-shell overlay pass | The fullscreen draw path is reachable from both the shell rendering path and the final overlay path when the tab UI is active, producing duplicate windows and duplicate interaction IDs | Make fullscreen minimap rendering have exactly one owner before judging input behavior |
| Map coordinate conversion | `ViewerApp_MinimapAndStatus.cs` `TryGetMinimapClickTarget` / `TeleportCameraToMinimapTile` and `MinimapHelpers` | Screen X/Y are translated into map tile Y/X and then into the existing `WoWConstants.MapOrigin` world convention | Preserve the current coordinate owner; test it independently from UI events |
| Active fog | `src/viewer/WoWViewer/Terrain/WorldScene.cs` | `WorldScene.Render` resolves lighting/LIT/DBC/global/user fog into `fogStart`/`fogEnd`; object visibility already receives `fogEnd` | The effective fog value is available and must remain the only source of truth |
| ADT streaming | `src/viewer/WoWViewer/Terrain/TerrainManager.cs` `UpdateAOI` / `ComputeStreamingTargets` | `UpdateAOI` passes `_terrainRenderer.Lighting.FogEnd`, but `ComputeStreamingTargets` explicitly discards it and derives targets from manual/default tile counts | Replace the ignored input with a bounded fog coverage contract while retaining explicit detail controls as a policy/diagnostic input |
| Tile ordering | `src/core/WowViewer.Core.Runtime/World/DirectionalTileSelector.cs` and `CameraTileWindowSelector.cs` | The directional selector protects a near-field square and then orders a bounded forward cone; selection and retained camera window are separate | Keep directional ordering for priority, but do not let it evict nearby side tiles inside the fog window |
| Object admission | `WorldScene.cs` and `WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs` | WMO/MDX collectors already accept `fogEnd` and apply object-specific cull distances; Phase 8P widened resident-neighbor WMO behavior | Feed the same normal fog coverage set to tile-owned object admission, preserving WMO containment and capture exceptions |
| Doodad batches | `Rendering/ModelRenderer.cs`, `M2Renderer.cs`, `WmoRenderer.cs`, `IGpuInstancedModelRenderer.cs`, `IGpuInstancedWmoRenderer.cs` | Existing renderer interfaces support batch/GPU-instance paths; WMO groups opaque doodads by renderer; some paths remain unbatched/fallback | Define compatibility and asset ownership above individual renderer calls before expanding batching |
| Existing related specs | Specs 136, 137, 142 | Spec 136 owns previous M2/WMO doodad optimizations; 137 owns minimap teleport consistency; 142 owns spatial/residency architecture | This feature is a bounded follow-up and must link its tasks to those owners rather than duplicate them |

## Decisions

1. **Use active fog, not a new fog setting.** `WorldScene` remains responsible for resolving the
   active source hierarchy. The streaming layer consumes that resolved value.
2. **Use conservative bounds intersection.** Tile centers are insufficient near boundaries; a tile
   whose bounds can contribute within `fogEnd` stays eligible.
3. **Keep two concepts separate.** Fog-bounded normal coverage controls ordinary admission and draw
   eligibility. Capture-path preload and full-load diagnostic modes are explicit leases/exceptions.
4. **Use directional ordering, not directional exclusion, inside the normal coverage window.** The
   old FOV selector can prioritize work, but it must not make nearby side/rear tiles disappear.
5. **Batch by compatibility key.** “One model” is not a safe batch key when materials, alpha,
   animation, particles, ribbons, or fade state differ. Static compatible buckets are the first
   target; correctness fallbacks remain named and measured.
6. **Repair interaction before renderer changes.** A broken minimap prevents reliable camera and
   capture validation, so the pure gesture contract is the first implementation phase.

## Known risks

- ImGui `IsMouseDoubleClicked`/mouse-release timing may not provide a reliable triple-click event
  when the parent fullscreen window and the invisible surface both observe input. A pure gesture
  state machine avoids making the renderer depend on frame timing.
- Duplicate fullscreen draw ownership can make an otherwise correct helper appear broken by
  processing the same pointer state twice.
- `UpdateAOI()` can run before the render pass resolves a new spatial fog value. The implementation
  must define a frame snapshot or an ordering handoff so streaming and drawing use the same value.
- An active fog range may change while the camera is stationary because LIT/DBC/local lighting is
  spatial. Residency invalidation must be cheap and hysteretic.
- A tile bounds radius can admit more tiles than the old manual count. Diagnostics must show why and
  preserve the manual detail control as a deliberate budget/quality policy where appropriate.
- Existing `MdxRenderer`/legacy adapters and transparent/effect paths do not all support GPU
  instancing. The first batch slice must fail open to the existing path.
- WMO internal doodad sets have placement/group/portal semantics that cannot be reduced to a global
  asset list without losing correctness.

## Non-decisions

- No claim is made here about WDL horizon quality, sky, stars, audio, shader parity, or a fixed FPS.
- No client-specific DBC schema or hardcoded client root is introduced.
- No whole-map load, whole-map scene graph rewrite, or reader duplication is authorized.
