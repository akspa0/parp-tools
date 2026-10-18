# Technical Design Plan: Client Ground-Truth Renderer Performance & Occlusion

**Feature Branch**: `v0.6.1-dev`
**Owning Spec**: `wow-viewer/specs/273-client-ground-truth-renderer-performance/spec.md`
**Ground Truth Source**: `H:\wowdecomps\3.3.5a\wowdecomp-main` (`wowdecomp`)

---

## 1. Ground Truth Architecture & Reference Points

The WoW 3.3.5 client (`wowdecomp`) handles massive scale, dense cities, and thousands of doodads efficiently using specific architectural principles:

### A. WMO Group Classification & Portal Traversal (`WoW/Source/WorldClient/MapObj.cpp`)
- **Exterior vs Interior Groups**:
  - Groups with flag `0x2000` (or lack of exterior batch count) are strictly **Interior**.
  - Groups without `0x2000` are **Exterior**.
- **Visibility Rules**:
  - When the camera is **outside**:
    - Exterior groups are tested against the main camera view frustum.
    - Interior groups are **never** admitted by simple frustum overlap. They are admitted **only** if an exterior portal on a visible exterior group faces the camera and intersects the frustum, creating a reduced portal view frustum (`CPortalView`) that penetrates into the interior.
  - When the camera is **inside**:
    - The camera's containing interior group is the traversal root.
    - Connected groups are traversed across open portals up to depth 10 (`s_maxPortalDepth = 10`).
    - Exterior groups are reached only if an exit portal is traversed.
- **Current Defect in `wow-viewer`**:
  `WmoRenderer.cs` lines 1813–1819 contains:
  ```csharp
  for (int groupIndex = 0; groupIndex < _frustumVisibleScratch.Length; groupIndex++)
  {
      if (!_frustumVisibleScratch[groupIndex]) continue;
      _runtimeVisibleGroups[groupIndex] = true;
  }
  ```
  This single block forcibly un-culls all interior groups whose AABBs touch the camera frustum, causing the 80,000+ draw-call disaster in Stormwind and dense settlements.

### B. Point Light Evaluation & Decoupled Instancing (`CShaderEffect.cpp`, `WorldObjectPassCoordinator.cs`)
- In `wowdecomp`, local lights are evaluated against object bounds. Placements not within range of any local point light use the standard directional/ambient pass.
- In `wow-viewer`, `WorldScene.cs:3177` currently evaluates `canBatch` as:
  ```csharp
  bool canBatch = _sceneLightManager.Count == 0 && renderer is IGpuInstancedWmoRenderer gpuRenderer ...
  ```
  If even one torch exists in the entire map, all WMO instancing is disabled.
  Furthermore, `WmoRenderer.cs:133` requires `_wmo.Portals.Count == 0`, disabling instancing for any model that possesses portal geometry even if only its exterior shell is drawn.

### C. Spatial Doodad Submission (`CM2Scene.cpp`, `WorldScene.cs`)
- In `wowdecomp`, `CM2Scene` only evaluates models resident within the camera's active sector/view volume.
- In `wow-viewer`, `RebuildSceneLights` loops over all `_mdxInstances` in the entire world every frame (`for (int i = 0; i < _mdxInstances.Count; i++)`), creating large recurring frame stalls.

---

## 2. Phased Implementation Roadmap

### Phase 1: WMO Portal & Interior/Exterior Group Occlusion
- Update `WmoPortalVisibilityEvaluator.cs` and `WmoRenderer.cs`.
- Separate exterior group admission (frustum culling) from interior group admission (portal traversal only).
- Remove the blunt `_frustumVisibleScratch` loop that marks all interior groups visible.
- Add safeguards for malformed or missing portal winding (fail-safe exterior-only or conservative mode with logged diagnostics).
- Ensure `DoodadRefs` are only submitted for actively admitted groups.

### Phase 2: Decoupled WMO Instancing from Scene Lights (`R-10`)
- Modify `WorldScene.cs` and `WorldObjectPassCoordinator.cs`:
  - Replace the global `_sceneLightManager.Count == 0` check with a per-placement bounds intersection test: `_sceneLightManager.IntersectsBounds(placementBounds)`.
  - Placements outside the range of any scene light are batched into GPU instanced draws.
  - Placements touched by a light fall back to individual draw with light uniforms uploaded.
- Relax `SupportsGpuInstancedOpaque` so models with portals can still have their exterior shells instanced when the camera is outdoors and no portals are traversed.

### Phase 3: Spatial M2 & Doodad Tile-Window Light Collection
- In `WorldScene.cs`, replace the linear loop over `_mdxInstances` in `RebuildSceneLights` with a spatial partition query:
  - Query only instances residing on loaded/active tiles (`_terrainManager.LastSelectedTiles` or within fog radius).
  - Use `FlatVisibilityBucket` or tile-indexed dictionary buckets.

### Phase 4: Terrain Shader Alignment & Verification
- Check `TerrainRenderer.cs` against `SVertexShaderConstants`:
  - Verify diffuse/ambient ratios and point light attenuation: $1.0 / (k_c + k_l \cdot d + k_q \cdot d^2)$.
  - Verify texture coordinate conventions.
- Record before/after benchmarks with `profile-render` and HUD telemetry.

---

## 3. Files Impacted

- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalVisibilityEvaluator.cs` (portal walk & interior gate)
- `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs` (runtime visibility & group admission)
- `wow-viewer/src/viewer/WoWViewer/Rendering/WmoDoodadController.cs` (group-gated doodad submission)
- `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` (instancing gate & spatial light collection)
- `wow-viewer/src/viewer/WoWViewer/Terrain/Scene/Frame/WorldObjectPassCoordinator.cs` (per-placement light planning)
- `wow-viewer/src/viewer/WoWViewer/Terrain/TerrainRenderer.cs` (shader constants and attenuation)
