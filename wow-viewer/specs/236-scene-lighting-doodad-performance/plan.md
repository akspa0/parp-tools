# Implementation Plan: Spec 236 — Unified Scene Lighting, Doodad Performance & Client-Constrained World Pipeline

**Feature Branch**: `v0.5.4-dev`

**Owner Spec**: `wow-viewer/specs/236-scene-lighting-doodad-performance/spec.md`

---

## 1. Technical Architecture & Design Decisions

### A. Lighting & Shading Architecture
1. **Normal Transformation & Facing Fix**:
   - In `ModelRenderer.cs` and `M2Renderer.cs`:
     - Delete `if (!gl_FrontFacing) surfaceNormal = -surfaceNormal;` from the main fragment shader.
     - Only invert normals on passes explicitly flagged as two-sided cutouts (`MdlGeoFlags.TwoSided` on transparent passes) where back-face viewing is intended.
2. **Half-Lambert Model Diffuse**:
   - Align MDX and M2 diffuse calculation with WMO shader:
     `float diff = (dot(surfaceNormal, lightDir) * 0.5 + 0.5);`
     `diff = diff * diff;`
3. **Unified Spatial SceneLightManager**:
   - Create `SceneLightManager.cs` owned by `WorldScene`.
   - Collects:
     - Ambient and directional lighting from active weather/LIT/DBC profiles.
     - Active WMO `MOLT` light sources transformed to world space.
     - Active placed doodad `LITE` light sources transformed to world space.
   - For any rendered object (WMO group, terrain chunk, doodad instance/batch):
     - Query `SceneLightManager.GetLightsInRadius(center, radius, maxCount: 8)`.
     - Upload up to 8 local point lights with attenuation start/end, color, and intensity to shader uniform arrays.
4. **WmoRenderer Shader Modernization**:
   - Add `uniform int uLocalLightCount;`, `uniform vec3 uLocalLightPos[8];`, `uniform vec3 uLocalLightColor[8];`, etc., to `WmoRenderer` fragment shader.
   - Accumulate point light contribution across WMO interior and exterior surfaces.

### B. Doodad Rendering Performance Architecture
1. **Multi-Placement GPU Batching**:
   - In `WorldScene.cs` and `WmoRenderer.cs`:
     - Aggregate identical doodad model instances into shared GPU instanced buffers across all visible WMOs, rather than splitting batches per WMO instance.
2. **Coarse Bounding-Sphere Culling**:
   - Test doodad cluster bounds against camera frustum before iterating individual instances.

### C. Client-Constrained Map Generator
1. **Dynamic Listfile Querying**:
   - In `NewMapCreatorService.cs` and `TemplatedTerrainGenerator.cs`:
     - Query `IDataSource.FileExists` or listfile entries.
     - If client is 0.5.3:
       - Map models to authentic `.mdx` paths (`World\Generic\Human\Passive Doodads\Fountains\StormwindFountain01.mdx` or similar verified Alpha listfile assets).
       - Map terrain textures to authentic Alpha tilesets (e.g., `TILESET\ELWYNN\ElwynnGrassBase.blp`, `TILESET\StormwindCity\SW_Cobble_A.blp`).
     - If client is Wrath: map to authentic `.m2` paths and Wrath tilesets.

### D. Map Save Pipeline (Spec 234) & Export Paths
1. **Root-Relative Export Directory**:
   - Change `ExportDir` in `ViewerApp.cs` from `AppDomain.CurrentDomain.BaseDirectory` to resolve relative to workspace directory or user-selected output folder.
2. **MapSaveService**:
   - Implement service to iterate active composed terrain chunks, liquid chunks, and placements.
   - Use `AlphaWdtWriter` to write Alpha 0.5.3 WDT + ADT tiles.
   - Use `LkAdtWriter` to write LK v18 ADT tiles.
   - Expose in Editor Data I/O and Archaeology UI.

---

## 2. Phased Implementation Breakdown

### Phase 1: Immediate Shading & Normal Fix (P0)
- Fix dark MDX shading by removing inappropriate `!gl_FrontFacing` inversion.
- Add Half-Lambert diffuse wrapping to `ModelRenderer.cs` and `M2Renderer.cs`.
- Validate standalone model viewer and world-placed doodads in 0.5.3.

### Phase 2: Multi-Surface Light Casting (P0)
- Implement `SceneLightManager.cs` to aggregate WMO `MOLT` and MDX `LITE` lights.
- Update `WmoRenderer` and `TerrainRenderer` shaders to accept local lights.
- Validate that torches and braziers cast light onto WMO floors/walls and terrain.

### Phase 3: Doodad Batching & Rendering Performance (P1)
- Overhaul doodad collection in `WorldScene.cs` to batch across WMO placements.
- Implement GPU instancing for WMO doodad groups.
- Measure FPS and draw call reduction in dense areas.

### Phase 4: Client-Constrained Map Generator (P1)
- Update `BiomePalette` and `TemplatedTerrainGenerator` to query loaded client listfile.
- Constrain 0.5.3 map generation to authentic Alpha `.mdx` models and tilesets.
- Add regression tests verifying no cross-expansion paths in generated 0.5.3 maps.

### Phase 5: Map Merge Save Pipeline & GLB Export Fix (P1)
- Fix GLB export path to target workspace root `output/export/`.
- Implement `MapSaveService` to save merged maps to Alpha WDT/ADT and LK ADT.
- Wire UI buttons in Editor Data I/O and Archaeology with receipt verification.
