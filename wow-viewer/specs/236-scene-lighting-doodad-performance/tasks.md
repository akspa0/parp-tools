# Tasks: Spec 236 — Unified Scene Lighting, Doodad Performance & Client-Constrained World Pipeline

**Feature Branch**: `v0.5.4-dev`

**Owner Spec**: `wow-viewer/specs/236-scene-lighting-doodad-performance/spec.md`

Governance: AGENTS.md §9 applies. Tasks are checked `[x]` only with verifiable receipts.

---

## Phase 1 — Shading & Normal Corrections (Immediate Fix)

- [x] **T001**: Remove unconditional `!gl_FrontFacing` normal inversion in `ModelRenderer.cs` and `M2Renderer.cs`. (Receipt: `evidence/phase1-shading-fix.md`)
- [x] **T002**: Implement Half-Lambert diffuse wrapping in `ModelRenderer.cs` and `M2Renderer.cs` fragment shaders. (Receipt: `evidence/phase1-shading-fix.md`)
- [ ] **T003**: Verify standalone model viewer lighting on 0.5.3 models (e.g. `HumanMale.mdx`, `StormwindStreetlamp01.mdx`) — ensure models are evenly illuminated without dark tinting. (Operator visual proof)
- [x] **Gate 1**: Solution builds with 0 errors; unit tests pass; receipt written to `evidence/phase1-shading-fix.md`.

---

## Phase 2 — Multi-Surface Light Casting

- [ ] **T010**: Create `SceneLightManager.cs` to collect outdoor ambient/sun, WMO `MOLT` point lights, and placed doodad `LITE` lights into a spatial lookup structure.
- [x] **T011**: Modernize `WmoRenderer.cs` shader to accept up to 8 local point lights (`uLocalLightPos`, `uLocalLightColor`, `uLocalLightIntensity`, `uLocalLightStart`, `uLocalLightEnd`). (Receipt: `evidence/phase2-wmo-light-casting-slice.md`)
- [ ] **T012**: Modernize `TerrainRenderer.cs` shader to accept nearby local lights for terrain illumination.
- [ ] **T013**: Wire `SceneLightManager` in `WorldScene.cs` to upload nearby lights during WMO, terrain, and doodad passes.
- [ ] **Gate 2**: Verify torches and braziers cast light onto surrounding WMO geometry and ground surfaces; receipt written to `evidence/phase2-light-casting.md`.

2026-09-16 slice note: `evidence/phase2-wmo-light-casting-slice.md` lands the source-only WMO shell consumer plus WMO `MOLT`, MDX `LITE`, M2 light, and WMO-internal doodad light collection. T010 remains open until outdoor ambient/sun handling is represented in the manager contract; T012/T013/Gate 2 remain open until terrain and doodad consumers plus operator runtime visual proof are complete.

---

## Phase 3 — Doodad Rendering Performance & GPU Instancing

- [ ] **T020**: Unify WMO doodad submission across placements in `WorldScene.cs` using `IGpuInstancedModelRenderer`.
- [ ] **T021**: Implement spatial frustum culling for doodad clusters.
- [ ] **T022**: Profile draw calls and frame times on dense scenes (Ironforge / Shattrath); verify reduction in unbatched draws.
- [ ] **Gate 3**: Measured draw call reduction receipt written to `evidence/phase3-doodad-performance.md`.

---

## Phase 4 — Client-Constrained Map Generator

- [ ] **T030**: Update `BiomePalette.ForTheme` and `TemplatedTerrainGenerator.cs` to dynamically query active client's listfile via `IDataSource`.
- [ ] **T031**: Constrain 0.5.3 Alpha generation to authentic Alpha `.mdx` models and Alpha tilesets; reject `.m2` and expansion paths.
- [ ] **T032**: Add automated unit test asserting generated 0.5.3 maps contain zero `.m2` models and zero `EXPANSION02` paths.
- [ ] **Gate 4**: Unit tests pass; receipt written to `evidence/phase4-client-constrained-generator.md`.

---

## Phase 5 — Map Merge Save Pipeline & GLB Export Fix

- [ ] **T040**: Change `ExportDir` in `ViewerApp.cs` to resolve to project workspace root `output/export/` instead of `bin/Debug`.
- [ ] **T041**: Implement `MapSaveService.cs` to save composed cartography layers to Alpha 0.5.3 WDT/ADT and LK v18 ADT formats (implementing Spec 234).
- [ ] **T042**: Wire Save buttons in Archaeology and Editor Data I/O pages with real-time status reporting.
- [ ] **T043**: Fix GLB viewport export buttons in `ViewerApp_Editor.cs` when viewing terrain maps.
- [ ] **Gate 5**: Verified saved Alpha WDT reloads in viewer and GLB file exports to `output/export/`; receipt written to `evidence/phase5-save-export-pipeline.md`.
