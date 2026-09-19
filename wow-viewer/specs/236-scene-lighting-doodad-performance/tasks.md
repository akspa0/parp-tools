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

- [x] **T010**: Create `SceneLightManager.cs` to collect outdoor ambient/sun, WMO `MOLT` point lights, and placed doodad `LITE` lights into a spatial lookup structure. (Receipt: `evidence/phase2-wmo-light-casting-slice.md`, `evidence/phase2-light-casting.md` — outdoor ambient/sun `SceneAmbientLight` contract added 2026-09-18)
- [x] **T011**: Modernize `WmoRenderer.cs` shader to accept up to 8 local point lights (`uLocalLightPos`, `uLocalLightColor`, `uLocalLightIntensity`, `uLocalLightStart`, `uLocalLightEnd`). (Receipt: `evidence/phase2-wmo-light-casting-slice.md`)
- [x] **T012**: Modernize `TerrainRenderer.cs` shader to accept nearby local lights for terrain illumination. (Receipt: `evidence/phase2-light-casting.md` — both chunk and tile programs)
- [ ] **T013**: Wire `SceneLightManager` in `WorldScene.cs` to upload nearby lights during WMO, terrain, and doodad passes. **Partial 2026-09-18**: WMO (prior slice), terrain, and the doodad/model external-light consumer for unbatched/state-hoisted/transparent + WMO-internal doodads all landed with receipts. Remaining: the GPU-**instanced** opaque doodad batch (cannot carry per-placement lights) and native (non-legacy) M2 — see `evidence/phase2-doodad-light-consumer.md`.
- [ ] **Gate 2**: Verify torches and braziers cast light onto surrounding WMO geometry and ground surfaces; receipt written to `evidence/phase2-light-casting.md`. **Operator-owned visual gate still open**; only the source receipt exists.

2026-09-16 slice note: `evidence/phase2-wmo-light-casting-slice.md` lands the source-only WMO shell consumer plus WMO `MOLT`, MDX `LITE`, M2 light, and WMO-internal doodad light collection.

2026-09-18 slice note: `evidence/phase2-light-casting.md` lands the outdoor ambient/sun manager contract (T010), both terrain shaders' nearby-light evaluation (T012), and the terrain consumer wiring in `WorldScene`. `evidence/phase2-doodad-light-consumer.md` lands the doodad/model external-light consumer (FR-007) for unbatched/state-hoisted/transparent world doodads and WMO-internal doodads, with a deliberate non-regressive boundary: GPU-instanced opaque doodad batches and native (non-legacy) M2 remain base-lit. T013 stays unchecked until that boundary closes (shared with Spec 242's per-placement instancing decision); Gate 2 stays unchecked because runtime visual proof is operator-owned.

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
