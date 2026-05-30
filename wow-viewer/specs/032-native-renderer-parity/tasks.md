# Tasks: Native Renderer Parity

**Input**: Design documents from `wow-viewer/specs/032-native-renderer-parity/`

**Prerequisites**: plan.md (required), spec.md (required)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US5)

---

## Phase 1: Terrain Mesh Topology and Distance LOD (US1 — P1)

**Goal**: Terrain renders with correct 145-vertex topology, per-cell diagonal splits, distance-based texture LOD, low-detail far mesh, and shadow overlay.

**Independent Test**: Load a terrain tile, compare close-range mesh detail and far-range LOD against native client screenshots.

- [ ] T001 [US1] Extend `WorldTerrainChunkData` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainChunkData.cs` — add 145-vertex layout (81 outer + 64 inner), cell diagonal split flags for 8x8 grid, and 256 face plane storage with triangulation variant index.
- [ ] T002 [US1] Extend `WorldTerrainHeightmapData` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainHeightmapData.cs` — add inner vertex height values (entries 81-144 of MCVT) and per-cell face plane normals (Y/Z/X byte packing, scale ~0.251388).
- [ ] T003 [P] [US1] Create `WorldTerrainLodSelector` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainLodSelector.cs` — compute LOD level per chunk from camera distance: AllLayers (<textureLodDist), FadingLayers (<textureLodDist+256), SingleLayer (>=textureLodDist+256), LowDetail (beyond fog distance). Expose fade alpha for FadingLayers.
- [ ] T004 [P] [US1] Create `WorldTerrainLowDetailBuilder` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainLowDetailBuilder.cs` — generate 17x17 fog-colored vertex mesh from subsampled outer vertices, with index buffer for 16x16 grid triangulation.
- [ ] T005 [US1] Create `TerrainRenderPipeline` in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/TerrainRenderPipeline.cs` — orchestrate per-layer terrain rendering: for each texture layer, set texture on Tex0, alpha mask on Tex1 (if `props & 0x100`), configure texgen matrices (detail + alpha, camera-relative), handle per-layer props (animated UV offset `props & 0x40`, lighting disable `props & 0x80`), select LOD level via `WorldTerrainLodSelector`, render 145-vertex triangle list.
- [ ] T006 [US1] Implement texture LOD fade in `TerrainRenderPipeline` — when LOD is FadingLayers, compute alpha = `(256 - (dist - textureLodDist)) * 128.0 / 256.0` clamped [0,1], apply to extra layer alpha; when SingleLayer, render only base texture; when LowDetail, use `WorldTerrainLowDetailBuilder` mesh with fog color.
- [ ] T007 [P] [US1] Create `WorldTerrainShadowOverlay` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainShadowOverlay.cs` — post-layer blend pass: set `MatDiffuse = shadowColor`, blend mode 2, shadow texture on Tex0 + shadow mod texture on Tex1, render same triangle list.
- [ ] T008 [US1] Wire `TerrainRenderPipeline` into `WorldFramePassCoordinator` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs` — add terrain render pass that uses the pipeline, including shadow overlay when enabled.
- [ ] T009 [P] [US1] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/TerrainLodSelectorTests.cs` — test LOD level computation (AllLayers, FadingLayers, SingleLayer, LowDetail), fade alpha clamping, edge cases (distance = 0, negative fade).
- [ ] T010 [US1] Validate against staged client — load terrain tile from `I:\parp\parp-tools\output\tmp\wowarchive-clients\`, compare close-range 145-vertex mesh detail, medium-range layer fade, and far-range low-detail mesh against native client screenshots.

**Checkpoint**: Terrain renders with correct topology, LOD, and shadow overlay. Build passes. Tests green.

---

## Phase 2: WMO Interior/Exterior Render Dispatch (US2 — P1)

**Goal**: WMO groups render with correct interior/exterior pass, per-batch MOMT flags, lightmap split, and interior fog.

**Independent Test**: Load a dungeon WMO (interior) and exterior WMO, compare lighting, fog, window brightness against native client.

- [ ] T011 [US2] Create `WorldWmoGroupRenderDispatch` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wmo/WorldWmoGroupRenderDispatch.cs` — select render path based on `group.flags & 0x48`: Interior (flags & 0x48 == 0 → MOCV, no dynamic lighting), Exterior (flags & 0x48 != 0 → dynamic lighting). Skip groups with `flags & 0x88`. Always-render groups with `flags & 0x10000` bypass portal walk.
- [ ] T012 [P] [US2] Create `WorldWmoBatchMaterialFlags` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wmo/WorldWmoBatchMaterialFlags.cs` — evaluate per-batch MOMT flags: bit0 (disable lighting), bit1 (disable fog), bit2 (disable culling), 0x10 (emissive — self-illuminated), 0x20 (window-lit — receive exterior sun in interior). Return a `WmoBatchRenderState` struct per batch.
- [ ] T013 [P] [US2] Create `WorldWmoLightmapPassSelector` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wmo/WorldWmoLightmapPassSelector.cs` — interior lightmap: lighting OFF, lightmap UV on tex1; exterior lightmap: lighting ON, no lightmap on tex1 (tex1 free for other use). Select based on group interior/exterior flag from dispatch.
- [ ] T014 [P] [US2] Create `WorldWmoInteriorFogState` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wmo/WorldWmoInteriorFogState.cs` — interior fog from `DayNightGetInfo()->intFog`: start, end, color. Apply only when camera is inside the WMO and `intFog != 0`. Provide fog parameters to render pipeline.
- [ ] T015 [US2] Create `WmoGroupRenderPipeline` in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/WmoGroupRenderPipeline.cs` — orchestrate full WMO group render: (1) dispatch via `WorldWmoGroupRenderDispatch`, (2) evaluate per-batch flags via `WorldWmoBatchMaterialFlags`, (3) select lightmap pass via `WorldWmoLightmapPassSelector`, (4) apply interior fog via `WorldWmoInteriorFogState`, (5) render each batch with accumulated state.
- [ ] T016 [US2] Wire `WmoGroupRenderPipeline` into `WorldFramePassCoordinator` — add WMO render pass that uses the pipeline, handling group visibility (portal walk or always-render) and interior/exterior dispatch.
- [ ] T017 [P] [US2] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WmoGroupRenderDispatchTests.cs` — test dispatch logic (interior vs exterior from flags), skip groups (0x88), always-render groups (0x10000), batch flag evaluation (each MOMT bit), lightmap Int vs Ext selection.
- [ ] T018 [US2] Validate against staged client — load interior dungeon WMO (verify MOCV lighting, no dynamic lights, interior fog) and exterior WMO (verify dynamic lighting, window-lit flag behavior).

**Checkpoint**: WMO groups render with correct pass selection, per-batch flags, lightmaps, and interior fog. Build passes. Tests green.

---

## Phase 3: Liquid Rendering with Animation and Type Dispatch (US3 — P2)

**Goal**: Water surfaces render with animated textures, correct interior/exterior behavior, and magma type dispatch.

**Independent Test**: View exterior water (ocean/river) and interior dungeon water, verify animation, fog difference, and magma rendering.

- [ ] T019 [US3] Create `WorldLiquidTypeDispatch` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Liquid/WorldLiquidTypeDispatch.cs` — water path for types 0/4/8, magma path for types 2/3/6/7. Return `LiquidRenderPath { Water, Magma }` enum.
- [ ] T020 [P] [US3] Create `WorldLiquidAnimationState` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Liquid/WorldLiquidAnimationState.cs` — 30-frame cycling: frame index = `(curTimeSec % secsPerLoop) * 30.0 / secsPerLoop`, per-type `secsPerLoop`. Expose current frame index and texture filter mode (LinearMipNearest default, Anisotropic if enabled, LinearMipLinear if trilinear).
- [ ] T021 [US3] Extend `WorldLiquidChunkData` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Liquid/WorldLiquidChunkData.cs` — add interior/exterior distinction (from parent WMO group flags), material `diffColor` for interior water, `DayNightGetInfo()->light.WaterArray[3]` for exterior water color.
- [ ] T022 [US3] Create `LiquidRenderPipeline` in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/LiquidRenderPipeline.cs` — orchestrate: (1) type dispatch via `WorldLiquidTypeDispatch`, (2) animation state via `WorldLiquidAnimationState`, (3) interior: vertex color from material diffColor + interior fog; exterior: color from WaterArray[3] + no interior fog, (4) magma: separate render path, (5) render.
- [ ] T023 [US3] Implement river texgen scrolling in `LiquidRenderPipeline` — for terrain chunk water, set river texture on Tex0, texgen on Tex1 (0.14 scale + camera translation offset), animated flow.
- [ ] T024 [P] [US3] Add specular water pixel shader stub in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/ShaderRegistry.cs` — register `psOcean0` shader path as opt-in, enabled when `enableSpecularWater` is true. Implementation defers to backend-specific shader compilation.
- [ ] T025 [US3] Wire `LiquidRenderPipeline` into `WorldFramePassCoordinator` — add liquid render pass after terrain and WMO passes.
- [ ] T026 [P] [US3] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/LiquidTypeDispatchTests.cs` — test type dispatch (0→Water, 2→Magma, 4→Water, 7→Magma), animation cycling (frame index at various times), interior vs exterior color selection.
- [ ] T027 [US3] Validate against staged client — view exterior ocean (animation, day/night color), interior dungeon water (interior fog, material color), magma pool (magma path).

**Checkpoint**: Liquid renders with animation, correct type dispatch, and interior/exterior behavior. Build passes. Tests green.

---

## Phase 4: Per-Chunk Lighting and Shadow System (US5 — P3)

**Goal**: Terrain and WMO groups are lit by sun + up to 7 local lights per chunk, matching the native client's `SelectLights` behavior.

**Independent Test**: View terrain with torches/lamps, verify local light contribution on nearby surfaces.

- [ ] T028 [US5] Create `WorldDayNightInfo` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Lighting/WorldDayNightInfo.cs` — time-of-day lighting singleton: sun direction, sun color, ambient color, exterior fog (start/end/color), interior fog (start/end/color, enabled flag), water color array (WaterArray[0..3]). Configurable via time-of-day setter.
- [ ] T029 [US5] Create `WorldLightSelectionState` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Lighting/WorldLightSelectionState.cs` — per-chunk light array: light 0 = sun (always), lights 1-7 = up to 7 local lights from `lightLinkList`, distance-priority selection, disable unused slots explicitly. Max 8 lights total.
- [ ] T030 [US5] Wire terrain per-chunk light selection into `TerrainRenderPipeline` — before rendering each chunk, call `SelectLights` to populate light array, pass to shader as uniform array (position, color, attenuation per light).
- [ ] T031 [US5] Wire WMO exterior lighting into `WmoGroupRenderPipeline` — for exterior groups, call `SelectLight` if within fog distance; for interior groups, lighting is already OFF (from Phase 2).
- [ ] T032 [P] [US5] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WorldLightSelectionTests.cs` — test 8-light cap, distance-priority ordering, sun always at index 0, empty slots disabled.
- [ ] T033 [US5] Validate against staged client — view terrain with local lights (torches), verify light contribution on nearby chunk surfaces; view WMO exterior with local lights.

**Checkpoint**: Per-chunk lighting with local lights works. Build passes. Tests green.

---

## Phase 5: Debug Toggle System (US4 — P3)

**Goal**: The `CWorld::enables` bitfield is runtime-toggled, enabling visual debugging of normals, wireframes, portals, culling, shadows, WMO textures/lightmaps, terrain LOD.

**Independent Test**: Toggle each debug bit, verify expected visual overlay.

- [ ] T034 [US4] Create `WorldDebugEnables` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Debug/WorldDebugEnables.cs` — 32-bit bitfield with named flags: Terrain (0x02), TerrainCull (0x20), Shadow (0x40), ShowPortals (0x100), PortalVis (0x200), MapObjLightMode (0x400), MapObjTextures (0x800), DebugBSP (0x10000), CrappyBatches (0x20000), LowDetail (0x4000000), Water (0x1000000), ShowTris (0x20000000), ShowNormals (0x40000000). Toggle methods, enable/disable individual bits, query methods.
- [ ] T035 [US4] Wire terrain-related enables — Terrain (skip terrain render when off), Shadow (gate shadow overlay pass), LowDetail (force 17x17 mesh), TerrainCull (visualize culled chunks), ShowNormals (normal lines), ShowTris (wireframe overlay).
- [ ] T036 [US4] Wire WMO-related enables — MapObjTextures (skip WMO textures when off), MapObjLightMode (toggle lightmaps vs vertex color), ShowPortals (render portal wireframes), DebugBSP (render BSP polygons), CrappyBatches (highlight low-quality batches).
- [ ] T037 [US4] Wire liquid enables — Water (skip water render when off).
- [ ] T038 [P] [US4] Create `WorldDebugOverlayRenderer` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Debug/WorldDebugOverlayRenderer.cs` — debug draw calls: vertex normal lines (from position along normal), wireframe triangle edges, portal semi-transparent colored quads, BSP polygon outlines. Produce overlay render packets consumed by backend.
- [ ] T039 [US4] Expose debug toggles in `WowViewerDesktopApp` — keyboard shortcuts (e.g., F1=ShowTris, F2=ShowNormals, F3=ShowPortals, F4=Shadow) or debug panel UI. Toggle bits on `WorldDebugEnables` instance.
- [ ] T040 [P] [US4] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WorldDebugEnablesTests.cs` — test toggle on/off, bit positions, combined states, default values.
- [ ] T041 [US4] Validate each toggle — verify ShowNormals renders lines, ShowTris renders wireframe, ShowPortals renders portal quads, Shadow toggles overlay, Terrain/Water toggles visibility.

**Checkpoint**: Debug toggle system works. Each bit produces expected visual result. Build passes. Tests green.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Terrain LOD)**: No hard dependency on other specs. Can start immediately. Soft dependency on spec 031 data structures (if not yet available, Phase 1 builds its own terrain data extensions).
- **Phase 2 (WMO Dispatch)**: No hard dependency on Phase 1. Can start in parallel with Phase 1. Soft dependency on spec 030 architecture doc (already written).
- **Phase 3 (Liquid)**: Depends on Phase 1 (terrain LOD affects water visibility) and Phase 2 (interior fog for interior water).
- **Phase 4 (Lighting)**: Depends on Phase 1 (terrain rendering exists for per-chunk lights) and Phase 2 (WMO rendering exists for exterior lighting).
- **Phase 5 (Debug Toggles)**: Depends on Phases 1-4 (toggles are meaningless without the rendering features they control).

### Parallel Opportunities

- **Phases 1 and 2** can run in parallel (different files, different render paths).
- Within Phase 1: T003 + T004 can run in parallel; T007 can run in parallel with T005/T006.
- Within Phase 2: T012 + T013 + T014 can run in parallel (different files).
- Within Phase 3: T020 and T024 can run in parallel with T019/T021.
- Within Phase 4: T028 and T029 can run in parallel.
- Within Phase 5: T034 + T038 + T040 can start in parallel.

### Execution Strategy

1. **Phase 1 + Phase 2 in parallel** (P1 priority, biggest visual wins)
2. **Phase 3** after Phases 1+2 complete and validated
3. **Phase 4** after Phases 1+2 complete and validated (can overlap with Phase 3)
4. **Phase 5** after Phases 1-4 all complete and validated

---

## Task Count

- **Total**: 41 tasks
- **Phase 1**: 10 tasks (terrain LOD)
- **Phase 2**: 8 tasks (WMO dispatch)
- **Phase 3**: 9 tasks (liquid rendering)
- **Phase 4**: 6 tasks (lighting)
- **Phase 5**: 8 tasks (debug toggles)
- **Parallel tasks**: 15 tasks marked [P]
