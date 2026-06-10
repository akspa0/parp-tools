# Tasks: Native Renderer Parity

**Input**: Design documents from `wow-viewer/specs/032-native-renderer-parity/`

**Prerequisites**: plan.md (required), spec.md (required)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US8)

---

## Phase 1: Lighting Data Model and LIT Reader (US6 — P1)

**Goal**: The `CurrentLight` data model exists with all 18+ tracks, `.lit` files can be loaded and evaluated, storm/clear blending works. Foundation for all downstream rendering.

**Independent Test**: Load a real `.lit` file, evaluate at noon/midnight/dawn, compare track values against MdxViewer LitLoader output.

- [ ] T001 [US6] Port `LitLoader.cs` data decode to `wow-viewer/src/core/WowViewer.Core.IO/Lighting/LitFileReader.cs` — support versions `0x80000003` through `0x80000005`, all 18+ color tracks (DirectColor=0, AmbientColor=1, SkyTop=2..SkyHorizon=7, ShadowOpacity=8, CloudArray=9-13, WaterArray=14-17), 4 param groups (Clear, Storm, ClearUnderwater, StormUnderwater), float band data (FogEnd, FogStartScaler, SkyFloatBands, ParameterBands). **DO NOT port `EvaluateLighting` spatial selection** — MdxViewer's spatial app has a known bug (horizontal wrapping instead of distance-based falloff). Port only the data decode; spatial selection will be rewritten in T006.
- [ ] T002 [P] [US6] Define `WorldDayNightLightingState` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Lighting/WorldDayNightLightingState.cs` — immutable record/struct with: DirectColor, AmbientColor (Vector3), SkyArray[0..5] (6 x Vector3), ShadowOpacity (Vector3), CloudArray[0..4] (5 x Vector3), WaterArray[0..3] (4 x Vector3), FogEnd (float), FogStartScalar (float), CloudData[1] (float), Darkness (float).
- [ ] T003 [US6] Build `WorldDayNightLightEvaluator` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Lighting/WorldDayNightLightEvaluator.cs` — `CalcLightColors`-style track interpolation: for color tracks (ID 0-0x11), linear byte-per-channel interpolation between adjacent time markers; for float tracks (0x12+), linear float interpolation. Midnight wrap: time range 0-2880, markers can span midnight boundary.
- [ ] T004 [US6] Implement storm/clear blending in `WorldStormBlendEvaluator` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Lighting/WorldStormBlendEvaluator.cs` — `result = clear * (100 - weight) / 100 + storm * weight / 100`; fog float tracks use `* 0.01` scale factor per `CalcLightColors`.
- [ ] T005 [US6] Build `WorldDayNightInfo` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Lighting/WorldDayNightInfo.cs` — singleton with settable `gameTime` (0-2880), `Evaluate()` method returning `WorldDayNightLightingState`, spatial light blending (default light + local zones with distance-based falloff weight), sun direction computation from `gameTime`.
- [ ] T006 [US6] Implement spatial light selection — **rewrite from scratch** (NOT ported from MdxViewer). MdxViewer's `EvaluateLighting` has a known bug: lights wrap horizontally (`><` around map edges) instead of using distance-based falloff from light center positions. The correct model (from Ghidra `CalcLightColors` + `LoadLightsAndFog`): each light has a world-space position, radius, and dropoff; default light has `ChunkX==-1 && ChunkY==-1`; local lights blend based on `clamp((dist - falloffStart) / (falloffEnd - falloffStart), 0, 1)` weight; no horizontal wrapping. For minimap rendering, only the global default light is needed (local zones are irrelevant for orthographic top-down captures).
- [ ] T007 [P] [US6] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WorldDayNightLightingTests.cs` — test track interpolation (noon, midnight, dawn marker boundaries), storm blending (weight 0 = pure clear, weight 100 = pure storm, weight 50 = 50/50), midnight wrap (markers spanning 0), spatial blending (camera near/inside/outside local zone).
- [ ] T008 [P] [US6] Add I/O tests in `wow-viewer/tests/WowViewer.Core.IO.Tests/LitFileReaderTests.cs` — test loading LIT files of each version, verify track counts, verify float band sizes, verify default light detection (ChunkX==-1, ChunkY==-1).
- [ ] T009 [US6] Validate against real `.lit` file — load from staged client MPQ, evaluate at gameTime=0/720/1440/2160/2880, compare DirectColor/AmbientColor/SkyArray/FogEnd against MdxViewer LitLoader output.

**Checkpoint**: Lighting data model produces correct values at any time of day. LIT files load correctly. Build passes. Tests green.

---

## Phase 2: Sky Dome and Fog from Lighting Data (US7 + US8 — P1)

**Goal**: 6-band sky dome, exterior fog from lighting data, interior fog inside WMOs, WMO area fog blending, time-of-day UI control.

**Independent Test**: Set time to noon vs. midnight. Verify sky dome bands change, fog distance changes, fog color changes.

- [ ] T010 [US7] Build `WorldSkyDomeRenderer` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Sky/WorldSkyDomeRenderer.cs` — hemisphere mesh (32 segments, 16 rings), 6-band gradient from `CurrentLight.SkyArray[0..5]` (zenith → below-horizon), camera-following (translates with camera), depth write OFF, depth test OFF. Band transitions via smoothstep or linear interpolation based on vertex height on dome.
- [ ] T011 [US7] Build `WorldSkyClearcolor` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Sky/WorldSkyClearcolor.cs` — clear color from `CurrentLight.SkyArray[5]` (fog/horizon band), or from `CurrentLight.SkyArray[0]` when underwater (matching native `DayNightRenderSky` liquid status check).
- [ ] T012 [P] [US8] Build `WorldExteriorFogState` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Fog/WorldExteriorFogState.cs` — fog parameters from `CurrentLight.FogEnd`, `CurrentLight.FogStartScalar`, fog color from `CurrentLight.SkyArray[5]`. Fog start = `FogEnd * (1.0 - FogStartScalar)`. Blend factor: `1.0 - (dist - fogStart) / (fogEnd - fogStart)` clamped [0,1], matching `ComputeFogBlend`.
- [ ] T013 [P] [US8] Build `WorldInteriorFogState` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Fog/WorldInteriorFogState.cs` — interior fog from `DayNightGetInfo()->intFogInfo` (start, end, color). Applied only when camera is inside a WMO and `intFog != 0`. Overrides exterior fog for WMO interior groups.
- [ ] T014 [US8] Build `WorldWmoAreaFogBlender` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Fog/WorldWmoAreaFogBlender.cs` — up to 4 `SMOFog` zones per WMO group, distance-based blending between overlapping zones using priority queue, matching `QueryCameraFog`.
- [ ] T015 [US7] Replace hardcoded `SkyRenderer.cs` in WowViewer.App with `WorldSkyDomeRenderer` — wire sky dome into frame rendering before terrain/WMO passes, clear color from `WorldSkyClearcolor`.
- [ ] T016 [US8] Wire exterior fog into terrain/WMO/liquid render pipelines — fog uniforms (color, start, end) set from `WorldExteriorFogState` each frame.
- [ ] T017 [US8] Wire interior fog into WMO render pipeline — when camera inside WMO and `intFog != 0`, override fog uniforms with interior fog parameters.
- [ ] T018 [US7] Add time-of-day slider to `WowViewerDesktopApp.cs` — expose `gameTime` control (0-2880, with labels for midnight/noon/dawn/dusk), updates `WorldDayNightInfo.gameTime`.
- [ ] T019 [US7] Validate sky dome + fog — set time to noon/midnight/dawn, verify sky dome band colors change, fog distance changes, fog color changes. Compare against native client screenshots at same time.

**Checkpoint**: Sky dome renders 6-band gradient from lighting data. Fog is driven by lighting data. Time-of-day control works. Build passes. Tests green.

---

## Phase 3: Terrain Mesh Topology and Distance LOD (US1 — P1)

**Goal**: 145-vertex topology, distance-based texture LOD, low-detail far mesh, shadow overlay.

**Independent Test**: Load terrain tile, compare close-range mesh detail and far-range LOD against native client.

- [ ] T020 [US1] Extend `WorldTerrainChunkData` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainChunkData.cs` — add 145-vertex layout (81 outer + 64 inner in interleaved LK order), cell diagonal split flags for 8x8 grid, and 256 face plane storage with triangulation variant index. **IMPORTANT**: The runtime consumes already-reinterleaved data from the I/O layer. Alpha 0.6.0 inner/outer semantic inversion is handled at read time by AlphaWdtReader/AlphaTerrainAdapter — the runtime must NOT re-decode raw MCVT or attempt to use un-reinterleaved Alpha vertex data.
- [ ] T021 [US1] Extend `WorldTerrainHeightmapData` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainHeightmapData.cs` — add inner vertex height values (entries 81-144 of MCVT) and per-cell face plane normals.
- [ ] T022 [P] [US1] Create `WorldTerrainLodSelector` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainLodSelector.cs` — compute LOD level per chunk: AllLayers (<textureLodDist), FadingLayers (<textureLodDist+256), SingleLayer (>=textureLodDist+256), LowDetail (beyond fog distance from `WorldExteriorFogState`). Expose fade alpha for FadingLayers.
- [ ] T023 [P] [US1] Create `WorldTerrainLowDetailBuilder` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainLowDetailBuilder.cs` — generate 17x17 fog-colored vertex mesh from subsampled outer vertices, fog color from `CurrentLight.SkyArray[5]`, index buffer for 16x16 grid triangulation.
- [ ] T024 [US1] Create `TerrainRenderPipeline` in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/TerrainRenderPipeline.cs` — per-layer terrain rendering: texture on Tex0, alpha mask on Tex1 (if `props & 0x100`), texgen matrices (detail + alpha, camera-relative), per-layer props (`props & 0x40` animated UV, `props & 0x80` lighting disable), LOD level from `WorldTerrainLodSelector`.
- [ ] T025 [US1] Implement texture LOD fade in `TerrainRenderPipeline` — FadingLayers: alpha = `(256 - (dist - textureLodDist)) * 128.0 / 256.0` clamped; SingleLayer: base texture only; LowDetail: `WorldTerrainLowDetailBuilder` mesh with fog color from lighting.
- [ ] T026 [P] [US1] Create `WorldTerrainShadowOverlay` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainShadowOverlay.cs` — post-layer blend: `MatDiffuse = shadowColor`, blend mode 2, shadow texture on Tex0 + mod texture on Tex1, blend intensity from `CurrentLight.ShadowOpacity`.
- [ ] T027 [US1] Wire `TerrainRenderPipeline` into `WorldFramePassCoordinator` — terrain render pass with shadow overlay when `enables & 0x40`.
- [ ] T028 [P] [US1] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/TerrainLodSelectorTests.cs` — LOD levels, fade alpha clamping, edge cases.
- [ ] T029 [US1] Validate against staged client — load terrain tile, compare close 145-vertex mesh, medium layer fade, far low-detail mesh, shadow overlay.

**Checkpoint**: Terrain renders with correct topology, LOD, shadow overlay from lighting data. Build passes. Tests green.

---

## Phase 4: WMO Interior/Exterior Render Dispatch (US2 — P1)

**Goal**: WMO groups render with correct interior/exterior pass, per-batch MOMT flags, lightmap split, interior fog.

**Independent Test**: Load dungeon WMO (interior) and exterior WMO, compare against native client.

- [ ] T030 [US2] Create `WorldWmoGroupRenderDispatch` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wmo/WorldWmoGroupRenderDispatch.cs` — interior (flags & 0x48 == 0) vs exterior (flags & 0x48 != 0). Skip groups with `flags & 0x88`. Always-render groups with `flags & 0x10000`.
- [ ] T031 [P] [US2] Create `WorldWmoBatchMaterialFlags` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wmo/WorldWmoBatchMaterialFlags.cs` — per-batch MOMT: bit0 (lighting), bit1 (fog), bit2 (culling), 0x10 (emissive), 0x20 (window-lit).
- [ ] T032 [P] [US2] Create `WorldWmoLightmapPassSelector` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wmo/WorldWmoLightmapPassSelector.cs` — interior: lighting OFF, lightmap on tex1; exterior: lighting ON, no lightmap on tex1.
- [ ] T033 [US2] Create `WmoGroupRenderPipeline` in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/WmoGroupRenderPipeline.cs` — dispatch → flags → lightmap → interior fog (from `WorldInteriorFogState`) → render.
- [ ] T034 [US2] Wire into `WorldFramePassCoordinator` — WMO render pass with group visibility and interior/exterior dispatch.
- [ ] T035 [P] [US2] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WmoGroupRenderDispatchTests.cs` — dispatch logic, skip/always-render, batch flags, lightmap selection.
- [ ] T036 [US2] Validate interior WMO (MOCV lighting, interior fog) + exterior WMO (dynamic lighting, window-lit) against staged client.

**Checkpoint**: WMO groups render with correct pass selection, flags, lightmaps, interior fog. Build passes. Tests green.

---

## Phase 5: Liquid Rendering with Animation and Type Dispatch (US3 — P2)

**Goal**: Animated water textures, correct interior/exterior behavior, magma dispatch, water color from `CurrentLight.WaterArray`.

**Independent Test**: View exterior ocean + interior dungeon water + magma pool.

- [ ] T037 [US3] Create `WorldLiquidTypeDispatch` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Liquid/WorldLiquidTypeDispatch.cs` — water (0/4/8) vs magma (2/3/6/7).
- [ ] T038 [P] [US3] Create `WorldLiquidAnimationState` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Liquid/WorldLiquidAnimationState.cs` — 30-frame cycling: `frameIdx = (timeSec % secsPerLoop) * 30.0 / secsPerLoop`, per-type secsPerLoop.
- [ ] T039 [US3] Extend `WorldLiquidChunkData` — interior/exterior flag, material diffColor for interior, `CurrentLight.WaterArray[3]` for exterior tint.
- [ ] T040 [US3] Create `LiquidRenderPipeline` in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/LiquidRenderPipeline.cs` — type dispatch → animation → color (WaterArray[3] exterior / diffColor interior) → interior fog check → render.
- [ ] T041 [US3] Implement river texgen scrolling in `LiquidRenderPipeline` — river texture on Tex0, texgen on Tex1 (0.14 scale + camera offset).
- [ ] T042 [P] [US3] Add specular water pixel shader stub in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/ShaderRegistry.cs` — register `psOcean0` as opt-in.
- [ ] T043 [US3] Wire into `WorldFramePassCoordinator` — liquid render pass after terrain + WMO.
- [ ] T044 [P] [US3] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/LiquidTypeDispatchTests.cs` — type dispatch, animation cycling, water color from lighting.
- [ ] T045 [US3] Validate exterior water (animation, WaterArray[3] color, river scrolling) + interior water + magma.

**Checkpoint**: Liquid renders with animation, type dispatch, lighting-driven color. Build passes. Tests green.

---

## Phase 6: Per-Chunk/Per-Group Local Light Selection with MDX + WMO Lights (US5 + US9 + US10 — P2)

**Goal**: Sun + up to 7 local lights per terrain chunk and per WMO group. Local lights from three sources: MDX model LITE chunks, WMO MOLT entries, CMapLight world lights. All share the same 8-light budget per rendering unit.

**Independent Test**: View dungeon WMO with torch MDX placements + MOLT entries. Verify warm light pools appear on nearby walls and floor.

- [ ] T046 [US5] Create `WorldLightSelectionState` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Lighting/WorldLightSelectionState.cs` — light 0 = sun (always), lights 1-7 = local lights from merged pool, distance-priority selection, max 8 total.
- [ ] T047 [US9] Create `WorldMdxLightCollector` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Lighting/WorldMdxLightCollector.cs` — extract LITE chunk lights from visible MDX model instances, create CGxLight-style entries (position, color, intensity, attenStart/attenEnd, type: directional vs omni). Skip models beyond fog distance (`camDist >= farFog`), matching `CMap::SelectLight`.
- [ ] T048 [US10] Create `WorldWmoLightCollector` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Lighting/WorldWmoLightCollector.cs` — consume existing `WmoLightDetail` data from MOLT reader, convert to CGxLight-style entries with WMO-to-world transform. Omni lights use attenuation; directional lights use rotation quaternion for direction.
- [ ] T049 [US9+10] Merge MDX + MOLT + CMapLight entries into a single per-frame light pool. Wire into `WorldLightSelectionState` — for each terrain chunk or WMO group, pick the nearest 7 lights from the pool (distance-priority) beyond the sun.
- [ ] T050 [US5] Wire terrain per-chunk light selection into `TerrainRenderPipeline` — `SelectLights` per chunk, pass light array to shader uniforms.
- [ ] T051 [US5] Wire WMO per-group light selection into `WmoGroupRenderPipeline` — same selection for each WMO group.
- [ ] T052 [P] [US9] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/MdxLightCollectorTests.cs` — LITE chunk extraction, attenuation, fog-distance skip, animated light state.
- [ ] T053 [P] [US10] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WmoLightCollectorTests.cs` — MOLT-to-CGxLight conversion, omni vs directional, attenuation, WMO-to-world transform.
- [ ] T054 [P] [US5] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WorldLightSelectionTests.cs` — 8-light cap, distance priority, sun at index 0, merged pool selection.
- [ ] T055 [US9+10] Validate terrain + WMO with MDX torch lights + MOLT lights against staged client — verify light pools on nearby surfaces.

**Checkpoint**: Local lights (MDX + MOLT) illuminate nearby surfaces. 8-light budget respected. Build passes. Tests green.

---

## Phase 7: Shader Family Reconstruction (US11 — P2)

**Goal**: GLSL shaders produce visually native-equivalent output, verified by side-by-side comparison. No "too reddish/yellow" noggit-style color deviation.

**Independent Test**: Render same scene at same time of day in viewer vs. native client. Compare pixel output.

- [ ] T056 [US11] Audit existing inline GLSL shaders in MdxViewer + wow-viewer — catalog which effect families each shader approximates, identify color-space assumptions, document noggit's known pitfall (wrong gamma/color-space = too-warm output).
- [ ] T057 [US11] Write `TerrainEffectFamily` GLSL shader in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/TerrainEffectFamily.cs` — equivalent to `psTerrain`/`psSpecTerrain` with CurrentLight-driven fog/lighting, 4-layer alpha blend, shadow overlay, MCCV tint. Verify no color shift vs. native.
- [ ] T058 [US11] Write `MapObjEffectFamily` GLSL shader in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/MapObjEffectFamily.cs` — equivalent to `MapObjDiffuse`/`MapObjSpecular` with interior MOCV blending + local light contributions, per-batch MOMT flag handling. Verify interior lighting matches native.
- [ ] T059 [US11] Write `LiquidEffectFamily` GLSL shader in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/LiquidEffectFamily.cs` — equivalent to `psOcean0` with animated textures, specular highlights, WaterArray[3] tint. Verify water color matches native at same time.
- [ ] T060 [US11] Write `ModelCombinerFamily` GLSL shader in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/ModelCombinerFamily.cs` — equivalent to `Combiners_Opaque`/`Combiners_Mod`/`Combiners_Mod2x` with alpha blending, specular gating (requires pixel shader support flag), LITE chunk local light support.
- [ ] T061 [US11] Write `SkyEffectFamily` GLSL shader in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/SkyEffectFamily.cs` — equivalent to `DNSky::Render` with 6-band gradient from SkyArray[0..5], clear color from SkyArray[5].
- [ ] T062 [US11] Build `ShaderRegistry` in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/ShaderRegistry.cs` — map native effect names to GLSL programs. Register: psTerrain, psSpecTerrain, MapObjDiffuse, MapObjSpecular, psOcean0, Combiners_Opaque, Combiners_Mod, DNSky. Query by effect name → compiled shader program.
- [ ] T063 [US11] Wire effect-family shaders into render pipelines — `TerrainRenderPipeline` uses `TerrainEffectFamily`, `WmoGroupRenderPipeline` uses `MapObjEffectFamily`, `LiquidRenderPipeline` uses `LiquidEffectFamily`, model rendering uses `ModelCombinerFamily`, `SkyRenderPipeline` uses `SkyEffectFamily`.
- [ ] T064 [US11] Validate color-space behavior — render same scene at same time of day in viewer vs. native client, compare terrain/WMO/water/model pixel values. Document any deviations. Verify no reddish/yellow shift (noggit pitfall).
- [ ] T065 [US11] Validate per-effect-family — terrain (psTerrain), WMO interior (MapObjDiffuse), water (psOcean0), model (Combiners_Opaque), sky (DNSky). Each produces visually native-equivalent output.

**Checkpoint**: GLSL shaders match native client output. No color-space deviation. Build passes. Tests green.

---

## Phase 8: Debug Toggle System (US4 — P3)

**Goal**: Runtime-toggled `CWorld::enables` bitfield for visual debugging.

**Independent Test**: Toggle each debug bit, verify expected overlay.

- [ ] T066 [US4] Create `WorldDebugEnables` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Debug/WorldDebugEnables.cs` — 32-bit bitfield: Terrain(0x02), TerrainCull(0x20), Shadow(0x40), ShowPortals(0x100), MapObjLightMode(0x400), MapObjTextures(0x800), DebugBSP(0x10000), CrappyBatches(0x20000), LowDetail(0x4000000), Water(0x1000000), ShowTris(0x20000000), ShowNormals(0x40000000). Toggle/query methods.
- [ ] T067 [US4] Wire terrain enables (Terrain, Shadow, LowDetail, TerrainCull, ShowNormals, ShowTris).
- [ ] T068 [US4] Wire WMO enables (MapObjTextures, MapObjLightMode, ShowPortals, DebugBSP, CrappyBatches).
- [ ] T069 [US4] Wire liquid enables (Water visibility) + sky enables (sky visibility, fog toggle).
- [ ] T070 [US4] Wire lighting enables (lighting toggle, model lights toggle, MOLT lights toggle).
- [ ] T071 [P] [US4] Create `WorldDebugOverlayRenderer` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Debug/WorldDebugOverlayRenderer.cs` — normal lines, wireframe edges, portal quads, BSP polygons.
- [ ] T072 [US4] Add keyboard shortcuts / debug panel in `WowViewerDesktopApp.cs` — F1=ShowTris, F2=ShowNormals, F3=ShowPortals, F4=Shadow, F5=Water, F6=LowDetail, etc.
- [ ] T073 [P] [US4] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WorldDebugEnablesTests.cs` — toggle on/off, bit positions, combined states.
- [ ] T074 [US4] Validate each toggle — ShowNormals, ShowTris, ShowPortals, Shadow, Terrain, Water, LowDetail, MapObjTextures.

**Checkpoint**: Debug toggle system works. Each bit produces expected visual result. Build passes. Tests green.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Lighting Data)**: No dependencies — can start immediately. This is the foundation.
- **Phase 2 (Sky + Fog)**: Depends on Phase 1 (needs `CurrentLight` values).
- **Phase 3 (Terrain LOD)**: Depends on Phase 2 (fog distance from lighting, low-detail fog color from lighting).
- **Phase 4 (WMO Dispatch)**: Depends on Phase 2 (interior fog from lighting). Can parallel with Phase 3.
- **Phase 5 (Liquid)**: Depends on Phases 2+3+4 (fog, terrain LOD, interior fog).
- **Phase 6 (Local Lights)**: Depends on Phases 3+4 (terrain + WMO rendering exist). Can overlap with Phase 5.
- **Phase 7 (Shader Families)**: Depends on Phases 1-6 (all rendering features exist for shader validation).
- **Phase 8 (Debug Toggles)**: Depends on Phases 1-7 (toggles need features to toggle).

### Parallel Opportunities

- **Phases 3 and 4** can run in parallel (different render paths, different files).
- **Phases 5 and 6** can run in parallel (liquid rendering vs light collection — different files).
- Within Phase 1: T002 can run in parallel with T001; T007 + T008 can run in parallel.
- Within Phase 2: T012 + T013 can run in parallel (different files).
- Within Phase 3: T022 + T023 + T026 can run in parallel with T024.
- Within Phase 4: T031 + T032 can run in parallel with T030.
- Within Phase 5: T038 + T042 can run in parallel with T037.
- Within Phase 6: T052 + T053 + T054 can run in parallel.
- Within Phase 7: T057-T061 can run in parallel (different effect families).
- Within Phase 8: T066 + T071 + T073 can start in parallel.

### Execution Strategy

1. **Phase 1** first (foundation — everything depends on it)
2. **Phase 2** after Phase 1 (sky + fog from lighting)
3. **Phases 3 + 4** in parallel after Phase 2
4. **Phases 5 + 6** in parallel after 3+4
5. **Phase 7** after 5+6
6. **Phase 8** last

---

## Task Count

- **Total**: 74 tasks
- **Phase 1**: 9 tasks (lighting data + LIT reader)
- **Phase 2**: 10 tasks (sky dome + fog)
- **Phase 3**: 10 tasks (terrain LOD)
- **Phase 4**: 7 tasks (WMO dispatch)
- **Phase 5**: 9 tasks (liquid rendering)
- **Phase 6**: 10 tasks (MDX + WMO local lights)
- **Phase 7**: 10 tasks (shader family reconstruction)
- **Phase 8**: 9 tasks (debug toggles)
- **Parallel tasks**: 25 tasks marked [P]
