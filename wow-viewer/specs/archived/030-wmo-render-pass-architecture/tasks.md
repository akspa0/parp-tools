# Tasks: WMO Render Pass Architecture
**Input**: Design documents from `wow-viewer/specs/030-wmo-render-pass-architecture/`
**Prerequisites**: plan.md (required), spec.md (required)

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US3)

---

## Phase 1: WMO Render Pass Dispatch (US1 + US2)
**Goal**: Implement correct interior/exterior pass selection and all 11 render pass functions.

**Independent Test**: Load dungeon WMO, verify interior groups use `RenderGroupColorTex_Int`, exterior groups use `RenderGroupColorTex_Ext`.

- [ ] T001 [US1] Create `WorldWmoGroupRenderDispatch.cs` in `wow-viewer/src/core/WowViewer.Core.Runtime/Wmo/` — evaluates `group.flags & 0x48`, selects interior (==0) vs exterior (!=0) path, skips `flags & 0x88`, handles `flags & 0x10000` always-render
- [ ] T002 [US1] Create `WmoPassTypes.cs` in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/` — enum with 11 passes: Int, Ext, ColorTex_Int, ColorTex_Ext, ColorTex, LightTex, Lightmap, LightmapTex_Int, LightmapTex_Ext, LightmapTex, Tex, Bsp (include Ghidra addresses as comments)
- [ ] T003 [US2] Create `WorldWmoBatchMaterialFlags.cs` in `wow-viewer/src/core/WowViewer.Core.Runtime/Wmo/` — evaluates MOMT flags: bit0=lighting, bit1=fog, bit2=culling, 0x10=emissive, 0x20=window-lit
- [ ] T004 [P] [US2] Create `WorldWmoLightmapPassSelector.cs` — interior: lighting OFF + lightmap on tex1; exterior: lighting ON + no lightmap on tex1
- [ ] T005 [P] [US2] Create `WorldWmoInteriorFogState.cs` — fog from `DayNightGetInfo()->intFogInfo`, applied when camera inside WMO and intFog != 0
- [ ] T006 [US2] Create `WmoRenderPipeline.cs` in `wow-viewer/src/core/WowViewer.Core.Runtime/Rendering/` — full orchestration: dispatch → flags → lightmap → fog → render
- [ ] T007 [US2] Wire `WmoRenderPipeline` into `WorldFramePassCoordinator` — WMO render pass with group visibility
- [ ] T008 [P] [US1] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WmoGroupRenderDispatchTests.cs` — dispatch logic, skip/always-render, flag evaluation
- [ ] T009 [US2] Add unit tests for batch material flags, lightmap selection
- [ ] T010 [US2] Validate interior WMO (Deadmines) — MOCV lighting, no dynamic lighting, interior fog if intFog != 0
- [ ] T011 [US2] Validate exterior WMO — dynamic lighting ON, sun + local lights, no interior fog

**Checkpoint**: WMO groups render with correct pass selection and per-batch flags. Build passes. Tests green.

---

## Phase 2: Per-Batch Material Flags and Lighting (US2)
**Goal**: Full MOMT flag evaluation and state application per batch.

**Independent Test**: Render WMO batch with bit0=0 (lighting off), verify lighting disabled for that batch only.

- [ ] T012 [US2] Extend `WorldWmoBatchMaterialFlags` for all 7 MOMT flag bits with individual toggle methods
- [ ] T013 [US2] Apply lighting state per batch (bit0): interior groups ignore, exterior groups use dynamic lighting
- [ ] T014 [P] [US2] Apply fog state per batch (bit1): enable/disable fog rendering for batch
- [ ] T015 [P] [US2] Apply culling state per batch (bit2): backface culling toggle
- [ ] T016 [P] [US2] Apply emissive state per batch (bit0x10): self-illuminated surfaces ignore scene lighting
- [ ] T017 [P] [US2] Apply window-lit state per batch (bit0x20): interior windows receive exterior sun lighting
- [ ] T018 [US2] Validate against native client screenshots — verify each flag produces correct visual
- [ ] T019 [P] [US2] Add unit tests for each flag combination (lighting+fog, emissive+window-lit, etc.)

**Checkpoint**: Per-batch MOMT flags correctly control rendering state. Tests green.

---

## Phase 3: Liquid Type Dispatch (US3)
**Goal**: Water (types 0/4/8) vs magma (types 2/3/6/7) with interior/exterior behavior.

**Independent Test**: View WMO with water interior + exterior, verify different fog behavior and color.

- [ ] T020 [US3] Add liquid type dispatch to `WmoRenderPipeline` — scan first tile for type, select water vs magma path
- [ ] T021 [US3] Implement interior water path — vertex color from material diffColor, interior fog if intFog != 0
- [ ] T022 [US3] Implement exterior water path — day/night lighting color from `WaterArray[3]`, normal = (0,0,1), no interior fog
- [ ] T023 [US3] Implement magma path — separate render for types 2/3/6/7 (different shader/color)
- [ ] T024 [P] [US3] Add unit tests for liquid type dispatch (water types 0/4/8, magma types 2/3/6/7)
- [ ] T025 [US3] Validate interior water (WMO with water, camera inside, intFog != 0)
- [ ] T026 [US3] Validate exterior water (WMO with water, camera outside)
- [ ] T027 [US3] Validate magma rendering (WMO with magma pools)

**Checkpoint**: Liquid renders with correct type dispatch and interior/exterior behavior. Tests green.

---

## Dependencies & Execution Order

### Phase Dependencies
- **Phase 1** → **Phase 2**: Dispatch must exist before per-batch flag evaluation
- **Phase 1** → **Phase 3**: WMO rendering must exist before liquid dispatch

### Parallel Opportunities
- T004 + T005 can run in parallel (different files)
- T014 + T015 + T016 + T017 can run in parallel (different flag bits)
- T024 + T025 + T026 + T027 can run in parallel (different validation scenarios)

### Execution Strategy
1. **Phase 1** first (foundation — dispatch + flags + lightmap + fog)
2. **Phase 2** after Phase 1 (per-batch flag application)
3. **Phase 3** after Phase 1 (liquid dispatch)

---

## Task Count
- **Total**: 27 tasks
- **Phase 1**: 11 tasks (dispatch + flags + lightmap + fog)
- **Phase 2**: 8 tasks (per-batch material flags)
- **Phase 3**: 8 tasks (liquid type dispatch)
- **Parallel tasks**: 8 tasks marked [P]
