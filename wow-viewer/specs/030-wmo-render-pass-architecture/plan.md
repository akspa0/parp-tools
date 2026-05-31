# Implementation Plan: WMO Render Pass Architecture
**Branch**: `030-wmo-render-pass-architecture` | **Date**: 2026-05-30 | **Spec**: [spec.md](./spec.md)

## Summary

Implement the WMO render pass architecture confirmed by Ghidra RE of build 3368 in `WowViewer.Core.Runtime`. The spec documents 11 render pass functions, per-batch MOMT material flags, interior/exterior dispatch, lightmap split, interior fog, and liquid type dispatch. This plan covers porting the architecture from `MdxViewer/WmoRenderer.cs` (read-only reference) into the wow-viewer native renderer.

## Technical Context

**Language/Version**: C# / .NET 10
**Primary Dependencies**: WowViewer.Core.IO (WMO readers), WowViewer.Core.Runtime (render pipeline)
**Testing**: `dotnet test wow-viewer/WowViewer.slnx -c Debug`; visual validation against staged client `output/tmp/wowarchive-clients/`
**Target Platform**: Windows x64 (Silk.NET.OpenGL)
**Performance Goals**: Correct pass selection < 16ms per WMO group
**Constraints**: No code in `gillijimproject_refactor/`; MdxViewer is reference only; one phase at a time

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All code in `wow-viewer/` |
| II. Library-First | PASS | Logic in Core.Runtime, app is thin host |
| III. Real-Data Validation | PASS | Validates against staged clients |
| IV. Residual Model Chain | N/A | No ML work |
| V. Streaming-First | N/A | Not a dataset pipeline |
| VI. No Game Client Assumptions | PASS | Uses staged clients only |
| Read-Only Reference | PASS | MdxViewer WmoRenderer is reference only |
| One Phase at a Time | PASS | Phases ordered by dependency |

## Project Structure

### Documentation (this feature)
```
specs/030-wmo-render-pass-architecture/
├── spec.md          # Feature specification (248 lines)
├── plan.md          # This file
└── tasks.md         # Task breakdown
```

### Source Code
```
wow-viewer/src/core/WowViewer.Core.Runtime/
├── Wmo/
│   ├── WorldWmoGroupRenderDispatch.cs   # NEW — interior/exterior pass selection
│   ├── WorldWmoBatchMaterialFlags.cs  # NEW — per-batch MOMT flag evaluation
│   ├── WorldWmoLightmapPassSelector.cs # NEW — lightmap Int/Ext selection
│   ├── WorldWmoInteriorFogState.cs    # NEW — interior fog parameters
│   └── WorldWmoGroupData.cs           # EXISTING — extend with render flags
├── Rendering/
│   ├── WmoRenderPipeline.cs            # NEW — full WMO group render pipeline
│   └── WmoPassTypes.cs                 # NEW — 11 pass function enums
└── World/
    └── WorldFramePassCoordinator.cs    # EXISTING — extend for WMO passes
```

## Implementation Phases

### Phase 1 — WMO Render Pass Dispatch (US1 + US2)
**Goal**: Implement correct interior vs exterior pass selection based on `group.flags & 0x48`. Implement all 11 render pass functions.

**Dependencies**: None.

**Approach**:
1. Build `WorldWmoGroupRenderDispatch` — reads group flags, selects `DAT_00ec1b98` (interior) vs `DAT_00ec1ca0` (exterior) callback path
2. Build `WmoPassTypes` enum — 11 passes: Int, Ext, ColorTex_Int, ColorTex_Ext, ColorTex, LightTex, Lightmap, LightmapTex_Int, LightmapTex_Ext, LightmapTex, Tex, Bsp
3. Build `WmoRenderPipeline` that orchestrates pass selection based on group flags and camera state
4. Handle skip groups (`flags & 0x88`) and always-render groups (`flags & 0x10000`)
5. Validate: Load a dungeon WMO (e.g., Deadmines), verify interior groups use correct pass

**Steps** (max 10):
1. Create `WorldWmoGroupRenderDispatch.cs` — flag evaluation, pass selection logic
2. Create `WmoPassTypes.cs` — 11 render pass enums with Ghidra addresses
3. Create `WorldWmoBatchMaterialFlags.cs` — MOMT flag evaluation (bit0-2, 0x10, 0x20)
4. Create `WorldWmoLightmapPassSelector.cs` — Int (lighting off, lightmap on tex1) vs Ext (lighting on)
5. Create `WorldWmoInteriorFogState.cs` — fog from `DayNightGetInfo()->intFog`
6. Create `WmoRenderPipeline.cs` — full orchestration: dispatch → flags → lightmap → fog → render
7. Wire into `WorldFramePassCoordinator`
8. Add unit tests for dispatch logic, flag evaluation, lightmap selection
9. Validate interior WMO (Deadmines) — MOCV lighting, no dynamic lighting
10. Validate exterior WMO — dynamic lighting, sun, no interior fog

---

### Phase 2 — Per-Batch Material Flags and Lighting (US2)
**Goal**: Evaluate per-batch MOMT flags (lighting, fog, culling, emissive, window-lit) and apply correct state.

**Dependencies**: Phase 1 (dispatch must exist).

**Approach**:
1. Extend `WorldWmoBatchMaterialFlags` — full MOMT flag evaluation per batch
2. Apply state: lighting on/off, fog on/off, backface culling, emissive, window-lit
3. Interior: lighting OFF, MOCV vertex color provides illumination
4. Exterior: lighting ON, sun + local lights
5. Window-lit flag: interior windows receive exterior sun lighting
6. Validate: Render WMO with varied batch flags, verify each flag works

**Steps** (max 8):
1. Extend `WorldWmoBatchMaterialFlags` for all 7 flag bits
2. Apply lighting state per batch (bit0)
3. Apply fog state per batch (bit1)
4. Apply culling state per batch (bit2)
5. Apply emissive state per batch (bit0x10)
6. Apply window-lit state per batch (bit0x20)
7. Add unit tests for each flag combination
8. Validate against native client screenshots (interior/exterior batches)

---

### Phase 3 — Liquid Type Dispatch (US3)
**Goal**: Dispatch liquid rendering based on type: water (0/4/8) vs magma (2/3/6/7), interior vs exterior.

**Dependencies**: Phase 1 (WMO rendering must exist).

**Approach**:
1. Build liquid type dispatch in `WmoRenderPipeline`
2. Interior water: vertex color from material diffColor, interior fog if intFog != 0
3. Exterior water: day/night lighting color from WaterArray[3], normal = (0,0,1)
4. Magma: separate render path for types 2/3/6/7
5. Validate: Render WMO with water interior, water exterior, magma

**Steps** (max 6):
1. Add liquid type dispatch to `WmoRenderPipeline`
2. Implement interior water path (diffColor, interior fog)
3. Implement exterior water path (WaterArray[3], no interior fog)
4. Implement magma path (types 2/3/6/7)
5. Add unit tests for type dispatch
6. Validate against native client (water + magma WMOs)

---

## Complexity Tracking
No constitution violations. All phases are rendering logic in `WowViewer.Core.Runtime`.
