# Implementation Plan: Native Renderer Parity

**Branch**: `032-native-renderer-parity` | **Date**: 2026-05-30 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `wow-viewer/specs/032-native-renderer-parity/spec.md`

## Summary

Implement the rendering pipeline changes needed for `WowViewer.Core.Runtime` to achieve visual parity with the native wowclient.exe (build 3368). The spec identifies 12 rendering gaps across terrain, WMO, water, and lighting. The plan decomposes this into 5 phases: terrain mesh+LOD, WMO interior/exterior dispatch, liquid rendering, lighting+shadow, and debug toggles. Each phase is independently validatable against staged client data.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET.OpenGL (current backend), WowViewer.Core.IO (terrain/WMO readers), WowViewer.Core (shared contracts)

**Storage**: N/A (runtime rendering — no persistent storage changes)

**Testing**: `dotnet test wow-viewer/WowViewer.slnx -c Debug`; visual validation against staged client at `I:\parp\parp-tools\output\tmp\wowarchive-clients\`

**Target Platform**: Windows desktop (OpenGL backend), future Vulkan

**Project Type**: Library + viewer host

**Performance Goals**: 60 fps terrain+WMO rendering at medium distance; LOD must keep frame rate stable at far zoom-out

**Constraints**: No code outside `wow-viewer/`; `gillijimproject_refactor` is read-only reference; one phase at a time with validation

**Scale/Scope**: 5 phases, ~20 tasks total, touches Core.Runtime (terrain/WMO/liquid/rendering) and WowViewer.App (debug toggles UI)

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All code in `wow-viewer/` |
| II. Library-First | PASS | Logic in Core.Runtime, app is thin host |
| III. Real-Data Validation | PASS | Each phase validates against staged clients |
| IV. Residual Model Chain | N/A | No ML work in this spec |
| V. Streaming-First | N/A | No dataset pipeline work |
| VI. No Game Client Assumptions | PASS | Uses staged clients only |
| Read-Only Reference | PASS | MdxViewer terrain/WMO renderers are reference only |
| One Phase at a Time | PASS | Phases ordered by dependency, each validated |
| Bite-Sized Plans | PASS | Max 10 steps per phase, each independently validatable |

## Project Structure

### Documentation (this feature)

```text
specs/032-native-renderer-parity/
├── spec.md          # Feature specification
├── plan.md          # This file
└── tasks.md         # Task breakdown (from speckit-tasks)
```

### Source Code (repository root)

```text
wow-viewer/src/core/WowViewer.Core.Runtime/
├── World/
│   ├── Terrain/
│   │   ├── WorldTerrainTileBuilder.cs      # EXISTING — extend for 145-vertex mesh
│   │   ├── WorldTerrainChunkData.cs         # EXISTING — extend for cell grid
│   │   ├── WorldTerrainHeightmapData.cs     # EXISTING — extend for inner vertices
│   │   ├── WorldTerrainVisualSnapshot.cs    # EXISTING — extend for LOD level
│   │   ├── WorldTerrainLodSelector.cs       # NEW — distance-based LOD decisions
│   │   ├── WorldTerrainShadowOverlay.cs     # NEW — shadow overlay pass data
│   │   └── WorldTerrainLowDetailBuilder.cs  # NEW — 17x17 far-distance mesh
│   ├── Wmo/
│   │   ├── WorldWmoGroupRenderDispatch.cs   # NEW — interior/exterior pass selection
│   │   ├── WorldWmoBatchMaterialFlags.cs    # NEW — per-batch MOMT flag evaluation
│   │   ├── WorldWmoInteriorFogState.cs      # NEW — interior fog parameters
│   │   └── WorldWmoLightmapPassSelector.cs  # NEW — lightmap Int vs Ext selection
│   ├── Liquid/
│   │   ├── WorldLiquidTileBuilder.cs        # EXISTING — extend for type dispatch
│   │   ├── WorldLiquidTileData.cs           # EXISTING — extend for animation state
│   │   ├── WorldLiquidChunkData.cs         # EXISTING — extend for interior/exterior
│   │   ├── WorldLiquidAnimationState.cs    # NEW — 30-frame cycling state
│   │   └── WorldLiquidTypeDispatch.cs      # NEW — water vs magma render path
│   ├── Lighting/
│   │   ├── WorldLightSelectionState.cs     # NEW — per-chunk light array (sun + up to 7 local)
│   │   └── WorldDayNightInfo.cs            # NEW — time-of-day lighting singleton
│   ├── Debug/
│   │   ├── WorldDebugEnables.cs            # NEW — CWorld::enables bitfield
│   │   └── WorldDebugOverlayRenderer.cs    # NEW — normals/wireframes/portals debug draws
│   └── Passes/
│       └── WorldFramePassCoordinator.cs    # EXISTING — extend for new pass types
├── Rendering/
│   ├── TerrainRenderPipeline.cs            # NEW — full terrain layer render pipeline
│   ├── WmoGroupRenderPipeline.cs          # NEW — WMO group render pipeline
│   ├── LiquidRenderPipeline.cs            # NEW — liquid render pipeline
│   └── ShaderRegistry.cs                  # NEW — pixel shader paths (psTerrain, psOcean0, etc.)
└── M2/                                     # EXISTING — not touched by this spec

wow-viewer/src/viewer/WowViewer.App/
├── WorldGpuPreviewRenderer.cs             # EXISTING — extend for new render paths
└── WowViewerDesktopApp.cs                 # EXISTING — add debug toggle UI

wow-viewer/tests/WowViewer.Core.Runtime.Tests/
├── TerrainLodSelectorTests.cs             # NEW
├── WmoGroupRenderDispatchTests.cs        # NEW
├── LiquidTypeDispatchTests.cs            # NEW
├── WorldLightSelectionTests.cs           # NEW
└── WorldDebugEnablesTests.cs             # NEW
```

**Structure Decision**: Library-first. All rendering logic in `WowViewer.Core.Runtime` under `World/` and `Rendering/` subdirectories. App host only adds debug toggle UI wiring.

## Implementation Phases

### Phase 1 — Terrain Mesh Topology and Distance LOD (P1)

**Goal**: Terrain renders with correct 145-vertex topology and distance-based texture LOD, matching the native client.

**Dependencies**: Spec 031 (terrain cell awareness) must provide the 145-vertex data and 8x8 cell grid. If spec 031 is not yet implemented, Phase 1 begins by building the terrain data structures needed (this is a bootstrapping concern, not a blocker — the data structures are simple).

**Approach**:
1. Extend `WorldTerrainChunkData` to carry the full 145-vertex layout (9x9 outer + 8x8 inner) and cell diagonal split information.
2. Build `WorldTerrainLodSelector` that computes LOD level per chunk based on distance: close (all layers), medium (alpha-fade), far (1 layer), very-far (low-detail 17x17).
3. Build `WorldTerrainLowDetailBuilder` for 17x17 fog-colored far-distance mesh.
4. Build `TerrainRenderPipeline` that orchestrates per-layer rendering: texture setup, alpha mask, texgen matrices, per-layer props (animated UV, lighting disable, alpha presence), LOD level selection.
5. Implement shadow overlay pass in `WorldTerrainShadowOverlay` — post-layer blend with `shadowColor`.
6. Validate against staged client: load a terrain tile, compare close-range mesh detail and far-range LOD behavior against native client screenshots.

**Steps** (max 10):
1. Add 145-vertex layout to `WorldTerrainChunkData` (inner vertices, cell diagonal flags)
2. Add per-cell face plane storage (256 planes, 2 triangulation variants)
3. Build `WorldTerrainLodSelector` with 4 LOD levels and distance thresholds
4. Build `WorldTerrainLowDetailBuilder` (17x17 fog-colored mesh)
5. Build `TerrainRenderPipeline` per-layer setup (texture, alpha mask, texgen, props)
6. Implement texture LOD fade logic (alpha-fade at `textureLodDist`, hard-cut at +256)
7. Implement shadow overlay pass
8. Wire `TerrainRenderPipeline` into `WorldFramePassCoordinator`
9. Add unit tests for LOD selection, low-detail builder, per-layer props
10. Validate against staged client terrain (close + far screenshots)

---

### Phase 2 — WMO Interior/Exterior Render Dispatch (P1)

**Goal**: WMO groups render with correct pass selection, per-batch MOMT flags, lightmap split, and interior fog, matching the native client.

**Dependencies**: Spec 030 (WMO render pass architecture) provides the documentation. The WMO I/O readers in `WowViewer.Core.IO/Wmo/` already provide MOMT flags, group flags, MOCV, lightmap UV, and liquid data.

**Approach**:
1. Build `WorldWmoGroupRenderDispatch` — selects interior vs exterior path based on `group.flags & 0x48`.
2. Build `WorldWmoBatchMaterialFlags` — evaluates per-batch MOMT flags: lighting (bit0), fog (bit1), culling (bit2), emissive (0x10), window-lit (0x20).
3. Build `WorldWmoLightmapPassSelector` — Int vs Ext lightmap behavior: interior (lighting off, lightmap on tex1) vs exterior (lighting on, no lightmap on tex1).
4. Build `WorldWmoInteriorFogState` — interior fog from `DayNightGetInfo()->intFog`, applied only when camera is inside the WMO.
5. Build `WmoGroupRenderPipeline` that orchestrates the full group render: pass selection → per-batch flag eval → lightmap setup → interior fog → render.
6. Handle skip groups (`flags & 0x88`) and always-render groups (`flags & 0x10000`).
7. Validate against staged client: load a dungeon WMO (interior) and an exterior WMO, compare lighting, fog, and window brightness.

**Steps** (max 10):
1. Build `WorldWmoGroupRenderDispatch` (interior/exterior path selection from flags)
2. Build `WorldWmoBatchMaterialFlags` (per-batch MOMT flag evaluation)
3. Build `WorldWmoLightmapPassSelector` (Int vs Ext lightmap behavior)
4. Build `WorldWmoInteriorFogState` (interior fog start/end/color)
5. Build `WmoGroupRenderPipeline` (full orchestration: dispatch → flags → lightmap → fog → render)
6. Handle skip groups (`0x88`) and always-render groups (`0x10000`)
7. Wire into `WorldFramePassCoordinator`
8. Add unit tests for dispatch logic, flag evaluation, lightmap selection
9. Validate interior WMO against native client (MOCV lighting, interior fog)
10. Validate exterior WMO against native client (dynamic lighting, window-lit flag)

---

### Phase 3 — Liquid Rendering with Animation and Type Dispatch (P2)

**Goal**: Water surfaces render with animated textures, correct interior/exterior behavior, and magma type dispatch, matching the native client.

**Dependencies**: Phase 1 (terrain LOD — distance affects water visibility) and Phase 2 (interior fog for interior water). Liquid I/O readers already exist in `WowViewer.Core.IO/Maps/AdtLiquidReader.cs` and `WowViewer.Core.IO/Maps/AdtMclqReader.cs`.

**Approach**:
1. Build `WorldLiquidTypeDispatch` — water (types 0/4/8) vs magma (types 2/3/6/7) render path selection.
2. Build `WorldLiquidAnimationState` — 30-frame cycling with per-type `secsPerLoop`, frame index computation.
3. Extend `WorldLiquidChunkData` with interior/exterior distinction and material color for interior water.
4. Build `LiquidRenderPipeline` — orchestrates type dispatch → animation state → interior/exterior color → render.
5. Implement river texgen scrolling (0.14 scale + camera offset on Tex1).
6. Add specular water pixel shader path (`psOcean0`) as opt-in.
7. Validate against staged client: view exterior water (ocean/river) and interior dungeon water.

**Steps** (max 10):
1. Build `WorldLiquidTypeDispatch` (water vs magma path selection)
2. Build `WorldLiquidAnimationState` (30-frame cycling, secsPerLoop, frame index)
3. Extend `WorldLiquidChunkData` for interior/exterior and material diffColor
4. Build `LiquidRenderPipeline` (type dispatch → animation → color → render)
5. Implement river texgen scrolling (0.14 scale + camera offset)
6. Add specular water path (`psOcean0`) as opt-in feature
7. Wire into `WorldFramePassCoordinator`
8. Add unit tests for type dispatch, animation cycling, interior/exterior color selection
9. Validate exterior water (animation, day/night color, river scrolling)
10. Validate interior water + magma against native client

---

### Phase 4 — Per-Chunk Lighting and Shadow System (P3)

**Goal**: Terrain and WMO groups are lit by sun + up to 7 local lights per chunk, and terrain receives shadow overlay, matching the native client.

**Dependencies**: Phase 1 (terrain rendering must exist for shadow overlay) and Phase 2 (WMO rendering must exist for exterior lighting). The lighting system is additive — the base rendering from Phases 1-3 uses sun-only lighting and works without this phase.

**Approach**:
1. Build `WorldDayNightInfo` — time-of-day lighting singleton providing sun direction/color/ambient, exterior fog, interior fog, water color array.
2. Build `WorldLightSelectionState` — per-chunk light array: light 0 = sun, lights 1-7 = local lights from `lightLinkList`, distance-priority selection.
3. Extend terrain render pipeline to call `SelectLights` per chunk and pass up to 8 lights to the shader.
4. Extend WMO exterior render pipeline to use `SelectLight` for map objects within fog distance.
5. Validate against staged client: view a terrain area with torches/lamps, verify local light contribution on nearby surfaces.

**Steps** (max 7):
1. Build `WorldDayNightInfo` (sun, exterior fog, interior fog, water colors, time-of-day)
2. Build `WorldLightSelectionState` (sun + up to 7 local, distance-priority)
3. Wire terrain per-chunk light selection into `TerrainRenderPipeline`
4. Wire WMO exterior lighting into `WmoGroupRenderPipeline`
5. Add unit tests for light selection (8-light cap, distance priority)
6. Validate terrain with local lights against native client
7. Validate WMO exterior lighting against native client

---

### Phase 5 — Debug Toggle System (P3)

**Goal**: The `CWorld::enables` bitfield is implemented as a runtime-toggled debug system, enabling visual debugging of normals, wireframes, portals, culling, shadows, WMO textures/lightmaps, and terrain LOD.

**Dependencies**: Phases 1-4 must be complete so that toggles are meaningful (can't toggle shadow overlay if shadow rendering doesn't exist).

**Approach**:
1. Build `WorldDebugEnables` — the 32-bit bitfield with named bits and toggle methods.
2. Wire each enable bit into the corresponding render pipeline: terrain visibility, shadow, water, WMO textures, WMO lightmaps, terrain LOD, low-detail, etc.
3. Build `WorldDebugOverlayRenderer` — normals (lines from vertices), wireframes (triangle edges), portal quads, BSP polygons.
4. Expose toggle UI in `WowViewerDesktopApp` — keyboard shortcuts or debug panel.
5. Validate each toggle produces expected visual result.

**Steps** (max 7):
1. Build `WorldDebugEnables` bitfield with all named bits from spec
2. Wire terrain-related enables (terrain, shadow, LOD, low-detail, culling, normals, tris)
3. Wire WMO-related enables (textures, lightmaps, portals, BSP, crappy batches)
4. Wire liquid enables (water visibility)
5. Build `WorldDebugOverlayRenderer` (normals, wireframes, portal quads)
6. Add keyboard shortcuts / debug panel in `WowViewerDesktopApp`
7. Validate each toggle against expected visual output

## Complexity Tracking

No constitution violations. All code stays in `wow-viewer/`, library-first, real-data validated.
