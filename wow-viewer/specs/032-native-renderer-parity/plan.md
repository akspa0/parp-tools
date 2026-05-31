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
│   │   ├── WorldDayNightLightingState.cs   # NEW — CurrentLight data model (all 18+ tracks)
│   │   ├── WorldDayNightLightEvaluator.cs # NEW — CalcLightColors-style track interpolation
│   │   ├── WorldDayNightInfo.cs            # NEW — time-of-day singleton + DayNightGetInfo bridge
│   │   ├── WorldLightSelectionState.cs     # NEW — per-chunk light array (sun + up to 7 local)
│   │   ├── WorldStormBlendEvaluator.cs     # NEW — storm/clear parameter blending
│   │   ├── WorldMdxLightCollector.cs       # NEW — MDX LITE chunk light extraction
│   │   └── WorldWmoLightCollector.cs       # NEW — WMO MOLT light extraction
│   ├── Sky/
│   │   ├── WorldSkyDomeRenderer.cs         # NEW — 6-band sky dome from SkyArray[0..5]
│   │   └── WorldSkyClearcolor.cs          # NEW — clear color from lighting data
│   ├── Fog/
│   │   ├── WorldExteriorFogState.cs        # NEW — fog from CurrentLight.FogEnd/FogStartScalar
│   │   ├── WorldInteriorFogState.cs        # NEW — intFogInfo interior fog
│   │   └── WorldWmoAreaFogBlender.cs      # NEW — multi-zone WMO fog blending
│   ├── Debug/
│   │   ├── WorldDebugEnables.cs            # NEW — CWorld::enables bitfield
│   │   └── WorldDebugOverlayRenderer.cs    # NEW — normals/wireframes/portals debug draws
│   └── Passes/
│       └── WorldFramePassCoordinator.cs    # EXISTING — extend for new pass types
├── Rendering/
│   ├── TerrainRenderPipeline.cs            # NEW — full terrain layer render pipeline
│   ├── WmoGroupRenderPipeline.cs          # NEW — WMO group render pipeline
│   ├── LiquidRenderPipeline.cs            # NEW — liquid render pipeline
│   ├── SkyRenderPipeline.cs               # NEW — sky dome + clear color pipeline
│   ├── ShaderRegistry.cs                  # NEW — effect-family to GLSL program mapping
│   ├── TerrainEffectFamily.cs             # NEW — psTerrain/psSpecTerrain GLSL shader
│   ├── MapObjEffectFamily.cs              # NEW — MapObjDiffuse/MapObjSpecular GLSL shader
│   ├── LiquidEffectFamily.cs              # NEW — psOcean0 GLSL shader
│   ├── ModelCombinerFamily.cs             # NEW — Combiners_Opaque/Mod GLSL shader
│   └── SkyEffectFamily.cs                 # NEW — DNSky 6-band GLSL shader
└── M2/                                     # EXISTING — not touched by this spec

wow-viewer/src/core/WowViewer.Core.IO/
├── Lighting/
│   └── LitFileReader.cs                   # NEW — Alpha .lit file reader (ported from MdxViewer LitLoader)
└── Maps/                                  # EXISTING

wow-viewer/src/viewer/WowViewer.App/
├── WorldGpuPreviewRenderer.cs             # EXISTING — extend for new render paths
└── WowViewerDesktopApp.cs                 # EXISTING — add debug toggle UI + time-of-day slider

wow-viewer/tests/WowViewer.Core.Runtime.Tests/
├── TerrainLodSelectorTests.cs             # NEW
├── WmoGroupRenderDispatchTests.cs        # NEW
├── LiquidTypeDispatchTests.cs            # NEW
├── WorldDayNightLightingTests.cs         # NEW
├── WorldLightSelectionTests.cs           # NEW
├── MdxLightCollectorTests.cs            # NEW
├── WmoLightCollectorTests.cs            # NEW
└── WorldDebugEnablesTests.cs             # NEW

wow-viewer/tests/WowViewer.Core.IO.Tests/
└── LitFileReaderTests.cs                 # NEW
```

**Structure Decision**: Library-first. All rendering logic in `WowViewer.Core.Runtime` under `World/` and `Rendering/` subdirectories. App host only adds debug toggle UI wiring.

## Implementation Phases

### Phase 1 — Lighting Data Model and LIT Reader (P1, US6)

**Goal**: The `CurrentLight` data model exists with all 18+ tracks, `.lit` files can be loaded and evaluated at any time of day, and storm/clear blending works. This is the **foundation** — every other phase depends on correct lighting values.

**Dependencies**: None. This phase is purely data model + I/O — no rendering required.

**Approach**:
1. Port `LitLoader.cs` from MdxViewer to `WowViewer.Core.IO/Lighting/LitFileReader.cs` — same format support, same track enumeration, same float band data.
2. Build `WorldDayNightLightingState` — the `CurrentLight` data model: DirectColor, AmbientColor, SkyArray[0..5], ShadowOpacity, CloudArray[0..4], WaterArray[0..3], FogEnd, FogStartScalar, CloudData[1], Darkness.
3. Build `WorldDayNightLightEvaluator` — `CalcLightColors`-style track interpolation: linear byte interp for color tracks, linear float interp for float tracks, midnight wrap at 2880.
4. Build `WorldStormBlendEvaluator` — `result = clear * (100 - weight) / 100 + storm * weight / 100`.
5. Build `WorldDayNightInfo` — singleton wrapping the evaluator, exposing settable `gameTime`, providing `CurrentLight` snapshot on demand.
6. Wire spatial light blending (default vs local light zone with falloff).
7. Validate: load a real `.lit` file, evaluate at noon/midnight/dawn, compare track values against MdxViewer `LitLoader` output.

**Steps** (max 10):
1. Port `LitLoader.cs` to `WowViewer.Core.IO/Lighting/LitFileReader.cs` (all 4 param groups, 18+ tracks, float bands)
2. Define `WorldDayNightLightingState` record/struct with all CurrentLight fields
3. Implement color track interpolation (byte-per-channel linear interp between markers)
4. Implement float track interpolation (linear interp between markers)
5. Implement midnight wrap (time 0-2880, markers can wrap around midnight)
6. Implement storm/clear parameter blending (percentage-based, fog float * 0.01)
7. Build `WorldDayNightInfo` singleton with settable `gameTime` and `Evaluate()` → `WorldDayNightLightingState`
8. Implement spatial light blending (default + local zones with falloff weight)
9. Add unit tests for track interpolation, storm blending, midnight wrap, spatial blending
10. Validate against real `.lit` file + MdxViewer LitLoader output

---

### Phase 2 — Sky Dome and Fog from Lighting Data (P1, US7 + US8)

**Goal**: The sky dome renders a 6-band gradient from `CurrentLight.SkyArray[0..5]`, exterior fog uses `CurrentLight.FogEnd/FogStartScalar` with correct color, and interior fog works inside WMOs. This makes every rendered frame visually match the native client at the current time of day.

**Dependencies**: Phase 1 (lighting data model must produce correct `CurrentLight` values).

**Approach**:
1. Replace hardcoded `SkyRenderer.cs` with `WorldSkyDomeRenderer` — 6-band hemisphere gradient from `CurrentLight.SkyArray[0..5]`, camera-following, depth write OFF.
2. Set scene clear color to `CurrentLight.SkyArray[5]` (fog/horizon color).
3. Build `WorldExteriorFogState` — fog parameters from `CurrentLight.FogEnd`, `CurrentLight.FogStartScalar`, fog color from `CurrentLight.SkyArray[5]`.
4. Build `WorldInteriorFogState` — interior fog from `DayNightGetInfo()->intFogInfo` (start, end, color), applied when camera inside WMO.
5. Build `WorldWmoAreaFogBlender` — multi-zone fog blending from `SMOFog` entries per WMO group.
6. Wire fog into all existing render pipelines (terrain, WMO, liquid).
7. Validate: set time to noon vs. midnight, verify sky dome bands change, fog distance changes, fog color changes.

**Steps** (max 10):
1. Build `WorldSkyDomeRenderer` (6-band hemisphere, SkyArray[0..5], smoothstep transitions)
2. Set clear color from `CurrentLight.SkyArray[5]` (fog/horizon color)
3. Build `WorldExteriorFogState` (FogEnd, FogStartScalar, fog color from SkyArray[5])
4. Build `WorldInteriorFogState` (intFogInfo start/end/color, camera-in-WMO check)
5. Build `WorldWmoAreaFogBlender` (SMOFog multi-zone, distance-based blending)
6. Replace hardcoded `SkyRenderer.cs` in WowViewer.App with `WorldSkyDomeRenderer`
7. Wire exterior fog into terrain/WMO/liquid render pipelines
8. Wire interior fog into WMO render pipeline
9. Add time-of-day slider to `WowViewerDesktopApp` UI
10. Validate sky dome + fog at noon/midnight/dawn against native client

---

### Phase 3 — Terrain Mesh Topology and Distance LOD (P1, US1)

**Goal**: Terrain renders with correct 145-vertex topology and distance-based texture LOD, matching the native client.

**Dependencies**: Phase 2 (fog from lighting data drives LOD distance and low-detail mesh fog color). Spec 031 (terrain cell awareness) must provide the 145-vertex data and 8x8 cell grid. **Critical Alpha constraint**: The runtime must consume already-reinterleaved 145-vertex arrays from the I/O layer. It must NOT re-decode raw MCVT bytes.

**Approach**:
1. Extend `WorldTerrainChunkData` to carry the full 145-vertex layout (9x9 outer + 8x8 inner) and cell diagonal split information.
2. Build `WorldTerrainLodSelector` that computes LOD level per chunk based on distance.
3. Build `WorldTerrainLowDetailBuilder` for 17x17 fog-colored far-distance mesh (fog color from `CurrentLight.SkyArray[5]`).
4. Build `TerrainRenderPipeline` that orchestrates per-layer rendering.
5. Implement shadow overlay pass — blend factor from `CurrentLight.ShadowOpacity`.
6. Validate against staged client.

**Steps** (max 10):
1. Add 145-vertex layout to `WorldTerrainChunkData` (inner vertices, cell diagonal flags)
2. Add per-cell face plane storage (256 planes, 2 triangulation variants)
3. Build `WorldTerrainLodSelector` with 4 LOD levels and distance thresholds
4. Build `WorldTerrainLowDetailBuilder` (17x17 fog-colored mesh, fog color from CurrentLight)
5. Build `TerrainRenderPipeline` per-layer setup (texture, alpha mask, texgen, props)
6. Implement texture LOD fade logic (alpha-fade at `textureLodDist`, hard-cut at +256)
7. Implement shadow overlay pass (blend factor from `CurrentLight.ShadowOpacity`)
8. Wire `TerrainRenderPipeline` into `WorldFramePassCoordinator`
9. Add unit tests for LOD selection, low-detail builder, per-layer props, shadow opacity
10. Validate against staged client terrain (close + far screenshots)

---

### Phase 4 — WMO Interior/Exterior Render Dispatch (P1, US2)

**Goal**: WMO groups render with correct pass selection, per-batch MOMT flags, lightmap split, and interior fog, matching the native client.

**Dependencies**: Phase 2 (interior fog from lighting data). Spec 030 provides documentation. WMO I/O readers already provide MOMT flags, group flags, MOCV, lightmap UV, and liquid data.

**Approach**:
1. Build `WorldWmoGroupRenderDispatch` — selects interior vs exterior path based on `group.flags & 0x48`.
2. Build `WorldWmoBatchMaterialFlags` — evaluates per-batch MOMT flags.
3. Build `WorldWmoLightmapPassSelector` — Int vs Ext lightmap behavior.
4. Build `WorldWmoInteriorFogState` — interior fog from `DayNightGetInfo()->intFog`.
5. Build `WmoGroupRenderPipeline` — full orchestration.
6. Handle skip groups (`flags & 0x88`) and always-render groups (`flags & 0x10000`).
7. Validate against staged client.

**Steps** (max 10):
1. Build `WorldWmoGroupRenderDispatch` (interior/exterior path selection from flags)
2. Build `WorldWmoBatchMaterialFlags` (per-batch MOMT flag evaluation)
3. Build `WorldWmoLightmapPassSelector` (Int vs Ext lightmap behavior)
4. Build `WorldWmoInteriorFogState` (interior fog start/end/color from DayNightGetInfo)
5. Build `WmoGroupRenderPipeline` (full orchestration: dispatch → flags → lightmap → fog → render)
6. Handle skip groups (`0x88`) and always-render groups (`0x10000`)
7. Wire into `WorldFramePassCoordinator`
8. Add unit tests for dispatch logic, flag evaluation, lightmap selection
9. Validate interior WMO against native client (MOCV lighting, interior fog)
10. Validate exterior WMO against native client (dynamic lighting, window-lit flag)

---

### Phase 5 — Liquid Rendering with Animation and Type Dispatch (P2, US3)

**Goal**: Water surfaces render with animated textures, correct interior/exterior behavior, and magma type dispatch. Water color from `CurrentLight.WaterArray[0..3]`.

**Dependencies**: Phases 2+3 (fog + terrain LOD) and Phase 4 (interior fog for interior water).

**Steps** (max 9):
1. Build `WorldLiquidTypeDispatch` (water vs magma path selection)
2. Build `WorldLiquidAnimationState` (30-frame cycling, secsPerLoop, frame index)
3. Extend `WorldLiquidChunkData` for interior/exterior and material diffColor
4. Build `LiquidRenderPipeline` (type dispatch → animation → color → render)
5. Implement water color from `CurrentLight.WaterArray[3]` for exterior, material diffColor for interior
6. Implement river texgen scrolling (0.14 scale + camera offset)
7. Add specular water path (`psOcean0`) as opt-in feature
8. Add unit tests for type dispatch, animation cycling, water color from lighting
9. Validate exterior + interior water + magma against native client

---

### Phase 6 — Per-Chunk/Per-Group Local Light Selection (P2, US5 + US9 + US10)

**Goal**: Sun + up to 7 local lights per terrain chunk and per WMO group. Local lights come from three sources: MDX model LITE chunks, WMO MOLT entries, and `CMapLight` world lights. All share the same 8-light budget per rendering unit.

**Dependencies**: Phases 3+4 (terrain and WMO rendering must exist). Phase 1 (lighting data model provides sun direction/color).

**Steps** (max 10):
1. Build `WorldLightSelectionState` (sun + up to 7 local, distance-priority, 8-light total cap)
2. Build `WorldMdxLightCollector` — extracts LITE chunk lights from visible MDX model instances, creates `CGxLight`-style entries with position, color, intensity, attenuation, type
3. Build `WorldWmoLightCollector` — consumes existing `WmoLightDetail` data from MOLT, converts to `CGxLight`-style entries with WMO-to-world transform
4. Merge MDX + MOLT + CMapLight entries into a single per-frame light pool
5. Wire terrain per-chunk light selection into `TerrainRenderPipeline` — `SelectLights` picks nearest 7 from pool for each chunk
6. Wire WMO per-group light selection into `WmoGroupRenderPipeline` — same selection for each group
7. Skip lights beyond fog distance (`camDist >= farFog`), matching `CMap::SelectLight`
8. Add unit tests for light selection (8-light cap, distance priority, fog skip, MDX/MOLT merge)
9. Validate terrain with MDX torch lights against native client
10. Validate WMO interior with MOLT lights against native client

---

### Phase 7 — Shader Family Reconstruction (P2, US11)

**Goal**: Replace the existing inline GLSL shaders with effect-family-aware shaders that produce visually native-equivalent output, verified by side-by-side comparison with the native client at the same time of day. No "too reddish/yellow" noggit-style color deviation.

**Dependencies**: Phases 1-6 (all rendering features must exist for shader validation). Phase 6 specifically (local lights need shader support).

**Approach**: Shader-family reconstruction — for each named effect family from the native client, write a modern GLSL shader that produces equivalent output using Ghidra-verified uniform bindings. Do NOT attempt BLS execution or auto-conversion. Learn from noggit's known pitfall (wrong color-space/gamma produces too-warm output) by validating against native client screenshots.

**Steps** (max 10):
1. Audit existing inline GLSL shaders in MdxViewer + wow-viewer — catalog which effect families each shader approximates, identify color-space assumptions
2. Write `TerrainEffectFamily` GLSL shader — equivalent to `psTerrain`/`psSpecTerrain` with correct `CurrentLight`-driven fog/lighting, no color shift
3. Write `MapObjEffectFamily` GLSL shader — equivalent to `MapObjDiffuse`/`MapObjSpecular` with interior MOCV + local light blending
4. Write `LiquidEffectFamily` GLSL shader — equivalent to `psOcean0` with animated textures and specular
5. Write `ModelCombinerFamily` GLSL shader — equivalent to `Combiners_Opaque`/`Combiners_Mod`/`Combiners_Mod2x` with alpha blending and specular gating
6. Write `SkyEffectFamily` GLSL shader — equivalent to `DNSky::Render` with 6-band gradient
7. Wire effect-family shaders into `ShaderRegistry` — map native effect names to GLSL programs
8. Validate color-space behavior — render same scene at same time of day in viewer vs. native client, compare pixel values (no reddish/yellow shift)
9. Validate per-effect-family — terrain (psTerrain), WMO interior (MapObjDiffuse), water (psOcean0), model (Combiners_Opaque)
10. Validate combined scene — full world frame with all effect families active

---

### Phase 8 — Debug Toggle System (P3, US4)

**Goal**: The `CWorld::enables` bitfield is runtime-toggled, enabling visual debugging of normals, wireframes, portals, culling, shadows, WMO textures/lightmaps, terrain LOD, and sky.

**Dependencies**: Phases 1-7 must be complete so toggles are meaningful.

**Steps** (max 8):
1. Build `WorldDebugEnables` bitfield with all named bits from spec
2. Wire terrain-related enables (terrain, shadow, LOD, low-detail, culling, normals, tris)
3. Wire WMO-related enables (textures, lightmaps, portals, BSP, crappy batches)
4. Wire liquid enables (water visibility)
5. Wire sky enables (sky visibility, fog toggle)
6. Wire lighting enables (lighting toggle, model lights toggle, MOLT lights toggle)
7. Build `WorldDebugOverlayRenderer` (normals, wireframes, portal quads)
8. Add keyboard shortcuts / debug panel in `WowViewerDesktopApp`

## Complexity Tracking

No constitution violations. All code stays in `wow-viewer/`, library-first, real-data validated.
