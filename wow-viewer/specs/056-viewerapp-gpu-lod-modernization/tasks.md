---
description: "Task list for spec 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization"
---

# Tasks: 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization

**Input**: Design documents from `/specs/056-viewerapp-gpu-lod-modernization/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/
**Branch**: `056-viewerapp-gpu-lod-modernization` (target: `v0.5.0-dev`)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US7 from spec.md)
- **[US7]** = the real-data validation surface (US7 spans all phases)
- Include exact file paths in descriptions

## Path Conventions

- Renderer library: `wow-viewer/src/core/WowViewer.Core.Renderer/`
- Renderer test project (NEW in Phase 0): `wow-viewer/tests/WowViewer.Core.Renderer.Tests/`
- Runtime (consumer, not modified here except as noted): `wow-viewer/src/core/WowViewer.Core.Runtime/World/`
- Viewer host (renderer code retires in Phase 6): `wow-viewer/src/viewer/WoWViewer/Rendering/`
- Validation tool: `wow-viewer/tools/validation-capture/`
- Spec artifacts: `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/`
- Memory bank: `wow-viewer/memory-bank/`

---

## Phase 0: Renderer Library Test Foundation + Project Topology Lock

**Purpose**: Create the new test project, lock the namespace split, document the backend-agnostic contract seams, and produce the deterministic capture baseline that downstream phases compare against. **No user story work can begin until this phase is complete.**

- [ ] T001 [P] [US7] Create `wow-viewer/tests/WowViewer.Core.Renderer.Tests/WowViewer.Core.Renderer.Tests.csproj` as an xUnit project (.NET 10) referencing `WowViewer.Core.Renderer`, `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`. Add to `wow-viewer/WowViewer.slnx`.
- [ ] T002 [P] [US7] Add `RendererError.cs`, `RendererLifecycleState.cs`, `RenderBackendKind.cs`, `AoiBounds.cs`, `Viewport.cs` (all backend-neutral, no GL types) under `wow-viewer/src/core/WowViewer.Core.Renderer/Contracts/`.
- [ ] T003 [P] [US7] Author `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/contracts/RenderScene.md` (already exists from `speckit-plan`; review and update if any data-model field is missing).
- [ ] T004 [P] [US7] Author `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/contracts/RenderBackend.md` (already exists; review).
- [ ] T005 [P] [US7] Author `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/contracts/RenderResources.md` (already exists; review).
- [ ] T006 [P] [US7] Author `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/contracts/TextureCache.md` (already exists; review).
- [ ] T007 [US7] Run a deterministic capture on staged `0_5_3_3368` Azeroth tile `(30, 48)` and on staged `3_3_5_12340` Azeroth tile `(30, 48)` through the *current* `WowViewer.Tool.ValidationCapture` path (pre-spec renderer) and store the PNG + per-frame-stats JSON under `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/pre-spec/`. Also capture a Stormwind tile for the city / WMO test (Phase 3 will use it).
- [ ] T008 [US7] Add a `compare-baseline` subcommand stub to `wow-viewer/tools/validation-capture/` that takes a build + tile, runs the current capture, and reports a per-pixel diff against the pre-spec baseline. (No actual diffing yet — just the CLI surface. Real diffing lands in Phase 7.)
- [ ] T009 [US7] Update `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` with the spec 056 entry: status, owner, scope, locked decisions, "in progress" = Phase 0.

**Checkpoint**: `dotnet build wow-viewer/WowViewer.slnx -c Debug` succeeds. The new test project builds and `dotnet test` runs an empty test list successfully. The contracts are committed and referenced by `spec.md`. The pre-spec capture baseline is recorded.

**Do not regress**: existing `WowViewer.Core.Tests`, `WowViewer.Core.PM4.Tests`, `WowViewer.Core.Anim.Tests` all still pass.

---

## Phase 1: Promote `WowViewer.Core.Renderer.Scene` to a Multi-Tile, Retained-Mode Core

**Purpose**: Make the existing `WowViewer.Core.Renderer` skeleton genuinely multi-tile, retained-mode (VBO/IBO/UBO), instanced, and frustum-culled. **US1 (multi-tile render) and US5 (per-frame diagnostics) are the primary user stories here.** Phase 1 is the precondition for Phases 2-5.

> **Phase 1 note**: this phase was split from a 14-task phase into 1a (contracts + scene extensions) and 1b (backend + SceneRenderer rewrite + validation). The split is the only deviation from the original 9-phase plan in `plan.md`; the total phase count is now 10, which is still under the 10-step "max per phase" rule. See the Phase Dependency Graph in `plan.md` for the updated view.

### Phase 1a — Contract surface + scene extensions

- [ ] T010 [P] [US1] Contract test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Contracts/RenderSceneTests.cs`: empty `Tiles` + empty `WorldObjects` + valid camera renders a no-op frame; null `Camera` raises `RendererError.InvalidScene`; null `TileData` skips the tile.
- [ ] T011 [P] [US1] Contract test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Contracts/RenderBackendContractTests.cs`: lifecycle (Initialize → BeginFrame → Submit → EndFrame → Dispose); double BeginFrame rejected; mid-frame Resize rejected; failed frame sets `LastError`.
- [ ] T012 [P] [US7] Integration test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Validation/HeadlessMultiTileTests.cs`: stage `3_3_5_12340` Azeroth, render 3×3 AOI through `OpenGLRenderBackend`, assert each tile's mesh is built and at least one frustum-culled tile is rejected (cull the corner tiles by pointing the camera at the opposite corner).
- [ ] T013 [P] [US1] Add `wow-viewer/src/core/WowViewer.Core.Renderer/Contracts/RenderScene.cs` per `data-model.md` §1 (record type, backend-neutral, references `WowViewer.Core.Runtime.World` only).
- [ ] T014 [US1] Extend `wow-viewer/src/core/WowViewer.Core.Renderer/Scene/SceneCamera.cs` with `ViewProjectionMatrix`, `InverseViewProjectionMatrix`, `Frustum` (six planes), `AoiBounds`, and `ComputeFromLookAt(...)` helper.
- [ ] T015 [US1] Extend `wow-viewer/src/core/WowViewer.Core.Renderer/Scene/FrustumCuller.cs` with `TestAabb` (existing) plus new helpers `TestTile(RenderTerrainTile)`, `TestWmo(RenderWorldObjectRef)`, `TestM2(RenderWorldObjectRef)`, all O(1) per test.
- [ ] T016 [US1] Extend `wow-viewer/src/core/WowViewer.Core.Renderer/Scene/RenderVariant.cs` with `TerrainLod`, `ObjectLod`, `WaterLod`, `LightLod`, `MipSelection`, `Quality` (per `data-model.md` §1); all backend-neutral.
- [ ] T017 [P] [US1] Add `wow-viewer/src/core/WowViewer.Core.Renderer/Contracts/IRenderBackend.cs`, `RenderBackendKind.cs`, `IRenderResources.cs`, `PerFrameRenderStats.cs` per `data-model.md` §10-12 and `contracts/RenderBackend.md`. No GL types in this file.
- [ ] T018 [US1, US5] Extend `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs` with the new `waterLOD`, `mapObjLightLOD`, `MaxLights` fields (per `data-model.md` §8). No behavior change yet — just the surface.

**Checkpoint for 1a**: all contract types compile, all contract tests fail (proving they're real tests), `RenderScene` and the extended `Scene` types are referenced from the contract tests. `dotnet build wow-viewer/WowViewer.slnx -c Debug` succeeds.

### Phase 1b — Backend skeleton + SceneRenderer rewrite + validation

- [ ] T019 [P] [US1] Add `wow-viewer/src/core/WowViewer.Core.Renderer/OpenGL/OpenGLRenderBackend.cs` (the v0.5.0-dev implementation; consumes `IRenderBackend`; owns its `GL` handle, its `OpenGLBufferFactory`, its `OpenGLShaderCache`).
- [ ] T020 [US1] Add `wow-viewer/src/core/WowViewer.Core.Renderer/OpenGL/OpenGLBufferFactory.cs` and `OpenGLShaderCache.cs` (VBO/IBO/UBO helpers + shader program cache, both retained-mode).
- [ ] T021 [US1] Rewrite `wow-viewer/src/core/WowViewer.Core.Renderer/Scene/SceneRenderer.cs` for multi-tile: it accepts a `RenderScene`, walks tiles + world objects + sky + fog, calls into the `IRenderBackend`, writes `PerFrameRenderStats`. Add `wow-viewer/src/core/WowViewer.Core.Renderer/Diagnostics/PerFrameRenderStatsWriter.cs` for the stats emission.
- [ ] T022 [US1] De-duplicate `wow-viewer/src/core/WowViewer.Core.Renderer/Output/*` into `wow-viewer/src/core/WowViewer.Core.Renderer/Headless/*` (FR-011). Update `WowViewer.Core.Renderer.Headless.HeadlessContext` to be the single headless entry point.
- [ ] T023 [US1, US5] Run the Phase 1 integration test (T012) and the Phase 0 baseline capture (T007) on staged `3_3_5_12340` and `0_5_3_3368`; record the result in `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/phase-1/`. Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.

**Checkpoint for 1b**: A staged `3_3_5_12340` outdoor map renders a 5×5 AOI of terrain tiles through the new shared renderer with no missing terrain at tile seams (SC-001). A 3×3 AOI render uses one material bind per unique material, not 9 (SC-002). Per-tile / per-WMO / per-M2 frustum culling works (FR-004). Headless capture path still works for `WowViewer.Tool.ValidationCapture` (FR-011). Deterministic capture parity with the Phase 0 baseline on `0_5_3_3368` and `3_3_5_12340` (FR-016, FR-017). `specs/020-renderer-culling-and-tile-capture` P1 culling fix is not regressed (FR-018).

---

## Phase 2: Terrain Renderer in the Shared Library (LOD-Aware, WDL Far Horizon)

**Purpose**: Port the terrain renderer from `wow-viewer/src/viewer/WoWViewer/Rendering/TerrainRenderer.cs` and the legacy `gillijimproject_refactor/src/MdxViewer/Rendering/TerrainRenderer.cs` (~1808 lines) into `WowViewer.Core.Renderer.Terrain`. Wire to `WorldTerrainLodSelector` and add WDL far-horizon. **US2 (terrain mesh LOD + WDL far horizon) is the primary user story.**

### Tests for Phase 2 (US2, US7) ⚠️

- [ ] T024 [P] [US2] Unit test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Terrain/TerrainLodSelectorTests.cs`: given `(distance, nearDistance, midDistance, farDistance, useWdlForFar)`, the renderer selects `Full / Reduced / WdlOnly / Culled` correctly across threshold boundaries; gradual fade (no popping) verified at the threshold by deterministic capture.
- [ ] T025 [P] [US2] Integration test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Terrain/TerrainMultiTileTests.cs`: stage `3_3_5_12340` Azeroth, render at three camera distances (near / mid / far), assert vertex density drops monotonically (count vertices in the captured mesh via the diagnostic stream).

### Implementation for Phase 2 (US2, US7)

- [ ] T026 [P] [US2] Add `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/RenderTerrainTile.cs`, `TerrainLodSettings.cs`, `TerrainLodBucket.cs` per `data-model.md` §3. No GL types.
- [ ] T027 [P] [US2] Decide (and record in a one-paragraph note in the commit message) whether `WorldTerrainLodSelector` gets a new `UseWdl` bucket or whether the renderer translates `LowDetail` into WDL. Default: extend the enum. If the runtime rejects the extension, fall back to the renderer-translation approach.
- [ ] T028 [US2] Rewrite `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/TerrainMeshBuilder.cs` for multi-tile: add `BuildFull(tile, 257)`, `BuildReduced(tile, 33)`, `BuildReduced(tile, 17)`, `BuildWdlPlaceholder(tile)`. Each builds a retained-mode VBO/IBO via the `IRenderResources` cache.
- [ ] T029 [US2] Rewrite `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/TerrainRenderer.cs` to consume the new builder: walk `RenderScene.Tiles`, for each tile call `FrustumCuller.TestTile(...)`, then call `TerrainLodSelector.Select(...)` (per-tile) for the bucket, then call the appropriate builder via `IRenderResources.Terrain.GetOrBuildMesh(...)`.
- [ ] T030 [US2] Port the WDL far-horizon path: add `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/WdlFarHorizonRenderer.cs` that consumes `WorldWdlTileData` and renders the WDL as a single quad mesh per far tile, with the appropriate `WorldWdlTileBuilder`-produced heightmap.
- [ ] T031 [US2] Extend `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/TerrainShader.cs` with the WDL far-horizon shader path (per FR-007); keep the existing near/mid shader paths intact.
- [ ] T032 [US2, US7] Update `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/phase-2/` with the three-distance capture and the Alpha MCAL parity check on `0_5_3_3368`. Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.

**Checkpoint**: Three visibly distinct vertex densities at near / mid / far distances on `3_3_5_12340` (SC-003). WDL is the only representation beyond the configured far distance (SC-003 acceptance 3, FR-007). LOD transitions are gradual (SC-003 acceptance 4). Alpha MCAL parity preserved on `0_5_3_3368` (FR-017). LK 3.3.5 parity preserved on `3_3_5_12340` (FR-016).

---

## Phase 3: WMO, M2, and MDX Renderers in the Shared Library (Instanced, LOD-Aware)

**Purpose**: Port the WMO, M2, and MDX renderers from the legacy `gillijimproject_refactor/src/MdxViewer/Rendering/*` (~2866 lines for MDX) into `WowViewer.Core.Renderer.{Wmo,M2,Mdx}`. Make them instanced, frustum-culled, and object-LOD-aware. **US3 (object LOD + draw-distance culling) is the primary user story.**

> **Phase 3 risk note**: if MDX requires more than 10 steps, the spec allows splitting into Phase 3a (M2 + WMO) and Phase 3b (MDX). The decision is made at T037 based on the size of the current `wow-viewer/src/viewer/WoWViewer/Rendering/MdxAnimator.cs` (the source of truth, not the legacy MdxRenderer).

### Tests for Phase 3 (US3, US7) ⚠️

- [ ] T033 [P] [US3] Unit test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Wmo/WmoLodTests.cs`: given `(distance, wmoDrawDistance)`, the renderer selects `Near / Far / Culled` correctly; culling past `wmoDrawDistance` is verified by zero draw-call count.
- [ ] T034 [P] [US3] Unit test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/M2/M2LodTests.cs`: same shape as T033 but for M2.
- [ ] T035 [P] [US3] Integration test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Validation/ObjectLodTests.cs`: stage `3_3_5_12340` Stormwind tile, render at near and far, assert (a) all visible objects render at near, (b) zero objects render beyond `wmoDrawDistance` at far, (c) per-frame `WorldObjectsByLodLevel[Near] / [Far] / [Culled]` distribution matches the expected LOD curve.

### Implementation for Phase 3 (US3, US7)

- [ ] T036 [P] [US3] Add `wow-viewer/src/core/WowViewer.Core.Renderer/Contracts/RenderWorldObjectRef.cs`, `ObjectLodSettings.cs`, `ObjectLodLevel.cs` per `data-model.md` §4.
- [ ] T037 [P] [US3] Decide (in a one-paragraph commit note) whether MDX needs its own phase (3b) or fits inside this phase. If MDX > 6 steps, split. The decision is based on the actual size of the current `wow-viewer/src/viewer/WoWViewer/Rendering/MdxAnimator.cs` (the source of truth, not the legacy MdxRenderer).
- [ ] T038 [US3] Move + improve WMO renderer: `wow-viewer/src/core/WowViewer.Core.Renderer/Wmo/WmoRenderer.cs` (rewrite for instanced + object-LOD; conform to Ghidra pass dispatch from `docs/architecture/wmo-render-pass-architecture-2026-05-30.md`: interior/exterior dispatch by `flags & 0x48`, per-batch MOMT flag testing, lightmap pass split, liquid type dispatch, portal-walk visibility, group flag filtering). Update `WmoMesh.cs` and `WmoShader.cs` for retained-mode. Consume `WorldObjectVisibilityCollector` (existing) — do not reimplement visibility selection.
- [ ] T039 [US3] Move + improve M2 renderer: `wow-viewer/src/core/WowViewer.Core.Renderer/M2/M2Renderer.cs` (NEW). Consume `M2RuntimeFramePipeline` and `M2SceneSubmissionCoordinator` from `WowViewer.Core.Runtime.M2` — do not reimplement the M2 frame pipeline. Update `M2MaterialPassProfile.cs` and `M2CameraPathRenderer.cs` if they exist in the current viewer-app surface; otherwise skip.
- [ ] T040 [US3] Move + improve MDX renderer: `wow-viewer/src/core/WowViewer.Core.Renderer/Mdx/MdxRenderer.cs` (NEW; largest single file in the cutover). Consume `WorldMdxRenderPlan` and `MdxEffectRuntime` from `WowViewer.Core.Runtime.World` and `WowViewer.Core.Runtime.Mdx`. Update `MdxAnimator.cs` if it exists in the current viewer-app surface; otherwise skip.
- [ ] T041 [US3] Wire `ObjectLodSettings` into the renderer: each `RenderWorldObjectRef` is checked against `ObjectLodSettings.WmoDrawDistance` / `M2DrawDistance`; the renderer either binds a near-detail handle, a far-detail handle, or skips the instance.
- [ ] T042 [US3, US7] Update `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/phase-3/` with the Stormwind capture and the per-LOD-bucket distribution from `PerFrameRenderStats`. Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.

**Checkpoint**: A staged `3_3_5_12340` route through a city tile shows visible M2 / WMO instances, with no missing meshes (SC-004 acceptance 1+2). Object LOD culls past draw distance (SC-004 acceptance 3). A small object occluded by a WMO wall is correctly culled (SC-004 acceptance 4). `WowViewer.Tool.ValidationCapture` still produces the same `object_visibility_mask` for `Azeroth_30_48` on `3_3_5_12340` (SC-008, FR-011, FR-018).

**Do not regress**:
- Spec 020 culling fix (FR-018).
- M2 runtime ownership stays in `WowViewer.Core.Runtime.M2`. The renderer is a **consumer** of `M2RuntimeFramePipeline` and `M2SceneSubmissionCoordinator`; it must not reimplement them.
- M2 parity recovery (out of scope per spec Out-of-Scope, tracked in 037/038).

---

## Phase 4: Liquid, Sky, Particle, and Bounding-Box Renderers in the Shared Library

**Purpose**: Port the liquid, sky, particle, and bounding-box renderers from the legacy `Rendering/*` into the shared library, with `waterLOD` and `mapObjLightLOD` controls wired to the runtime pass coordinator. **US4 (water LOD + light LOD) is the primary user story.**

> **Phase 4 note**: this phase was split from an 11-task phase into 4a (Liquid + Sky, smaller) and 4b (Light LOD + Particle + BoundingBox + Minimap, larger). The split is the only deviation from the original 9-phase plan; the total phase count is now 10 (the 10th is the bookkeeping phase). See the Phase Dependency Graph in `plan.md` for the updated view.

### Phase 4a — Liquid + Sky (smaller half)

### Tests for Phase 4a (US4, US7) ⚠️

- [ ] T043 [P] [US4] Unit test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Liquid/WaterLodTests.cs`: given `(distance, waterLodSettings)`, the renderer selects `Full / Reduced / Culled` correctly; the per-frame `LiquidTilesByLodBucket` distribution matches the expected curve.

### Implementation for Phase 4a (US4, US7)

- [ ] T044 [P] [US4] Add `wow-viewer/src/core/WowViewer.Core.Renderer/Liquid/RenderLiquidTile.cs`, `WaterLodSettings.cs`, `WaterLodBucket.cs` per `data-model.md` §5.
- [ ] T045 [US4] Port the liquid renderer: `wow-viewer/src/core/WowViewer.Core.Renderer/Liquid/LiquidRenderer.cs` (rewrite for instanced + `waterLOD`). Update `LiquidShader.cs` with the reduced far-distance shader path.
- [ ] T046 [US4] Port the sky renderer: `wow-viewer/src/core/WowViewer.Core.Renderer/Sky/SkyRenderer.cs` (rewrite for multi-tile-aware; sky still per-frame, but the cached state is keyed by `SkyState`).
- [ ] T047 [US4, US7] Update `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/phase-4a/` with the water LOD and sky captures. Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.

**Checkpoint for 4a**: `waterLOD` produces a measurable per-frame draw-call drop at far distance (SC-005 acceptance 1). Sky state is keyed correctly and cache is invalidated on `SkyState` change. Alpha + LK parity preserved.

### Phase 4b — Light LOD + Particle + BoundingBox + Minimap (larger half)

### Tests for Phase 4b (US4, US7) ⚠️

- [ ] T048 [P] [US4] Unit test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Scene/LightLodTests.cs`: given `(maxLights, perObjectLightSelectionPolicy)`, the renderer selects the correct N lights per object; the per-frame `ActiveLightCount` is bounded by `MaxLights`.
- [ ] T049 [P] [US4] Integration test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Validation/LightLodIntegrationTests.cs`: stage `3_3_5_12340` map with many lights, render with `MaxLights=4`, assert `ActiveLightCount <= 4` per object and per-frame.

### Implementation for Phase 4b (US4, US7)

- [ ] T050 [P] [US4] Add `wow-viewer/src/core/WowViewer.Core.Renderer/Scene/LightLodSettings.cs`, `PerObjectLightSelectionPolicy.cs` per `data-model.md` §8; wire into `WorldFramePassCoordinator` (already extended in Phase 1a T018).
- [ ] T051 [US4] Port the particle renderer: `wow-viewer/src/core/WowViewer.Core.Renderer/Particle/ParticleRenderer.cs` (NEW) + `ParticleSystem.cs` (NEW). Consume `M2ParticleRibbonRuntime` for M2-attached particles.
- [ ] T052 [US4] Port the bounding-box renderer: `wow-viewer/src/core/WowViewer.Core.Renderer/BoundingBox/BoundingBoxRenderer.cs` (NEW). Used for the existing selection/inspector UX.
- [ ] T053 [US4] Port the minimap renderer: `wow-viewer/src/core/WowViewer.Core.Renderer/Minimap/MinimapRenderer.cs` (NEW). The compositor itself stays in `WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs` — the renderer is a consumer.
- [ ] T054 [US4, US7] Update `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/phase-4b/` with the light-LOD, particle, bounding-box, and minimap captures. Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.

**Checkpoint for 4b**: `mapObjLightLOD` and `MaxLights` are observable in the per-frame diagnostic stream (SC-005 acceptance 2+3, FR-009). Particle and bounding-box renderers work in deterministic capture. Minimap overlay works in deterministic capture. Alpha + LK parity preserved.

**Do not regress**:
- `TerrainMinimapCompositor` ownership stays in `WowViewer.Core.IO`. The renderer is a **consumer** of the compositor; the renderer does not redefine compositing.
- Particle/renderer ownership does not leak format-specific logic back into the app.

---

## Phase 5: TextureCache Mipmap Selection + Per-Frame Diagnostic Surface Finalization

**Purpose**: Extend the existing `WowViewer.Core.Renderer.Texture.TextureCache` with BLP mip-level selection (FR-010), and finalize the per-frame diagnostic surface (FR-004, SC-005). **US5 (BLP mipmaps) is the primary user story.**

### Tests for Phase 5 (US5, US7) ⚠️

- [ ] T055 [P] [US5] Unit test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Texture/MipSelectionTests.cs`: `SelectMip(entry, distance, settings)` is deterministic; the per-frame `MipSelectedDistribution` matches the expected curve when sampled at near/mid/far distances.
- [ ] T056 [P] [US5] Unit test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Texture/TextureCacheRefcountTests.cs`: `Acquire` twice + `Release` once keeps the entry; `Release` again evicts; `EvictAll` defers eviction for refcounted entries.
- [ ] T057 [P] [US5] Integration test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Validation/MipBandwidthTests.cs`: stage `3_3_5_12340` Azeroth, render a fixed camera route twice (with and without mip selection), assert `TextureBandwidthBytes` is lower with mip selection and visual output is within tolerance.

### Implementation for Phase 5 (US5, US7)

- [ ] T058 [P] [US5] Add `wow-viewer/src/core/WowViewer.Core.Renderer/Texture/TextureCacheEntry.cs` (NEW), `TextureResidency.cs` (NEW), `MipSelectionSettings.cs` (NEW in Scene, per `data-model.md` §9). No GL types in these files.
- [ ] T059 [US5] Extend `wow-viewer/src/core/WowViewer.Core.Renderer/Texture/TextureCache.cs` with `Acquire(path) -> TextureCacheEntry`, `Release(entry)`, `SelectMip(entry, distance, settings)`, `EvictAll()`, `OnMapSwitch()`. The existing immediate handle API is preserved as a deprecated path that calls through the new entry-based API.
- [ ] T060 [US5] Wire `SelectMip` into the renderer: during `Submit`, the renderer calls `SelectMip` for every bound texture and binds the selected mip level. The selection is recorded in `PerFrameRenderStats.MipSelectedDistribution`.
- [ ] T061 [US5] Wire `Acquire` / `Release` into the renderer: every `BindTexture` is paired with `Acquire` before and `Release` after. The `IRenderResources` cache wraps the calls so consumers don't see them.
- [ ] T062 [US5, US7] Update `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/phase-5/` with the bandwidth measurement. Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.

**Checkpoint**: BLP mip selection reduces per-frame texture bandwidth by a measurable amount on `3_3_5_12340` (SC-006, FR-010). Per-frame diagnostic stream exposes mip-selection counts, draw-call count, instance count, terrain LOD bucket count, water LOD bucket count, active light count, `mapObjLightLOD` value, `MaxLights` value (FR-004, SC-005, SC-010 acceptance). Existing `WowViewer.Core.Tests` integration tests pass with the extended diagnostic stream.

**Do not regress**:
- `WowViewer.Core.IO/Wmo/WmoMinimapAssetResolver.cs`, `WowViewer.Core.IO/Blp/AlphaBlpCompatibilityService.cs`, `WowViewer.Core.IO/Blp/BlpSummaryReader.cs`. Texture truth stays in IO; the renderer's texture cache is a **consumer** of these surfaces.

---

## Phase 6: Host Cutover — Retire `WoWViewer/Rendering/*` and Shrink `ViewerApp.cs`

**Purpose**: Switch `WowViewer.App` (and `WowViewer.Tool.ValidationCapture`) over to the new shared renderer. Retire the viewer-app `Rendering/*` namespace. `ViewerApp.cs` becomes substantially smaller. **US6 (ViewerApp-as-thin-host) is the primary user story.**

### Tests for Phase 6 (US6, US7) ⚠️

- [ ] T063 [P] [US6] Integration test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Validation/HostCutoverTests.cs`: stage `3_3_5_12340` Azeroth, invoke the headless `OpenGLRenderBackend` directly (no viewer-app code), assert the same scene renders as through `WowViewer.Tool.ValidationCapture`. This proves the host cutover has not changed the rendering path.
- [ ] T064 [P] [US6] Static check in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Validation/HostCutoverStaticTests.cs`: assert that `wow-viewer/src/viewer/WoWViewer/Rendering/*` is empty (or the directory does not exist); assert that `WowViewer.csproj` and `WowViewer.CrossPlatform.csproj` do not reference any `wow-viewer/src/viewer/WoWViewer/Rendering/*` source file.

### Implementation for Phase 6 (US6, US7)

- [ ] T065 [P] [US6] Audit `wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj` and `wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj`: list every `Compile Include` that points at `wow-viewer/src/viewer/WoWViewer/Rendering/*`. Record the list in the commit message.
- [ ] T066 [US6] Update `wow-viewer/src/viewer/WoWViewer/ViewerApp_RenderQuality.cs` to wire `RenderVariant` to the shared library (not to viewer-local types). This is the host-side wiring; the actual values come from `ViewerApp`'s `RenderQualitySettings` field.
- [ ] T067 [US6] Update `wow-viewer/src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs` and `wow-viewer/src/viewer/WoWViewer/Terrain/VlmTerrainManager.cs` to consume the shared renderer's terrain stage. The adapter becomes a wiring layer (calls into the shared library), not an implementation.
- [ ] T068 [US6] Update `wow-viewer/tools/validation-capture/` to consume the shared `OpenGLRenderBackend` (FR-011). The capture CLI surface is preserved.
- [ ] T069 [US6] Delete the files under `wow-viewer/src/viewer/WoWViewer/Rendering/*` (FR-013). Update both csproj files to drop the `Compile Include` entries. Verify `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` succeeds.
- [ ] T070 [US6, US7] Update `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/phase-6/` with the post-cutover capture; assert parity with the pre-spec baseline. Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.

**Checkpoint**: `wow-viewer/src/viewer/WoWViewer/Rendering/*` is empty or absent (SC-010, FR-013). `WowViewer.App` no longer compiles any renderer implementation code outside the shared library (FR-014, SC-010). `ViewerApp.cs` is "substantially smaller" per the user's D5 (no numeric target; code review by maintainer) (SC-007). The viewer app, the validation-capture tool, and the headless renderer all invoke the same shared renderer entry point (FR-012 acceptance). A staged `3_3_5_12340` map renders identically through the cutover (FR-016, FR-017, FR-018). `WowViewer.Tool.ValidationCapture` continues to produce the same `object_visibility_mask` for `Azeroth_30_48` (SC-008).

**Do not regress**:
- Spec 044 dockspace shell, spec 045 scene graph workbench, spec 049 viewer UI consolidation. They are viewer-app UX; this phase must not silently break their host surfaces.

---

## Phase 7: Real-Data Validation Suite (Cross-Phase Parity)

**Purpose**: Land a permanent validation surface that exercises the new shared renderer on staged `0_5_3_3368` and `3_3_5_12340`, producing deterministic captures and structured reports. This is the harness that the engine plan and AGENTS.md require for any change touching terrain, liquid, WMO, or M2 rendering. **US7 (real-data validation suite) is the primary user story.**

### Tests for Phase 7 (US7) ⚠️

- [ ] T071 [P] [US7] Test in `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Validation/ParitySuiteTests.cs`: run the full parity suite on staged `0_5_3_3368` Azeroth and `3_3_5_12340` Azeroth; assert each capture is within tolerance of the pre-spec baseline; assert the per-frame diagnostic counters are within expected ranges.

### Implementation for Phase 7 (US7)

- [ ] T072 [P] [US7] Add `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Validation/ParityFixture.cs` (loads staged clients, runs the headless renderer, captures the output, compares against baselines).
- [ ] T073 [US7] Extend `wow-viewer/tools/validation-capture/Program.cs` with the real `compare-baseline` subcommand: takes `--build`, `--tile`, `--baseline-path`, runs the headless renderer, produces a per-pixel diff report (PNG diff + JSON summary), exits non-zero on regression.
- [ ] T074 [US7] Wire the parity suite into the build pipeline: add a CI step (PowerShell or GitHub Actions YAML) that runs the parity suite on every PR. Document the step in `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/quickstart.md`.
- [ ] T075 [US7] Update `wow-viewer/memory-bank/progress.md` with the final validation harness entry (Phase 7 completion).

**Checkpoint**: `dotnet test wow-viewer/tests/WowViewer.Core.Renderer.Tests` runs the parity suite on `0_5_3_3368` Azeroth and `3_3_5_12340` Azeroth and passes. The harness exits non-zero on any visual regression above documented tolerance. The harness exposes per-frame diagnostic counters (draw calls, instances, mip selection, terrain LOD bucket, water LOD, light count) (FR-004, SC-005, SC-010).

**Do not regress**:
- Existing `WowViewer.Tool.ValidationCapture` capture output schema.
- Existing baseline captures in `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/`.

---

## Phase 8: Spec 036 Archive, Memory-Bank Sync, Final Spec Audit

**Purpose**: Archive `specs/036-renderer-improvements` with a forward pointer to this spec, sync the memory bank, and run a final spec-vs-code audit.

- [ ] T076 [P] Move `wow-viewer/specs/036-renderer-improvements/` to `wow-viewer/specs/archived/036-renderer-improvements/`. Add a banner at the top of `spec.md` saying "Superseded by `specs/056-viewerapp-gpu-lod-modernization/`."
- [ ] T077 [P] Append a `wow-viewer/specs/archived/036-renderer-improvements/ARCHIVED.md` entry: "Superseded by spec 056 (2026-06-10). Renderer-improvements convergence work is now owned by 056."
- [ ] T078 [P] Update `wow-viewer/docs/architecture/speckit-doc-audit-2026-05-18.md`: change the `wow-viewer-library-completeness-plan-2026-05-06.md` row to "completed by 056"; change the `game-viewer-host-plan-2026-05-13.md` row to "slices 3-6 covered by 056"; change the `wow-viewer-full-porting-roadmap.md` row to "Phase I Priority 5 in-progress under 056".
- [ ] T079 [P] Author `wow-viewer/docs/architecture/speckit-doc-audit-2026-06-XX.md` (new audit) with the post-spec-056 state: which docs are implemented, partial, planned, stale.
- [ ] T080 [US6] Compress `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` to ≤ 200 lines each. The "in progress" lane is now spec 056 → next phase, or "all phases done" if at end of spec.
- [ ] T081 [US7] Run the full `quickstart.md` validation suite (build + all tests + capture on both staged clients) and record the result in `wow-viewer/memory-bank/progress.md`.

**Checkpoint**: `specs/036-renderer-improvements` is under `wow-viewer/specs/archived/`. `specs/archived/ARCHIVED.md` lists 036 with a forward pointer to 056. `memory-bank/activeContext.md` and `memory-bank/progress.md` are compressed and current (≤ 200 lines each). `dotnet build wow-viewer/WowViewer.slnx -c Debug` and `dotnet test` both pass. A new `docs/architecture/speckit-doc-audit-2026-06-XX.md` exists and reflects 056.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 0**: No dependencies. Creates the test project, locks the contracts, records the pre-spec baseline.
- **Phase 1**: Depends on Phase 0. Promotes the renderer skeleton to multi-tile + retained-mode + instanced + frustum-culled. **BLOCKS Phases 2-5.**
- **Phase 2**: Depends on Phase 1. Terrain renderer in the shared library. **Can run in parallel with Phase 3 and Phase 4** (different files; both consume Phase 1's multi-tile core).
- **Phase 3**: Depends on Phase 1. WMO + M2 + MDX renderers in the shared library. **Can run in parallel with Phase 2 and Phase 4** (different files; both consume Phase 1's multi-tile core).
- **Phase 4**: Depends on Phase 1. Liquid + Sky + Particle + BoundingBox + Minimap renderers in the shared library. **Can run in parallel with Phase 2 and Phase 3** (different files; both consume Phase 1's multi-tile core).
- **Phase 5**: Depends on Phase 1. TextureCache mipmap selection. **Can run in parallel with Phases 2-4** (different files), but the renderer wiring (T060, T061) depends on Phases 2-4 having their renderer calls in place.
- **Phase 6**: Depends on Phases 1-5. The host cutover. **BLOCKS Phase 7.**
- **Phase 7**: Depends on Phase 6. The permanent validation harness. **BLOCKS Phase 8.**
- **Phase 8**: Depends on Phase 7. The bookkeeping phase. Final.

### Within Each Phase

- Tests (T010-T012, T024-T025, T033-T035, T043, T048, T056-T058, T066-T067, T073) MUST be written and FAIL before implementation.
- Models / types before services.
- Service / renderer implementations before consumer wiring.
- Story complete before moving to the next priority.
- The phase's `quickstart.md` checklist items pass before declaring the phase done.

### Parallel Opportunities

- **Phase 0**: T001 (test project), T002 (Contracts types), T003-T006 (contract docs) can all run in parallel. T007 (baseline capture), T008 (CLI stub), T009 (memory bank update) are sequential after T001-T006.
- **Phase 1a (1 of 1)**: T013 (RenderScene record), T017 (Contracts interfaces) can run in parallel. T014-T016 (Scene extensions) are sequential after T013. T018 (WorldFramePassCoordinator extension) is independent of the renderer code and can run in parallel.
- **Phase 1b (2 of 1)**: T019 (OpenGL backend skeleton) can run in parallel with T020 (buffer/shader factory). T021 (SceneRenderer rewrite) depends on T013-T020. T022 (Headless de-dup) can run in parallel with T021. T023 (validation) is the final step.
- **Phases 2, 3, 4a, 4b**: each has its own concern and its own files. They can be worked on by different developers in parallel. The MDX files in Phase 3 are the highest-risk for over-running the 10-task budget; if MDX needs its own phase, split into Phase 3a and Phase 3b (decision at T037).
- **Phase 5**: T059 (new types) can run in parallel with T060 (TextureCache extension). T061-T062 (renderer wiring) depend on Phases 2-4. T063 (validation) is final.
- **Phase 6**: T068 (csproj audit) is the first step. T069-T071 (host wiring) can run in parallel after T068. T072 (delete legacy files) is sequential after T069-T071. T073 (validation) is final.
- **Phase 7**: T074 (test), T075 (fixture), T076 (CLI subcommand) can run in parallel. T077 (CI wiring) depends on T074-T076. T078 (memory bank) is final.
- **Phase 8**: T079 (archival + audit) can all run in parallel. T080-T081 (memory bank compression + final validation) are sequential at the end.

### User Story Traceability

- US1 (multi-tile render): **Phase 1a (T010-T012, T013-T018) + Phase 1b (T019-T023)**.
- US2 (terrain mesh LOD + WDL far horizon): **Phase 2** (T024-T032).
- US3 (object LOD + draw-distance culling): **Phase 3** (T033-T042).
- US4 (water LOD + light LOD): **Phase 4a (T043-T047) + Phase 4b (T048-T054)**.
- US5 (BLP mipmaps): **Phase 5** (T055-T063).
- US6 (ViewerApp-as-thin-host): **Phase 6** (T064-T073).
- US7 (real-data validation): **All phases** (T001, T007-T009, T012, T023, T032, T042, T047, T054, T063, T073, T074-T078, T081).

---

## Implementation Strategy

### MVP First (Phase 0 + Phase 1)

1. Complete Phase 0 (test project, contracts, baseline).
2. Complete Phase 1 (multi-tile + retained-mode + instanced + frustum-culled core).
3. **STOP and VALIDATE**: the renderer can render a multi-tile AOI on `3_3_5_12340` through the headless capture path. The pre-spec baseline is preserved.
4. The MVP is "shared renderer can drive a real map" — even without LOD, without object rendering, without liquid.

### Incremental Delivery

1. Phase 0 + Phase 1 → foundation ready. Headless capture works for terrain-only scenes.
2. Phase 2 → terrain LOD. The visible result: distant terrain is cheaper, near terrain is full.
3. Phase 3 → object LOD. The visible result: cities render, distant objects cull.
4. Phase 4 → liquid + sky + particle. The visible result: outdoor scenes are complete.
5. Phase 5 → mipmap selection. The visible result: bandwidth drops, no visual change.
6. Phase 6 → host cutover. The visible result: `ViewerApp.cs` shrinks, `Rendering/*` is gone.
7. Phase 7 → permanent validation. The visible result: regressions are caught.
8. Phase 8 → bookkeeping. The visible result: 036 is archived, memory bank is current.

### Parallel Team Strategy

With multiple developers:

1. Team completes Phase 0 + Phase 1 together.
2. Once Phase 1 is done:
   - Developer A: Phase 2 (terrain)
   - Developer B: Phase 3 (WMO + M2)
   - Developer C: Phase 4 (liquid + sky + particle + minimap)
   - Developer D: Phase 5 (mipmap)
3. Phases 2-5 complete and integrate independently.
4. Phase 6 is sequential after all of 2-5 (the cutover touches everything).
5. Phase 7 is sequential after 6.
6. Phase 8 is sequential after 7.

---

## Notes

- `[P]` tasks = different files, no dependencies.
- `[Story]` label maps task to spec user story (US1-US7) for traceability.
- Each user story is independently completable and testable.
- Verify tests fail before implementing (the `Tests for ... ⚠️` blocks).
- Commit after each task or logical group. Each commit is a single concern.
- Stop at any checkpoint to validate the story independently.
- Avoid: vague tasks, same-file conflicts, cross-story dependencies that break independence.
- The `[P]` flag assumes a parallel-friendly file layout. If two `[P]` tasks touch the same `Compile Include` glob in the csproj, they are NOT truly parallel; the first one to land updates the glob, the second lands with a "compile after first" comment.
- All tasks that touch terrain must validate against `0_5_3_3368` MCAL parity (terrain alpha risk area).
- All tasks that touch the renderer must not regress the `specs/020-renderer-culling-and-tile-capture` P1 culling fix.
- No task may modify a `WowViewer.Core.IO` parser. If a parser change is needed, escalate to the user; this is RULE 3.
- No task may reference `H:\CLIENTS`. Staged clients only (RULE 9).
- Total task count: **81 tasks** across 9 phases (Phase 1 split into 1a + 1b; Phase 4 split into 4a + 4b). Maximum per sub-phase: 10. Bite-sized rule honored.

### Source-of-truth rules (binding for every task)

1. **Source code source-of-truth** = `wow-viewer/src/viewer/WoWViewer/Rendering/*` and the current viewer-app `Terrain/*` files. The new shared library is built by *improving and moving* this code.
2. **Forbidden source** = `wow-viewer/src/viewer/WowViewer.App.Defunct/*`. Do not read, do not port, do not reference. Treated as a poisoned source per user instruction 2026-06-10.
3. **Read-only correctness reference** = `gillijimproject_refactor/src/MdxViewer/Rendering/*` (RULE 1). Not the source of truth; may be consulted for cross-checks at most.
4. **Correctness oracle** = `docs/architecture/wmo-render-pass-architecture-2026-05-30.md` (Ghidra-confirmed 3.3.5 renderer research). The new WMO renderer must conform to the dispatch logic in that doc (interior/exterior, MOMT flag testing, lightmap pass split, liquid type dispatch, portal-walk visibility, group flag filtering). It is **not** a code source; conformance is the test.
5. If a task description above says "if it exists in the legacy surface" or similar, that text means the **current viewer-app** surface (`wow-viewer/src/viewer/WoWViewer/*`), not the legacy `MdxViewer`. The legacy surface is not to be ported from.

---

*End of tasks. Next: load `speckit-implement` (when ready to execute) or `speckit-checklist` (to verify alignment).*
