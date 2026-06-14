# Research: 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization

**Phase 0 output. Companion to `plan.md`.**
**Date**: 2026-06-10.

This file is the Phase 0 research that the plan's Phase 0 step "lock the contract seams" relies on. It inventories the existing surfaces the new shared renderer will consume, port from, or coordinate with.

---

## 1. Existing Surfaces the New Renderer Consumes (Do Not Rewrite)

### 1.1 `WowViewer.Core.Runtime.World` — LOD, visibility, pass routing (already exists)

| File | Role | Plan phase that depends on it |
|---|---|---|
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainLodSelector.cs` | Pure-function LOD selector. Takes `(chunk, distance, textureLodDistance, fogEndDistance)` and returns `WorldTerrainLodSelection { Level, ActiveTextureLayerCount, OverlayFadeFactor, RenderableCellCount, UsesLowDetailMesh }`. Enum is currently `FullDetail / FadeToBaseLayer / BaseLayerOnly / LowDetail`. | Phase 2 (terrain). The current enum is **almost** the right shape, but it does not yet have a "UseWdl" bucket (FR-007). Phase 2 must add that bucket OR the renderer must translate `LowDetail` into the WDL representation. Decision is made in Phase 2 tasks. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs` | Pure-function visibility collector. Returns `WorldVisibleMdxEntry[]` / `WorldVisibleWmoEntry[]`. | Phase 3 (WMO/M2/MDX). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityContext.cs` | Context object for the visibility collector. | Phase 3. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibilityFrame.cs` | Per-frame visibility snapshot. | Phase 3. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldObjectInstance.cs` | Per-instance world object (position, rotation, scale, model path, doodad set, etc.). | Phase 3. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldMdxRenderPlan.cs` | Per-MDX render plan. | Phase 3. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderCompositionBuilder.cs` | Builds the per-frame render composition (terrain + objects + liquid + sky). | Phase 1, 2, 3, 4. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderCompositionFrame.cs` | Per-frame composition result. | Phase 1, 2, 3, 4. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs` | Per-frame statistics (draw calls, instances, etc.). | Phase 1, 5. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderOptimizationAdvisor.cs` | Optimization advisor (suggests LOD, draw distance, etc.). | Phase 1, 5. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldSkyboxBackdropClassifier.cs` | Skybox backdrop classifier. | Phase 4. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs` | Per-frame pass coordinator. | Phase 1, 4, 5 (extends with `waterLOD`, `mapObjLightLOD`, `MaxLights`). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs` | Per-object pass coordinator. | Phase 3. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassFrame.cs` | Per-object pass result. | Phase 3. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldTileStageSummary.cs`, `WorldTileStageSummaryBuilder.cs` | Per-tile stage summary. | Phase 2. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldVisibleMdxPassRoute.cs` | Per-MDX pass route. | Phase 3. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Liquid/*` | Liquid domain data (per-tile, per-chunk, per-layer). | Phase 4. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrain*` (8 files) | Terrain domain data. | Phase 2. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wdl/WorldWdlTile*` | WDL domain data. | Phase 2 (FR-007 far horizon). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/HeadlessValidationCaptureSession.cs` | Headless capture harness. | Phase 1, 7. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/IValidationWorldSceneAdapter.cs` | Adapter interface for validation. | Phase 1, 7. |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/ValidationCapture*` (20 files) | Validation capture machinery. | Phase 7. |

**Key finding**: The runtime-side LOD / visibility / pass-routing / validation surface is already in place. The new shared renderer is a **consumer** of these, not a parallel owner. No changes to `WowViewer.Core.Runtime` are required by the new spec except for the explicit LOD/light controls extensions listed in `plan.md` (Phase 1 for `waterLOD` / `mapObjLightLOD` / `MaxLights`; Phase 2 for the WDL bucket if `LowDetail` is not enough).

### 1.2 `WowViewer.Core.M2` / `WowViewer.Core.Runtime.M2` — M2 runtime (already exists)

| File | Role | Plan phase |
|---|---|---|
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2RuntimeFramePipeline.cs` | Per-frame M2 pipeline. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2SceneSubmissionCoordinator.cs` | M2 scene submission. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2SceneSubmissionEntryBuilder.cs` | M2 scene submission entry. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2BonePoseEvaluator.cs`, `M2TrackSampler.cs`, `M2EffectRecipe.cs` | M2 animation. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2SkinProfileRuntime*.cs` | M2 skin profile. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2SkinnedRenderModelBuilder.cs`, `M2StaticRenderModelBuilder.cs`, `M2StaticRenderModel.cs` | M2 render model builders. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2ParticleRibbonRuntime.cs` | M2 particle/ribbon. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2AnimatedRenderStateEvaluator.cs` | M2 animated state. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2ExternalAnimationRuntime*.cs` | M2 external animation. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2RenderFrame.cs`, `M2RenderConsumerFrameState.cs` | M2 frame data. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2CameraPathOverlayBuilder.cs`, `M2CameraPathVisualization.cs` | M2 camera path. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2SoftwareVisualSnapshot.cs` | M2 software snapshot. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2RuntimeGoldenFrame.cs` | M2 golden frame (parity). | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/M2dx/MdxEffectRuntime.cs` | MDX effect runtime. | Phase 3 (consumer). |
| `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldMdxRenderPlan.cs` | MDX render plan. | Phase 3 (consumer). |

**Key finding**: The M2/MDX runtime is **deep** in `WowViewer.Core.Runtime` and is NOT to be ported into the new renderer. The renderer is a consumer of the runtime; the runtime stays put.

### 1.3 `WowViewer.Core.Renderer` skeleton (already exists, will be grown)

| File | Current state | Plan phase |
|---|---|---|
| `Scene/SceneRenderer.cs` | Single-tile stub. | Phase 1 (rewrite for multi-tile). |
| `Scene/FrustumCuller.cs` | Exists, single-AABB test. | Phase 1 (extend with per-tile / per-WMO / per-M2 helpers). |
| `Scene/SceneCamera.cs` | Exists. | Phase 1 (extend with view-projection helpers, fog distances, AOI helpers). |
| `Scene/RenderVariant.cs` | Exists. | Phase 1 (extend with `waterLOD`, `mapObjLightLOD`, `MaxLights`, terrain LOD thresholds). |
| `Terrain/TerrainRenderer.cs`, `TerrainMesh.cs`, `TerrainMeshBuilder.cs`, `TerrainShader.cs`, `TerrainConstants.cs` | Exists, single-tile. | Phase 2 (port to multi-tile + LOD-aware). |
| `Wmo/WmoRenderer.cs`, `WmoMesh.cs`, `WmoShader.cs` | Exists, single-instance. | Phase 3 (port to instanced + object-LOD). |
| `Liquid/LiquidRenderer.cs`, `LiquidShader.cs` | Exists. | Phase 4 (port to instanced + `waterLOD`). |
| `Sky/SkyRenderer.cs` | Exists. | Phase 4 (port). |
| `Texture/TextureCache.cs` | Exists. | Phase 5 (extend with mip selection + residency tracking). |
| `Headless/HeadlessContext.cs`, `FrameCapture.cs`, `PngWriter.cs`, `RenderSurface.cs` | Exists. | Phase 1, 7 (extend; this is the validation path). |
| `Output/FrameCapture.cs`, `PngWriter.cs` | Duplicate of Headless. | Phase 1 (de-duplicate to one Headless path; FR-011). |
| `Validation/NativeValidationWorldSceneAdapter.cs` | Exists. | Phase 1, 7 (consume). |

**Key finding**: The new renderer is built on top of this skeleton, not parallel to it. The skeleton is sparse but has the right shape (Scene, Terrain, Wmo, Liquid, Sky, Texture, Headless, Validation, Output). M2 and MDX are missing as namespaces — Phase 3 adds them.

### 1.4 Source-of-truth map (WoWViewer = source; MdxViewer = reference only; Defunct = forbidden)

> **Three sources, three roles** (per user clarification 2026-06-10):
>
> 1. **`wow-viewer/src/viewer/WoWViewer/Rendering/*`** = the current active viewer renderer. This is the **source of truth for current behavior**. The new shared library is built by *improving and moving* this code.
> 2. **`gillijimproject_refactor/src/MdxViewer/Rendering/*`** = the old legacy MdxViewer. Read-only reference per RULE 1. **Not** the source of truth. May be consulted for cross-checks at most; never ported.
> 3. **`wow-viewer/src/viewer/WowViewer.App.Defunct/*`** = **forbidden**. Do not read, do not port, do not reference. Explicit user instruction 2026-06-10.
>
> The Ghidra-confirmed 3.3.5 renderer research at `docs/architecture/wmo-render-pass-architecture-2026-05-30.md` is the **correctness oracle** for the new WMO renderer. It tells us *what the renderer must do* (interior/exterior dispatch, per-batch MOMT flag testing, lightmap pass split, liquid type dispatch, portal-walk visibility). It is **not** a code source. The new renderer conforms to it because the native client confirms that is the only correct 3.3.5 behavior; we are not "porting the decompilation."

### 1.4 Legacy / viewer-local surfaces to retire in Phase 6

| File | Source | Lines | Plan phase | Role |
|---|---|---|---|
| `wow-viewer/src/viewer/WoWViewer/Rendering/BlendStateManager.cs` | Viewer-app | (size TBD) | Phase 1 (port to `WowViewer.Core.Renderer.OpenGL/BlendStateManager.cs`) / Phase 6 (retire). |
| `wow-viewer/src/viewer/WoWViewer/Rendering/Camera.cs` | Viewer-app | (size TBD) | Already superseded by `WowViewer.Core.Renderer.Scene.SceneCamera`; Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/FrustumCuller.cs` | Viewer-app | (size TBD) | Already superseded by `WowViewer.Core.Renderer.Scene.FrustumCuller`; Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/IAnimationController.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer.M2`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/IModelRenderer.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer.Contracts`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/ISceneRenderer.cs` | Viewer-app | (size TBD) | Phase 1 (port to `WowViewer.Core.Renderer.Contracts.RenderBackend`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/LoadingScreen.cs` | Viewer-app | (size TBD) | Phase 4 (port to `WowViewer.Core.Renderer.LoadingScreen`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/M2CameraPathRenderer.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer.M2`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/M2MaterialPassProfile.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer.M2`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/M2Renderer.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer.M2`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/M2RouteDecision.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer.M2`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/M2RouteDiagnostics.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer.M2`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/M2RuntimeAnimator.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer.M2`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/Material.cs` | Viewer-app | (size TBD) | Phase 1 (port to `WowViewer.Core.Renderer.Material`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/MdxAnimator.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer.Mdx`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/MinimapRenderer.cs` | Viewer-app | (size TBD) | Phase 4 (port to `WowViewer.Core.Renderer.Minimap`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs` | Viewer-app | (size TBD) | Phase 3 (port to `WowViewer.Core.Renderer`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/ParticleRenderer.cs`, `ParticleSystem.cs` | Viewer-app | (size TBD) | Phase 4 (port to `WowViewer.Core.Renderer.Particle`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/RenderQualitySettings.cs` | Viewer-app | (size TBD) | Phase 1 (port to `WowViewer.Core.Renderer.RenderQualitySettings`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/RenderQueue.cs` | Viewer-app | (size TBD) | Phase 1 (port to `WowViewer.Core.Renderer.OpenGL`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/ReplaceableTextureResolver.cs` | Viewer-app | (size TBD) | Phase 5 (port to `WowViewer.Core.Renderer.Texture`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/ShaderProgram.cs` | Viewer-app | (size TBD) | Phase 1 (port to `WowViewer.Core.Renderer.OpenGL`); Phase 6 retire. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs` | Viewer-app | (size TBD) | Phase 4 (move + improve into `WowViewer.Core.Renderer.Sky`); Phase 6 delete. | **Source of truth**. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/WarcraftNetM2Adapter.cs` | Viewer-app | (size TBD) | Phase 3 (move + improve into `WowViewer.Core.Renderer.M2`); Phase 6 delete. | **Source of truth**. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs` | Viewer-app | (size TBD) | Phase 3 (move + improve into `WowViewer.Core.Renderer.Wmo`); Phase 6 delete. | **Source of truth**. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/WoWConstants.cs` | Viewer-app | (size TBD) | Phase 1 (move + improve into `WowViewer.Core.Renderer`); Phase 6 delete. | **Source of truth**. |
| `wow-viewer/src/viewer/WoWViewer/Rendering/WowViewerM2RuntimeBridge.cs` | Viewer-app | (size TBD) | Phase 3 (move + improve into `WowViewer.Core.Renderer.M2`); Phase 6 delete. | **Source of truth**. |
| `wow-viewer/src/viewer/WowViewer.App.Defunct/*.cs` (M2GpuPreviewRenderer, MdxGpuPreviewRenderer, WmoGpuPreviewRenderer, WorldGpuPreviewRenderer, WorldMinimapRenderer, ModelOutputGpuRenderer, etc.) | **Forbidden** | (size TBD) | **N / A — DO NOT READ, DO NOT PORT, DO NOT REFERENCE.** | **Poisoned source**. Excluded by user instruction 2026-06-10. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/M2Renderer.cs` | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs` (~1500+ lines) | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/TerrainRenderer.cs` (~1808 lines) | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/MdxRenderer.cs` (~2866 lines) | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/LiquidRenderer.cs` (~500 lines) | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/SkyDomeRenderer.cs` (~300 lines) | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/FrustumCuller.cs` (~150 lines) | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/ShaderProgram.cs` (~112 lines) | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/RenderQueue.cs` (~200 lines) | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `gillijimproject_refactor/src/MdxViewer/Rendering/Material.cs` (~100 lines) | Read-only ref | (size TBD) | **Not** the source of truth. Correctness cross-check at most. | RULE 1. |
| `docs/architecture/wmo-render-pass-architecture-2026-05-30.md` | Ghidra research | (Ghidra doc) | **Correctness oracle** for the new WMO renderer. Tells us *what the renderer must do* (interior/exterior dispatch, per-batch MOMT flag testing, lightmap pass split, liquid type dispatch, portal-walk visibility). | **Not** a code source. The new renderer conforms to it because that is the only correct 3.3.5 behavior. |

**Key finding**: The legacy `MdxViewer/Rendering/*` files are not "game client file reading tooling" (RULE 3). They are renderer implementation. The user is explicitly asking to refactor them out of the legacy repo into a shared library, so this is allowed and intended — done by **moving**, not **rewriting** (consistent with the analysis D1 answer "Build a new shared renderer from scratch" means carefully porting; not blowing it up).

### 1.5 `WowViewer.Core.Renderer.Headless/*` (the validation surface)

`Headless/HeadlessContext.cs`, `Headless/FrameCapture.cs`, `Headless/PngWriter.cs`, `Headless/RenderSurface.cs` and the duplicated `Output/FrameCapture.cs`, `Output/PngWriter.cs` are the validation surface. `WowViewer.Tool.ValidationCapture` consumes them. Per FR-011, this path must keep working. Phase 1 de-duplicates `Output/*` into `Headless/*`.

### 1.6 `WowViewer.Tool.ValidationCapture` and the world validation capture flow

`wow-viewer/tools/validation-capture/` and `WowViewer.Core.Runtime/World/Validation/HeadlessValidationCaptureSession.cs` form the end-to-end validation flow. Phase 1, 6, 7 exercise this flow.

---

## 2. Existing Plans/Specs that Map onto Phases

| Plan / spec | Section | Mapped to phase |
|---|---|---|
| `wow-engine-modernization-plan-2026-05-14.md` | viewer-first, UE bridge. Replaced 2026-06-14. | This spec supersedes the old engine-phases framing. OpenGL modernization is viewer-internal — no Vulkan, no UE backend here. |
| `wow-engine-editor-and-interop-plan-2026-05-14.md` | Renderer Layer Model | Phase 1 (Contracts). |
| `game-viewer-plan-pack-2026-05-14/gv-14-render-layer-contracts.md` | Render layer contracts | Phase 1 (Contracts). |
| `game-viewer-plan-pack-2026-05-14/gv-15-terrain-and-liquid-render-packets.md` | Terrain + liquid render packets | Phase 2 + 4. |
| `game-viewer-plan-pack-2026-05-14/gv-16-object-model-render-packets.md` | Object + model render packets | Phase 3. |
| `game-viewer-plan-pack-2026-05-14/gv-17-backend-bridge-vulkan-opengl.md` | Vulkan/OpenGL backend bridge | Phase 1 (the OpenGL half; Vulkan half is follow-on). |
| `game-viewer-host-plan-2026-05-13.md` | Slice 3 (World Session Closure) | Phase 1, 2. |
| `game-viewer-host-plan-2026-05-13.md` | Slice 4 (Terrain/Liquid Shader Baseline) | Phase 2, 4. |
| `game-viewer-host-plan-2026-05-13.md` | Slice 5 (Skybox And Lighting Parity) | Phase 4. |
| `game-viewer-host-plan-2026-05-13.md` | Slice 6 (Standalone Asset Consumer Closure) | Phase 3. |
| `wow-viewer-full-porting-roadmap.md` | Phase I Priority 5 (Viewer/Editor) | All phases. |
| `wow-viewer-full-porting-roadmap.md` | Phase D Priority 1 (Deep Format Readers) | **Out of scope**, separate lane (RULE 3). |
| `wow-viewer-library-completeness-plan-2026-05-06.md` | Section 2.3 (Rendering System table) | All phases. |
| `wow-viewer-library-completeness-plan-2026-05-06.md` | Section 3 Phase F (Renderer Architecture) | All phases. |
| `specs/020-renderer-culling-and-tile-capture` | All user stories | Hard prerequisite; do not regress. |
| `specs/030-wmo-render-pass-architecture` | All user stories | Phase 3 (WMO). |
| `specs/031-terrain-cell-awareness` | All user stories | Phase 2 (terrain). |
| `specs/032-native-renderer-parity` | All user stories | Phase 1, 2, 3, 4 (the "parity" framing). |
| `specs/036-renderer-improvements` | All user stories | **Superseded** by this spec (Phase 8 archives it). |
| `specs/044-viewer-shell-usability` | All user stories | Out of scope (viewer-app UX, not renderer). |
| `specs/045-scene-graph-workbench` | All user stories | Out of scope (viewer-app UX, not renderer). |
| `specs/049-viewer-ui-consolidation` | All user stories | Out of scope. |
| `specs/055-unreal-engine-bridge` | All user stories | Out of scope (post-v0.5.0-dev). |

**Key finding**: This spec is the single owner plan for renderer modernization on `v0.5.0-dev`. The other overlapping specs either feed into a phase (030, 031, 032, 020) or are explicitly out of scope (044, 045, 049, 055).

---

## 3. Reuse-and-Adapt Map (Renderer-Internal)

The new shared renderer should reuse, not reinvent, these existing building blocks:

| Block | Where it lives | Reuse strategy |
|---|---|---|
| `WorldTerrainLodSelector` | `WowViewer.Core.Runtime.World.Terrain` | Consume in Phase 2. Extend with WDL bucket if `LowDetail` is insufficient. |
| `WorldObjectVisibilityCollector` | `WowViewer.Core.Runtime.World.Visibility` | Consume in Phase 3. |
| `WorldRenderCompositionBuilder` | `WowViewer.Core.Runtime.World` | Consume in Phase 1, 2, 3, 4. |
| `WorldFramePassCoordinator` | `WowViewer.Core.Runtime.World.Passes` | Consume in Phase 1, 4, 5. Extend with `waterLOD` / `mapObjLightLOD` / `MaxLights`. |
| `M2RuntimeFramePipeline`, `M2SceneSubmissionCoordinator` | `WowViewer.Core.Runtime.M2` | Consume in Phase 3. Do NOT reimplement. |
| `WorldWdlTileData`, `WorldWdlTileBuilder` | `WowViewer.Core.Runtime.World.Wdl` | Consume in Phase 2 (WDL far horizon). |
| `WorldRenderFrameStats` | `WowViewer.Core.Runtime.World` | Consume in Phase 1, 5. |
| `FrustumCuller` | `WowViewer.Core.Renderer.Scene` | Extend in Phase 1 with per-tile / per-WMO / per-M2 helpers. |
| `SceneCamera` | `WowViewer.Core.Renderer.Scene` | Extend in Phase 1. |
| `RenderVariant` | `WowViewer.Core.Renderer.Scene` | Extend in Phase 1. |
| `TextureCache` | `WowViewer.Core.Renderer.Texture` | Extend in Phase 5. |
| `HeadlessContext`, `FrameCapture`, `PngWriter`, `RenderSurface` | `WowViewer.Core.Renderer.Headless` | De-duplicate `Output/*` in Phase 1. |

---

## 4. Risks Specific to the Porting Phases

In addition to the 10 risks in the analysis doc, the porting phases have these specific risks:

| # | Risk | Phase | Mitigation |
|---|---|---|---|
| P1 | The legacy `MdxRenderer.cs` is ~2,866 lines and may be hard to break into a multi-tile instanced renderer in one bounded pass. | Phase 3 | Split MDX port into (3a) M2 first (smaller), (3b) MDX second (bigger), with separate validation after each. tasks.md enforces ≤ 10 steps per phase, so the phase itself is bounded, but MDX may need to be its own phase. **Decision is made in tasks.md for Phase 3 — if MDX exceeds 10 steps, split into Phase 3 (M2 + WMO) and Phase 3.5 (MDX).** |
| P2 | The `WorldTerrainLodSelector` enum (`FullDetail / FadeToBaseLayer / BaseLayerOnly / LowDetail`) does not yet model the WDL far-horizon bucket. | Phase 2 | Either (a) extend the enum with a `UseWdl` bucket, or (b) translate `LowDetail` into a WDL representation in the renderer. tasks.md records which is chosen. |
| P3 | The existing `WowViewer.Core.Renderer` skeleton uses immediate-mode GL in places (e.g. `SceneRenderer.GetOrCreateMesh` builds with `TerrainMeshBuilder` and likely uses immediate-mode VBO setup). | Phase 1 | Replace with retained-mode VBO/IBO + UBO in Phase 1. Verified by `dotnet build` and a deterministic capture baseline check. |
| P4 | The legacy `Rendering/ShaderProgram.cs` may use OpenGL 3.0-era patterns. | Phase 1 | Port to `WowViewer.Core.Renderer.OpenGL.OpenGLShaderCache` with explicit versioning. |
| P5 | The viewer-app `WoWViewer.csproj` may have a hard-coded reference to viewer-app `Rendering/*` files that depends on the `WoWViewer.CrossPlatform.csproj`. | Phase 6 | Phase 6 step 1 audits both csproj files and removes the rendering source references before retiring the namespace. |
| P6 | The headless capture pipeline uses `WowViewer.Core.Renderer.Headless` and `Output/*` in parallel. | Phase 1 | Phase 1 step 1 de-duplicates `Output/*` into `Headless/*` and updates the validation-capture consumer. |
| P7 | The viewer-app `WoWViewer.Rendering.M2Renderer.cs` may reach into `WowViewer.Core.Runtime.M2` via `WowViewerM2RuntimeBridge.cs`. | Phase 3 | The renderer must consume `M2RuntimeFramePipeline` directly, not via the bridge. tasks.md records the consumer shape. |
| P8 | The terrain alpha parity check on `0_5_3_3368` may fail after the multi-tile rewrite (Phase 1) because the single-tile path was a special case. | Phase 1 | Phase 1 step 1 records the Alpha baseline capture and step N re-checks parity. If parity fails, the cutover is blocked until it is restored. |
| P9 | The MDX port may regress existing M2 runtime ownership seams. | Phase 3 | The MDX renderer must consume `WorldMdxRenderPlan.cs` and `M2RuntimeFramePipeline.cs`-equivalent (or analog) and not reimplement. |
| P10 | The Vulkan follow-on spec will need a stable backend-neutral contract surface. | Phase 1 | The `Contracts/*` and `OpenGL/*` split in Phase 1 deliberately creates the seam. Vulkan half is documented as out of scope. |

---

## 5. Open Decisions Deferred to Phases

| Decision | Phase | Decided by |
|---|---|---|
| Whether `WorldTerrainLodSelector` gets a new `UseWdl` bucket or whether the renderer translates `LowDetail` into WDL. | Phase 2 | tasks.md (Phase 2) step 1. |
| Whether MDX is its own phase (3.5) or part of Phase 3. | Phase 3 | tasks.md (Phase 3) step 1; if MDX steps > 10, split. |
| Whether the BLP mip selection lives in the existing `TextureCache` or in a new `MipSelector` helper. | Phase 5 | tasks.md (Phase 5) step 1. |
| Whether `Output/*` is removed or moved into `Headless/*` in Phase 1. | Phase 1 | tasks.md (Phase 1) step 1. |
| Whether `LoadingScreen.cs` is part of the renderer library or stays in the host. | Phase 4 | tasks.md (Phase 4) step 1. |
| Whether `WoWConstants.cs` becomes `WowViewer.Core.Renderer.WoWConstants` or moves to a shared `WowViewer.Core` location. | Phase 1 | tasks.md (Phase 1) step 1. |

---

## 6. Validation Surface Already Available

- `dotnet build wow-viewer/WowViewer.slnx -c Debug`
- `dotnet test wow-viewer/tests/WowViewer.Core.Tests -c Debug`
- `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests -c Debug`
- `dotnet test wow-viewer/tests/WowViewer.Core.Anim.Tests -c Debug`
- (new in Phase 0) `dotnet test wow-viewer/tests/WowViewer.Core.Renderer.Tests -c Debug`
- (new in Phase 7) `dotnet run --project wow-viewer/tools/validation-capture/ -c Debug -- capture --tile 30 48 --build 3_3_5_12340` and the matching `compare-baseline` subcommand.
- (existing) `dotnet run --project wow-viewer/tools/harvest/ -c Debug -- harvest-map-mpq ...` and the other harvest/convert/inspect commands.

These are documented in `quickstart.md`.

---

*End of research. Next: data model + contracts + quickstart, then `speckit-tasks`.*
