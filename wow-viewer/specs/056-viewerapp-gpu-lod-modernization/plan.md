# Implementation Plan: 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization

**Branch**: `056-viewerapp-gpu-lod-modernization` | **Date**: 2026-06-10 | **Spec**: [`specs/056-viewerapp-gpu-lod-modernization/spec.md`](spec.md)

**Input**: Feature specification from `specs/056-viewerapp-gpu-lod-modernization/spec.md`. Locked decisions D1–D7 are in the spec.

**Parents**:
- `docs/architecture/wow-engine-modernization-plan-2026-05-14.md` (Pillar C — backend separation)
- `docs/architecture/spec056-viewerapp-gpu-lod-modernization-analysis-2026-06-10.md` (audit)
- `wow-viewer-library-completeness-plan-2026-05-06.md` Section 2.3 + Phase F (now this spec)
- `game-viewer-host-plan-2026-05-13.md` slices 3-6 (now this spec)
- `specs/036-renderer-improvements` (superseded)

**Supersedes**: `specs/036-renderer-improvements` (archived by this spec).

---

## Summary

`wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` is 621k bytes of partial-class host code with renderer implementation spread across 28+ files in `wow-viewer/src/viewer/WoWViewer/Rendering/*`, in parallel with a thin stub at `wow-viewer/src/core/WowViewer.Core.Renderer/*` that is single-tile, has no LOD, no instancing, and no frustum culling. The runtime LOD/visibility/pass-routing surface at `wow-viewer/src/core/WowViewer.Core.Runtime/World/*` is already stubbed (`WorldTerrainLodSelector`, `WorldObjectVisibilityCollector`, `WorldFramePassCoordinator`).

This plan consolidates that work into 9 dependency-ordered phases that:

1. Promote `WowViewer.Core.Renderer` to a real shared library by carefully porting legacy `MdxViewer`/`WoWViewer.Rendering/*` code in.
2. Make the renderer multi-tile, retained-mode (VBO/IBO/UBO), instanced, frustum-culled, with per-tile/per-frame diagnostics.
3. Add a full LOD matrix: terrain mesh LOD, object LOD, water LOD, light LOD, WDL far horizon, BLP mipmap selection.
4. Move the `WowViewer.App` host off its own renderer fork; the host becomes a thin wiring layer over the shared renderer.
5. Keep `WowViewer.Tool.ValidationCapture` and the headless capture path working through the new shared renderer (FR-011, FR-018).
6. Maintain real-data parity on staged `0_5_3_3368` (Alpha, MCAL alpha-mask risk area) and `3_3_5_12340` (LK) at every phase boundary.

GPU backend: **OpenGL via Silk.NET**, retained-mode. Vulkan primary is a follow-on spec. Compute shaders and async streaming are out of scope.

---

## Technical Context

**Language/Version**: C# / .NET 10. All new code targets `net10.0`.

**Primary Dependencies**:
- `Silk.NET.OpenGL` (existing, `WowViewer.Core.Renderer.csproj`).
- `System.Numerics` (existing).
- Existing: `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`, `WowViewer.Core.PM4`, `WowViewer.Core.Anim`.

**Storage**: N/A. Renderer is a stateless-on-disk transform over in-memory world data.

**Testing**: xUnit. New test project: `wow-viewer/tests/WowViewer.Core.Renderer.Tests` (does not exist yet — Phase 0 creates it). Existing: `WowViewer.Core.Tests`, `WowViewer.Core.PM4.Tests`, `WowViewer.Core.Anim.Tests`.

**Target Platform**: Windows + Linux desktop (x64). macOS best-effort.

**Project Type**: Library (`WowViewer.Core.Renderer`) plus a thin host (`WowViewer.App`) that consumes it.

**Performance Goals**:
- Stable 60 FPS frame pacing for a 3×3 terrain AOI on `3_3_5_12340` outdoor scenes (recorded, then improved by each LOD phase).
- Per-frame draw-call and instance-count diagnostics emitted to a structured per-frame log.
- Reduced texture bandwidth after BLP mipmap selection (measured against pre-spec baseline).

**Constraints**:
- No `H:\CLIENTS` (RULE 9) — staged clients only.
- `gillijimproject_refactor` is read-only (RULE 1).
- Terrain alpha risk area (Alpha MCAL): every cutover step must validate against staged `0_5_3_3368`.
- `specs/020-renderer-culling-and-tile-capture` P1 culling fix is a hard prerequisite and must not regress (FR-018).
- No CUDA-only assumptions; backend seams must stay open (per `wow-viewer` AGENTS.md).
- `AlphaWdtWriter.cs` is frozen (RULE 10) — not touched.

**Scale/Scope**:
- Source: ~28 files in `wow-viewer/src/viewer/WoWViewer/Rendering/*` to retire + ~12 files in `WowViewer.Core.Renderer/*` skeleton to grow + a new `WowViewer.Core.Renderer.OpenGL/*` namespace + new test project.
- Estimated phases: 9 (per the engine plan's max-10 rule). Estimated steps per phase: 5-10. The phase order is itself a hard "one phase at a time" guardrail (RULE 8 + engine plan).

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|---|---|---|
| I. Repo independence | **Pass** | All work inside `wow-viewer/`. No cross-repo references. |
| II. Library-first | **Pass, load-bearing** | This spec exists to move renderer code from app into a shared library. |
| III. Real-data validation | **Pass, load-bearing** | FR-016 mandates staged-client validation at every phase boundary. |
| IV. Residual model chain | N/A | Not an ML feature. |
| V. Streaming-first dataset pipeline | N/A | Not a dataset feature. |
| VI. No game client path assumptions | **Pass** | All clients come from `output/tmp/wowarchive-clients/`. |
| Read-only ref codebase | **Pass** | `gillijimproject_refactor` is read-only reference. |
| Format reader/writer ownership | **Pass** | No `WowViewer.Core.IO` parsers are modified. |
| Terrain alpha risk area | **Pass** | FR-017 mandates Alpha MCAL parity check. |
| `AlphaWdtWriter` is frozen | **Pass** | Not touched. |
| One phase at a time | **Pass, load-bearing** | Plan is 9 phases; each is bounded and independently validatable. |
| Spec docs are source of truth | **Pass** | Spec exists. This plan references it. |
| Bite-sized plans | **Pass** | Max 10 phases; each phase has ≤10 steps (enforced in tasks.md). |
| Library-first renderer ownership | **Pass** | New code goes to `WowViewer.Core.Renderer/*` and `WowViewer.Core.Renderer.OpenGL/*`; viewer app stops owning renderer code. |

**Verdict: constitution-clean. No violations to justify.**

---

## Project Structure

### Documentation (this feature)

```text
specs/056-viewerapp-gpu-lod-modernization/
├── spec.md              (created by speckit-specify, 2026-06-10)
├── plan.md              (this file, created by speckit-plan)
├── research.md          (Phase 0 output, created next)
├── data-model.md        (Phase 1 output, created next)
├── quickstart.md        (Phase 1 output, created next)
├── contracts/           (Phase 1 output, created next)
│   ├── RenderScene.md
│   ├── RenderBackend.md
│   ├── RenderResources.md
│   └── TextureCache.md
└── tasks.md             (Phase 2 output, created by speckit-tasks)
```

### Source Code (repository root)

```text
wow-viewer/
├── src/
│   ├── core/
│   │   ├── WowViewer.Core/
│   │   ├── WowViewer.Core.IO/
│   │   ├── WowViewer.Core.PM4/
│   │   ├── WowViewer.Core.Runtime/
│   │   │   └── World/                                  (consumed, not modified here)
│   │   │       ├── Terrain/WorldTerrainLodSelector.cs   (existing)
│   │   │       ├── Visibility/                         (existing)
│   │   │       └── Passes/                             (existing)
│   │   └── WowViewer.Core.Renderer/                    (GROWS — primary landing zone)
│   │       ├── Scene/                                  (SceneCamera, FrustumCuller, RenderVariant)
│   │       ├── Terrain/                                (TerrainRenderer, TerrainMesh, TerrainShader)
│   │       ├── Wmo/                                    (WmoRenderer, WmoMesh, WmoShader)
│   │       ├── Liquid/                                 (LiquidRenderer, LiquidShader)
│   │       ├── Sky/                                    (SkyRenderer)
│   │       ├── M2/                                     (NEW — M2 instanced renderer)
│   │       ├── Mdx/                                    (NEW — MDX instanced renderer)
│   │       ├── Particle/                               (NEW — particle renderer)
│   │       ├── Minimap/                                (NEW — minimap overlay)
│   │       ├── BoundingBox/                            (NEW)
│   │       ├── Texture/TextureCache.cs                 (extended with mip selection)
│   │       ├── Diagnostics/                            (NEW — per-frame draw-call + instance counter)
│   │       ├── Contracts/                              (NEW — RenderScene, RenderBackend, RenderResources)
│   │       ├── OpenGL/                                 (NEW namespace — backend-agnostic split)
│   │       │   ├── OpenGLRenderBackend.cs
│   │       │   ├── OpenGLBufferFactory.cs
│   │       │   ├── OpenGLShaderCache.cs
│   │       │   └── OpenGLRenderResources.cs
│   │       └── Headless/                               (existing — extended)
│   └── viewer/
│       ├── WoWViewer/                                  (HOST — renderer code retires from here)
│       │   ├── ViewerApp.cs                            (SHRINKS — cutover target)
│       │   ├── Rendering/                              (RETIRED by Phase 6)
│       │   ├── Terrain/                                (host wiring only)
│       │   ├── ...                                     (other host partials — untouched)
│       │   └── WoWViewer.csproj                        (loses Rendering/* refs by Phase 6)
│       └── WowViewer.App/                              (future long-range host; empty for v0.5.0-dev)
├── tools/
│   ├── validation-capture/                             (consumes shared renderer; FR-011)
│   ├── harvest/, converter/, inspect/, animfarm/       (untouched)
├── tests/
│   ├── WowViewer.Core.Renderer.Tests/                  (NEW — created in Phase 0)
│   ├── WowViewer.Core.Tests/                           (extended with renderer integration tests)
│   ├── WowViewer.Core.PM4.Tests/                       (untouched)
│   └── WowViewer.Core.Anim.Tests/                      (untouched)
├── data-harvester/                                     (untouched)
├── docs/
│   └── architecture/
│       └── spec056-viewerapp-gpu-lod-modernization-analysis-2026-06-10.md  (audit, exists)
└── memory-bank/
    ├── activeContext.md                                (updated each phase)
    └── progress.md                                     (updated each phase)
```

**Structure Decision**: Single repo, single solution (`WowViewer.slnx`). New code lives in the existing `WowViewer.Core.Renderer` project. A new test project `WowViewer.Core.Renderer.Tests` is created in Phase 0. The viewer app's `Rendering/*` namespace is retired in Phase 6 (the final host-cutover phase).

**What this structure explicitly does NOT do**:
- It does not create a new `WowViewer.Core.Renderer.Vulkan` project. Vulkan is a follow-on spec.
- It does not create a new `WowViewer.Core.Renderer.Metal` project. Cross-platform is best-effort through Silk.NET.
- It does not extract `WowViewer.App` from `WoWViewer`. The viewer app stays in `wow-viewer/src/viewer/WoWViewer/`; the long-range "App vs Viewer" split is the engine plan's Phase E5, not this spec.
- It does not modify the data harvester, the VLM/PM4 workbench, the audio engine, or the Unreal bridge.

---

## Implementation Phases

**Strict dependency order. Each phase ends with a real-data validation pass on staged `0_5_3_3368` and `3_3_5_12340` and an updated `memory-bank/activeContext.md`. Each phase has ≤10 steps (enforced by `tasks.md`).**

### Phase 0 — Renderer Library Test Foundation + Project Topology Lock

**Goal**: Create the test project, lock the namespace split, document the backend-agnostic contract seams, and produce the validation harness for downstream phases.

**Why first**: Every other phase needs (a) a place to put renderer tests, (b) a locked contract surface so subsequent phases don't have to renegotiate types, and (c) a deterministic capture baseline against which parity is measured.

**Touches**:
- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/` (new xUnit project, references `WowViewer.Core.Renderer`)
- `WowViewer.slnx` (add new test project)
- `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/contracts/RenderScene.md` (new — contract surface)
- `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/contracts/RenderBackend.md` (new)
- `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/contracts/RenderResources.md` (new)
- `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/contracts/TextureCache.md` (new)
- `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/research.md` (Phase 0 research — existing `WorldTerrainLodSelector` and `WorldObjectVisibilityCollector` consumption plan)
- `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/data-model.md` (Phase 1 data model — `RenderScene`, `RenderBackend`, `RenderResources`)
- `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/quickstart.md` (Phase 1 quickstart — validation commands and parity baselines)
- `wow-viewer/memory-bank/activeContext.md` and `memory-bank/progress.md` (initial entry for spec 056)

**Steps (≤10)**: see `tasks.md` (Phase 0).

**Proof / exit criteria**:
- `dotnet build wow-viewer/WowViewer.slnx -c Debug` succeeds.
- `dotnet test wow-viewer/tests/WowViewer.Core.Renderer.Tests` runs an empty test list successfully.
- Contracts are committed and referenced by `spec.md` FR-001 through FR-012.
- A deterministic capture baseline is recorded for `0_5_3_3368` Azeroth and `3_3_5_12340` Azeroth through the *current* shared renderer + legacy viewer-app renderer (whatever the pre-spec renderer is), and the baseline image hashes are stored under `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/`.

**Do not regress**: existing `WowViewer.Core.Tests`, `WowViewer.Core.PM4.Tests`, `WowViewer.Core.Anim.Tests` all still pass.

---

### Phase 1 — Promote `WowViewer.Core.Renderer.Scene` to a Multi-Tile, Retained-Mode Core

**Goal**: Make the existing `WowViewer.Core.Renderer` skeleton genuinely multi-tile, retained-mode (VBO/IBO/UBO), instanced, and frustum-culled. Stop being a single-tile stub.

**Why second**: The renderer library needs a real multi-tile core before any specific asset family (terrain, WMO, M2, MDX, liquid, sky) can be modernized in the new shared namespace. Multi-tile is the precondition for LOD (Phase 3-4) and for the host cutover (Phase 6).

**Touches**:
- `wow-viewer/src/core/WowViewer.Core.Renderer/Scene/SceneRenderer.cs` (rewrite for multi-tile)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Scene/FrustumCuller.cs` (extend with per-tile / per-WMO / per-M2 helpers, per FR-004)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Scene/SceneCamera.cs` (extend with view-projection helpers, fog distances, AOI helpers)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Scene/RenderVariant.cs` (extend with `waterLOD`, `mapObjLightLOD`, `MaxLights`, terrain LOD threshold controls, per FR-006, FR-008, FR-009)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Contracts/` (new — `RenderScene`, `RenderBackend`, `RenderResources` per Phase 0 contracts)
- `wow-viewer/src/core/WowViewer.Core.Renderer/OpenGL/` (new namespace — `OpenGLBufferFactory`, `OpenGLShaderCache`, `OpenGLRenderResources`, FR-002 + FR-003)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Diagnostics/` (new — per-frame draw-call and instance counters, FR-004)
- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Scene/` (new tests)

**Steps (≤10)**: see `tasks.md` (Phase 1).

**Proof / exit criteria**:
- A staged `3_3_5_12340` outdoor map renders a 5×5 AOI of terrain tiles through the new shared renderer with no missing terrain at tile seams (SC-001).
- A 3×3 AOI render uses one material bind per unique material, not 9 (SC-002).
- Per-tile / per-WMO / per-M2 frustum culling works (FR-004, US1 acceptance 3).
- Headless capture path still works for `WowViewer.Tool.ValidationCapture` (FR-011, FR-018).
- Deterministic capture parity with Phase 0 baseline on `0_5_3_3368` and `3_3_5_12340` (FR-016, FR-017).

**Do not regress**:
- Alpha MCAL alpha-mask parity (terrain alpha risk area).
- `specs/020-renderer-culling-and-tile-capture` P1 culling fix (FR-018).
- `WowViewer.Tool.ValidationCapture` produces the same `object_visibility_mask` (SC-008).

---

### Phase 2 — Terrain Renderer in the Shared Library (LOD-Aware, WDL Far Horizon)

**Goal**: Improve and move the terrain renderer (`TerrainRenderer`, `TerrainMesh`, `TerrainMeshBuilder`, `TerrainShader`) from the current `wow-viewer/src/viewer/WoWViewer/Terrain/TerrainRenderer.cs` (the source of truth) into `WowViewer.Core.Renderer.Terrain`. Wire it to `WorldTerrainLodSelector` for the near / mid / WDL-far LOD buckets (FR-005, FR-006, FR-007).

**Why third**: Terrain is the most-visible and most-tested surface; the LOD matrix depends on it landing first. After this phase, all subsequent phases (objects, water, light, mipmaps) can hook into the same terrain-stage pass coordinator.

**Touches**:
- `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/TerrainRenderer.cs` (port, multi-tile, LOD-aware)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/TerrainMesh.cs` (port, add near/mid/far mesh builder hooks)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/TerrainMeshBuilder.cs` (port, extend with mid-res 33×33 / 17×17 buckets)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/TerrainShader.cs` (port + add WDL far-horizon shader path)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Terrain/TerrainConstants.cs` (existing — extend with LOD bucket distances)
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainLodSelector.cs` (existing — verify it has hooks for the 3 buckets; if not, add `FullDetail` / `ReducedMesh` / `UseWdl` bucket helpers in this spec)
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wdl/WorldWdlTileData.cs` (existing — consumer of the WDL data path)
- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Terrain/` (new — terrain LOD tests)

**Steps (≤10)**: see `tasks.md` (Phase 2).

**Proof / exit criteria**:
- Three visibly distinct vertex densities at near / mid / far distances on `3_3_5_12340` (SC-003 acceptance 1+2).
- WDL is the only representation beyond the configured far distance (SC-003 acceptance 3, FR-007).
- LOD transitions are gradual, not popping (SC-003 acceptance 4) — verified by deterministic capture at the threshold.
- Alpha MCAL parity preserved on `0_5_3_3368` (FR-017).
- LK 3.3.5 parity preserved on `3_3_5_12340` (FR-016).

**Do not regress**: existing single-tile `SceneRenderer.RenderTile` API (must remain a valid one-tile path; new API is additive).

---

### Phase 3 — WMO, M2, and MDX Renderers in the Shared Library (Instanced, LOD-Aware, Ghidra-Compliant)

**Goal**: Improve and move the WMO, M2, and MDX renderers from the current viewer-app `wow-viewer/src/viewer/WoWViewer/Rendering/{WmoRenderer.cs,M2Renderer.cs,MdxAnimator.cs}` (the source of truth) into `WowViewer.Core.Renderer.{Wmo,M2,Mdx}`. Make them instanced, frustum-culled, and object-LOD-aware (FR-003, FR-004, FR-008, FR-018). The new WMO renderer must conform to the Ghidra-confirmed 3.3.5 pass-dispatch in `docs/architecture/wmo-render-pass-architecture-2026-05-30.md` (correctness oracle): interior/exterior dispatch by `flags & 0x48`, per-batch MOMT flag testing, lightmap pass split (`RenderGroupLightmapTex_Int` vs `_Ext`), liquid type dispatch (water vs magma), portal-walk visibility, group flag filtering (skip `0x88`, always-render `0x10000`).

**Why fourth**: Object LOD is half the user's stated ask. Objects depend on the multi-tile + frustum-culled core from Phase 1, and they share the per-frame UBO update path that Phase 1 introduced. They can be ported in parallel sub-steps within this phase.

**Touches**:
- `wow-viewer/src/core/WowViewer.Core.Renderer/Wmo/WmoRenderer.cs` (port, instanced for WMO group doodad repeats)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Wmo/WmoMesh.cs` (port)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Wmo/WmoShader.cs` (port)
- `wow-viewer/src/core/WowViewer.Core.Renderer/M2/M2Renderer.cs` (NEW — port from `gillijimproject_refactor/src/MdxViewer/Rendering/M2Renderer.cs`, `WoWViewer/Rendering/M2Renderer.cs`, plus consume `WowViewer.Core.Runtime.M2.M2RuntimeFramePipeline.cs` and `M2SceneSubmissionCoordinator.cs`)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Mdx/MdxRenderer.cs` (NEW — port from `gillijimproject_refactor/src/MdxViewer/Rendering/MdxRenderer.cs`, ~2866 lines; this is the largest single file in the cutover and the most likely source of phase-blowout risk)
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs` (existing — consumer; no changes unless hooks are missing)
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityContext.cs` (existing — same)
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldObjectInstance.cs` (existing — same)
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs` (existing — same)
- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/{Wmo,M2,Mdx}/` (new)

**Steps (≤10)**: see `tasks.md` (Phase 3).

**Proof / exit criteria**:
- A staged `3_3_5_12340` route through a city tile shows visible M2 / WMO instances, with no missing meshes (SC-004 acceptance 1+2).
- Object LOD culls past draw distance (SC-004 acceptance 3).
- A small object occluded by a WMO wall is correctly culled (SC-004 acceptance 4).
- `WowViewer.Tool.ValidationCapture` still produces the same `object_visibility_mask` for `Azeroth_30_48` on `3_3_5_12340` (SC-008, FR-011, FR-018).

**Do not regress**:
- Spec 020 culling fix (FR-018).
- `M2` runtime ownership stays in `WowViewer.Core.Runtime.M2` (the renderer is a *consumer* of `M2RuntimeFramePipeline` / `M2SceneSubmissionCoordinator`; it must not reimplement the M2 frame pipeline).
- M2 parity recovery (out of scope per spec Out-of-Scope, tracked in 037/038).

---

### Phase 4 — Liquid, Sky, Particle, and Bounding-Box Renderers in the Shared Library

**Goal**: Port the liquid, sky, particle, and bounding-box renderers from viewer-app `Rendering/*` into the shared library, with `waterLOD` and `mapObjLightLOD` controls wired to the runtime pass coordinator (FR-009, US4).

**Why fifth**: These are the smaller-asset renderers. They depend on the multi-tile + UBO + frustum-culled core but not on each other. They can be ported in parallel sub-steps within this phase, and they complete the asset-family surface that the host cutover (Phase 6) will retire into.

**Touches**:
- `wow-viewer/src/core/WowViewer.Core.Renderer/Liquid/LiquidRenderer.cs` (existing — port to instanced + `waterLOD` bucketing)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Liquid/LiquidShader.cs` (existing — extend with `waterLOD` reduced shader path)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Sky/SkyRenderer.cs` (existing — port)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Particle/ParticleRenderer.cs` (NEW — port from `wow-viewer/src/viewer/WoWViewer/Rendering/ParticleRenderer.cs`, `ParticleSystem.cs`, ~500+500 lines)
- `wow-viewer/src/core/WowViewer.Core.Renderer/BoundingBox/BoundingBoxRenderer.cs` (NEW — port from `wow-viewer/src/viewer/WoWViewer/Terrain/BoundingBoxRenderer.cs`)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Minimap/MinimapRenderer.cs` (NEW — port from `wow-viewer/src/viewer/WoWViewer/Rendering/MinimapRenderer.cs`, ~400 lines, plus the existing shared `TerrainMinimapCompositor.cs` in `WowViewer.Core.IO`)
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs` (existing — extend with `waterLOD`, `mapObjLightLOD`, `MaxLights` controls)
- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/{Liquid,Sky,Particle,BoundingBox,Minimap}/` (new)

**Steps (≤10)**: see `tasks.md` (Phase 4).

**Proof / exit criteria**:
- `waterLOD` produces a measurable per-frame draw-call drop at far distance (SC-005 acceptance 1).
- `mapObjLightLOD` and `MaxLights` are observable in the per-frame diagnostic stream (SC-005 acceptance 2+3, FR-009).
- Particle and bounding-box renderers work in deterministic capture.
- Minimap overlay works in deterministic capture (regression-checked against the existing `wow-viewer/src/viewer/WoWViewer/MinimapHelpers.cs`).
- Alpha + LK parity preserved.

**Do not regress**:
- `TerrainMinimapCompositor` ownership stays in `WowViewer.Core.IO` (the renderer is a *consumer* of the compositor; the renderer does not redefine compositing).
- Particle/renderer ownership does not leak format-specific logic back into the app.

---

### Phase 5 — TextureCache Mipmap Selection + Per-Frame Diagnostic Surface

**Goal**: Extend the existing `WowViewer.Core.Renderer.Texture.TextureCache` with BLP mip-level selection by sampling distance (FR-010, US5), and finalize the per-frame diagnostic surface (FR-004, SC-005).

**Why sixth**: BLP mipmaps are an explicit part of the user's LOD ask. They are also essentially free once a retained-mode texture cache exists. Doing this after the asset renderers land means the per-tile diagnostic surface already has a place to record per-frame mip-selection counts.

**Touches**:
- `wow-viewer/src/core/WowViewer.Core.Renderer/Texture/TextureCache.cs` (existing — extend with mip-level selection and per-texture residency tracking)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Diagnostics/PerFrameRenderStats.cs` (NEW — write-side helper for the renderer)
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs` (existing — extend to read mip-selection counts)
- `wow-viewer/src/core/WowViewer.Core.Renderer/Contracts/RenderResources.md` (Phase 0 contract — verify mip selection is described; if not, update)
- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Texture/` (new — mip-selection tests)

**Steps (≤10)**: see `tasks.md` (Phase 5).

**Proof / exit criteria**:
- BLP mip selection reduces per-frame texture bandwidth by a measurable amount on `3_3_5_12340` (SC-006, FR-010).
- Per-frame diagnostic stream exposes mip-selection counts, draw-call count, instance count, terrain LOD bucket count, water LOD bucket count, active light count, `mapObjLightLOD` value, `MaxLights` value (FR-004, SC-005, SC-010 acceptance).
- Existing `WowViewer.Core.Tests` integration tests pass with extended diagnostic stream.

**Do not regress**:
- `WowViewer.Core.IO/Wmo/WmoMinimapAssetResolver.cs`, `WowViewer.Core.IO/Blp/AlphaBlpCompatibilityService.cs`, `WowViewer.Core.IO/Blp/BlpSummaryReader.cs` (texture truth stays in IO; the renderer's texture cache is a *consumer* of these surfaces).

---

### Phase 6 — Host Cutover: Retire `WoWViewer/Rendering/*` and Shrink `ViewerApp.cs`

**Goal**: Switch `WowViewer.App` (and `WowViewer.Tool.ValidationCapture`) over to the new shared renderer. Retire the viewer-app `Rendering/*` namespace. `ViewerApp.cs` becomes substantially smaller (FR-013, FR-014, FR-015, US6).

**Why seventh**: This is the user's stated sub-goal #1. It is also the highest-risk phase, because the host wiring touches ImGui, the dockspace shell (spec 044), and the existing capture pipeline. It must be the last renderer-content phase so the cutover has a real renderer to point at.

**Touches**:
- `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` (cutover target — host wiring only; renderer calls go through shared library)
- `wow-viewer/src/viewer/WoWViewer/ViewerApp_RenderQuality.cs` (host wiring only — wires to shared `RenderVariant`)
- `wow-viewer/src/viewer/WoWViewer/Rendering/*` (retire by deletion; FR-013)
- `wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj` (drop renderer source references; reference `WowViewer.Core.Renderer` instead)
- `wow-viewer/src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs` (existing — verify it now consumes the shared renderer's terrain stage; if not, refactor the adapter to be a host-side wiring only)
- `wow-viewer/src/viewer/WoWViewer/Terrain/VlmTerrainManager.cs` (existing — same)
- `wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj` (same as `WoWViewer.csproj`)
- `wow-viewer/tools/validation-capture/` (consume shared renderer; FR-011)
- `WowViewer.slnx` (no change)

**Steps (≤10)**: see `tasks.md` (Phase 6).

**Proof / exit criteria**:
- `wow-viewer/src/viewer/WoWViewer/Rendering/*` is empty or absent (SC-010, FR-013).
- `WowViewer.App` no longer compiles any renderer implementation code outside the shared library (FR-014, SC-010).
- `ViewerApp.cs` is "substantially smaller" per the user's D5 (no numeric target; code review by maintainer) (SC-007).
- The viewer app, the validation-capture tool, and the headless renderer all invoke the same shared renderer entry point (FR-012 acceptance).
- A staged `3_3_5_12340` map renders identically through the cutover (FR-016, FR-017, FR-018).
- `WowViewer.Tool.ValidationCapture` continues to produce the same `object_visibility_mask` for `Azeroth_30_48` (SC-008).

**Do not regress**:
- Spec 044 dockspace shell, spec 045 scene graph workbench, spec 049 viewer UI consolidation. They are viewer-app UX; this phase must not silently break their host surfaces.

---

### Phase 7 — Real-Data Validation Suite (Cross-Phase Parity)

**Goal**: Land a permanent validation surface that exercises the new shared renderer on staged `0_5_3_3368` and `3_3_5_12340`, producing deterministic captures and structured reports. This is the harness that the engine plan and AGENTS.md require for any change touching terrain, liquid, WMO, or M2 rendering (FR-016, FR-017, US7).

**Why eighth**: Every previous phase runs its own ad-hoc parity check. Phase 7 turns those ad-hoc checks into a single reusable CI / on-demand harness, so future changes can be validated without rebuilding the parity surface.

**Touches**:
- `wow-viewer/tools/validation-capture/` (extend with a `compare-baseline` subcommand that runs the new shared renderer against the baseline captures from Phase 0)
- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Validation/` (new — runs the headless renderer against the baseline captures and asserts pixel/image-hash tolerance)
- `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/quickstart.md` (existing — extend with the new validation commands)
- `wow-viewer/memory-bank/progress.md` (record the validation harness)

**Steps (≤10)**: see `tasks.md` (Phase 7).

**Proof / exit criteria**:
- `dotnet test wow-viewer/tests/WowViewer.Core.Renderer.Tests` runs the parity suite on `0_5_3_3368` Azeroth and `3_3_5_12340` Azeroth and passes.
- The harness exits non-zero on any visual regression above documented tolerance.
- The harness exposes per-frame diagnostic counters (draw calls, instances, mip selection, terrain LOD bucket, water LOD, light count) (FR-004, SC-005, SC-010).

**Do not regress**:
- Existing `WowViewer.Tool.ValidationCapture` capture output schema.
- Existing baseline captures in `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/`.

---

### Phase 8 — Spec 036 Archive, Memory-Bank Sync, Final Spec Audit

**Goal**: Archive `specs/036-renderer-improvements` with a forward pointer to this spec, sync the memory bank, and run a final spec-vs-code audit.

**Why ninth and last**: This is the bookkeeping phase. It guarantees that the "this spec is the one owner plan" decision in D6 is honored, and that future sessions do not route renderer work back into 036.

**Touches**:
- `wow-viewer/specs/036-renderer-improvements/` (move to `wow-viewer/specs/archived/`, add a "Superseded by 056" banner)
- `wow-viewer/specs/036-renderer-improvements/ARCHIVED.md` (NEW — forward pointer)
- `wow-viewer/memory-bank/activeContext.md` (final update — `viewerapp-gpu-lod-modernization` is the active renderer lane)
- `wow-viewer/memory-bank/progress.md` (final update — log the spec landing and the cutover completion)
- `wow-viewer/docs/architecture/speckit-doc-audit-2026-05-18.md` (update audit table to reflect that 036 is archived and 056 is the active owner)
- `wow-viewer/docs/architecture/wow-viewer-library-completeness-plan-2026-05-06.md` (mark Section 2.3 + Phase F as completed by 056)
- `wow-viewer/docs/architecture/game-viewer-host-plan-2026-05-13.md` (mark slices 3-6 as covered by 056)
- `wow-viewer/docs/architecture/wow-viewer-full-porting-roadmap.md` (mark Phase I Priority 5 as in-progress under 056)

**Steps (≤10)**: see `tasks.md` (Phase 8).

**Proof / exit criteria**:
- `specs/036-renderer-improvements` is under `wow-viewer/specs/archived/`.
- `specs/archived/ARCHIVED.md` lists 036 with a forward pointer to 056.
- `memory-bank/activeContext.md` and `memory-bank/progress.md` are compressed and current (≤ 200 lines each, per the memory-bank rule).
- `dotnet build wow-viewer/WowViewer.slnx -c Debug` and `dotnet test` both pass.
- A new `docs/architecture/speckit-doc-audit-2026-06-XX.md` exists and reflects 056.

**Do not regress**:
- Existing archived specs in `specs/archived/`.
- Existing audit-trail files in `docs/architecture/`.

---

## Cross-Phase Discipline

These rules apply to every phase, not just one:

1. **One phase at a time.** Phases run in numerical order. A phase is not "done" until its proof / exit criteria pass and `memory-bank/activeContext.md` is updated. RULE 8 + engine plan.
2. **Bite-sized steps.** Max 10 steps per phase, enforced by `tasks.md`. doc-hygiene rule.
3. **Real-data validation.** Every phase ends with a staged-client run on `0_5_3_3368` and `3_3_5_12340`. Constitution Principle III.
4. **Library-first.** New code goes into `WowViewer.Core.Renderer/*`. The viewer app is a *consumer* of the library. Constitution Principle II.
5. **No `H:\CLIENTS`.** Staged clients only. RULE 9.
6. **Terrain alpha risk area.** Every phase that touches terrain validates against `0_5_3_3368` MCAL parity.
7. **Spec 020 must not regress.** The P1 culling fix is a hard prerequisite for capture correctness.
8. **Format readers are frozen.** No `WowViewer.Core.IO` parser is modified by this spec. RULE 3.
9. **No CUDA-only assumptions.** Backend seams stay open. AGENTS.md dataset-builder guardrail.
10. **No code-style drift.** The shared renderer is one style, not a hybrid of `MdxViewer` style + `wow-viewer` style. doc-hygiene rule.
11. **Memory bank discipline.** `activeContext.md` and `progress.md` are updated at the end of each phase, compressed aggressively (≤ 200 lines each). RULE 11 / memory-bank rule.
12. **Single-commit-per-step.** Each bite-sized step lands as its own focused commit with a "why" message. AGENTS.md.

---

## Validation Language Rule (Per AGENTS.md + Engine Plan)

- **Library compile + tests in `wow-viewer` are primary proof.**
- **Real-data captures on `0_5_3_3368` and `3_3_5_12340` are required** for any change touching terrain, liquid, WMO, or M2 rendering.
- **Legacy `MdxViewer` evidence is compatibility evidence, not ownership evidence.**
- **Do not claim "modern replacement engine"** until the new shared renderer can load and render real worlds through its own runtime/backend stack without legacy ownership seams.
- The full validation surface lives in `quickstart.md` (Phase 1) and is operationalized in Phase 7.

---

## Complexity Tracking

> *Fill ONLY if Constitution Check has violations that must be justified.*

No constitution violations. Section intentionally empty per template guidance.

---

## Phase Dependency Graph (Compact)

```
Phase 0 (test foundation, contracts, baselines)
   |
   v
Phase 1 (multi-tile, retained-mode, instanced, frustum-culled core)
   |
   +----------+----------+
   |          |          |
   v          v          v
Phase 2   Phase 3     Phase 4
(terrain) (WMO/M2/Mdx)(liquid/sky/particle/bbox/minimap)
   |          |          |
   +-----+----+----------+
         |
         v
       Phase 5 (mipmap + diagnostics)
         |
         v
       Phase 6 (host cutover, retire Rendering/*)
         |
         v
       Phase 7 (permanent validation harness)
         |
         v
       Phase 8 (archive 036, sync memory bank, audit)
```

Phases 2, 3, 4 can be done in any order after Phase 1, but they are listed in priority order (terrain first because the user said terrain is most visible). Phases 5, 6, 7, 8 are strictly sequential.

---

*End of plan. Next: create Phase 0 research, Phase 1 data model, Phase 1 quickstart, and Phase 1 contracts (research.md, data-model.md, quickstart.md, contracts/*). Then run `speckit-tasks` to break each phase into ≤10 steps each.*
