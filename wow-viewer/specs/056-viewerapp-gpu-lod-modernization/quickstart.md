# Quickstart: 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization

**Phase 1 quickstart. Companion to `plan.md`, `research.md`, `data-model.md`.**

This file is the **on-ramp** for any developer or operator working on spec 056. It lists the build/test/validate commands, the staged-client paths, the per-phase parity baselines, and the diagnostic surface the renderer exposes.

---

## 1. Prerequisites

- .NET 10 SDK.
- A staged `0_5_3_3368` client at `output/tmp/wowarchive-clients/0_5_3_3368/` (Alpha; for the terrain alpha risk area).
- A staged `3_3_5_12340` client at `output/tmp/wowarchive-clients/3_3_5_12340/` (LK; for the standard outdoor / city / WMO validation).
- (Optional) A staged `0_5_5_3494` client at `output/tmp/wowarchive-clients/0_5_5_3494/` (Alpha variant; used in some phases).
- (Optional) A staged `4_0_0_11927` client at `output/tmp/wowarchive-clients/4_0_0_11927/` (Cata; used in some phases for split-ADT parity).

`H:\CLIENTS` is **forbidden** (RULE 9).

## 2. Build

```powershell
# Full solution (existing command; the spec adds no new top-level projects)
dotnet build wow-viewer/WowViewer.slnx -c Debug

# Just the renderer library (faster, for renderer-internal work)
dotnet build wow-viewer/src/core/WowViewer.Core.Renderer/WowViewer.Core.Renderer.csproj -c Debug

# Just the new test project (Phase 0+)
dotnet build wow-viewer/tests/WowViewer.Core.Renderer.Tests/WowViewer.Core.Renderer.Tests.csproj -c Debug
```

## 3. Test

```powershell
# All existing test projects (must still pass)
dotnet test wow-viewer/tests/WowViewer.Core.Tests -c Debug
dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests -c Debug
dotnet test wow-viewer/tests/WowViewer.Core.Anim.Tests -c Debug

# New test project (Phase 0+)
dotnet test wow-viewer/tests/WowViewer.Core.Renderer.Tests -c Debug

# Renderer integration tests (run against a real GL context)
dotnet test wow-viewer/tests/WowViewer.Core.Renderer.Tests --filter "Category=Integration" -c Debug
```

## 4. Headless Validation Capture (per Phase 1+)

```powershell
# Single-tile capture on LK 3.3.5 Azeroth
dotnet run --project wow-viewer/tools/validation-capture/ -c Debug -- capture --tile 30 48 --build 3_3_5_12340

# Multi-tile capture on LK 3.3.5 Azeroth (Phase 1+)
dotnet run --project wow-viewer/tools/validation-capture/ -c Debug -- capture --aoi 30 47 35 52 --build 3_3_5_12340

# Single-tile capture on Alpha 0.5.3 Azeroth (terrain alpha risk area)
dotnet run --project wow-viewer/tools/validation-capture/ -c Debug -- capture --tile 30 48 --build 0_5_3_3368

# Compare capture against the baseline (Phase 7)
dotnet run --project wow-viewer/tools/validation-capture/ -c Debug -- compare-baseline --build 3_3_5_12340 --tile 30 48
```

The capture output goes to `wow-viewer/output/validation-capture/<build>/<tileX>_<tileY>.png` plus a JSON sidecar with the per-frame diagnostic stats.

## 5. Per-Phase Parity Baselines (Recorded in Phase 0)

`wow-viewer/specs/056-viewerapp-gpu-lod-modernization/baselines/` contains the pre-spec capture hashes:

```
baselines/
├── pre-spec/
│   ├── 0_5_3_3368/
│   │   └── Azeroth_30_48.png            (Alpha MCAL parity; terrain alpha risk area)
│   ├── 3_3_5_12340/
│   │   ├── Azeroth_30_48.png            (standard LK tile)
│   │   ├── Azeroth_30_48.json           (per-frame stats sidecar)
│   │   ├── Stormwind_32_48.png          (city / WMO test; Phase 3+)
│   │   └── Stormwind_32_48.json
│   └── ...
└── README.md
```

Each phase records its own post-step capture under `baselines/phase-N/` and compares against `baselines/pre-spec/` via `compare-baseline` (Phase 7).

## 6. Diagnostic Surface

The renderer emits a per-frame diagnostic record. The host can read it via:

```csharp
var stats = renderer.Stats;
// stats.DrawCallCount, stats.InstanceCount, stats.TextureBindCount, etc.
```

The same data is also written to the JSON sidecar of each headless capture, so headless runs can be diffed across phases.

The diagnostic fields are:

```text
DrawCallCount
InstanceCount
StateChangeCount
TextureBindCount
ShaderSwitchCount
TerrainTileCount
TerrainTilesByLodBucket[4]      // Full, Reduced, WdlOnly, Culled
WorldObjectCount
WorldObjectsByLodLevel[3]        // Near, Far, Culled
LiquidTileCount
LiquidTilesByLodBucket[3]        // Full, Reduced, Culled
ActiveLightCount
MipSelectedDistribution[N]       // per mip level
TextureBandwidthBytes            // estimated
FrameCpuTimeMs
FrameGpuTimeMs                   // if available
Backend                          // OpenGL, Vulkan (future)
```

The full data model is in `data-model.md`.

## 7. Per-Phase Validation Checklist

Every phase ends with this checklist:

1. `dotnet build wow-viewer/WowViewer.slnx -c Debug` passes.
2. `dotnet test wow-viewer/tests/WowViewer.Core.Tests -c Debug` passes.
3. `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests -c Debug` passes.
4. `dotnet test wow-viewer/tests/WowViewer.Core.Anim.Tests -c Debug` passes.
5. `dotnet test wow-viewer/tests/WowViewer.Core.Renderer.Tests -c Debug` passes (Phase 0+).
6. `dotnet run --project wow-viewer/tools/validation-capture/ -c Debug -- capture --tile 30 48 --build 3_3_5_12340` produces a capture visually equivalent to the `pre-spec/3_3_5_12340/Azeroth_30_48.png` baseline (Phase 7+).
7. `dotnet run --project wow-viewer/tools/validation-capture/ -c Debug -- capture --tile 30 48 --build 0_5_3_3368` produces a capture visually equivalent to the `pre-spec/0_5_3_3368/Azeroth_30_48.png` baseline (Alpha MCAL parity).
8. `memory-bank/activeContext.md` and `memory-bank/progress.md` are updated and ≤ 200 lines.
9. The phase's exit criteria from `plan.md` are met.
10. `git status` shows only intended files changed.

## 8. Per-Phase Commit Hygiene

Each bite-sized step lands as its own focused commit. The commit message follows the "why not what" rule (AGENTS.md). Example:

```text
Phase 1 step 4: introduce IRenderBackend interface in Contracts/

The WowViewer.Core.Renderer.Scene.SceneRenderer was a single-tile stub with no
backend-neutral contract. This step introduces IRenderBackend as the seam for
the future OpenGL/Vulkan split and the seam for the host cutover in Phase 6.
No behavior change; this is a pure addition.
```

## 9. Cross-Phase Discipline

- **One phase at a time.** Phases run in numerical order (0-8). A phase is not "done" until its exit criteria pass.
- **Bite-sized steps.** Max 10 steps per phase, enforced by `tasks.md`.
- **Real-data validation.** Every phase ends with a staged-client run on `0_5_3_3368` and `3_3_5_12340`.
- **Library-first.** New code goes into `WowViewer.Core.Renderer/*`. The viewer app is a *consumer* of the library.
- **No `H:\CLIENTS`.** Staged clients only.
- **Terrain alpha risk area.** Every phase that touches terrain validates against `0_5_3_3368` MCAL parity.
- **Spec 020 must not regress.** The P1 culling fix is a hard prerequisite for capture correctness.
- **Format readers are frozen.** No `WowViewer.Core.IO` parser is modified by this spec.
- **No CUDA-only assumptions.** Backend seams stay open.

## 10. Where to Read More

- Spec: `specs/056-viewerapp-gpu-lod-modernization/spec.md`
- Plan: `specs/056-viewerapp-gpu-lod-modernization/plan.md`
- Research: `specs/056-viewerapp-gpu-lod-modernization/research.md`
- Data model: `specs/056-viewerapp-gpu-lod-modernization/data-model.md`
- Contracts: `specs/056-viewerapp-gpu-lod-modernization/contracts/`
- Tasks: `specs/056-viewerapp-gpu-lod-modernization/tasks.md` (Phase 2 output, generated by `speckit-tasks`)
- Audit: `docs/architecture/spec056-viewerapp-gpu-lod-modernization-analysis-2026-06-10.md`
- Program direction: `docs/architecture/wow-engine-modernization-plan-2026-05-14.md` (viewer-first + UE bridge)
- Constitution: `wow-viewer/.specify/memory/constitution.md`
- Workspace guardrails: `AGENTS.md`
- Ghidra-correctness-oracle (WMO dispatch, 3.3.5): `docs/architecture/wmo-render-pass-architecture-2026-05-30.md`

## 11. Source-of-truth map (binding)

| Role | Path | Notes |
|---|---|---|
| **Source of truth (current behavior)** | `wow-viewer/src/viewer/WoWViewer/Rendering/*` and `wow-viewer/src/viewer/WoWViewer/Terrain/*` | The new shared library is built by *improving and moving* this code. |
| **Forbidden** | `wow-viewer/src/viewer/WowViewer.App.Defunct/*` | Do not read, do not port, do not reference. User instruction 2026-06-10. |
| **Read-only reference** | `gillijimproject_refactor/src/MdxViewer/Rendering/*` | RULE 1. Not the source of truth. May be consulted for cross-checks at most. |
| **Correctness oracle** | `docs/architecture/wmo-render-pass-architecture-2026-05-30.md` | Ghidra-confirmed 3.3.5 WMO pass dispatch. The new WMO renderer conforms to it because that is the only correct 3.3.5 behavior. Not a code source. |

## 11. Quick Reference: Where the New Code Lands

```text
WowViewer.Core.Renderer/                                  (the renderer library)
├── Contracts/                                            (Phase 1)
│   ├── IRenderBackend.cs                                 (RenderBackend.md)
│   ├── RenderScene.cs                                    (RenderScene.md)
│   ├── IRenderResources.cs                               (RenderResources.md)
│   ├── PerFrameRenderStats.cs
│   ├── RendererError.cs
│   └── RendererLifecycleState.cs
├── OpenGL/                                               (Phase 1+)
│   ├── OpenGLRenderBackend.cs
│   ├── OpenGLBufferFactory.cs
│   ├── OpenGLShaderCache.cs
│   └── OpenGLRenderResources.cs
├── Scene/                                                (Phase 1)
│   ├── SceneRenderer.cs                                  (rewritten multi-tile)
│   ├── FrustumCuller.cs                                  (extended)
│   ├── SceneCamera.cs                                    (extended)
│   └── RenderVariant.cs                                  (extended with LOD controls)
├── Terrain/                                              (Phase 2)
│   ├── TerrainRenderer.cs                                (rewritten multi-tile, LOD-aware)
│   ├── TerrainMeshBuilder.cs                             (extended with mid-res buckets)
│   ├── TerrainShader.cs                                  (extended with WDL far-horizon)
│   ├── RenderTerrainTile.cs
│   └── TerrainLodSettings.cs
├── Wmo/                                                  (Phase 3)
├── M2/                                                   (Phase 3)
├── Mdx/                                                  (Phase 3)
├── Liquid/                                               (Phase 4)
├── Sky/                                                  (Phase 4)
├── Particle/                                             (Phase 4)
├── BoundingBox/                                          (Phase 4)
├── Minimap/                                              (Phase 4)
├── Texture/TextureCache.cs                               (Phase 5 extended)
├── Diagnostics/                                          (Phase 1, 5)
└── Headless/                                             (Phase 1 de-duped)

WowViewer.Core.Runtime/                                   (consumer, not modified by this spec except for explicit extensions)
└── World/
    ├── Terrain/WorldTerrainLodSelector.cs                (existing; maybe extended with WDL bucket in Phase 2)
    ├── Visibility/                                       (existing; consumed)
    ├── Passes/WorldFramePassCoordinator.cs                (existing; extended with waterLOD/lightLOD in Phase 1)
    └── ...

wow-viewer/src/viewer/WoWViewer/Rendering/*                (RETIRED in Phase 6)

wow-viewer/tests/WowViewer.Core.Renderer.Tests/           (NEW in Phase 0)
└── Contracts/, Scene/, Terrain/, Wmo/, M2/, Mdx/, Liquid/, Sky/, Particle/, BoundingBox/, Minimap/, Texture/, Diagnostics/, Validation/

wow-viewer/tools/validation-capture/                      (consumes shared renderer; FR-011)
```
