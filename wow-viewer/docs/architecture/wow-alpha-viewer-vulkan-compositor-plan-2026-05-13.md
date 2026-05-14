# WoWAlphaViewer Vulkan Compositor Plan (Deep-Dive Audit)

## Status

- status: active
- date: 2026-05-13
- scope: renderer architecture reset
- baseline strategy: `0.5.3` behavior parity first, modern renderer architecture first-class, compatibility up through `4.0.0`

---

## 1) Audit: Why legacy `MdxViewer` became CPU-bound and brittle

### 1.1 Monolithic world orchestration owns too many responsibilities

Legacy world orchestration in [`WorldScene.cs`](gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs) is a very large, mixed-responsibility surface (visibility, culling policy, pass scheduling, rendering calls, debug overlays, PM4 diagnostics, hover picking, capture concerns).

Observed consequence:
- high coupling between runtime logic and rendering/backend state,
- difficult optimization because policy and draw submission are interwoven,
- costly maintenance and regression risk.

### 1.2 OpenGL submission pattern is still largely CPU-driven

In the legacy path, frame passes are scheduled on CPU and then perform many per-instance decisions and calls (see [`WorldScene.Render()`](gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs:7746)).

Even after useful pass extraction, the execution style remains:
- CPU visibility buckets
- CPU sorting
- frequent state changes and per-object draw routes

The old render queue concept in [`RenderQueue`](gillijimproject_refactor/src/MdxViewer/Rendering/RenderQueue.cs:12) is correct at a high level (opaque/transparent ordering), but still CPU-centric and not GPU-driven scheduling.

### 1.3 Terrain path improved draw-call count, but still has CPU-heavy prep/upload behavior

Tile batching in [`TerrainTileMeshBuilder`](gillijimproject_refactor/src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs:12) reduces draw calls by merging chunks per tile, but each tile still performs substantial CPU assembly and upload work.

Terrain management in [`TerrainManager`](gillijimproject_refactor/src/MdxViewer/Terrain/TerrainManager.cs:17) added practical safeguards (AOI, upload budgets, concurrency limits), yet remains primarily a CPU stream manager with GPU used as a downstream consumer rather than a primary planner.

### 1.4 Asset path resolution and deferred loading are practical but expensive at scale

[`WorldAssetManager`](gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs:69) carries large caching and deferred loading logic. It solved real stability issues, but still spends significant CPU budget in path probing, cache checks, and load orchestration during active world sessions.

### 1.5 WMO/MDX renderers are feature-rich but backend-specific and tightly coupled

Both [`WmoRenderer`](gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs:35) and [`MdxRenderer`](gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs:81) embed large amounts of backend behavior and model policy in one place. This makes parity fixes possible, but scale-out modernization (Vulkan + tiny renderers) expensive without architectural reset.

---

## 2) Audit: Current `wow-viewer` runtime/library readiness and gaps

### 2.1 Strong extraction already exists on the runtime-policy side

`wow-viewer` already has useful extracted seams:
- pass sequencing: [`WorldFramePassCoordinator`](wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs:55)
- visibility/culling policy: [`WorldObjectVisibilityCollector`](wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs:6)
- composition summaries: [`WorldRenderCompositionBuilder`](wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderCompositionBuilder.cs:8)

### 2.2 Critical current gap: runtime has no render-backend abstraction

[`WowViewer.Core.Runtime.csproj`](wow-viewer/src/core/WowViewer.Core.Runtime/WowViewer.Core.Runtime.csproj:1) intentionally has no graphics backend dependency, which is good, but there is currently no explicit backend interface package (Vulkan/OpenGL/Headless) and no composited GPU execution graph.

### 2.3 Current composition notes confirm renderer incompleteness

[`WorldRenderCompositionBuilder`](wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderCompositionBuilder.cs:82) still documents WMO/MDX geometry as marker-represented in the bounded preview context. This is explicit proof that the runtime seam is ahead of the real renderer implementation.

### 2.4 App surface is reset baseline (correct for restart)

The app reset created a clean baseline in [`Program.Main()`](wow-viewer/src/viewer/WowViewer.App/Program.cs:7) and Layer 0 registry contracts in [`WoWAlphaViewerLayerRegistry`](wow-viewer/src/viewer/WowViewer.App/WoWAlphaViewerLayerRegistry.cs:3). This is now the right launch point for a non-monolithic renderer rebuild.

---

## 3) Target Architecture: Vulkan-first compositor with tiny specialized renderers

## 3.1 Core principle

`WoWAlphaViewer` renderer is a **compositor** that consumes runtime frame data and dispatches many small renderer modules, not one mega-renderer.

### 3.2 Layered rendering stack

1. **Runtime policy layer** (`WowViewer.Core.Runtime`)
   - visibility sets
   - pass routes
   - per-layer source sets
   - LOD decisions

2. **Render graph layer** (`WowViewer.Core.Runtime.Rendering` new)
   - backend-agnostic frame graph nodes/resources
   - pass prerequisites
   - transient resource lifetime

3. **Backend layer** (`wow-viewer/src/viewer/WowViewer.App/Rendering/Vulkan` new)
   - Vulkan device/swapchain/resource allocators
   - command recording
   - descriptor management

4. **Tiny renderer modules** (backend consumers)
   - `SkyRenderer`
   - `SkyboxRenderer`
   - `WdlFarTerrainRenderer`
   - `TerrainRenderer`
   - `LiquidRenderer`
   - `WmoOpaqueRenderer`
   - `MdxOpaqueRenderer`
   - `MdxTransparentRenderer`
   - `OverlayRenderer`

5. **Compositor layer**
   - pass ordering
   - inter-pass resource transitions
   - composite and post chain

### 3.3 CPU/GPU responsibility split (non-negotiable)

CPU owns:
- data decode, streaming policy, visibility and LOD decisions,
- per-frame scene graph assembly,
- upload scheduling and resource residency policy.

GPU owns:
- material evaluation,
- terrain/liquid shading,
- object rasterization and blending,
- depth/overdraw rejection,
- as much per-instance expansion and draw dispatch as practical (targeting indirect workflows over time).

### 3.4 Vulkan-first backend policy

- Primary backend: Vulkan (all supported production targets assumed Vulkan-capable).
- OpenGL fallback architecture.
- Optional debug backend can exist later, but architecture must not be shaped around legacy GL constraints.

---

## 4) Re-implementation inventory (library first, then UI)

## 4.1 Library/runtime re-implementation backlog

### A. Runtime render contracts (must land first)
- new backend-agnostic render contracts in runtime:
  - frame graph packet model
  - pass input/output contracts
  - material/texture indirection contracts
  - terrain/layer sampling contracts

### B. Visibility + LOD parity closure
- complete `0.5.3`-aligned visibility tuning from extracted collector policies,
- close projected-size + cone + fog range policy with explicit profile tables.

### C. Terrain/liquid rendering data contracts
- tile/chunk render packet format
- alpha/shadow/MCCV channel contracts
- liquid family/type-driven render packet contracts

### D. Skybox and lighting contracts
- sky source selection and zone-driven lighting packet contracts,
- explicit bridge for later-era data differences.

### E. Object renderer packet contracts
- WMO packet layout (opaque/trans routes)
- MDX/M2 packet layout (opaque/trans routes, animation state bridge)

## 4.2 App/UI re-implementation backlog

### A. Minimal host shell (post-library)
- renderer diagnostics panel
- frame timings and pass stats
- backend state and resource residency views

### B. World session UI
- staged client attach
- map/tile session control
- pass toggles and LOD profile controls

### C. Converter UX integration
- guided flows for conversion commands
- provenance + validation summary in UI
- actionable error surfacing (no hidden failures)

---

## 5) Phased implementation plan (strict gates)

### Phase R0 — Contracts and graph scaffolding (current next slice)

Deliver:
- `World` runtime render contracts package (`Core.Runtime.Rendering` namespace)
- backend-independent pass graph primitives

Validation gate:
- build + focused contract tests
- deterministic `layer0-status` + `renderer-contract-status` CLI diagnostics

### Phase R1 — Vulkan device + frame lifecycle baseline

Deliver:
- Vulkan instance/device/swapchain management
- command buffer lifecycle
- synchronization skeleton

Validation gate:
- headless or windowed clear-frame proof + frame pacing metrics

### Phase R2 — Terrain + liquid shader baseline

Deliver:
- terrain shader module
- liquid shader module
- terrain/liquid pass nodes in compositor graph

Validation gate:
- staged `0.5.3` map capture + pass stats

### Phase R3 — Visibility/LOD policy closure

Deliver:
- integrate runtime visibility output directly into draw packet generation
- tune to `0.5.3` reference behavior

Validation gate:
- fixed camera path regression with submitted-instance deltas and CPU frame-time reductions

### Phase R4 — WMO + MDX tiny renderers

Deliver:
- separate WMO opaque renderer
- separate MDX opaque renderer
- separate MDX transparent renderer

Validation gate:
- per-pass counters + image checks on staged scenes

### Phase R5 — Skybox + lighting parity baseline

Deliver:
- real skybox path
- world lighting model path

Validation gate:
- reproducible captures with controlled time-of-day and fog settings

### Phase R6 — UI shell and tool UX integration

Deliver:
- world session UI
- diagnostics UI
- conversion GUI workflows

Validation gate:
- operator-run acceptance checklist without command-line reconstruction

---

## 6) Parity and proof governance

Tracking file:
- [`wow-alpha-viewer-parity-matrix-template-2026-05-13.md`](wow-viewer/docs/architecture/wow-alpha-viewer-parity-matrix-template-2026-05-13.md)

Rules:
1. Every feature port row links to one renderer/runtime layer.
2. Every row includes objective proof artifact.
3. No phase closure without matrix row closure for that phase.

---

## 6.1 Safety Gate: Do not regress AlphaWDT liquid semantics while rebuilding renderer

Renderer modernization must not mutate conversion/file-writing semantics.

Hard rule:
- No renderer phase may change AlphaWDT liquid flag generation or liquid type classification in converter/writer paths.

Protected surfaces:
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs`

Required gate before any merge that touches liquid-adjacent code:
1. run focused round-trip tests for LK↔Alpha,
2. run staged real-data conversion smoke on a known map,
3. confirm no change to current working liquid behavior contract documented in memory-bank continuity.

If any liquid regression appears, renderer work pauses until converter parity is restored.

---

## 7) Immediate next executable slice

Start `Phase R0` with new runtime render contracts and graph scaffolding in `wow-viewer` only:

- add runtime namespace: `WowViewer.Core.Runtime.Rendering`
- add core contracts:
  - `WorldRenderGraphFrame`
  - `WorldRenderGraphPass`
  - `WorldRenderPacket` variants (terrain/liquid/wmo/mdx/sky/overlay)
  - backend-neutral `IRenderBackend`
- wire `WoWAlphaViewer` CLI status command to print contract readiness.

This keeps the rebuild disciplined: library first, then backend, then UI.
