# Spec 136: M2 Doodad Rendering Performance Optimization

## Overview

On maps with high density of placed M2 doodads (trees, rocks, fences, grass, clutter), rendering framerates drop from 60+ FPS down to <1 FPS when doodad rendering is enabled. Toggling doodad rendering off (`DoodadsVisible = false`) immediately restores performance to 30-3000 FPS.

Audit of the rendering pipeline reveals three related bottlenecks:
1. **Forced Unbatched Rendering**: the legacy M2 adapter path historically forced every M2 instance onto the unbatched `RenderWithTransform` fallback path, causing full OpenGL shader program rebinding (`glUseProgram`), 8–10 GL uniform uploads, and state changes per object instance per frame.
2. **WMO Doodad Submission Churn**: `WmoRenderer` historically submitted every opaque doodad placement independently, repeating batch setup for each visible object even when placements shared the same `IModelRenderer`.
3. **Redundant Animation Updates**: `WorldScene` iterates over all $N$ visible placement instances every frame and calls `renderer.UpdateAnimation()`. When dozens or hundreds of placed instances share the same model (e.g. 500 instances of `tree.m2`), `UpdateAnimation()` is executed 500 times per frame on the exact same renderer object.
4. **Placement-Multiplied Client I/O**: deferred WMO doodad model loads were advanced from every
   `WmoRenderer.RenderWithTransform` call. A WMO placed many times could therefore perform one
   synchronous model read per placement in one frame, multiplying the intended load budget. The
   minimap loader also used four background readers against the shared client data source.

## User Stories

### User Story 1 - Enable Batched Instancing for M2 Doodads (Priority: P1)

As a viewer user, I want M2 doodad models without active particle or ribbon emitters to use the high-throughput `BeginBatch()` / `RenderInstance()` rendering path, so that thousands of static doodads can be rendered smoothly without per-instance shader rebinding or redundant GL state calls.

**Acceptance Criteria**:
1. `ModelRenderer.RequiresUnbatchedWorldRender` returns `false` for M2 models that do not have active particle or ribbon emitters.
2. `WorldObjectPassCoordinator` groups opaque M2 doodad instances by model key and renders them via `BeginBatch()` + `RenderInstance()` per instance.
3. `M2Renderer` exposes the wrapped legacy renderer's particle/ribbon fallback requirement; native-runtime M2s remain isolated until a backend-specific batch key exists.
4. Framerates on dense doodad maps remain high (>60 FPS) when doodad rendering is enabled.

### User Story 1A - Batch Opaque WMO Doodads (Priority: P1)

As a viewer user, I want opaque WMO doodad placements that share a renderer to share one batch
setup, so that doodad sets do not repeat shader and uniform state work for every placement.

**Acceptance Criteria**:
1. Opaque visible WMO doodads group by `IModelRenderer` and use one `BeginBatch()` call per group.
2. Particle/ribbon renderers retain the unbatched fallback.
3. Transparent doodads retain distance-ordered submission and are not reordered by opaque grouping.

### User Story 2 - Deduplicate Per-Frame Model Animation Updates (Priority: P1)

As a viewer developer, I want `WorldScene` to advance model animations (`UpdateAnimation()`) at most once per unique `IModelRenderer` per frame, so that CPU time is not wasted re-evaluating animation curves and particle emitters hundreds of times for identical shared models.

**Acceptance Criteria**:
1. `WorldScene` collects unique visible `IModelRenderer` instances per frame and invokes `UpdateAnimation()` once per unique renderer.
2. Profiling metrics (`MdxAnimationMs`) reflect single-digit millisecond time even when thousands of doodad instances are visible.

### User Story 3 - Visual & Functional Parity Verification (Priority: P2)

As a viewer developer, I want to verify that batched M2 rendering maintains exact visual parity (lighting, fog, distance fading, alpha testing, and bounds) with the unbatched path and that unit test suites pass.

**Acceptance Criteria**:
1. Opaque and alpha-tested M2 doodads render identically under batched and unbatched modes.
2. Automated test suites pass without regression.

### User Story 4 - Bound Client I/O During Rendering (Priority: P1)

As a viewer user, I want deferred WMO doodad loading and minimap reads to obey scene-wide limits,
so that the number of visible placements or minimap tiles cannot turn client-file I/O into a
render-loop stall.

**Acceptance Criteria**:

1. Deferred WMO doodad model loads are advanced at most once per scene frame through the shared
   `WorldAssetManager` budget, independent of the number of visible WMO placements.
2. `WmoRenderer.RenderWithTransform` performs no deferred doodad model read.
3. Minimap decoding keeps archive reads bounded to one background reader and uploads completed
   textures through the existing render-thread upload budget.
4. Runtime diagnostics continue to distinguish pending asset loads, deferred WMO doodad loads,
   minimap pending/uploaded/failed tiles, and stage CPU time.

---

## Technical Approach

1. **`ModelRenderer.cs`**:
   - Change `RequiresUnbatchedWorldRender`:
     ```csharp
     public bool RequiresUnbatchedWorldRender => _particleEmitters.Count > 0 || _mdx.RawParticleEmitterCount > 0 || _mdx.RawRibbonEmitterCount > 0;
     ```
2. **`WorldScene.cs`**:
   - In `WorldPassCoordinator` / `ExecuteVisibleMdxAnimation`:
     - Deduplicate the set of `IModelRenderer` targets before calling `UpdateAnimation()`.

3. **`M2Renderer.cs` and `WmoRenderer.cs`**:
   - Allow static legacy-backed M2s to use the existing shared batch path while preserving the
     particle/ribbon fallback and isolating the native runtime backend's distinct state path.
   - Group compatible opaque WMO doodad placements by renderer and call `BeginBatch()` once before
     issuing their `RenderInstance()` submissions. Keep transparent placements in their existing
     back-to-front order.

The GPU phase is implemented in the renderer types but is currently held out of the production
world MDX route. Direct Alpha MDX and adapted M2 placements use the established per-instance
`RenderWithTransform()` path while the GPU batch path is revalidated against real client visuals.
The native runtime backend, transparent layers, particle/ribbon models, and unsupported
fade/material states remain on their existing fallback routes. Visual parity and real-scene
frame-time proof remain user-run gates; no MDX batching performance claim is active until the
models visibly render again.

Deferred client I/O is now scene-bounded: `WorldAssetManager` advances WMO doodad model loads once
per frame instead of once per WMO placement, and `MinimapRenderer` uses one shared-data-source
reader. This containment slice does not claim that model parsing, terrain upload, or GPU time is
now within budget; the user-run real-client capture remains the proof owner.

The MDX streaming admission path also keeps unresolved placements loadable. An MDDF instance
without model bounds starts with a small placement-centered fallback AABB; performance projected-
size culling must not reject that placeholder before the model can be queued. Such instances are
admitted from placement/frustum visibility into the existing per-frame unique-asset queue, then
return to normal projected-size culling after the loaded model supplies transformed bounds.
