# Spec 136: M2 Doodad Rendering Performance Optimization

## Overview

On maps with high density of placed M2 doodads (trees, rocks, fences, grass, clutter), rendering framerates drop from 60+ FPS down to <1 FPS when doodad rendering is enabled. Toggling doodad rendering off (`DoodadsVisible = false`) immediately restores performance to 30-3000 FPS.

Audit of the rendering pipeline reveals two major bottlenecks:
1. **Forced Unbatched Rendering**: `ModelRenderer.RequiresUnbatchedWorldRender` returns `true` for all `_isM2AdapterModel` instances (100% of M2 models loaded in the viewer). This forces every single M2 doodad instance onto the unbatched `RenderWithTransform` fallback path, causing full OpenGL shader program rebinding (`glUseProgram`), 8–10 GL uniform uploads, and state changes per object instance per frame.
2. **Redundant Animation Updates**: `WorldScene` iterates over all $N$ visible placement instances every frame and calls `renderer.UpdateAnimation()`. When dozens or hundreds of placed instances share the same model (e.g. 500 instances of `tree.m2`), `UpdateAnimation()` is executed 500 times per frame on the exact same renderer object.

## User Stories

### User Story 1 - Enable Batched Instancing for M2 Doodads (Priority: P1)

As a viewer user, I want M2 doodad models without active particle or ribbon emitters to use the high-throughput `BeginBatch()` / `RenderInstance()` rendering path, so that thousands of static doodads can be rendered smoothly without per-instance shader rebinding or redundant GL state calls.

**Acceptance Criteria**:
1. `ModelRenderer.RequiresUnbatchedWorldRender` returns `false` for M2 models that do not have active particle or ribbon emitters.
2. `WorldObjectPassCoordinator` groups opaque M2 doodad instances by model key and renders them via `BeginBatch()` + `RenderInstance()` per instance.
3. Framerates on dense doodad maps remain high (>60 FPS) when doodad rendering is enabled.

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
