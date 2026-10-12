# Contract: RenderBackend

**Phase 1 contract. Companion to `plan.md`, `research.md`, `data-model.md`.**

This is the **backend-neutral** GPU interface. The v0.5.0-dev implementation is `OpenGLRenderBackend`. A future `VulkanRenderBackend` conforms to the same interface and lands in a follow-on spec.

## Shape

```csharp
public enum RenderBackendKind
{
    OpenGL,
    Vulkan   // future
}

public interface IRenderBackend : IDisposable
{
    RenderBackendKind Kind { get; }

    void Initialize(Viewport viewport);
    PerFrameRenderStats BeginFrame(RenderScene scene, RenderVariant variant);
    void Submit(
        IReadOnlyList<RenderTerrainTile> terrain,
        IReadOnlyList<RenderLiquidTile> liquids,
        IReadOnlyList<RenderWorldObjectRef> worldObjects,
        SkyState sky);
    PerFrameRenderStats EndFrame();
    void Resize(int width, int height);

    PerFrameRenderStats Stats { get; }
    RendererLifecycleState Lifecycle { get; }
    RendererError? LastError { get; }
}
```

## Ownership

- `IRenderBackend` lives in `WowViewer.Core.Renderer.Contracts` namespace.
- It MUST NOT reference any backend-specific types (no `Silk.NET.OpenGL.GL`, no `Vulkan.*`).

## Producer

- `WowViewer.Core.Renderer.Scene.SceneRenderer` selects a backend at construction time.
- For v0.5.0-dev, only `OpenGLRenderBackend` exists.
- Backend selection is **constructor injection**, not runtime polymorphism (we ship one backend in v0.5.0-dev).

## Consumer

- `WowViewer.App` (the viewer host).
- `WowViewer.Tool.ValidationCapture` (the headless capture tool).
- Future editor host.

## Lifecycle invariants

- `Initialize` MUST be called exactly once before any frame is submitted.
- `BeginFrame` / `Submit` / `EndFrame` MUST be called in that order, exactly once per frame.
- `EndFrame` MUST be called even if `Submit` was a no-op.
- `Resize` MAY be called between frames; it MUST NOT be called mid-frame.
- `Dispose` MUST be called exactly once; after `Dispose`, no other method MAY be called.

## Error invariants

- If a frame fails, the backend MUST set `LastError` to a non-None value and continue (not throw).
- `Stats` after a failed frame MUST reflect the partial work, not stale data.
- `LastError` is sticky until the next successful `BeginFrame`.

## Threading

- The interface is **single-threaded**. The host calls `BeginFrame` / `Submit` / `EndFrame` on the render thread.
- `OpenGLRenderBackend` is **not** thread-safe; it requires the GL context to be current on the calling thread.

## Performance invariants

- `BeginFrame` MUST do at most O(1) work (no allocation, no per-frame state reset beyond the stats reset).
- `EndFrame` MUST update `Stats` in O(1) work.
- Per-frame allocation MUST be bounded; the renderer is held to a no-alloc-in-steady-state policy (this is verified by Phase 7's validation harness).

## Versioning

- The interface is a contract; adding a method is breaking for any implementation. New optional behavior should go through `RenderVariant` or a new settings record, not new interface methods.
- `RenderBackendKind` is an enum; adding a new kind is non-breaking (consumers should default-switch on it).

## Tests

- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Contracts/RenderBackendContractTests.cs`:
  - Tests run against `OpenGLRenderBackend` with a real GL context (or a stub if no GL is available in CI).
  - Lifecycle: `Initialize` -> `BeginFrame` -> `Submit` -> `EndFrame` x N -> `Dispose` succeeds.
  - Lifecycle violation: `BeginFrame` twice without `EndFrame` is rejected.
  - Resize mid-frame is rejected.
  - `Dispose` twice is a no-op (or rejected; documented in the test).
  - Failed frame sets `LastError` and `Stats` reflects the failure.

---

## OpenGL Implementation Notes (Phase 1+)

`OpenGLRenderBackend` is the v0.5.0-dev implementation. It lives in `WowViewer.Core.Renderer.OpenGL`. The split between `Contracts` (this file) and `OpenGL` (the implementation) is the seam that the future Vulkan backend plugs into.

- All `GL.*` calls live in `WowViewer.Core.Renderer.OpenGL/*`. No GL types are exposed from the public `Contracts` namespace.
- The implementation owns its own `GL` handle and its own resource caches (`OpenGLBufferFactory`, `OpenGLShaderCache`, `OpenGLRenderResources`).
- The implementation is responsible for translating the backend-neutral `RenderScene` / `RenderVariant` into GL calls.

## Vulkan Implementation Notes (Follow-on Spec, Out of Scope)

`VulkanRenderBackend` is **not** in this spec. When it lands, it conforms to `IRenderBackend` and lives in `WowViewer.Core.Renderer.Vulkan`. The current `Contracts/*` types are designed to be backend-neutral enough to support Vulkan; if a Vulkan-specific data type is needed (e.g. a descriptor-set binding model), it goes into a follow-on contract addendum, not into the OpenGL implementation.
