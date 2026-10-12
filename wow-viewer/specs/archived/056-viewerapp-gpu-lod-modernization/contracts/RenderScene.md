# Contract: RenderScene

**Phase 1 contract. Companion to `plan.md`, `research.md`, `data-model.md`.**

This is the **backend-neutral** input to the renderer. It carries everything the renderer needs to draw one frame: camera, terrain tiles, world objects, sky/fog state, AOI bounds.

## Shape

```csharp
public sealed record RenderScene(
    SceneCamera Camera,
    IReadOnlyList<RenderTerrainTile> Tiles,
    IReadOnlyList<RenderWorldObjectRef> WorldObjects,
    SkyState Sky,
    FogState Fog,
    RenderWdlOverride? WdlOverride,
    AoiBounds AoiBounds);
```

## Ownership

- `RenderScene` lives in `WowViewer.Core.Renderer.Contracts` namespace.
- It is a **value type** (record) — no event subscriptions, no IDisposable.
- It MUST NOT reference any Silk.NET, OpenGL, Vulkan, or other backend-specific types.
- It MAY reference types from `WowViewer.Core.Runtime.World` (those are already backend-neutral).

## Producer

- The host (`WowViewer.App`, `WowViewer.Tool.ValidationCapture`, future editor) builds the `RenderScene` from the runtime composition.
- The runtime (`WowViewer.Core.Runtime.World.WorldRenderCompositionBuilder`) produces the per-tile and per-object data; the host wraps it into a `RenderScene`.

## Consumer

- `WowViewer.Core.Renderer.Scene.SceneRenderer` (multi-tile rewrite in Phase 1).
- `WowViewer.Core.Renderer.OpenGL.OpenGLRenderBackend` (Phase 1+).
- Future `WowViewer.Core.Renderer.Vulkan.VulkanRenderBackend` (follow-on spec).

## Invariants

- `RenderScene.Camera` MUST be non-null.
- `Tiles` MAY be empty (the renderer must handle a no-tile scene without crashing).
- `WorldObjects` MAY be empty.
- `AoiBounds` MUST be valid (`MaxTileX >= MinTileX`, `MaxTileY >= MinTileY`).
- `WdlOverride` MAY be null; when present, the renderer uses it for the far-horizon tiles.

## Threading

- `RenderScene` is **per-frame** and **immutable** within a frame.
- It is built on the host thread, consumed on the render thread.
- The host MUST NOT mutate the `RenderScene` after passing it to the renderer.

## Failure modes

- If `Camera` is null, the renderer MUST log a `RendererError.InvalidScene` and skip the frame.
- If a tile's `TileData` is null, the renderer MUST log a `RendererError.ResourceNotResident` and skip that tile.
- If `WdlOverride.SourceTile` references a tile not in `Tiles`, the renderer MUST log `RendererError.WdlMissingForFarHorizon` and fall back to a flat low-res mesh.

## Versioning

- Additive changes to `RenderScene` are non-breaking.
- Removing or renaming a field is a breaking change and requires a major version bump of the renderer contract.
- New optional fields (e.g. `RenderWdlOverride?`) are non-breaking.

## Tests

- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Contracts/RenderSceneTests.cs`:
  - Empty `Tiles` + empty `WorldObjects` + valid camera renders a no-op frame.
  - Null `Camera` raises `RendererError.InvalidScene`.
  - Null `TileData` for a tile skips the tile and logs an error.
  - `WdlOverride` for a tile not in `Tiles` falls back to a flat mesh.
