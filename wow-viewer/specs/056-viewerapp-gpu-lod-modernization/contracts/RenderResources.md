# Contract: RenderResources

**Phase 1 contract. Companion to `plan.md`, `research.md`, `data-model.md`.**

This is the **per-tile / per-frame** resource surface. The renderer caches GPU resources (VBOs, IBOs, UBOs, textures, shaders) keyed by tile, model path, or material. The contract makes the caching policy explicit and backend-neutral.

## Shape

```csharp
public interface IRenderResources : IDisposable
{
    ITerrainResourceCache Terrain { get; }
    IWmoResourceCache Wmo { get; }
    IM2ResourceCache M2 { get; }
    IMdxResourceCache Mdx { get; }
    ILiquidResourceCache Liquid { get; }
    ISkyResourceCache Sky { get; }
    TextureCache Textures { get; }
    IShaderCache Shaders { get; }

    void EvictAll();
    void OnMapSwitch();
}

public interface ITerrainResourceCache
{
    RenderTerrainMesh GetOrBuildMesh(RenderTerrainTile tile, TerrainLodBucket lod);
    void Evict(int tileX, int tileY);
}

public interface IWmoResourceCache
{
    RenderWmoHandle GetOrBuild(string modelPath);
    void Evict(string modelPath);
}

public interface IM2ResourceCache
{
    RenderM2Handle GetOrBuild(string modelPath);
    void Evict(string modelPath);
}

public interface IMdxResourceCache
{
    RenderMdxHandle GetOrBuild(string modelPath);
    void Evict(string modelPath);
}

public interface ILiquidResourceCache
{
    RenderLiquidHandle GetOrBuild(RenderLiquidTile tile, WaterLodBucket lod);
    void Evict(int tileX, int tileY);
}

public interface ISkyResourceCache
{
    RenderSkyHandle GetOrBuild(SkyState state);
    void Invalidate();
}
```

## Ownership

- `IRenderResources` lives in `WowViewer.Core.Renderer.Contracts`.
- The OpenGL implementation (`OpenGLRenderResources`) lives in `WowViewer.Core.Renderer.OpenGL`.
- Texture caching is delegated to the existing `WowViewer.Core.Renderer.Texture.TextureCache` (extended in Phase 5).

## Producer

- `OpenGLRenderResources` (Phase 1+) is the v0.5.0-dev implementation.
- Future `VulkanRenderResources` (follow-on spec) plugs into the same interface.

## Consumer

- `IRenderBackend` (the GPU backend) reads from caches during `Submit`.
- The host (viewer app, validation capture) does not directly touch `IRenderResources`; the renderer manages them.

## Invariants

- `GetOrBuild*` MUST be idempotent: the same input returns the same handle.
- `Evict*` removes the cached handle; the next `GetOrBuild*` rebuilds it.
- `EvictAll` removes all handles; the next frame rebuilds what it needs.
- `OnMapSwitch` is called when the host begins a map switch; it MUST release all per-map resources but MAY keep per-profile resources (e.g. common shaders).
- `Dispose` releases everything; after `Dispose`, no method may be called.

## Lifetime invariants

- A `RenderTerrainMesh` (VBO + IBO + UBO) lives as long as its tile is in the cache.
- A `TextureCacheEntry` lives as long as any model or terrain still references it (refcounted).
- A shader lives as long as `IShaderCache` keeps it; shaders are evicted only on `EvictAll` or `Dispose`.

## Threading

- The interface is **single-threaded**. The render thread owns it.
- Phase 1 does NOT introduce a streaming worker thread; the renderer is synchronous on the render thread.

## Eviction policy

- Terrain: cache by `(tileX, tileY, lod)`; eviction on map switch + on `Evict(tileX, tileY)`.
- WMO / M2 / MDX: cache by `modelPath`; eviction on map switch + on `Evict(modelPath)`.
- Liquid: cache by `(tileX, tileY, lod)`; eviction on map switch + on `Evict(tileX, tileY)`.
- Sky: cache by `SkyState`; invalidated on sky change.
- Texture: refcounted; release on last reference.
- Shader: never evicted during a map; only on `EvictAll` or `Dispose`.

## Versioning

- Adding a new resource cache (e.g. `ParticleResourceCache`) is a breaking change to `IRenderResources`. New caches should be added in the same phase that introduces them, not after.
- Adding a new method to a sub-cache (e.g. `ITerrainResourceCache.GetOrBuildAsync`) is breaking for any custom implementation; for v0.5.0-dev, there is only one implementation (`OpenGLRenderResources`), so this is not a problem in practice.

## Tests

- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Contracts/RenderResourcesContractTests.cs`:
  - `GetOrBuildMesh` twice with the same input returns the same handle.
  - `Evict(tileX, tileY)` invalidates the tile; the next `GetOrBuild` rebuilds.
  - `OnMapSwitch` clears per-map resources but keeps common shaders.
  - `EvictAll` clears everything.
  - `Dispose` makes subsequent calls throw.
  - Texture refcounting: `Acquire` -> `Release` -> `Evict` works; `Acquire` twice requires `Release` twice.
