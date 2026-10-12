# Contract: TextureCache

**Phase 1 + Phase 5 contract. Companion to `plan.md`, `research.md`, `data-model.md`.**

This is the **texture caching** surface. It is built on the existing `WowViewer.Core.Renderer.Texture.TextureCache`. Phase 5 extends it with mip-level selection (FR-010) and per-texture residency tracking.

## Current State (Phase 1)

The existing `WowViewer.Core.Renderer.Texture.TextureCache`:

- Caches BLP textures by source path.
- Owns a Silk.NET.OpenGL texture handle per cached entry.
- Decides eviction by some existing policy (Phase 1 must read the code to confirm).
- Does NOT track mip selection (Phase 5 adds it).
- Does NOT track residency as a first-class concept (Phase 5 adds it).

## Phase 5 Extensions

```csharp
public enum TextureResidency
{
    Resident,         // uploaded to GPU, ready to bind
    Streaming,        // currently being uploaded (future; out of scope for v0.5.0-dev)
    NotResident       // not uploaded; bind will fail
}

public sealed class TextureCacheEntry
{
    public TextureHandle Handle { get; }
    public string SourcePath { get; }
    public int Width { get; }
    public int Height { get; }
    public int MipCount { get; }
    public int BytesPerPixel { get; }
    public TextureResidency Residency { get; }
    public int LastSelectedMip { get; internal set; }   // updated per frame by the renderer
    public float LastSelectionDistance { get; internal set; }
    public int RefCount { get; }                         // refcounted; release on last
}

public sealed class TextureCache
{
    public TextureCacheEntry Acquire(string sourcePath);
    public void Release(TextureCacheEntry entry);
    public int SelectMip(TextureCacheEntry entry, float distance, MipSelectionSettings settings);
    public void EvictAll();
    public void OnMapSwitch();
}
```

## Ownership

- `WowViewer.Core.Renderer.Texture.TextureCache` (existing, extended in Phase 5).
- `WowViewer.Core.Renderer.Texture.TextureCacheEntry` (NEW in Phase 5; the existing cache returns a handle only — Phase 5 introduces the entry object so per-frame mip selection can be tracked).
- `WowViewer.Core.Renderer.Texture.TextureResidency` (NEW in Phase 5).
- `WowViewer.Core.Renderer.Texture.MipSelectionSettings` (NEW in Phase 1 in the Scene namespace per `data-model.md`; consumed by `SelectMip` in Phase 5).

## Producer

- Phase 5 extends `TextureCache` to call `SelectMip` per frame on every bound texture.

## Consumer

- `IRenderBackend` (during `Submit`) calls `Acquire` before binding and `Release` after.
- The validation harness (Phase 7) reads `LastSelectedMip` and `LastSelectionDistance` to compute mip-selection distribution.

## Invariants

- `Acquire(path)` MUST be matched by exactly one `Release(entry)`.
- `SelectMip(entry, distance, settings)` MUST be deterministic: same `(entry, distance, settings)` -> same mip.
- `LastSelectedMip` is updated only on `SelectMip`, not on `Acquire`.
- `RefCount` starts at 1 after `Acquire`; it MUST reach 0 before the entry is evictable.
- An entry with `RefCount > 0` MUST NOT be evicted by `EvictAll` (the eviction is recorded but deferred until refcount reaches 0).

## Mip selection policy (Phase 5)

Default policy:

```text
if (distance < settings.NearDistance)        -> mip 0
else if (distance < settings.MidDistance)    -> mip 1
else if (distance < settings.FarDistance)    -> mip 2
else                                         -> highest mip
```

The default is configurable per call; a future phase may add LOD bias, anisotropic-aware selection, or per-texture overrides.

## Threading

- Single-threaded on the render thread.
- Acquire/Release pairs are not allowed to cross thread boundaries.

## Tests

- `wow-viewer/tests/WowViewer.Core.Renderer.Tests/Texture/TextureCacheTests.cs`:
  - `Acquire` + `Acquire` of the same path returns the same entry with `RefCount == 2`.
  - `Release` decrements `RefCount`; when 0, the entry is evictable.
  - `SelectMip` is deterministic across calls.
  - `LastSelectedMip` updates only on `SelectMip`.
  - `EvictAll` defers eviction for entries with `RefCount > 0` and evicts them when refcount reaches 0.
  - `OnMapSwitch` releases per-map textures but keeps common (e.g. atlas) textures.
