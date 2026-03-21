# WoW 4.0.0.11927 Engine Performance Recovery Guide

## Purpose

This guide captures the first Ghidra-backed performance findings from `wow.exe` build `4.0.0.11927` and maps them onto the active `MdxViewer` codebase.

The point is not to claim that the viewer should become a full reimplementation of Blizzard's engine. The point is to stop taking blind performance guesses when the binary already exposes clear subsystems for:

- archive and streamed file I/O
- cache layers and residency limits
- shader/effect selection through `CGx` and `.bls` assets
- async background work
- visibility and occlusion structures

## Evidence Base

- Binary: `wow.exe` build `4.0.0.11927`
- Method: Ghidra string survey, namespace/class inventory, and import-table inspection
- Goal: identify high-confidence engine seams that are relevant to viewer-side performance work

## High-Confidence Findings

### 1. The client has an explicit shader/effect stack, not just ad hoc fixed-function rendering

High-confidence strings and classes:

- `0x009f5a88`: `./ShaderEffectManager.cpp`
- `0x009f5aa4`: `Shaders\Effects\%s`
- `0x00a2ee70`: `Shaders\Vertex`
- `0x00a2ee80`: `Shaders\Hull`
- `0x00a2ee90`: `Shaders\Domain`
- `0x00a2eea0`: `Shaders\Geometry`
- `0x00a2eeb4`: `Shaders\Pixel`
- `0x00a2eec4`: `Shaders\Compute`
- `0x00a2ee60`: `%s\%s\%s.bls`
- namespaces/classes include `CGxShader`, `CShaderEffect`, `CSceneMaterial`, `CGxTexCache`, `CMapDynamicTexture`, `CMapFootprintTexture`, and `CMapRenderChunk`

Interpretation:

- the real client has an effect-selection layer on top of raw shader blobs
- it chooses from named shader directories and `.bls` shader assets
- material/effect identity is a first-class concept, not only a set of one-off render flags

Why this matters for `MdxViewer`:

- the active viewer mostly compiles local GLSL programs directly in renderers like `ModelRenderer`, `WmoRenderer`, `TerrainRenderer`, `LiquidRenderer`, and `BoundingBoxRenderer`
- that is workable, but it means state grouping and effect reuse are currently local implementation details rather than a shared effect system
- any performance pass around shaders should prefer stable material/effect descriptors and shared program reuse over piling on more one-off branches

### 2. The client has multiple archive, cache, and streamed-file layers

High-confidence strings:

- `0x009f2228`: `Failed to open archive %s.`
- `0x00a25830`: `archive %s opened`
- `0x00a2585c`: `SArchive`
- `0x00a26640`: `...\packages\mopaq\SFileArchives.cpp`
- `0x00a26460`: `...\packages\mopaq\Mpqstack_Cache.cpp`
- `0x009f06b0`: `streaming.MPQ`
- `0x009f069c`: `streamingloc.MPQ`
- `0x00a25a4c`: `Data/SoundCache.MPQ`
- `0x00a33f88`: `./FileCache.cpp`
- `0x00a2c9ae`: ` A.\M2Cache.cpp`

High-confidence imports:

- file and mapping: `CreateFileA`, `SetFilePointer`, `CreateFileMappingA`, `MapViewOfFile`, `UnmapViewOfFile`
- async/overlapped I/O: `ReadFileEx`, `WriteFileEx`, `CancelIo`, `GetOverlappedResult`
- I/O completion: `CreateIoCompletionPort`, `GetQueuedCompletionStatus`

Interpretation:

- the client is not treating asset access as one flat `open/read/close` path
- it has archive objects, cache-specific codepaths, and streaming-specific resources
- the import set strongly suggests overlapped/asynchronous file work, and the mapping imports suggest that not all asset reads are pure buffered copies

Why this matters for `MdxViewer`:

- the active viewer already has meaningful groundwork in `MpqDataSource` and `WorldAssetManager`:
  - file-set indexing
  - canonical path maps
  - raw-byte read cache
  - background prefetch queue with separate `NativeMpqService` workers
- that is the correct direction, but the engine evidence says this should be treated as a primary performance track, not a minor helper

### 3. The client exposes explicit cache knobs and cache-specific subsystems

High-confidence strings:

- `0x009f0df0`: `Texture cache size (%d meg) greater than maximum allowed for your system (%d meg).`
- `0x009f0e44`: `Texture cache size set to %d meg.`
- `0x009f1e6c`: `textureCacheSize`
- `0x00a24a14`: `gxTextureCacheSize`
- `0x00a1e0a4`: `../../Common/TextureCache.cpp`
- `0x00a1e0c4`: `TextureCacheGetInfo() Wait: %s`
- `0x00a24898`: `Disabling BSP node cache.`
- `0x00a248c8`: `Enabling BSP node cache (first time - starting up)`
- `0x00a24c98`: `bspcache`
- `0x00a1ae68`: `Max sound size that will be cached, larger files will be streamed instead`

Interpretation:

- the real client treats cache behavior as configurable and subsystem-specific
- texture caching, BSP-node caching, and sound streaming all have separate concerns
- at least one texture-info path has observable wait states, which implies synchronization or worker interaction worth respecting

Why this matters for `MdxViewer`:

- texture residency is still split across multiple local caches:
  - `TerrainRenderer` keeps its own diffuse texture cache
  - `MinimapRenderer` keeps a separate cache
  - model-side texture reuse exists, but not all GPU texture caching is centralized
- the engine evidence supports a dedicated cache/residency pass instead of leaving each renderer to manage its own unrelated cache policy

### 4. The client has explicit async worker structures

High-confidence classes/namespaces:

- `CAsyncObject`
- `CAsyncQueue`
- `CAsyncThread`
- `CStreamingEntry`

High-confidence strings/imports:

- `0x00a34c84`: `CreateIoCompletionPort failed`
- `0x00a8c490`: `ReadDirectoryChangesW`
- imports: `CreateThread`, `WaitForMultipleObjectsEx`, `CreateEventA`, `SignalObjectAndWait`, `CreateIoCompletionPort`, `GetQueuedCompletionStatus`

Interpretation:

- the client has named async queue/thread concepts and not just a few random worker helpers
- it also has the Windows primitives required to coordinate those systems

Why this matters for `MdxViewer`:

- background work in the viewer should stay deliberate:
  - prefer read-only worker services and immutable handoff payloads
  - do not share one mutable MPQ reader across threads
  - push discovery, byte warming, and decode preparation off the UI thread when possible

### 5. The client has world-visibility structures beyond plain frustum culling

High-confidence classes:

- `CWorldOccluder`
- `CWorldAntiOccluder`
- `CWorldOcclusionVolume`

Interpretation:

- the engine has explicit occlusion-side world structures
- this does not mean the viewer should jump straight into a full occlusion system, but it does mean that visibility is richer than just distance and frustum tests

Why this matters for `MdxViewer`:

- current viewer visibility is mostly:
  - `FrustumCuller` plane tests
  - distance caps/fades in `WorldScene`
  - chunk culling in `TerrainRenderer`
  - WMO doodad culling in `WmoRenderer`
- those are valuable, but the next performance slice should first improve batching and broad-phase rejection before attempting true occlusion

## Mapping To The Active Viewer Codebase

### Archive and file I/O

Active files:

- `src/MdxViewer/DataSources/MpqDataSource.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Services/NativeMpqService.cs`
- `src/MdxViewer/Terrain/WorldAssetManager.cs`

Current viewer strengths:

- path index and canonical-path lookup already exist
- raw-byte read cache already exists
- background prefetch already exists with separate MPQ workers
- renderer residency defaults are already less destructive than older branches

Current viewer gaps:

- repeated path probing and fallback search still exist in hot codepaths
- prefetch is not yet driven by a broader scene/streaming policy
- there is no single measured view of cache-hit rate, bytes read, or queue latency in normal runtime use

### Shader and material stack

Active files:

- `src/MdxViewer/Rendering/ModelRenderer.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`
- `src/MdxViewer/Terrain/TerrainRenderer.cs`
- `src/MdxViewer/Terrain/LiquidRenderer.cs`
- `src/MdxViewer/Terrain/BoundingBoxRenderer.cs`
- `reference_data/wowdev.wiki/BLS.md`
- `reference_data/wowdev.wiki/WFX.md`

Current viewer strengths:

- shared `ShaderProgram` exists for several terrain-side renderers
- recent M2 and WMO parity work already separated some material families better than before

Current viewer gaps:

- there is still no shared effect/material descriptor layer comparable to `CShaderEffect`
- renderer-specific GLSL compilation and state setup remain fragmented
- shader-family choice is still more heuristic and local than the real client's effect stack suggests

### Visibility, culling, and batching

Active files:

- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/Rendering/FrustumCuller.cs`
- `src/MdxViewer/Terrain/TerrainRenderer.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`

Current viewer strengths:

- frustum culling is present and already notes the original client frustum update path
- terrain, WMO, MDX, and PM4 all have some broad-phase visibility checks

Current viewer gaps:

- draw submission and state grouping are still largely per-renderer and per-instance
- there is no occlusion-volume analogue yet
- PM4 debugging features should remain optional and not silently become always-on per-frame cost

## Recommended One-Shot Passes

### 1. Archive I/O and path-resolution pass

Primary target:

- reduce repeated file lookup, fallback probing, and archive read cost for hot asset families

Good files to start with:

- `src/MdxViewer/DataSources/MpqDataSource.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Services/NativeMpqService.cs`
- `src/MdxViewer/Terrain/WorldAssetManager.cs`

Expected output:

- better cache-hit instrumentation
- fewer redundant probes in `ReadFile`-heavy codepaths
- tighter prefetch targeting for upcoming world assets

### 2. Texture/cache residency pass

Primary target:

- unify or at least coordinate texture residency decisions across terrain, model, minimap, and preview paths

Good files to start with:

- `src/MdxViewer/Terrain/TerrainRenderer.cs`
- `src/MdxViewer/Terrain/WorldAssetManager.cs`
- `src/MdxViewer/Rendering/MinimapRenderer.cs`
- `src/MdxViewer/Terrain/WdlPreviewCacheService.cs`

Expected output:

- lower duplicate decode/upload cost
- visible cache stats and budgets
- explicit eviction policy instead of accidental fragmentation

### 3. BLS/effect parity pass

Primary target:

- move the viewer closer to the real client's effect-family model so material choice and shader reuse become more stable

Good files to start with:

- `src/MdxViewer/Rendering/ModelRenderer.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`
- `reference_data/wowdev.wiki/BLS.md`
- `reference_data/wowdev.wiki/WFX.md`

Expected output:

- shared material/effect descriptors
- less duplicated shader setup
- clearer separation of opaque, cutout, alpha, additive, modulated, env-map, and related families

### 4. Scene culling and batching pass

Primary target:

- reduce per-frame CPU work and unnecessary state churn before attempting deeper occlusion work

Good files to start with:

- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/Rendering/FrustumCuller.cs`
- `src/MdxViewer/Terrain/TerrainRenderer.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`

Expected output:

- better submission counts
- lower repeated AABB and distance work in the hot render loop
- preserved PM4 tooling while keeping debug cost opt-in

## What Not To Do

- Do not claim parity or safety from build success alone.
- Do not start with speculative engine rewrites that ignore the current viewer architecture.
- Do not widen background threading by sharing one mutable MPQ reader across threads.
- Do not treat shader work as only a visual-parity problem; it is also a batching and state-stability problem.
- Do not treat cache work as only an archive concern; texture and scene residency are separate layers.

## Validation Rules

- Always build `src/MdxViewer/MdxViewer.sln` after a performance slice.
- If runtime validation on real data is not performed, say so explicitly.
- If a pass only adds instrumentation or planning artifacts, say so explicitly.
- Use the fixed development data paths in `memory-bank/data-paths.md` when runtime checks are needed.

## Companion Prompt Plans

The focused one-shot prompt plans created for this guide are:

- `.github/prompts/archive-io-performance-plan.prompt.md`
- `.github/prompts/cache-residency-performance-plan.prompt.md`
- `.github/prompts/bls-shader-parity-performance-plan.prompt.md`
- `.github/prompts/scene-culling-batching-performance-plan.prompt.md`