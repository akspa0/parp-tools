# Architecture & Technical Design: Spec 275 Asset Streaming for MPQ and CASC

**Spec**: [Spec 275: High-Performance Asset Streaming for MPQ and CASC](spec.md)  
**Parent**: Epic 249 (Renderer Performance, Lighting & Correctness)  
**Governance**: `AGENTS.md` §4, §9, §10

---

## 1. System Overview & Data Flow

Currently, asset streaming suffers from two major bottlenecks:
1. **Synchronous CPU execution on the render thread**: File reads, archive decompressions, M2 vertex construction, and WMO group parsing execute inside `ProcessPendingLoads` on the main loop.
2. **Stale FIFO queue order and uncoordinated prefetching**: Older requests block newer near-camera requests, while MPQ group files and textures are not prefetched ahead of time.

### Target Multi-Stage Streaming Pipeline

```
[Camera & Scene Visibility]
         │
         ▼
[WorldAssetPriorityQueue]  ◄── Dynamic scoring: (DistanceSq * (InFrustum ? 0.25 : 1.0))
         │
   ┌─────┴────────────────────────┐
   ▼                              ▼
[Archive Prefetcher]       [WorldAssetDecodeWorker Pool]
(MPQ / CASC Lookahead)     (Multi-threaded CPU Workers)
   │                              │
   │ Raw bytes hot in RAM         │ 1. Read bytes from IDataSource (RAM hit)
   └─────────────────────────────►│ 2. Parse M2StaticRenderModel / WmoV14Data
                                  │ 3. Extract vertex/index arrays & bounding boxes
                                  │ 4. Queue dependent textures to WorldTextureStreamer
                                  ▼
                     [ConcurrentQueue<PreparedModelPackage>]
                                  │
                                  ▼ (Render Thread — strictly budgeted <= 2.5 ms)
                     [WorldAssetManager.ProcessPendingLoads]
                                  │
                                  ├── Fast glGenVertexArrays / glGenBuffers
                                  ├── Direct glBufferData (pinned arrays)
                                  └── Register loaded IModelRenderer / WmoRenderer
```

---

## 2. Component Design

### 2.1 Component 1: In-Memory WMO Parsing (`WowViewer.Core.IO.Converters`)
- Add overload `public WmoV14Data ParseWmoV14(byte[] wmoData)` in `WmoV14ToV17Converter.cs` (and `MemoryStream` / `ReadOnlySpan<byte>` wrapper).
- In `WorldAssetManager.LoadWmoDataModel`:
  - Eliminate `Path.Combine(Path.GetTempPath(), $"wmo_{Guid.NewGuid():N}.tmp")` and `File.WriteAllBytes`.
  - Parse `wmoData` directly from the byte array.
  - Guarantees compliance with `AGENTS.md` §9.3 (Write Containment) and eliminates disk I/O latency.

### 2.2 Component 2: MPQ Dependency Lookahead Prefetching (`WoWViewer.DataSources` & `Terrain`)
- Extend MPQ prefetching to mirror the behavior of `CascDataSource`:
  - When an M2 path is prefetched:
    - Inspect the M2 header / chunk bytes.
    - If `M2ChunkedFileIds` or classic texture names are present, immediately prefetch candidate skins (`.skin`) and candidate textures (`.blp`).
  - When a WMO root path is prefetched:
    - Parse the 64-byte `MOHD` chunk from root bytes to get `GroupCount`.
    - Automatically enqueue group files `[base]_[000..GroupCount-1].wmo` into `MpqDataSource.PrefetchFile`.
    - Parse `MOTX` chunk (texture string block) and prefetch all referenced `.blp` material textures.
- Result: By the time the worker requests group files or textures, they are already warm in RAM cache, dropping archive read latency to zero.

### 2.3 Component 3: Off-Thread Model & Mesh Decoding Worker (`WorldAssetDecodeWorker`)
- Class `WorldAssetDecodeWorker`:
  - Worker pool using `System.Threading.Channels.Channel<AssetDecodeRequest>` with bounded concurrency ($\max(2, \text{ProcessorCount} / 2)$).
  - Handles two request types:
    - `MdxDecodeRequest`: Reads M2 bytes, resolves candidate skins, builds `M2StaticRenderModel` via `WowViewerM2RuntimeBridge.BuildStaticRenderModel`, computes world bounds, and extracts contiguous vertex/index byte buffers.
    - `WmoDecodeRequest`: Reads root WMO bytes, reads all group files, parses `WmoV14Data` / `WmoV17Data`, resolves materials and portals, and prepares group vertex/index buffers.
  - Emits `PreparedModelPackage` into a lock-free `ConcurrentQueue<PreparedModelPackage>`:
    ```csharp
    public sealed class PreparedModelPackage
    {
        public required string NormalizedKey { get; init; }
        public required bool IsMdx { get; init; }
        public M2StaticRenderModel? M2Model { get; init; }
        public WmoV14ToV17Converter.WmoV14Data? WmoData { get; init; }
        public string? ResolvedPath { get; init; }
        public Vector3 BoundsMin { get; init; }
        public Vector3 BoundsMax { get; init; }
        public Exception? Error { get; init; }
    }
    ```
- Render-Thread Upload:
  - In `WorldAssetManager.ProcessPendingLoads(maxLoads, maxBudgetMs)`:
    - Dequeue prepared packages from `CompletedPackages`.
    - If `IsMdx`, instantiate `M2Renderer` with the pre-built `M2StaticRenderModel` (only performs `glGenBuffers` and `glBufferData`).
    - If `WMO`, instantiate `WmoRenderer` with the pre-built `WmoData` (only performs group `glBufferData`).
    - Average time per model on the render thread drops from $\sim 35\text{ ms}$ to $\le 0.3\text{ ms}$!

### 2.4 Component 4: Dynamic Frustum & Distance-Ranked Priority Queue (`WorldAssetPriorityQueue`)
- Class `WorldAssetPriorityQueue`:
  - Replaces the simple `Queue<string>` with a camera-aware priority ranking:
    $$\text{Score} = \text{DistanceSq} \times (\text{InViewFrustum} ? 0.25 : 1.0)$$
  - Lower score = higher priority.
  - When the camera rotates or travels at high speed, active candidate scores are updated.
  - Stale entries that have moved beyond the maximum streaming radius ($\ge 800\text{ yards}$) are culled from the backlog, preventing queue bloat and memory leaks.

### 2.5 Component 5: Unified WMO Material Texture Off-Thread Streaming
- Generalize texture decoding to support WMO materials:
  - Migrate `WmoMaterialManager.ProcessDeferredMaterialTextureLoads` from synchronous render-thread `ImageSharp` decode to the background worker pipeline.
  - Read BLP bytes off-thread.
  - Parse DXT compressed blocks (DXT1, DXT3, DXT5) directly using `BlpCompressedTexture`.
  - On the render thread: upload via `glCompressedTexImage2D` with pre-authored mipmaps, completely eliminating CPU RGBA expansion and `glGenerateMipmap` GPU overhead.
  - Deduplicate material textures across WMO instances using a shared texture registry (`SharedTextureRegistry`).

---

## 3. Thread Safety & Architectural Boundaries

1. **No OpenGL Calls Off-Thread**:
   - Silk.NET OpenGL context belongs strictly to the render thread. Background workers only handle byte reading, parsing, math, and memory allocations.
2. **Thread-Safe Data Sources**:
   - `IDataSource.ReadFile` and `PrefetchFile` are thread-safe in both `MpqDataSource` and `CascDataSource`.
3. **God-Class Freeze (`AGENTS.md` §10)**:
   - All streaming logic is implemented in self-contained service classes under `WoWViewer.Terrain.Streaming` and `WoWViewer.Rendering`.
   - `WorldAssetManager` holds references to the new services and delegates streaming operations. `WorldScene.cs` remains clean.

---

## 4. Phase Roadmap

- **Phase 1: In-Memory WMO Parsing & Write Containment**
  - Add in-memory parsing to `WmoV14ToV17Converter`.
  - Remove temporary disk file creation from `WorldAssetManager.LoadWmoDataModel`.
- **Phase 2: MPQ Lookahead Dependency Prefetching**
  - Implement M2 and WMO dependency extraction in `MpqDataSource` / `WorldAssetManager`.
  - Auto-enqueue `.skin`, `_000.wmo`, and `.blp` dependencies when root models are prefetched.
- **Phase 3: Off-Thread Model Decoding Worker**
  - Implement `WorldAssetDecodeWorker` and `PreparedModelPackage`.
  - Route deferred M2 and WMO loads through the background worker.
  - Fast-path GL buffer allocation in `WorldAssetManager.ProcessPendingLoads`.
- **Phase 4: Dynamic Frustum & Camera-Distance Priority Queue**
  - Implement `WorldAssetPriorityQueue` with distance + frustum weighting.
  - Wire camera position and frustum updates into the priority queue.
- **Phase 5: WMO Material Texture Off-Thread Streaming & DXT Upload**
  - Connect WMO material textures to off-thread BLP decoding.
  - Replace uncompressed RGBA upload with direct `glCompressedTexImage2D`.
  - Implement cross-WMO shared texture caching.
- **Phase 6: Integration, Validation & Benchmarks**
  - Run full test suite (`dotnet test`).
  - Measure frame time and hitch duration under high-speed flight across MPQ and CASC maps.
