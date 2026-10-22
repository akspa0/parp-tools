# Spec 275 — High-Performance Asset Streaming for MPQ and CASC

**Feature Branch**: `v0.6.1-dev`  
**Created**: 2026-10-18  
**Status**: Specified; awaiting phase implementation  
**Parent**: [Epic 249 (Renderer Performance, Lighting & Correctness)](../249-epic-renderer-performance-and-correctness/spec.md) & [Epic 248 (Formats, Readers & Conversion)](../248-epic-formats-and-conversion/spec.md)  
**Governance**: `AGENTS.md` §4 (Code Ownership & Architectural Boundaries), §9 (Governance, Scope Fidelity & Receipts), §10 (God-Class Freeze)

---

## 1. Problem Statement & Baseline Evidence

Streaming game assets (M2 models, WMO buildings, skins, and textures) across both MPQ archives (Classic 0.5.3–3.3.5a) and modern CASC installations (4.3.4, 6.0+, 1.60.1) suffers from severe frame hitches, dropped frames, and visual pop-in during camera navigation and flight:

1. **Synchronous Render-Thread Model Decoding & Parsing**:
   - In `WorldAssetManager.cs` (`ProcessPendingLoads`), when a deferred M2 or WMO is popped from the queue, `LoadMdxModel` or `LoadWmoModel` runs directly on the render thread.
   - For M2s: `WowViewerM2RuntimeBridge.BuildStaticRenderModel` performs heavy CPU vertex extraction, normal calculation, and bone mapping on the render thread (5–30 ms per model).
   - For WMOs: `LoadWmoDataModel` writes a temporary file to `Path.GetTempPath()`, parses the root WMO, and synchronously reads and parses every WMO group file (`_000.wmo` … `_NNN.wmo`) on the render thread (20–100+ ms for complex structures).
   - Because `LoadBudget.CanStartAnotherLoad` is evaluated *before* beginning a load, a single 80 ms WMO load instantly blows the 6.0 ms frame budget, causing massive stutter (dropping below 15 FPS).

2. **WMO Material Textures Decoded Uncompressed on Render Thread**:
   - While Spec 256 introduced `M2TextureStreamer` for M2 textures, `WmoMaterialManager.cs` still runs on the render thread.
   - For every WMO material, it calls `_dataSource.ReadFile` synchronously, decodes BLP to uncompressed 32-bit RGBA via ImageSharp on the CPU, uploads via `glTexImage2D`, and invokes `_gl.GenerateMipmap` on the GPU.
   - A single building with 30 materials takes 30 frames to load its textures, generating continuous CPU/GPU hitches and redundant texture memory consumption without cross-WMO texture deduplication.

3. **MPQ Dependency Prefetching Deficiency**:
   - `CascDataSource.cs` features dependency parsing (`EnqueueDependencies`) for `SFID`, `TXID`, `GFID`, and `MODI`.
   - In contrast, `MpqDataSource.cs` has zero dependency prefetching. When a WMO or M2 is queued in MPQ mode, group files (`_000.wmo`) and textures are not prefetched. When the loader executes, every group and texture incurs a cache miss and blocks on synchronous disk I/O.

4. **Stale FIFO Backlog & Out-of-Frustum Loading**:
   - `WorldAssetManager._priorityMdxLoads` and `_priorityWmoLoads` are strict FIFO queues (`Queue<string>`).
   - When the user turns around or flies quickly across terrain, objects that were near the camera seconds ago remain at the front of the queue, while objects directly in front of the camera are pushed to the back.
   - Distant and out-of-view objects are decoded while visible near objects pop in late.

5. **Disk Write Containment Violation**:
   - `WorldAssetManager.LoadWmoDataModel` writes temporary files to `Path.GetTempPath()` (`File.WriteAllBytes(tmpPath, data)`) on every v14 WMO load, violating `AGENTS.md` §9.3 (Write Containment) and adding unnecessary filesystem overhead.

---

## 2. User Stories

- **US1 (P1 — Smooth Flight & Zero-Hitch Traversal)**: As a viewer operator, I want models (M2) and buildings (WMO) to stream into the world smoothly during fast camera flight (e.g. taxi routes at 150+ yards/sec) without frame drops below 60 FPS on both MPQ and CASC backends.
- **US2 (P1 — Off-Thread Model Decoding)**: As an engine developer, I want all CPU-intensive file reading, archive decompression, M2 vertex construction, and WMO group parsing to occur on background worker tasks so that the render thread only performs lightweight OpenGL buffer allocation.
- **US3 (P1 — Dynamic Camera-Prioritized Streaming)**: As a viewer operator, I want objects directly in the camera's view frustum and closest to the viewpoint to load first, pruning or deprioritizing assets that are behind the camera or out of range.
- **US4 (P2 — Unified Asynchronous Texture Streaming & Deduplication)**: As an engine developer, I want WMO materials and terrain textures to utilize off-thread BLP decoding and direct DXT compressed GPU uploads (`glCompressedTexImage2D`), sharing identical textures across all scene models.
- **US5 (P2 — MPQ Dependency Lookahead Prefetching)**: As a viewer operator loading Classic/Wrath MPQs, I want WMO group files, skins, and textures to be prefetched into cache as soon as the root asset is requested, eliminating synchronous cache misses.

---

## 3. Acceptance Criteria

- **AC-001 (Off-Thread Model Decode)**: `WorldAssetManager` decodes M2 and WMO geometry on background worker threads. The render thread spends $\le 1.0\text{ ms}$ per model creating GL buffers.
- **AC-002 (In-Memory WMO Parsing & Write Containment)**: `WmoV14ToV17Converter` exposes an in-memory byte/stream parse method. `WorldAssetManager` does not write temporary files to `Path.GetTempPath()`.
- **AC-003 (Frustum-Aware Distance Priority Queue)**: The asset priority queue orders pending loads by camera distance and frustum containment. Nearest objects in front of the camera pop before distant or behind-camera objects.
- **AC-004 (WMO Material Texture Off-Thread Streaming)**: WMO material textures are decoded off-thread into direct DXT/compressed pixel packets and uploaded via `glCompressedTexImage2D` with pre-authored mipmaps, eliminating `glGenerateMipmap` overhead and CPU RGBA conversions on the render thread.
- **AC-005 (Cross-Asset Texture Deduplication)**: Identical textures shared across multiple WMOs and M2s are cached by normalized virtual path and uploaded exactly once to GPU memory.
- **AC-006 (MPQ Dependency Prefetching)**: When an M2 or WMO root is requested in MPQ mode, its dependent `.skin` files, group files (`_000.wmo` … `_NNN.wmo`), and texture paths are enqueued into `MpqDataSource` prefetch workers.
- **AC-007 (Performance Metric & Budget Enforcement)**: Under heavy streaming (e.g. entering Dalaran or Stormwind), `ProcessPendingLoads` strictly honors the frame budget ($\le 3.0\text{ ms}$ total), maintaining $\ge 60\text{ FPS}$ on dedicated hardware.
- **AC-008 (Compilation & Test Parity)**: `dotnet build WowViewer.slnx -c Debug` builds with 0 errors and 0 warnings. Existing unit and integration tests continue to pass.

---

## 4. Architectural Constraints & Boundaries

1. **`AGENTS.md` §4 Format Readers**: Low-level MPQ archive readers (`MpqArchiveCatalog.cs`, `NativeMpqService.cs`) and CASC readers (`CascStorage.cs`) remain untouched. All streaming optimizations live in `WorldAssetManager`, `IDataSource` prefetch adapters, and renderer preparation services.
2. **`AGENTS.md` §10 God-Class Freeze**: No new fields, methods, or state in `WorldScene.cs` or `ViewerApp.cs`. Streaming logic lives in dedicated service classes:
   - `WorldAssetDecodeWorker.cs`: Off-thread CPU model and mesh decoding.
   - `WorldAssetPriorityQueue.cs`: Camera-distance and frustum-ranked priority queue.
   - `WorldTextureStreamer.cs`: Generalized texture decoding and DXT compressed packet management.
3. **`AGENTS.md` §9 Scope Fidelity & Receipts**: Each phase must be verified with automated test suites and concrete receipts before setting tasks to `[x]`.

---

## 5. Operator amendment — CASC local read path (2026-10-22, CASC-1)

**Origin:** operator: "the next thing needed is to stream data from casc a bit better, as that is
holding the CASC support back in terms of performance. Everything else is fine, once things
stabilize."

**Amendment to §4.1:** the constraint that `CascStorage.cs` remains untouched is lifted for the
**local read path only**, by this operator direction. The change is additive: the stock TACTSharp
read stays as the fallback, and no format reader or parser changes.

**Finding (measured, `evidence/casc-read-ab-azeroth-2026-10-22.txt`, `evidence/read-cost-probe/`):**
TACTSharp's local read (`CDN.TryGetLocalFile`) stats the archive, maps the whole `data.NNN` file,
creates a view over it, copies the entry, unmaps and closes - per file - and then copies the bytes a
second time. A synthetic probe measured 250-385 us per read regardless of size, with 8 threads only
~1.5x one thread, against 7-18 us for a cached handle with positional reads (4-64 KB entries).

- **FR-CASC-1**: Local CASC reads MUST return byte-identical results to the stock TACTSharp read and
  MUST fall back to it on any local miss, short read or error (CDN fill stays with the stock path).
- **FR-CASC-2**: The fast path MUST check itself: the first reads are compared with the stock path
  and a single difference MUST switch the fast path off. A documented switch
  (`PARP_CASC_FAST_READS=0`) MUST start it off.
- **FR-CASC-3**: Which path served reads, and how often a loader blocked on a read the prefetch
  cache did not hold, MUST be visible in the Perf panel.
- **AC-009**: On a real install, a file set read through both paths is byte-identical and the fast
  path is faster sequentially and in parallel (`inspect casc bench --read-ab-only`).
- **SC-CASC-1**: Operator-owned: Perf panel loader-miss count/time and streaming hitches are no worse
  than with `PARP_CASC_FAST_READS=0`, ideally better. FPS/visual claims come only from operator captures.

**Not in this amendment (proposed only, awaiting the operator's capture):** camera-distance prefetch
ordering (Phase 4 here), in-flight read coalescing, re-prefetch after cache eviction, caching
synchronous reads, and BLTE decode allocation. None is started.
