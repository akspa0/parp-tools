# Ghidra Performance Report: Why WoW Alpha Engine Is Fast (and Why C# Ports Can Be 1–5 FPS)

Date: 2026-02-13  
Binary: `WoWClient.exe` (Build 3368)

---

## Executive Summary

The original engine is fast not because each draw is “cheap,” but because it is built around:

1. **Aggressive culling before draw submission**
2. **Intrusive-list, low-allocation data flow**
3. **Stable render pass ordering and bounded hot loops**
4. **Chunk-path bifurcation (simple static vs complex dynamic)**
5. **Asynchronous streaming infrastructure (but with explicit wait points)**

Your C# 1–5 FPS result is consistent with one or more of these port mismatches:

- Blocking waits on update/render thread
- Excessive GC/allocation churn in cull/render lists
- State-change spam (shader/texture/blend) per layer/draw
- Frequent dynamic mesh rebuilds for chunk seams/holes
- Tiny lock/draw/unlock bursts instead of aggregated submissions

---

## 1) High-Impact Function Map (Ghidra)

## Frame orchestration
- `0x0066A9D0` `CWorldScene::Render`
  - Top-level render loop; sets frame states, culls, executes ordered passes.

## Visibility and sort
- `0x0066CA50` `CWorldScene::CullSortTable`
- `0x0066D3F0` `CWorldScene::CullChunks`
- `0x0066CDC0` `CWorldScene::CullEntitys`
- `0x0066CF50` `CWorldScene::CullDoodads`
- `0x0066E850` `CWorldScene::CullMapObjDefs`

## Terrain draw
- `0x0066DE50` `CWorldScene::RenderChunks`
- `0x006A5910` `CMapChunk::Render`
- `0x006A5D80` `CMapChunk::RenderLayers` (static path)
- `0x006A64B0` `RenderLayersDyn` (dynamic path)

## Streaming / I/O
- `0x0067F7F0` `CMap::Load`
- `0x006847F0` `CMap::PrepareChunks`
- `0x00477220` `AsyncFileReadWait`
- `0x004773D0` `AsyncFileReadWaitAll`
- `0x0063CB70` `SFile::Open`, `0x0063D6D0` `SFile::Read`, `0x0063DD20` seek

## GX submission and state
- `0x0058E150` `GxRsSet`
- `GxBufLock` / `GxBufRender` / `GxBufUnlock` wrappers around device vtable calls

---

## 2) What the Engine Optimizes Well

## A) Culling before expensive work
- `CullSortTable` fans out over fixed bucket count (`0x1A`) and performs cull passes before render.
- `CullChunks` performs early rejects (marker checks, frustum, clip-buffer), then only accepted chunks advance.
- This keeps heavy terrain/layer paths bounded by visible subset.

## B) Low-allocation runtime structures
- Uses intrusive linked lists heavily in visible/update paths (pointer relinking, not container rebuild allocations).
- Sort-table clearing and relinking is branchy but avoids heap churn.
- This pattern minimizes allocator overhead and cache-unfriendly object churn.

## C) Terrain path split keeps steady-state cheap
- `CMapChunk::Render` selects:
  - **Simple static path** (reuses static gx buffer)
  - **Complex dynamic path** only for holes/seam/LOD stitch cases
- Fast case remains static if LOD remap doesn’t change.

## D) Fixed pipeline ordering and scope-limited state
- Render pass order in `Render` is deterministic.
- Uses push/pop patterns for transform/state scope.
- Keeps global-state contamination lower and debugability higher.

## E) Data-driven layer reduction at distance
- Layer rendering can reduce active tested layers by distance (`nLayersTest` behavior).
- Optional effects (terrain shader/shadow branches) are conditional.

---

## 3) Where the Original Engine Still Stalls (and why this matters to your port)

## A) Blocking waits are real
- `AsyncFileReadWait` and `WaitAll` are polling loops with `OsSleep(1)`.
- If your C# port does equivalent waits on frame thread, FPS collapses.

## B) State-change heavy terrain layer loops
- `RenderLayers` / `RenderLayersDyn` do many per-layer state/texture/shader decisions.
- If port does these without sorting/state cache, CPU overhead explodes.

## C) Dynamic path rebuild churn
- Frequent transitions to dynamic seam/hole path can trigger repeated rebuilds/alloc/free patterns.
- In managed code, this becomes significantly worse with allocations and bound checks unless carefully designed.

---

## 4) Why C# Can Be 1–5 FPS Here (Likely Root Causes)

Most likely causes (ranked by impact):

1. **Main-thread blocking on chunk/asset readiness**
2. **Per-frame allocations in cull lists / draw commands / temporary arrays**
3. **No material/state sort => too many backend state changes**
4. **Per-chunk tiny dynamic buffer lock/unlock cycles**
5. **Rebuilding seam/hole index data too often**
6. **AoS + scattered object traversal in hot cull loops**
7. **Debug/validation overhead left enabled in hot loops**

---

## 5) Porting Strategy (Concrete, Prioritized)

## Priority 1 — Remove frame-thread stalls
- Never block render/update on streaming completion.
- Use chunk lifecycle states: `Requested -> IOReady -> CPUReady -> GPUReady -> Visible`.
- Promote states asynchronously; render only `GPUReady` chunks.

## Priority 2 — Kill allocation churn
- Pool visible lists, command buffers, temporary cull arrays (`ArrayPool<T>` or custom pools).
- Replace per-frame `List<T>` growth with fixed-capacity frame arenas.
- Avoid LINQ and iterator allocations in cull/render hot paths.

## Priority 3 — State batching and dirty-state cache
- Build a render command key (shader/material/blend/texture set).
- Sort commands by key before submission.
- Track current GPU state and skip redundant sets.

## Priority 4 — Stable terrain mesh cache
- Cache seam/hole index buffers by neighbor LOD mask + hole mask.
- Rebuild only on actual topology change, not every frame.
- Separate immutable index data from mutable vertex data.

## Priority 5 — Buffer strategy
- Use ring/persistent mapped buffers for dynamic terrain writes.
- Aggregate many chunk/layer submissions per lock window.
- Avoid lock/unlock around tiny draws.

## Priority 6 — Culling data layout
- Move hot cull data to SoA (bounds, flags, links, distance) for cache/SIMD friendliness.
- Keep heavy object payloads out of cull-critical arrays.

---

## 6) Immediate Diagnostic Checklist (for your project)

Measure these first:

- Frame time split: `Culling`, `Chunk Build`, `Render Submit`, `GPU`.
- Count per frame:
  - visible chunks
  - chunk topology rebuilds
  - state changes (`shader/texture/blend`) 
  - dynamic buffer lock calls
  - allocations + GC pauses
  - streaming waits on frame thread

If any of these are high, they directly explain 1–5 FPS.

---

## 7) Suggested “Parity Baseline” Mode

Implement a temporary baseline mode to isolate architecture issues:

- disable dynamic seam/hole path (force static if possible)
- disable shadows/effects layers
- force single terrain layer
- disable object/doodad passes
- no streaming waits (render only ready chunks)

If FPS jumps massively, problem is not raw language speed — it is pipeline architecture mismatch.

---

## 8) Address Reference

- `CWorldScene::Render` `0x0066A9D0`
- `RenderChunks` `0x0066DE50`
- `CullSortTable` `0x0066CA50`
- `CullChunks` `0x0066D3F0`
- `CMap::PrepareChunks` `0x006847F0`
- `AsyncFileReadWait` `0x00477220`
- `AsyncFileReadWaitAll` `0x004773D0`
- `CMapChunk::Render` `0x006A5910`
- `CMapChunk::RenderLayers` `0x006A5D80`
- `RenderLayersDyn` `0x006A64B0`
- `GxRsSet` `0x0058E150`

---

## 9) Final Takeaway

Your FPS result does make sense if the port diverged from the original engine’s **data flow and frame contract**. The binary indicates performance came from predictable hot loops, strict culling, low allocation, and controlled submission patterns — not from any single “magic optimization.”

If you want next, I can create a second document that is a literal C# subsystem blueprint (`ChunkScheduler`, `VisibilitySystem`, `TerrainSubmissionQueue`, `GpuBufferRing`) with APIs and thread boundaries.