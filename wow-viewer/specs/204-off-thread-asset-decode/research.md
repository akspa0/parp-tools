# Phase 0 Research: Off-Thread Asset Decode

**Date**: 2026-09-01

**Method**: read of the live load path — `WorldScene.ProcessDeferredAssetLoads`,
`WorldAssetManager.ProcessPendingLoads` / `ProcessDeferredWmoDoodadLoads` / `PrefetchModelBytes`,
`DeferredLoadBudget`, `MpqDataSource`, `MdxRenderer` — plus a 2048-frame operator flight over
Mogu Ruins / Dread Wastes on MoP 5.0.1.

Everything below is read out of the source or the flight, not inferred from symptoms.

## R1 — The I/O is already off-thread; the decode is not

`MpqDataSource` runs `PrefetchWorkerCount = 2` background workers draining a
`ConcurrentQueue<PrefetchRequest>`, and `WorldAssetManager.QueueMdxLoad` / `QueueWmoLoad` both call
`PrefetchModelBytes` at enqueue time. So by the time a load is dequeued, its **root** bytes are
usually already in the read cache.

`WorldAssetManager` itself contains **no threading at all** — no `Task`, no `Thread`, no
`Parallel`, no concurrent collection. Every parse, adaptation, image decode and GL upload runs
inside `ProcessDeferredAssetLoads`, on the render thread, inside the frame.

**This is why the operator's framing ("despite the data being served from an SSD, via our fast MPQ
reader code") is correct and the symptom persists anyway.** The reader is not the bottleneck and
was never going to be. Nothing after the read has ever left the render thread.

## R2 — The budget admits one unbounded load per frame, on purpose

```csharp
// DeferredLoadBudget.CanStartAnotherLoad
if (loadsStartedThisFrame <= 0)
{
    if (CheapestPredictedCostMs() > budgetMs)
        OversizedAdmissionCount++;
    return true;          // guaranteed progress
}
```

Without this an asset costlier than the whole budget would never become resident. With it, a 68 ms
model produces a 68 ms frame regardless of budget. The class documents this precisely and names the
fix:

> *"Removing that residual requires moving decode off the render thread so the budget governs upload
> alone — Spec 153 Phase 5 step 2, deliberately not attempted here."*

and calls `OversizedAdmissionCount` *"the honest measure of the residual the off-thread decode still
owes."*

**`ProcessDeferredAssetLoads` then calls two such loops per frame** — `ProcessPendingLoads(...)` and
`_assets.ProcessDeferredWmoDoodadLoads()` (with its defaults, `maxLoads: 1, maxBudgetMs: 2.0`) —
each entitled to its own unconditional first load. Up to **two** unbounded loads per frame, both
inside one timer.

## R3 — The CPU throttle costs throughput and buys nothing

```csharp
double previousFrameCpuMs = LastRenderFrameStats.TotalCpuMs;
if (previousFrameCpuMs >= 33.0)      { maxLoads = Math.Min(maxLoads, 1);  maxBudgetMs = Math.Min(maxBudgetMs, 1.0); }
else if (previousFrameCpuMs >= 20.0) { maxLoads = Math.Min(maxLoads, 2);  maxBudgetMs = Math.Min(maxBudgetMs, 1.5); }
```

Because the first load is admitted unconditionally (R2), clamping `maxLoads` to 1 **does not reduce
the hitch** — the frame still pays that load in full. It only removes the cheap loads that would
have followed. The throttle therefore cuts streaming throughput roughly six-fold while leaving the
worst-case frame cost untouched.

And it is self-reinforcing: slow frames throttle loading, oversized loads keep frames slow. At the
measured ~10 FPS this yields on the order of 10 models/second against 25,684 placements, which is
the "very slow to render objects in" the operator has reported twice.

## R4 — Prefetch covers root files only

`PrefetchModelBytes` requests the canonical model path, an alternate path, and skin candidates.
Everything else an asset needs is discovered **after** the root is parsed:

- WMO **group** files (`_000.wmo`, …)
- every BLP texture
- skins beyond the resolved candidate

Those reads are synchronous inside the load today. Moving decode off-thread converts them from a
frame cost into a worker cost, which is already a win — but they stay serialised behind the root
parse until prefetch follows the parse.

## R5 — Thread-safety of the parse layer is unestablished, and there are known hazards

This is the **primary risk** of the whole change, and it must be settled before anything runs
concurrently. Confirmed hazards found by inspection:

| Hazard | Location | Nature |
|---|---|---|
| Static GL shader program, uniform locations, `_shaderInitialized`, `_shaderRefCount` | `MdxRenderer` | GL objects on static fields. `InitShaders` **must** stay on the render thread. |
| `MdxTextureDiagnosticLogger.Initialize(mdxName)` / `.Close()` | `ModelRenderer.cs:16` | A **process-global** `StreamWriter` re-opened per model from the renderer constructor. Concurrent construction re-enters this lifecycle. It has a lock, but the lock does not make a per-model global writer correct. |
| `MdxRenderer` constructor | `ModelRenderer.cs` | Does `InitShaders()`, `InitBuffers()`, `LoadTextures()` inline — i.e. GL work in a constructor. The constructor cannot run on a worker as it stands. |
| `ReplaceableTextureResolver` | `Rendering/` | Static lookup tables are read-only and safe; **instance** caches are not audited. |
| `WarcraftNetM2Adapter` | `Rendering/` | Static methods appear pure; `FindSkinInFileList` and file-list access are not audited. |
| `WorldAssetManager` dictionaries + LRU | `Terrain/` | `_mdxModels`, `_wmoModels`, LRU maps, `_bestSkinPathCache` — all plain, all mutated during load. |

**Do not assume any of these are safe.** The audit is Phase 1 and is the gate on Phase 2.

## R6 — A second, unbudgeted load path hides inside the draw pass

`MdxRenderer.RenderGeosets` begins with `ProcessDeferredTextureLoads()`. Texture decode and upload
therefore happen **inside submission**, are billed to `MdxOpaqueSubmission` rather than
`DeferredAssetLoads`, and are governed by **no budget at all**.

This is invisible in the current attribution and will still be there after US1 is delivered. It is
US3 for that reason, not a footnote.

## R7 — What can and cannot move

The viewer holds a single GL context on one thread.

| Phase | Thread | Work |
|---|---|---|
| **Fetch** | worker (exists) | MPQ read + decompress, for *every* file the asset needs (R4) |
| **Decode** | worker (new) | model parse, M2→runtime adaptation, skin resolve, BLP→pixels, vertex/index array build |
| **Upload** | render | `glGen*`, `glBufferData`, `glTexImage2D`, VAO setup, shader program binding |

The upload phase is small and bounded, which is what lets the frame budget govern it honestly and
lets R2's unconditional oversized admission be retired.

## R8 — What this means for the design

1. Split renderer construction into a CPU-parse phase and a GPU-upload phase. This is the bulk of
   the work and the reason it is not a small change: the constructors currently interleave both.
2. Audit and fix the R5 hazards **before** running any of it concurrently.
3. Re-point `DeferredLoadBudget` at upload cost, then remove the unconditional oversized admission
   and confirm `OversizedAdmissionCount` stops rising.
4. Fix the R3 throttle once the hitch is gone — it is only defensible while a single load can cost
   68 ms.
5. Extend prefetch to follow the parse (R4), and pull texture loading out of the draw pass (R6).

**Alternatives considered.** *A second GL context with shared lists*, letting workers upload
directly — rejected: driver-dependent, historically fragile across vendors, and it would not remove
the decode cost that is the actual problem. *Tuning the existing budget* — rejected by R2: the
budget cannot subdivide one synchronous load, and its own author wrote that down.
