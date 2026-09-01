# Phase 0 Research: Unified Model Batching

**Date**: 2026-09-01
**Method**: read of the live submission path in `WorldScene`, `WorldObjectPassCoordinator`,
`IGpuInstancedModelRenderer`, `M2Renderer`; operator flight telemetry from a 5.0.1 map.

## R1 — "Batched" already means two different things

The batched callback in `WorldScene` has two arms, and the counter does not distinguish them:

```csharp
if (renderer is IGpuInstancedModelRenderer gpuRenderer
    && gpuRenderer.SupportsGpuInstancedOpaque
    && visible.OpaqueFade >= 0.999f)
{
    if (gpuBatchRenderers.Add(gpuRenderer)) gpuRenderer.BeginGpuInstanceBatch(...);
    gpuRenderer.QueueGpuInstance(visible.Instance.Transform, visible.OpaqueFade);
}
else
{
    if (immediateBatchRenderers.Add(renderer)) renderer.BeginBatch(...);
    renderer.RenderInstance(visible.Instance.Transform, RenderPass.Opaque, visible.OpaqueFade);
}
```

- **GPU instancing** — queues transforms, one draw per renderer. Fewer draw calls.
- **Immediate batch** — hoists state setup, then still issues **one draw per instance**. Same
  draw count, less per-draw setup.

Both increment `OpaqueBatchedMdxCount`. So a scene reported as "batched" may be issuing exactly
as many draw calls as an unbatched one. **This is a second metric conflation on top of the
M2/MDX one spec 201 addresses**, and it must be fixed in the same pass or the before/after for
this spec is unreadable too.

## R2 — The batch key is the renderer object, i.e. the model

`gpuBatchRenderers` and `immediateBatchRenderers` are `HashSet<>`s of renderer instances, and a
renderer is resolved per `ModelKey`. Nothing groups across models by shared state.

**Consequence — the floor.** With per-model instancing working perfectly, opaque draw calls
converge on **the number of distinct visible models**, not the number of instances. Going below
that requires batching across models by shared texture/shader state, which is a materially
larger change and is explicitly staged separately in the plan.

**Quantify it before building it.** The status bar reads `MDX 13180/20115 (429 ok/0 fail)`. If
429 is the distinct loaded-model count, then 13,180 instances collapsing to ~429 draws is a
~30x reduction and per-model instancing alone is the whole win. That reading is **not
confirmed** — T001 measures distinct visible models directly rather than inferring it from a
status-bar field whose meaning has not been checked.

## R3 — Three independent gates force unbatched, and they are not equivalent

| Gate | Meaning | Fix |
|---|---|---|
| `RequiresUnbatchedWorldRender` | route/loader has no batch path | spec 201 Phase 2 (native batch key) |
| `SupportsGpuInstancedOpaque` | renderer implements instancing | falls back to immediate batch, not to per-draw state |
| `visible.OpaqueFade >= 0.999f` | **any fading instance is forced out of instancing** | needs fade in the instance payload |

The third is easy to miss and may be large in practice: distance-faded doodads are exactly the
dense population, and every one of them drops to the immediate path. Its size is unknown and is
measured in T002.

## R4 — 0 batched / 13,180 unbatched means the batched callback never ran

The measured frame reported **zero** batched. Everything took the `renderUnbatched` arm, which
means `RequiresUnbatchedWorldRender` was true for every instance — either because the operator
had the spec 153 US3 toggle off (the documented baseline), or because every model on that map
is on a route with no batch path.

**These are indistinguishable in the current telemetry**, which is precisely why spec 201 is a
precondition rather than a nicety. T003 settles it.

## R5 — Animation is already deduplicated per model, submission is not

`ExecuteVisibleMdxAnimation` guards with `UpdatedMdxModelKeys.Add(...)` so each model animates
once per frame regardless of instance count. The same idea is simply absent on the submission
side. The pattern to follow already exists in the codebase.

## R6 — What this means for the design

1. Fix the metric first (spec 201, plus R1's batched/instanced split). Without it nothing here
   is measurable.
2. Make batchability a decision about **state**, taken in one place, rather than a property each
   renderer opts into.
3. Get per-model instancing working for every route — that is the ~30x claim to confirm.
4. Only then consider cross-model batching, which is a different and larger problem.

**Alternatives considered.** Making every renderer implement `IGpuInstancedModelRenderer`
independently — rejected: that is the current design, and it produced four different answers to
"can I batch" depending on which loader ran. A single decision point is the change.
