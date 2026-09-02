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

## R7 — The toggle was not the cause, and R4 is settled without a flight

`WorldScene.MdxOpaqueBatchingEnabled` is declared `{ get; set; } = true`. So the recorded
`0 batched / 13,180 unbatched` was **not** the spec 153 US3 toggle being off. Every instance
took the unbatched arm because `RequiresUnbatchedWorldRender` was true for all of them.

R4 called these two possibilities indistinguishable in the telemetry. They are distinguishable
in the source, and the source answers it. The T005 flight is no longer needed for this question.

## R8 — GPU instancing was unreachable, and would have been wrong if reached

Two facts, both in the source:

```csharp
// MdxRenderer
public bool SupportsGpuInstancedOpaque => false;      // hardcoded

// M2Renderer — delegates to the inner MdxRenderer, i.e. to false
public bool SupportsGpuInstancedOpaque
    => _legacyRenderer is IGpuInstancedModelRenderer gpu && gpu.SupportsGpuInstancedOpaque;
```

**No model renderer in the codebase could return true.** `QueueGpuInstance` was dead code on the
world path, so the `instanced` count was provably always zero and "batched" has only ever meant
"state-hoisted" — R1's conflation was not merely dangerous in principle, it was total in fact.

And the reason it was held out was sound. The CPU side was complete: instance VBO, attribute
pointers at locations 6–10, `VertexAttribDivisor`, `DrawElementsInstanced`, bone upload. **The
vertex shader declared only locations 0–5.** It never read the instance data, and
`EndGpuInstanceBatch` sets `uModel` to identity, so enabling the flag as it stood would have
drawn every doodad in the scene stacked on the world origin. The comment — *"held out until it
has portable compile and visual parity proof"* — was accurate, and the missing piece was the
shader, not the flag.

## R9 — The native-M2 batch path would save nothing today

`M2Renderer.RenderInstance` and `M2Renderer.RenderWithTransform` both call the same `RenderCore`,
and `RenderCore` re-uploads all ten shared uniforms — view, projection, three fog values, camera
position, light direction, light colour, ambient — on **every call**. `BeginBatch` stores the
same values in fields and changes nothing about the work done per instance.

So spec 201 US3 / spec 202 T301 — flipping `RequiresUnbatchedWorldRender` for native-route M2 —
is **churn until the shared uniforms are hoisted out of `RenderCore` first**. The ordering matters
and was not visible from the interface: a renderer can implement the batch contract without the
batch contract buying anything.

## R10 — Local MDX lights are per-instance state, and are not in the payload

`RenderInstance` calls `UploadMdxLights(modelMatrix)`, which transforms each light's pivot into
world space **per instance**. `BeginGpuInstanceBatch` uploads them once with an identity matrix.
A batched lamp or brazier would therefore light every copy as though it stood at the world origin.

This is contract C2's failure mode precisely: state that differs per instance, is not in the
batch key, and is not in the instance payload. Resolved by keeping models with
`_mdx.Lights.Count > 0` out of instancing — the conservative half of C2, chosen because the
population is small and the artefact would be conspicuous.

## R11 — Every M2 uses the native renderer, so nothing M2 can batch

Measured from an operator flight after R8's shader fix landed: the panel still reported
`route requires unbatched render` for every opaque M2, with `instanced 0`.

```csharp
public static bool PreferNativeStaticRenderer
{
    get
    {
        string? value = Environment.GetEnvironmentVariable(NativeRendererSettingName);
        if (string.IsNullOrWhiteSpace(value))
            return true;          // <-- unset means TRUE
        ...
    }
}

public static bool ShouldUseNativeStaticRenderer(MdxFile? adaptedMdx)
    => adaptedMdx == null || PreferNativeStaticRenderer;
```

The environment variable is normally unset, so `ShouldUseNativeStaticRenderer` is **always true** and
`WowViewerM2RuntimeBridge.CreateRenderer` always returns the native-only
`M2Renderer(gl, runtimeModel, ...)` — the constructor that leaves `_legacyRenderer` null. And:

```csharp
public bool RequiresUnbatchedWorldRender => _legacyRenderer is null || ...;
```

So **every M2 in the world is unconditionally unbatchable**, regardless of route decision, regardless
of the toggles, and regardless of R8's shader work. The route decision still records `AdapterSkin`,
which is why the panel's per-path breakdown attributes them there — the *route* was AdapterSkin, the
*renderer* is native.

**This is the same trap as R5's two decoders.** A fix landed in `MdxRenderer`, which M2 content never
constructs. Before concluding a renderer fix did nothing, check which renderer the path under test
actually builds.

## R6 — What this means for the design

1. Fix the metric first (spec 201, plus R1's batched/instanced split). Without it nothing here
   is measurable. **Done** — and it was the metric work that made R7–R10 worth looking for.
2. Make batchability a decision about **state**, taken in one place, rather than a property each
   renderer opts into.
3. Get per-model instancing working for every route — that is the ~30x claim to confirm.
4. Only then consider cross-model batching, which is a different and larger problem.

**Alternatives considered.** Making every renderer implement `IGpuInstancedModelRenderer`
independently — rejected: that is the current design, and it produced four different answers to
"can I batch" depending on which loader ran. A single decision point is the change.
