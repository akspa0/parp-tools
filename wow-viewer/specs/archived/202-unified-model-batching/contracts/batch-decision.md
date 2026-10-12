# Contract: Model Batch Decision

**Date**: 2026-09-01

In-process contract between `ModelBatchPlanner` and the renderers. Stated as behaviour so
`ModelBatchPlannerTests` can pin it headless.

## C1 — A decision never carries both a batch and a refusal

```
ModelBatchDecision is valid  ⟺  exactly one of { Batch, UnbatchableReason } is set
```

Same invariant as spec 198's `PermutationRequest` and spec 199's `AdtLayerDecodeOutcome`. It
exists for the same reason: the current design lets "batched" and "not actually fewer draws"
coexist, which is how research.md R1's conflation survived.

## C2 — The key decides, and the key is total

```
BatchKey(a) == BatchKey(b)  ⟹  a and b may share a draw
```

Any per-instance state **not** in the key must be either (a) uploaded per instance in the
instance payload, or (b) documented as deliberately ignored. A third option — state that
differs but is neither in the key nor in the payload — is the defect that makes instancing
change appearance. `OpaqueFade` moves from gate to payload in Phase 3 for exactly this reason.

## C3 — Batched means fewer draw calls

```
instanced   ⟹  one draw call per batch
stateHoisted ⟹  one draw call per instance, reduced setup
```

These are reported separately (T002). A counter that adds them together and calls the sum
"batched" is not permitted — it is the current defect and it makes the spec's own success
criteria unverifiable.

## C4 — The decision is taken in one place

```
No renderer may decide its own batchability.
```

`RequiresUnbatchedWorldRender` and `SupportsGpuInstancedOpaque` become *inputs* the planner
consults, not verdicts renderers return. Four loaders currently give four different answers to
the same question; that is the thing being fixed.

## C5 — Order is preserved where it is observable

```
Opaque:      order not observable ⟹ batching may reorder freely
Transparent: order observable     ⟹ order wins, instance stays unbatched with that reason
```

## C6 — The planner is headless

```
No member of the planner, key, or decision touches a GL context.
```

It lives in `Core.Runtime` beside `WorldObjectPassCoordinator`, so the decision is unit-testable
without a window and the defunct-app bridge can share it.

## C7 — Off reproduces the old path exactly

```
Batching disabled ⟹ output byte-identical to the pre-change renderer
```

Verified by pixel comparison (SC-006), and the reason the toggle must bypass the planner rather
than run it and discard the result.
