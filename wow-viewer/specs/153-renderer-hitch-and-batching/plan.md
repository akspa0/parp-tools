# Implementation Plan: Renderer hitch elimination and MDX batching restoration

**Branch**: `v0.5.3-dev` | **Date**: 2026-08-15 | **Spec**: [spec.md](spec.md)

## Summary

Diagnosis is done. Four defects are identified with measurements; this plan fixes them in order of
confidence and risk. The detector from Spec 152 is in place, so every step here has a before/after.

Ordering principle: **name it, then fix it.** Phase 0 exists because one defect is localised to a
pass but not yet to a call, and guessing at that boundary is how `MdxVisibility` at 0.2 ms got
blamed for a 350 ms frame.

## Technical Context

**Language/Version**: C# / .NET 10, Silk.NET.OpenGL

**Measurement**: in-viewer frame history (Utilities > Perf), peaks and per-stage max ordering

**Benchmark scene**: Stranglethorn Vale, Alpha 0.5.3 Azeroth. Thousands of objects per view; it
exposed both defects where Un'goro did not.

**Baseline to beat** (Stranglethorn, 2026-08-15):

| Metric | Baseline |
| --- | --- |
| median frame | 22.05 ms |
| p99 frame | 259.70 ms |
| max frame | 707.23 ms |
| hitches | 117 / 584 frames (20%) |
| pass gap peak | 259.04 ms (314.15 in South Seas) |
| MDX opaque batched | 0 of 312 |
| SceneMaintenance max | 454.8 ms |
| DeferredAssetLoads max | 58.1 ms (budget 3.5) |

## Constitution Check

| Principle | Status | Note |
| --- | --- | --- |
| III. Real-Data Validation | PASS | All measurement against staged real clients. |
| VI. No Client Path Assumptions | PASS | Client roots stay configuration. |
| One Phase at a Time | PASS | Phase gates below. |
| Bite-Sized Plans | PASS | Four independent defects, each landing separately. |

---

## Phase 0 — Name the periodic stall

**Gate to enter**: none. Sub-probes are already landed and unread.

**Work**

1. Capture Stranglethorn with the sub-probes: `PrepareObjectPhase total`, `AudioRuntime.Update`,
   `PM4 overlay window`.
2. If one peak matches the ~212 ms hitch magnitude, it is named — proceed to Phase 1.
3. If none match, subdivide `PrepareObjectPhase` further (frustum update, `GetChunkInfoAt`, GL state)
   before proposing any fix.

**Exit**: the stall has an operation name, not a region name.

**Deliverable**: `research.md` recording the capture.

---

## Phase 1 — Close the instrumentation hole (FR-001)

**Gate**: none; independent and safe.

`WorldFramePasses` has eleven passes and ten timers. `PrepareObjectPhase` has none, which is why a
200 ms stall could hide in plain sight for the whole investigation.

**Work**

1. Give `PrepareObjectPhase` a stage timer like every other pass.
2. Add it to `WorldRenderStage`, the stats struct, and the history.
3. Assert in a test that the number of passes equals the number of stage timers, so a future pass
   cannot be added without one.

**Exit**: unaccounted time in steady state approaches zero; no pass can hide cost by construction.

---

## Phase 2 — Move periodic residency work off the render thread (US2)

**Gate**: Phase 0 named the operation.

**Work depends on what Phase 0 names.** Likely shapes:

- **If PM4 overlay window loading**: make the window change enqueue work rather than load
  synchronously; overlays become resident a few frames later instead of stalling the frame.
- **If audio residency**: bound the per-frame work and spread emitter resolution across frames.
- **Either way**: the operation keeps its correctness contract — content still becomes resident, just
  not synchronously inside `Render`.

**Exit**: the periodic hitch pattern is *absent* from the frame history on the Stranglethorn route
(SC-001), not merely reduced. Verified by before/after capture.

**Revert**: single call-site change behind a switch.

---

## Phase 3 — Restore MDX batching (US3)

**Gate**: independent of Phases 0–2; may run in parallel.

100% of opaque MDX submit unbatched while WMO batches cleanly at 198/198. Something in the MDX path
either never reaches the instanced route or fails its capability check.

**Work**

1. Find where the opaque MDX path chooses batched vs unbatched. WMO's working path is the reference —
   the two diverged, and the diff is the bug.
2. Determine whether the MDX renderer implements `IGpuInstancedModelRenderer`, whether the check is
   failing, or whether the planner never groups by model.
3. Fix one cause at a time, measuring `OpaqueBatchedMdxCount` after each.
4. Keep a clean per-instance fallback with a visible count for models that genuinely cannot instance.
5. Confirm visual equivalence before accepting (FR-006).

**Exit**: majority of opaque MDX batched (SC-002); `MdxOpaqueSubmission` p99 beats baseline by more
than the noise floor (SC-003).

**Note**: this is the sustained cost, not the periodic stall. It will not remove the gallop on its
own, and it must not be credited with doing so.

---

## Phase 4 — Bound scene maintenance (US4)

**Gate**: Phases 2–3 measured, so this is evaluated against a cleaner baseline.

`SceneMaintenance` median 0.02 ms, max 454.8 ms. `RebuildInstanceLists` /
`RebuildSceneGraphObjectIndex` run in full when `_instancesDirty`, which streaming sets often.

**Work**: make the rebuild incremental over changed tiles, or budget it across frames.

**Exit**: `SceneMaintenance` max below the hitch threshold (SC-004).

---

## Phase 5 — Enforce the deferred load budget (US5)

**Gate**: independent.

`ProcessPendingLoads` checks its budget only *between* loads, so one long load runs to completion —
observed 58.1 ms against a 3.5 ms budget. It also calls `LoadMdxModel` synchronously on the render
thread.

**Work**

1. Check remaining budget before starting each load *and* bound what a single load may do in one
   frame.
2. Longer term, move decode off the render thread entirely; the budget then governs upload only.

**Exit**: `DeferredAssetLoads` max within the same order as its budget (SC-005).

---

## Sequencing

| Phase | Depends on | Parallel? |
| --- | --- | --- |
| 0 Name the stall | — | start now |
| 1 Close instrumentation hole | — | yes, independent |
| 2 Move residency off render thread | 0 | no |
| 3 Restore MDX batching | — | **yes, independent** |
| 4 Bound scene maintenance | 2, 3 measured | no |
| 5 Enforce load budget | — | yes, independent |

Phases 0, 1, 3 and 5 can all start immediately. Phase 3 is the largest user-visible win in dense
scenes; Phase 2 is the one that removes the gallop.

## Risks

| Risk | Mitigation |
| --- | --- |
| Phase 0 names nothing and we guess | Explicit exit criterion: subdivide further rather than propose a fix |
| Batching changes visual output | Visual equivalence is an acceptance gate, with a clean fallback path |
| Deferring residency makes content pop in late | Correctness contract preserved; lateness is measured, not assumed acceptable |
| Fixing the sustained cost is mistaken for fixing the gallop | They are separate defects with separate success criteria; SC-001 and SC-003 cannot substitute for each other |
| A quiet zone hides a regression | Stranglethorn is the standard benchmark for every before/after |

## Relationship to Spec 152

Spec 152 keeps the measurement infrastructure (frame history, hitch detection, unaccounted time,
region probes) and the independent per-era terrain lighting work.

**Spec 152 Phases 3–4 (flatten the scene graph into retained draw lists) are suspended.** They were
justified by an allocation-churn premise that the measurements refuted: median world-render CPU is
sub-millisecond to single-digit, and the hitches are neither allocation-shaped nor in the traversal.
The churn fixes already landed were worth doing and are kept. Flattening may return later if evidence
supports it, but not on the original reasoning.
