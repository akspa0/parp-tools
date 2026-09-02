# Tasks: Off-Thread Asset Decode

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Research**: [research.md](research.md)
| **Created**: 2026-09-01

Phases are sequential. **Phase 1 is a hard gate** — no concurrency lands before the audit exists.

Legend: `[ ]` open · `[x]` done · `[-]` in progress · **(operator)** = user-run, agent prepares the
command and stops.

---

## Phase 0 — Make the number mean something

No behaviour changes in this phase.

- [ ] **T001** Split the `DeferredAssetLoads` stage timer into **fetch**, **decode** and **upload**
      (FR-012, contract C3). Today one figure covers all three, and research R1 suspects fetch is
      already warm — so the current number cannot say what to fix.
- [ ] **T002** Attribute the two loops inside the timer separately: `ProcessPendingLoads` and
      `ProcessDeferredWmoDoodadLoads`. Research R2: each is entitled to its own unconditional
      oversized admission, so a frame can pay **two** unbounded loads inside one reported number.
- [ ] **T003** ~~Surface `OversizedAdmissionCount`, `BudgetDeferralCount`, `WorstObservedLoadMs`
      and per-kind predicted costs in the frame panel.~~ **Done 2026-09-01** — landed while
      diagnosing; these counters existed and nothing read them.
- [ ] **T004** **(operator)** Fly the baseline route. Record the fetch/decode/upload split.

**Phase 0 exit**: the 26.4–68.1 ms hitch is attributed to a phase. **If upload dominates rather
than decode**, Phase 3 is re-scoped toward upload batching before any concurrency is written — see
plan.md.

---

## Phase 1 — Thread-safety audit (GATE)

Nothing in Phase 3 starts until this is written down.

- [ ] **T101** Audit every static and shared structure on the load path. Research R5 lists the known
      hazards: `MdxRenderer`'s static shader program and uniform locations, the process-global
      `MdxTextureDiagnosticLogger` re-opened per model from a constructor, `ReplaceableTextureResolver`
      instance caches, `WarcraftNetM2Adapter` file-list access, `WorldAssetManager`'s dictionaries
      and LRU maps, `_bestSkinPathCache`.
- [ ] **T102** Record a verdict per item: **safe** / **made safe** / **confined to render thread**.
      Anything not on that list has not been audited and may not be touched off-thread.
- [ ] **T103** Fix `MdxTextureDiagnosticLogger`. A process-global `StreamWriter` re-opened per model
      from a renderer constructor cannot survive concurrent construction; its lock protects writes,
      not the Initialize/Close lifecycle.
- [ ] **T104** Verify `MpqDataSource`'s read cache and prefetch workers are safe for the additional
      concurrency (spec Assumptions says verified, not assumed).

**Phase 1 exit**: a written verdict list. No concurrency before it exists.

---

## Phase 2 — Split construction, still single-threaded

- [ ] **T201** Add `DecodedAssetPayload` — CPU arrays only, no GL handle (contract C2).
- [ ] **T202** Split `MdxRenderer` construction into a parse phase producing a payload and an upload
      phase consuming it. The constructor currently calls `InitShaders`, `InitBuffers` and
      `LoadTextures` inline; `InitShaders` touches static GL state and stays on the render thread.
- [ ] **T203** Same split for `M2Renderer` (native and legacy-backed) and `WmoRenderer`.
- [ ] **T204** Unit-test the seam headless: a payload can be built with no GL context present.
- [ ] **T205** **(operator)** Fly the route. Frame time and appearance **unchanged** — this phase
      moves a seam, it does not cross it. A difference here means the split is wrong, and finding
      that out while everything is still single-threaded is the point.

---

## Phase 3 — Cross the seam

- [ ] **T301** Add `AssetLoadRequest` (key, kind, priority by camera distance, cancellation) and
      `AssetLoadPipeline` in `Core.Runtime` — headless, unit-testable.
- [ ] **T302** Bounded worker pool running fetch + decode (FR-010). Render thread drains an upload
      queue.
- [ ] **T303** Implement contract C5 (cancellation), C6 (invisible until uploaded), C7 (one request
      per key in flight, failures recorded once).
- [ ] **T304** Runtime toggle (FR-009) that bypasses the worker stage and runs inline, per contract
      C9.
- [ ] **T305** Debug-build assertion that no GL call occurs off the render thread (SC-005, C1).
- [ ] **T306** Unit-test the pipeline: coalescing, cancellation before and after decode, failure
      recorded once, deterministic drain.
- [ ] **T307** **(operator)** Fly the route. SC-001: no `DeferredAssetLoads` stage exceeds the budget
      by more than one upload. SC-005 holds.

---

## Phase 4 — Retire the oversized admission

- [ ] **T401** Re-point `DeferredLoadBudget` at upload cost (contract C4).
- [ ] **T402** **Remove** the unconditional first-load admission, now that upload is bounded. Confirm
      `OversizedAdmissionCount` stays at zero under load (SC-002) — that counter going quiet is the
      proof the residual is actually paid off, not merely rarer.
- [ ] **T403** Fix the research R3 throttle. Clamping `maxLoads` on a slow previous frame cut
      throughput ~6x while leaving the hitch untouched, because the first load was unconditional.
      Once that is gone the throttle needs to mean something or go.
- [ ] **T404** **(operator)** SC-003: cold-cache time to a stable resident count, at a fixed camera
      position, at least 5x faster.

---

## Phase 5 — Textures out of the draw pass

- [ ] **T501** Remove `ProcessDeferredTextureLoads()` from `MdxRenderer.RenderGeosets`. Research R6:
      texture decode and upload currently run **inside submission**, billed to `MdxOpaqueSubmission`,
      governed by no budget.
- [ ] **T502** Move texture decode to the worker stage and texture upload to the budgeted queue.
- [ ] **T503** **(operator)** Confirm no file read or image decode occurs inside a submission stage,
      and that `MdxOpaqueSubmission` falls accordingly (US3, SC-007).

---

## Phase 6 — Prefetch follows the parse

- [ ] **T601** Request WMO group files once the root is parsed (research R4).
- [ ] **T602** Request BLP textures once the texture list is known.
- [ ] **T603** **(operator)** Compare prefetch hit/miss for groups and textures against Phase 0.

---

## Phase 7 — Era coverage and determinism

- [ ] **T701** **(operator)** Fly one map per era — 0.5.3, LK, Cata, MoP — recording the three
      timings and frame distribution. SC-004.
- [ ] **T702** SC-006: a reference scene is pixel-identical to the synchronous path once fully
      resident.
- [ ] **T703** Confirm capture and harvest paths still complete deterministically via the drain
      (FR-011, contract C8).
- [ ] **T704** Verify contract C9: with the toggle off, output matches the pre-change loader.

---

## Notes

- **Do not tune the budget as a substitute.** Research R2: the budget cannot subdivide a single
  synchronous load, and `DeferredLoadBudget`'s own documentation says so. That road has been walked.
- **The operator's premise was correct.** The SSD and the MPQ reader are not implicated — two
  prefetch workers already exist and root bytes are usually warm. Everything after the read is what
  never left the render thread.
- **This supersedes Spec 153 Phase 5 step 2**, which named this work and deferred it.
- Baseline to beat, 2048-frame MoP flight: median **75.26 ms**, p95 **144.59**, p99 **235.37**, max
  **541.65**, **2047/2048** frames over 33.3 ms; `DeferredAssetLoads` hitches **26.4–68.1 ms**.
- Known test baseline: `WowViewer.Core.Tests` has **9 pre-existing failures**. Compare to 9.
- The viewer holds the built exe open while running; a build during a live session will fail with
  MSB3027. Close it before rebuilding.
