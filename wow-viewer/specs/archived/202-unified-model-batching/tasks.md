# Tasks: Unified Model Batching

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Created**: 2026-09-01

Phases are strictly sequential and each ends in a **number**. Do not start a phase before the
previous phase's measurement exists — plan.md records why, and spec 153 US3 is the cautionary
example of building against an ambiguous baseline.

Legend: `[ ]` open · `[x]` done · `[-]` in progress · **(operator)** = user-run, agent prepares
the command and stops.

---

## Phase 0 — Make the numbers mean something

No rendering behaviour changes in this phase.

- [x] **T001** Land spec 201 Phase 1 (per-render-path attribution). This spec consumes it; do
      not duplicate the counters.
- [x] **T002** Split the `batched` counter into **instanced** (one draw per renderer, via
      `QueueGpuInstance`) and **state-hoisted** (`BeginBatch` + per-instance
      `RenderInstance`). research.md R1: these are counted identically today, and the second
      does **not** reduce draw calls.
- [x] **T003** Report actual **draw calls** issued for opaque models, separately from instance
      count. Draw calls are the thing being optimised; instances are not.
- [x] **T004** Report which of the three gates (research.md R3) forced each unbatched
      instance: `RequiresUnbatchedWorldRender`, `!SupportsGpuInstancedOpaque`, or
      `OpaqueFade < 0.999`.
- [x] **T005** ~~Re-fly with the toggle on, then off, to resolve R4's ambiguity.~~
      **Answered from source instead (R7)**: `MdxOpaqueBatchingEnabled` is declared `= true`, so
      the recorded `0 batched / 13,180 unbatched` was *not* the toggle. No flight needed for
      this question.

**Phase 0 exit**: batched/instanced/state-hoisted/unbatched/unbatchable are five distinct
numbers, draw calls are reported, and every unbatched instance has a named gate.

**T001-T004 landed 2026-09-01.** `ModelSubmissionAccounting.cs` in `Core.Runtime` carries the
decomposition; `WorldScene` records one outcome plus one gate per submitted instance; the frame
panel's "Submission efficiency" node reports it. The old aggregate pair is kept and shown
greyed-out as the FR-005 sum check, not as a performance figure.

**Draw calls are counted at the GL call sites**, not inferred. `ModelDrawCallCounter.Record()`
sits at the four model draw sites (three in `ModelRenderer`, one in `M2Renderer`) and the pass
brackets it with `Since()`. This is not an implementation preference: a model draws once per
geoset (legacy MDX) or once per section (native M2), so no arithmetic over the instance counters
can produce the draw-call number, and the panel's new `per instance` column is the figure that
says whether batching bought anything at all.

---

## Phase 1 — Measure the ceiling before building toward it

- [x] **T101** Count **distinct visible models** per frame. This is the floor per-model
      instancing can reach (research.md R2).
- [x] **T102** Confirm or refute that the status bar's `429` in `MDX 13180/20115 (429 ok/0 fail)`
      is the distinct loaded-model count. **Do not infer it** — the field's meaning has not
      been checked, and this project has a history of reading meaning into unverified fields.
- [-] **T103** Measure the share of instances blocked by each of the three gates, so the
      largest is known. The fade gate is a live suspicion: distance-faded doodads are exactly
      the dense population and every one drops out of instancing today.
- [ ] **T104** **(operator)** Fly the route and record T101–T103. Compute the projected
      reduction: `13,180 instances → N distinct models`.

**T101 landed** as `DistinctModelCount` on each pass tally — distinct model keys actually
submitted, which is exactly the floor rather than a proxy for it. Folded into Phase 0's code so
T005 and T104 can be the **same single flight**; the gate this phase enforces is on acting
on the number, and none of Phase 3 has been touched.

**T102 answered by reading the code, and the answer is a qualified yes.** The status bar's `429`
is `WorldAssetManager.MdxModelsLoaded`, defined as `_mdxModels.Count(kv => kv.Value != null)` —
so it is a **distinct-model count**, confirming R2 in kind. But it counts distinct models
**resident in the asset manager**, not distinct models **visible this frame**, so it is an upper
bound on the floor, not the floor. The ~30x figure in research.md stands only as a ceiling on the
prize. `DistinctModelCount` measures the real thing. The same read disposes of the `13180/20115`
note: those are `VisibleMdxCount`/`MdxInstanceCount`, so the ~7,000 gap is **culled placements**,
not lost ones.

**Phase 1 exit**: the theoretical floor is a number and the biggest gate is named. **If the
floor is close to 13,180** — i.e. instances are nearly all distinct models — then per-model
instancing is not the answer and Phase 3 is re-scoped toward cross-model batching before any
code is written.

---

## Phase 2 — One decision point, behaviour unchanged

- [ ] **T201** Add `ModelBatchKey` — the render state that makes two draws combinable.
      Everything excluded from the key must be justified in a comment (FR-002); unlisted
      per-instance state is how instancing changes appearance.
- [ ] **T202** Add `ModelBatchDecision`: batched / unbatched-by-state / unbatchable+reason,
      with the same mutual-exclusion invariant used in specs 198 and 199 — no decision may
      carry both a batch and a reason it could not batch.
- [ ] **T203** Add `ModelBatchPlanner` in `Core.Runtime` and route
      `WorldObjectPassCoordinator` through it. Renderers stop deciding batchability.
- [ ] **T204** Unit-test the planner headless: identical state batches, differing state does
      not, and each unbatchable reason is reachable.
- [ ] **T205** **(operator)** Fly the route. Frame time and appearance must be **unchanged** —
      this phase moves the decision, it does not change it. A difference here means the key is
      wrong, and finding that out now is the point.

---

## Phase 3 — Close the three gates

**Brought forward ahead of the Phase 1 flight, deliberately.** The gate exists to stop us
building toward an *unmeasured* ceiling. R8 replaced that measurement with something stronger:
the instanced count was **provably zero** because no renderer could return
`SupportsGpuInstancedOpaque = true`, so no flight could have reported anything else. Making
instancing reachable was a precondition for the measurement, not a bet on its outcome.

- [ ] **T301** Give native-route M2 a real batch path (absorbs spec 201 Phase 2). **This is the
      whole remaining blocker — see R11.** `WowViewerM2RuntimeBridge.PreferNativeStaticRenderer`
      returns **`true` when its env var is unset**, so `ShouldUseNativeStaticRenderer` always wins
      and *every* M2 gets the native-only constructor with `_legacyRenderer == null`, making
      `M2Renderer.RequiresUnbatchedWorldRender` unconditionally true. Three sub-parts, in order:
      - **T301a** Hoist the ten shared uniforms out of `RenderCore` into `BeginBatch` state. R9:
        `RenderInstance` and `RenderWithTransform` call the same `RenderCore`, which re-uploads
        view, projection, three fog values, camera position, light direction, light colour and
        ambient **per instance** — so flipping the flag without this buys literally nothing.
      - **T301b** Give the native `M2Renderer` its own instance attributes and shader branch, the
        same surgery already done on `MdxRenderer` in R8. `QueueGpuInstance` currently delegates to
        the null `_legacyRenderer` and is a no-op.
      - **T301c** Only then narrow `RequiresUnbatchedWorldRender` so it stops returning true merely
        because `_legacyRenderer is null`.
- [-] **T302** Carry fade in the GPU instance payload so `OpaqueFade < 0.999` no longer forces
      an instance out of instancing. **Half done**: the shader now carries `aInstanceFade` and
      multiplies it into the final alpha, so the payload exists. The `>= 0.999` gate is
      **deliberately retained** — `RenderGeosets` decides blend state once per batch from a
      single `fadeAlpha`, so mixing faded and unfaded instances in one batch would render the
      faded ones opaque. Closing it properly means splitting faded instances into their own
      blended batch. The new `GatedOpaqueFadeBelowThreshold` counter says how much that is
      worth before it is built.
- [x] **T303** Give every route a per-model instancing path, or a recorded reason it cannot
      have one (FR-003). Legacy-MDX and legacy-backed M2 now instance; native-route M2's reason
      is R9; models with local MDX lights are excluded with the reason in R10.
- [ ] **T304** Absorb spec 153 US3's mechanism into the planner rather than leaving two
      systems (Constitution II).
- [ ] **T305** **(operator)** Fly the route with **"GPU instancing for opaque models"** on, then
      off — both toggles now sit together in *Submission efficiency*. SC-001: opaque draw calls
      fall toward `distinct models`; every remaining unbatched instance has a named gate.
      SC-002: median frame time against the 84.44 ms baseline. This single flight also closes
      T103, T104 and spec 201's T005/T006.
- [ ] **T306** **(operator)** Capture comparison. SC-004: no appearance change. The specific
      things to look at are **lamps, braziers and campfires** (R10 excludes locally-lit models
      from instancing — confirm they are unchanged) and **distance-faded doodads** at the far
      edge of the draw distance (T302 keeps them off the instanced path).

---

## Phase 4 — Every era, not just 5.0.1

The operator's report is that this kills performance for **all** versions. A 5.0.1-only fix
has not delivered.

- [ ] **T401** **(operator)** Fly one map per era — 0.5.3, LK, Cata, MoP — recording draw
      calls, unbatched counts and frame time.
- [ ] **T402** Fix any era where instancing does not engage, with the gate named.
- [ ] **T403** **(operator)** SC-005: 0.5.3 reference scene pixel-identical to before.
- [ ] **T404** Verify SC-006: disabling batching reproduces pre-change output exactly.

---

## Phase 5 — Transparent

- [ ] **T501** Extend the planner to transparent submission, honouring sort order.
- [ ] **T502** Where batching would reorder draws, order wins and the instance is counted with
      that reason (FR-005).
- [ ] **T503** **(operator)** Confirm the 260 transparent unbatched draws fall, with no
      blending artefacts.

---

## Phase 6 — Cross-model batching (optional)

**Gated on Phase 1.** Only if the per-model floor proved too high.

- [ ] **T601** Re-measure the need against the Phase 3 result.
- [ ] **T602** If not needed, record that and close the spec.

---

## Notes

- **`MDX 13180/20115 (429 ok/0 fail)`** — ~7,000 placed models are not drawn and nothing
  reports a failure. Whether they are culled, unloaded, or lost is unknown. Out of scope here,
  but T102 touches the same field; if the answer falls out, record it.
- Animation already deduplicates per model via `UpdatedMdxModelKeys` (research.md R5). The
  submission side is simply missing the same idea — follow the existing pattern.
- Known test baseline: `WowViewer.Core.Tests` has **9 pre-existing failures**. Compare to 9.
