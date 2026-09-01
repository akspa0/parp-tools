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

- [ ] **T001** Land spec 201 Phase 1 (per-render-path attribution). This spec consumes it; do
      not duplicate the counters.
- [ ] **T002** Split the `batched` counter into **instanced** (one draw per renderer, via
      `QueueGpuInstance`) and **state-hoisted** (`BeginBatch` + per-instance
      `RenderInstance`). research.md R1: these are counted identically today, and the second
      does **not** reduce draw calls.
- [ ] **T003** Report actual **draw calls** issued for opaque models, separately from instance
      count. Draw calls are the thing being optimised; instances are not.
- [ ] **T004** Report which of the three gates (research.md R3) forced each unbatched
      instance: `RequiresUnbatchedWorldRender`, `!SupportsGpuInstancedOpaque`, or
      `OpaqueFade < 0.999`.
- [ ] **T005** **(operator)** Re-fly the Wandering Isle route with the spec 153 US3 toggle
      **on**, then **off**. Record both. This resolves R4's ambiguity: whether the recorded
      `0 batched / 13180 unbatched` was the toggle or the absence of any batch path.

**Phase 0 exit**: batched/instanced/state-hoisted/unbatched/unbatchable are five distinct
numbers, draw calls are reported, and every unbatched instance has a named gate.

---

## Phase 1 — Measure the ceiling before building toward it

- [ ] **T101** Count **distinct visible models** per frame. This is the floor per-model
      instancing can reach (research.md R2).
- [ ] **T102** Confirm or refute that the status bar's `429` in `MDX 13180/20115 (429 ok/0 fail)`
      is the distinct loaded-model count. **Do not infer it** — the field's meaning has not
      been checked, and this project has a history of reading meaning into unverified fields.
- [ ] **T103** Measure the share of instances blocked by each of the three gates, so the
      largest is known. The fade gate is a live suspicion: distance-faded doodads are exactly
      the dense population and every one drops out of instancing today.
- [ ] **T104** **(operator)** Fly the route and record T101–T103. Compute the projected
      reduction: `13,180 instances → N distinct models`.

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

- [ ] **T301** Implement the backend-specific batch key for native-route M2 (absorbs spec 201
      Phase 2). `M2Renderer.RequiresUnbatchedWorldRender` stops returning true merely because
      `_legacyRenderer is null`.
- [ ] **T302** Carry fade in the GPU instance payload so `OpaqueFade < 0.999` no longer forces
      an instance out of instancing.
- [ ] **T303** Give every route a per-model instancing path, or a recorded reason it cannot
      have one (FR-003).
- [ ] **T304** Absorb spec 153 US3's mechanism into the planner rather than leaving two
      systems (Constitution II).
- [ ] **T305** **(operator)** Fly the route. SC-001: opaque draw calls fall toward the Phase 1
      floor; every remaining unbatched instance has a reason. SC-002: median frame time against
      the 84.44 ms baseline.
- [ ] **T306** **(operator)** Capture comparison. SC-004: no appearance change.

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
