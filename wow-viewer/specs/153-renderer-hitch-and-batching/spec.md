# Feature Specification: Renderer hitch elimination and MDX batching restoration

**Feature Branch**: `v0.5.3-dev`

**Created**: 2026-08-15

**Status**: Draft — diagnosis complete, fixes not started

**Input**: Spec 152 built the detector and used it. This spec owns the defects the detector found.

## Why this spec exists

Spec 152's Phase 1 was written as an explicit decision point: *"if the hypothesis is refuted, Phase 3
is re-planned against the real cause before any flattening work begins. The plan must not proceed on
momentum."*

**That gate has now fired.** The measurements refuted the leading hypothesis. This spec records what
was actually measured and owns the fixes; Spec 152 retains the measurement infrastructure and the
per-era lighting work.

## Measured findings

All numbers are from the in-viewer frame history against real staged clients (Alpha 0.5.3, Kalimdor
and Azeroth), captured 2026-08-15. Zones: Un'goro Crater, Dustwallow/Dragonmurk, Stranglethorn Vale,
South Seas.

### The allocation-churn hypothesis was wrong

Spec 152 assumed per-frame allocation in the scene graph was the hitch mechanism, and Phases 3–4
proposed flattening the scene into retained ordered draw lists on that basis.

**Median world-render CPU is 0.33–8.58 ms depending on zone.** A renderer with a sub-millisecond
median in ordinary scenes does not have an allocation-throughput problem. The churn work landed in
Spec 152 (traversal is now allocation-free in steady state, verified by test) and was worth doing on
its own merits, but **it did not cause the gallop and did not fix it.**

The flattening work in Spec 152 Phases 3–4 must not proceed on the original justification.

### Defect A — periodic stall in an untimed pass

- Hitches recur at **~47–50 frame intervals** at **~212 ms** each. Fixed size, regular period.
- **All unaccounted time is in the pass gap.** Region probes: prologue peak 0.27 ms, **pass gap peak
  314.15 ms**, epilogue peak 0.00 ms.
- `WorldFramePasses` declares eleven passes. Ten assign a stage timer. **`PrepareObjectPhase` assigns
  none**, so its entire cost is invisible to the stage table by construction — and it is the region
  the probes point at.
- `PrepareObjectPhase` contains two periodic residency operations consistent with a recurring
  fixed-size stall: `AudioRuntime.Update(...)` and `EnsurePm4OverlayMatchesCameraWindow(...)`, the
  latter loading PM4 overlays when the camera crosses into a new tile window.

Sub-probes for both are landed but **not yet captured**. The specific operation is not yet named.

### Defect B — MDX submits one draw call per instance

Confirmed in every zone tested:

| Zone | MDX opaque batched | unbatched | WMO draw calls |
| --- | --- | --- | --- |
| Stranglethorn | 0 | 312 (100%) | 198 total / 198 batched |
| South Seas | 0 | 220 (100%) | 198 total / 198 batched |
| Dustwallow | 0 | (100%) | — |

**WMO batching works. MDX batching is completely inert.** With 7912 of 18663 MDX admitted in
Dustwallow at max fog, `MdxOpaqueSubmission` reached median 2.03 / p99 30.75 / max 41.0 ms.

This is independent of Defect A: it is a sustained per-frame cost, not a periodic stall.

### Defect C — SceneMaintenance rare enormous spike

`SceneMaintenance` in Stranglethorn: median 0.02 ms, **max 454.8 ms** — the single largest stage
observation recorded. It is timed, so it is not part of the unaccounted gap, but a near-half-second
stall is its own defect. `RebuildInstanceLists` / `RebuildSceneGraphObjectIndex` run here when
`_instancesDirty`, which streaming sets frequently.

### Defect D — deferred asset load budget is not enforced

`ProcessPendingLoads` checks `stopwatch.Elapsed < maxBudgetMs` **only between loads**, so once a load
starts it runs to completion. Observed `DeferredAssetLoads` at 46.58 ms and 58.1 ms against a 3.5 ms
nominal budget. It also calls `LoadMdxModel` **synchronously on the render thread**.

### Ruled out, with evidence

Recording these so they are not re-proposed:

- **Decoded-asset caching / LRU thrash.** `MaxMdxCached = 0` (unlimited, no eviction) and 554 loaded
  models serve 18663 instances. Nothing is evicted or re-decoded. The cost is submission, not
  loading.
- **Scene graph traversal.** Timed under `WmoVisibility`, max 0.22 ms.
- **Per-frame diagnostic logging.** The `_renderDiagPrinted` guard does disarm after the first frame.

### Detector lessons worth keeping

- **p99 hides rare hitches.** A 1-in-100 hitch lands at p100 under nearest-rank. Max and
  over-threshold count carry the signal; percentile-only reporting reproduces a false null.
- **Ranking stages by p99 buries the culprit.** A stage that fires rarely and costs enormously has a
  near-zero p99. Sorting by max surfaced `DeferredAssetLoads` and `SceneMaintenance`.
- **Unaccounted time must be reported.** Naming the largest instrumented stage when instrumented work
  is 2% of the frame points at the wrong place — it fingered `MdxVisibility` at 0.2 ms for a 350 ms
  frame.
- **Stranglethorn Vale is the benchmark scene.** Thousands of objects per view forced both defects
  into the open where quieter zones hid them.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Name the periodic stall (Priority: P1)

As the maintainer, I need the ~212 ms periodic stall attributed to a specific operation inside
`PrepareObjectPhase`, so the fix targets the real call rather than the region.

**Independent Test**: Fly Stranglethorn until hitches recur, read the sub-probe peaks, and confirm
one operation accounts for the stall magnitude.

**Acceptance Scenarios**:

1. **Given** a capture with recurring hitches, **When** the breakdown is read, **Then** one probed
   operation's peak matches the hitch magnitude within tolerance.
2. **Given** no probed operation matches, **Then** `PrepareObjectPhase` is subdivided further rather
   than a fix being guessed.

---

### User Story 2 - Periodic residency work leaves the render thread (Priority: P1)

As a user, I want the viewer not to stall for ~200 ms when crossing a tile window, so movement is
smooth.

**Acceptance Scenarios**:

1. **Given** the camera crosses a tile window boundary, **When** frames are recorded, **Then** no
   frame exceeds the hitch threshold as a result.
2. **Given** the work is moved or bounded, **When** measured on the same route, **Then** the periodic
   hitch pattern is gone from the frame history, not merely smaller.
3. **Given** the work is deferred, **When** its results are needed, **Then** correctness is preserved
   — overlays and audio still become resident, just not synchronously.

---

### User Story 3 - MDX draws batched (Priority: P1)

As a user, I want dense doodad scenes to render without per-instance draw calls.

**Acceptance Scenarios**:

1. **Given** a scene with many instances of the same model, **When** submitted, **Then** the majority
   of opaque MDX report as batched.
2. **Given** batching is restored, **When** measured in Stranglethorn, **Then** `MdxOpaqueSubmission`
   p99 improves against the recorded baseline by more than the noise floor.
3. **Given** batching is active, **When** the scene is compared visually, **Then** output is
   equivalent to the unbatched path.
4. **Given** a model cannot be instanced, **When** it is submitted, **Then** it falls back cleanly and
   the fallback count is visible.

---

### User Story 4 - Bounded scene maintenance (Priority: P2)

As a user, I want no half-second stall when instance lists rebuild.

**Acceptance Scenarios**:

1. **Given** `_instancesDirty` is set, **When** the rebuild runs, **Then** it is incremental or
   budgeted rather than a full rebuild in one frame.
2. **Given** a rebuild is in progress, **When** frames render, **Then** no frame exceeds the hitch
   threshold because of it.

---

### User Story 5 - Asset loads respect their budget (Priority: P2)

As a user, I want deferred loading to stay within its frame budget.

**Acceptance Scenarios**:

1. **Given** a nominal budget, **When** loads are processed, **Then** the stage does not exceed it by
   an order of magnitude.
2. **Given** a single load would exceed the remaining budget, **Then** it is split, bounded, or moved
   off the render thread rather than run to completion synchronously.

### Edge Cases

- A hitch that survives every fix — the frame history must still name where it went.
- Instancing that changes visual output subtly — visual equivalence is a gate, not an assumption.
- Deferring residency work far enough that assets visibly pop in late.
- A zone with few objects hiding a regression that Stranglethorn would expose.

## Requirements *(mandatory)*

- **FR-001**: Every pass in `WorldFramePasses` MUST have a stage timer, so no pass can hide cost by
  construction.
- **FR-002**: The periodic stall MUST be attributed to a named operation before any fix is applied.
- **FR-003**: Periodic residency work MUST NOT block the render thread for longer than the hitch
  threshold.
- **FR-004**: Opaque MDX instances sharing a model MUST submit as batches where the renderer supports
  instancing.
- **FR-005**: Batched and unbatched submission counts MUST remain observable.
- **FR-006**: Instanced output MUST be visually equivalent to unbatched output.
- **FR-007**: Instance-list rebuilds MUST be incremental or budgeted.
- **FR-008**: Deferred asset loading MUST enforce its budget rather than checking only between loads.
- **FR-009**: Every fix MUST have a before/after capture on the same zone and route.
- **FR-010**: Fixes MUST land one at a time and be individually revertible.
- **FR-011**: A fix whose measured effect does not exceed the noise floor MUST be reverted.

## Success Criteria *(mandatory)*

- **SC-001**: The periodic ~212 ms hitch pattern is absent from the frame history on the Stranglethorn
  route.
- **SC-002**: Majority of opaque MDX report as batched in dense scenes, down from 100% unbatched.
- **SC-003**: `MdxOpaqueSubmission` p99 in Stranglethorn improves against baseline by more than the
  stated noise floor.
- **SC-004**: `SceneMaintenance` max drops below the hitch threshold, down from 454.8 ms.
- **SC-005**: `DeferredAssetLoads` max stays within the same order of magnitude as its budget, down
  from 13× over.
- **SC-006**: Every pass reports a stage timer; unaccounted time in steady state approaches zero.
- **SC-007**: Interactive movement through Stranglethorn is smooth by the maintainer's own judgement —
  the numbers are the gate, but this is the point of the work.

## Assumptions

- Alpha 0.5.3 Kalimdor/Azeroth on the staged client library remain the measurement targets.
- Stranglethorn Vale is the standard benchmark scene for this work.
- The maintainer runs captures; automated camera-path runs are available but interactive flight has
  been the faster loop so far.
- Absolute timings are machine-specific; before/after on the same machine and route is the comparison
  that counts.

## Non-Goals

- Flattening the scene graph into retained draw lists. Spec 152 Phases 3–4 proposed this on an
  allocation-churn premise the measurements refuted. It may return later on its own evidence.
- Backend change.
- Visual-fidelity work beyond preserving equivalence.
