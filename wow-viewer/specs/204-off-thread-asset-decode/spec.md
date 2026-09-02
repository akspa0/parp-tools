# Feature Specification: Off-Thread Asset Decode

**Feature Branch**: `204-off-thread-asset-decode`

**Created**: 2026-09-01

**Status**: Draft

**Input**: Operator observation, 2026-09-01: *"asset loads seem to be really really slow, despite
the data being served from an ssd, via our fast mpq reader code."*

## Context

The operator is right that it is not the disk, and right that it is not the MPQ reader.

`MpqDataSource` already runs **two prefetch worker threads**, and `WorldAssetManager.QueueWmoLoad`
/ `QueueMdxLoad` already call `PrefetchModelBytes` when a placement is queued. Raw root-file I/O
is therefore **already off the render thread**.

What is still on the render thread is everything after the read: model parse, M2→runtime adaptation,
skin resolution, BLP decode, vertex/index array construction, and GL upload. `WorldAssetManager`
contains **no threading whatsoever** — no `Task`, no `Thread`, no `Parallel`, no concurrent
collection.

**Measured, 2048-frame flight over Mogu Ruins / Dread Wastes (MoP 5.0.1):**

| Metric | Value |
|---|---|
| Median frame | 75.26 ms |
| p95 / p99 / max | 144.59 / 235.37 / 541.65 ms |
| Frames over 33.3 ms | 2047 of 2048 |
| Recent hitches attributed to `DeferredAssetLoads` | 12 of 13 listed, **26.4–68.1 ms each** |
| Remaining listed hitches | `WmoSubmission`, 26.4–38.9 ms |
| Models resident vs placed | 926 distinct loaded; 1,489 visible of 25,684 placed |

`DeferredLoadBudget` already knows this is the problem and says so in its own documentation:

> *"A single synchronous load that is larger than the entire budget still costs what it costs; the
> policy only guarantees it is never added to time already spent. Removing that residual requires
> moving decode off the render thread so the budget governs upload alone — Spec 153 Phase 5 step 2,
> deliberately not attempted here."*

This spec is that step.

**Why the budget cannot rescue it.** `DeferredLoadBudget.CanStartAnotherLoad` admits the first load
of every frame unconditionally, by design, so that an asset costlier than the whole budget still
eventually becomes resident. A 68 ms model therefore produces a 68 ms frame no matter how the
budget is tuned. The class counts these as `OversizedAdmissionCount` and calls that count *"the
honest measure of the residual the off-thread decode still owes."*

**Why it gets worse the worse it gets.** `ProcessDeferredAssetLoads` throttles on the *previous*
frame's CPU time: at ≥33 ms it clamps to one load per frame. But the guaranteed-progress rule
admits that one load at full cost anyway — so the clamp **cuts streaming throughput roughly
six-fold without reducing the hitch at all**. Slow frames throttle loading; oversized loads keep
frames slow. At ~10 FPS that is ~10 models per second against 25,684 placements, which is the
"very slow to render objects in" the operator has now reported twice.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Flying does not stall on asset decode (Priority: P1)

An operator flies across a populated zone and the frame time does not spike when new assets become
resident.

**Why this priority**: `DeferredAssetLoads` owns almost every entry in the measured hitch list. It
is the single largest identified contributor to frame instability.

**Independent Test**: Fly a fixed route and compare the `DeferredAssetLoads` stage distribution
before and after. Delivers a measurable reduction with no change to what is drawn.

**Acceptance Scenarios**:

1. **Given** a queued asset whose decode costs more than the whole frame budget, **When** it is
   loaded, **Then** the render thread pays only its GPU upload, not its decode.
2. **Given** a frame in which several assets finish decoding at once, **When** they are uploaded,
   **Then** the number uploaded is governed by the frame budget and the rest wait.

---

### User Story 2 - Streaming keeps up with the camera (Priority: P1)

Objects appear promptly as the camera moves, instead of trickling in over minutes.

**Why this priority**: Equal to US1 and caused by it. The throughput collapse and the hitching are
the same defect seen from two sides, and fixing the hitch without fixing throughput would leave the
operator's original complaint standing.

**Independent Test**: From a cold cache, park at a fixed position and measure wall-clock time until
the resident model count stops rising.

**Acceptance Scenarios**:

1. **Given** a cold cache and a stationary camera, **When** assets stream in, **Then** the resident
   count rises at a rate set by decode throughput rather than by frame rate.
2. **Given** the previous frame was slow, **When** the loader throttles, **Then** it reduces work
   that actually costs the render thread, not work that no longer does.

---

### User Story 3 - Texture upload is budgeted and attributed (Priority: P2)

Texture residency work is governed by the same budget and reported as its own cost.

**Why this priority**: `ProcessDeferredTextureLoads()` is called from the top of
`MdxRenderer.RenderGeosets` — texture decode and upload therefore happen **inside the draw pass**,
are billed to `MdxOpaqueSubmission`, and are subject to no budget at all. It is a second unbudgeted
render-thread load path hiding behind a submission timer, and it will still be there after US1.

**Independent Test**: Confirm no texture decode occurs inside a submission stage, and that texture
upload appears as its own reported cost.

**Acceptance Scenarios**:

1. **Given** a model whose textures are not yet resident, **When** it is drawn, **Then** the draw
   does not perform file reads or image decoding.
2. **Given** textures pending upload, **When** a frame runs, **Then** their upload is admitted by
   the frame budget and reported separately from submission.

---

### User Story 4 - Prefetch covers what a load actually reads (Priority: P2)

Every file an asset needs is fetched by a worker, not just its root file.

**Why this priority**: `PrefetchModelBytes` covers the root model and skin candidates. WMO **group**
files, and every BLP texture, are discovered only after the root is parsed, so they are read
synchronously today. Moving decode off-thread makes those reads a worker cost rather than a frame
cost, but they remain serialised behind the root parse unless prefetch follows.

**Independent Test**: Compare prefetch hit/miss counts for group files and textures before and after.

**Acceptance Scenarios**:

1. **Given** a WMO root has been parsed, **When** its group files are needed, **Then** they were
   already requested by a worker.
2. **Given** a model's texture list is known, **When** its textures are needed, **Then** they were
   already requested.

---

### Edge Cases

- A tile unloads while one of its assets is mid-decode — the result must be discarded without
  touching GL and without corrupting residency bookkeeping.
- LRU eviction selects an asset that has an in-flight decode.
- The same asset is requested twice before the first decode completes.
- Decode throws. The failure must be recorded once, on the render thread, with the same
  `MdxModelsFailed` semantics as today, and must not retry in a loop.
- A decode completes for an asset the camera has moved far away from — it must not jump the upload
  queue ahead of nearer assets.
- Shutdown with work in flight.
- The headless capture and harvest paths, which expect deterministic completion rather than
  best-effort streaming.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: All GL calls MUST remain on the render thread. The viewer holds a single GL context;
  this spec moves CPU work only.
- **FR-002**: File read, parse, format adaptation, image decode, and CPU-side mesh construction MUST
  happen off the render thread.
- **FR-003**: The render thread's per-asset cost MUST be GPU resource creation and upload only.
- **FR-004**: Upload MUST be governed by the existing frame budget, and the budget MUST no longer
  need an unconditional oversized admission to guarantee progress.
- **FR-005**: An asset MUST NOT be submitted for rendering until its upload has completed.
- **FR-006**: Load failures MUST be reported with the same counts and suppression behaviour as
  today, and MUST NOT be retried in an unbounded loop.
- **FR-007**: Cancellation MUST be supported for assets whose need disappears before upload.
- **FR-008**: The change MUST NOT alter what is drawn once an asset is resident.
- **FR-009**: Off-thread work MUST be switchable at runtime, so the synchronous path remains
  reachable for same-session before/after comparison.
- **FR-010**: The number of worker threads MUST be bounded and configurable.
- **FR-011**: Deterministic completion MUST remain available for capture and harvest paths, which
  cannot accept best-effort streaming.
- **FR-012**: Per-stage timings MUST distinguish fetch, decode, and upload, so the residual after
  this change is attributable rather than lumped into one number.

### Key Entities

- **Load request**: A queued asset key with its kind, priority (camera distance), and cancellation
  state.
- **Decoded payload**: The CPU-side result of fetch + decode — parsed model, vertex/index arrays,
  decoded texture pixels — carrying no GL handle.
- **Upload queue**: Decoded payloads awaiting render-thread GPU creation, ordered by priority.
- **Load budget**: The existing `DeferredLoadBudget`, re-pointed at upload cost rather than total
  load cost.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On the baseline route, no `DeferredAssetLoads` stage exceeds the configured frame
  budget by more than one upload's cost. The measured 26.4–68.1 ms hitches do not recur.
- **SC-002**: `OversizedAdmissionCount` stops rising during steady-state flight.
- **SC-003**: From a cold cache at a fixed camera position, wall-clock time to reach a stable
  resident model count falls by at least 5x against the current path.
- **SC-004**: p99 frame time on the baseline route improves against the recorded 235.37 ms, and the
  share of frames over 33.3 ms falls from 2047/2048.
- **SC-005**: No GL call is issued from a non-render thread, verified by an assertion active in
  debug builds rather than by inspection.
- **SC-006**: A reference scene is pixel-identical to the synchronous path once fully resident.
- **SC-007**: Fetch, decode and upload are three separately reported timings.

## Assumptions

- The viewer holds one GL context on one thread. Shared-context uploads are explicitly out of scope;
  they are a larger change with driver-dependent behaviour.
- `MpqDataSource`'s existing prefetch workers and read cache are correct and thread-safe; this spec
  builds on them rather than replacing them. Their thread-safety is nonetheless **verified in
  research, not assumed** — see research.md R5.
- The measured MoP flight is representative of the problem. Other eras are expected to benefit but
  are verified, not assumed.
- Spec 153 Phase 5 step 2 is the same work. This spec supersedes that line item rather than
  competing with it.
- Whether the parse layer is thread-safe today is **unknown and is the main risk**. Static caches in
  the adapter, the texture resolver, the format-profile registry and the diagnostic loggers are
  suspected and must be audited before any of it runs concurrently.
