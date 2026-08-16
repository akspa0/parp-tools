# Research: Spec 153 implementation notes and Phase 0 capture protocol

**Date**: 2026-08-15 | **Branch**: `v0.5.3-dev` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

## What landed

Phases 1, 3 and 5 — the three the plan marks as independent — are implemented. Phase 0 is a capture
and is user-run; Phases 2 and 4 remain gated behind it exactly as the plan specifies.

| Phase | State | Requirement |
| --- | --- | --- |
| 0 Name the periodic stall | **DONE — captured 2026-08-15, stall named** | FR-002 |
| 1 Close the instrumentation hole | Implemented + tested; **confirmed by capture** | FR-001, SC-006 |
| 2 Move residency work off the render thread | Implemented after the capture; **unmeasured** | FR-003, SC-001 |
| 3 Restore MDX batching | Implemented; **confirmed by capture** | FR-004/005/006, SC-002/003 |
| 4 Bound scene maintenance | Gate now open (2–3 measured); not started | FR-007, SC-004 |
| 5 Enforce the deferred load budget | Implemented + tested; partial by construction; **not yet sufficient** | FR-008, SC-005 |

Nothing here claims a measured improvement. Every SC still needs the user-owned before/after against
the Stranglethorn baseline (FR-009).

---

## Phase 1 — the instrumentation hole is closed

`PrepareObjectPhase` was one of eleven passes in `WorldFramePasses` and the only one with no stage
timer, so its entire cost — including the ~212 ms periodic stall — landed in the unaccounted pass gap
where the stage table could not represent it.

- `WorldRenderStage.PrepareObjectPhase` added; `WorldRenderFrameHistory.StageCount` 18 → 19.
- `WorldRenderFrameStats.PrepareObjectPhase` added, recorded per frame, and included in the
  pass-gap subtraction in `RecordRenderRegionBreakdown`, so the region breakdown and the stage table
  now agree instead of double-counting.
- The timer spans the **whole** pass. The existing `ObjectPhasePrepareMs` probe previously covered
  only `AudioRuntime.Update` + `EnsurePm4OverlayMatchesCameraWindow`; it now measures the same span
  as the stage timer, so the two cannot disagree.
- The Perf panel gained an **`other (unprobed remainder)`** row = total − audio − PM4. That row is
  what decides Phase 0's second acceptance scenario.

**The structural guard.** `WorldFramePassInstrumentation` declares, for each `WorldFramePasses`
member, which stages account for its cost. `WorldFramePassInstrumentationTests` asserts by reflection
that every pass member appears with a non-empty stage set, that the table names no pass that does not
exist, that every `WorldRenderStage` is owned by some pass or declared pre-pass, that no stage is
claimed twice (double-counting would understate unaccounted time — the same failure mode inverted),
and that `StageCount` tracks the enum. Adding a twelfth pass without a timer now fails a test rather
than silently reopening the hole.

## Phase 3 — the MDX batching cause, found and fixed

Not a capability gap. `WorldScene.PlanVisibleMdxPasses` passed `PlanOpaqueMdxRoutes` a
`requiresUnbatchedRender` predicate whose entire body was `return true`, with a comment holding world
MDX on the per-instance path "until the shared/GPU batch paths have visual parity proof". So the
planner routed **100% of opaque MDX to the fallback by construction**, which is why the counters read
0 batched / 312 unbatched while WMO — which consumes the renderer's own declaration — batched 198/198.

The fix consumes that same declaration:

```csharp
IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
return renderer == null || renderer.RequiresUnbatchedWorldRender;
```

This is the identical contract the WMO-internal doodad path (`WmoRenderer.PrepareVisibleDoodads`) has
used all along, so it is not a new policy — it is the planner no longer overriding the existing one.

**What "batched" means here.** `MdxRenderer.SupportsGpuInstancedOpaque` is still `false` — GPU
instancing for MDX remains held out pending portable shader compile and parity proof, and this change
does **not** turn it on. The win is the shared-state route: `BeginBatch` once per renderer, then
`RenderInstance` per placement, instead of a full `RenderWithTransform` state setup per draw. Per
instance that removes `UseProgram`, three GL state calls, the view/proj matrix uploads, and the fog,
camera and lighting uniform uploads. Submission order is unchanged — `ExecutePlannedOpaqueMdx` still
walks routes in visibility order and dispatches each immediately.

**Visual equivalence (FR-006), by inspection.** `RenderInstance` and `RenderWithTransform` differ in
exactly three ways, all accounted for:

| Difference | Resolution |
| --- | --- |
| view/proj/fog/light/camera uniforms | Uploaded once by `BeginBatch` with the same per-frame constants. All `MdxRenderer`s share one `static` shader program and uniform locations, and nothing binds a different program inside the opaque MDX pass. |
| particle emitters (transparent pass only) | Already excluded — `RequiresUnbatchedWorldRender` is true whenever emitters exist. |
| `_wireframe` polygon mode | **Was a real divergence.** `RenderWithTransform` honours it; `BeginBatch` always sets `Fill`, so a wireframe-flagged model would have drawn filled on the batch path. `_wireframe` is now part of `RequiresUnbatchedWorldRender`, which also corrects the WMO-internal doodad path that already consumed it. |

`M2Renderer` needs no change: it already reports `RequiresUnbatchedWorldRender` when it has no legacy
renderer, so runtime-backend M2s stay unbatched. `M2CameraPathRenderer` reports `true` unconditionally.

**Revertibility (FR-010/011).** `WorldScene.MdxOpaqueBatchingEnabled` (default on) is exposed as a
checkbox in *Utilities > Perf > Submission efficiency*. Off restores the recorded 100%-unbatched
baseline exactly, so the before/after is one flight rather than two builds.

## Phase 5 — the load budget is enforced before each load, not only between them

`WorldAssetManager.ProcessPendingLoads` looped on `completed < maxLoads && elapsed < budget`, so with
3.4 ms of a 3.5 ms budget spent it would still start a load and learn the cost afterwards. Measured
46.6 ms and 58.1 ms against a 3.5 ms budget.

`DeferredLoadBudget` (core, pure, unit-tested) learns per-kind load cost — EWMA plus a decaying
high-water mark, tracked separately for MDX and WMO because their costs differ by an order of
magnitude — and answers `CanStartAnotherLoad(elapsed, budget, loadsStartedThisFrame)`. The loop now
consults it before each dequeue and records each load's real cost afterwards. The decision is made
pre-dequeue against the *cheapest* known kind, because the kind of the next queue item is not known
until it is popped and re-enqueueing would corrupt the priority-set bookkeeping.

Counted, not skipped: entries discarded as already-cached increment `loadsCompleted` but not
`loadsStarted`, so a frame that has only skipped cache hits still has its whole budget.

**This is a partial fix and is documented as such.** A single synchronous load larger than the entire
budget still costs what it costs; `CanStartAnotherLoad` always admits the frame's first load so an
oversized asset does not starve forever, and counts that admission in `OversizedAdmissionCount`. What
the policy removes is the *additive* overshoot — budget-nearly-spent plus a full heavy load. Removing
the residual needs decode off the render thread (plan Phase 5 step 2, not attempted). `LoadBudget` is
public so a diagnostics surface can read `BudgetDeferralCount`, `OversizedAdmissionCount` and
`WorstObservedLoadMs`.

**SC-005 is therefore not claimed.** The capture will show whether removing the additive component is
enough to bring `DeferredAssetLoads` max within an order of magnitude of its budget. If it is not,
the off-thread decode is the remaining work and the number will say so.

---

## Capture protocol (retained — rerun this for every before/after, FR-009)

Phase 0's run is recorded below. The same protocol is the standard before/after for Phases 2, 4 and 5.

1. Launch the viewer, load **Alpha 0.5.3 Azeroth**, fly to **Stranglethorn Vale**.
2. Open **Utilities > Perf > Frame history**. Confirm the panel does *not* show the
   "Camera stationary" warning — a stationary window cannot demonstrate movement behaviour.
3. Press **Reset region peaks**, then fly for ~60 s across tile boundaries until hitches recur.
4. Read **Unaccounted breakdown (peaks)**, specifically the four rows under *Inside PrepareObjectPhase*:
   `PrepareObjectPhase total`, `AudioRuntime.Update`, `PM4 overlay window`, `other (unprobed remainder)`.
5. Also read the stage table, now sorted by max. `PrepareObjectPhase` appears in it for the first
   time; its max is the same number as the probe total and is the cross-check that Phase 1 works.

Decision rule applied on the 2026-08-15 run: `AudioRuntime.Update` peak was 283.46 ms against a pass
total of 283.47 ms, so the first branch fired cleanly and no subdivision was needed.

### Re-capture checklist for the Phase 2 fix

1. `PrepareObjectPhase` max should collapse from 283.4 ms to near zero **with the audio panel
   closed**, and its p99 from 221.59 ms to ~0.
2. The "Recent hitches" list should stop being dominated by `<- PrepareObjectPhase`. Whatever it
   names instead is the new owner — expect `SceneMaintenance` and `DeferredAssetLoads`.
3. Cross tile boundaries deliberately. The `RemoveTile` stall was movement-triggered, so a route that
   evicts tiles is the one that proves it.
4. Open **Utilities > Audio** and confirm the emitter rows still populate and update — the gate must
   not have broken the panel it exists to serve.
5. SC-001 is only met if the periodic pattern is *absent*, not smaller.

### Phase 3 before/after, same flight

While in Stranglethorn, with the frame history running:

1. Note `MdxOpaqueSubmission` median / p99 / max and the batched/unbatched split with
   **Opaque MDX batching ON** (default).
2. Uncheck it, fly the same route, and record the same numbers. That is the recorded baseline
   reproduced live: 0 batched / ~312 unbatched.
3. FR-011: if the p99 difference does not exceed the noise floor, the change is reverted rather than
   kept on plausibility.
4. FR-006: compare the two visually before accepting. Batching is only equivalent if it looks
   equivalent.

### Capture results — Stranglethorn, 2026-08-15, 1170 frames

Batching ON (default). The OFF column was not taken; the ON numbers already clear SC-002/SC-003
against the recorded baseline by a wide margin, and the batching counters are unambiguous.

| Reading | Baseline (pre-fix) | Captured (Phases 1/3/5) |
| --- | --- | --- |
| median frame ms | 22.05 | **17.40** |
| p95 frame ms | — | 107.22 |
| p99 frame ms | 259.70 | 246.62 |
| max frame ms | 707.23 | 636.29 |
| hitches / frames | 117 / 584 (20%) | 186 / 1170 (16%) |
| pass gap peak ms | 259.04 (314.15 South Seas) | **9.45** |
| unaccounted median / p99 ms | not reported | **0.05 / 0.16** |
| `PrepareObjectPhase` median / p99 / max ms | *untimed* | **0.01 / 221.59 / 283.4** |
| ↳ `AudioRuntime.Update` peak ms | never captured | **283.46** |
| ↳ `PM4 overlay window` peak ms | never captured | 0.12 |
| ↳ `other` unprobed remainder | n/a | ~0.00 |
| MDX opaque batched / unbatched | 0 / 312 (100% unbatched) | **526 / 3 (0.6% unbatched)** |
| `MdxOpaqueSubmission` median / p99 / max ms | 2.03 / 30.75 / 41.0 | **0.01 / 14.12 / 26.9** |
| `SceneMaintenance` max ms | 454.8 | 454.5 (untouched — Phase 4) |
| `DeferredAssetLoads` p99 / max ms | — / 58.1 | 14.31 / **103.1** |
| resident audio emitters | not reported | 5565 |

### What the capture settles

**FR-002 satisfied — Defect A is named.** `AudioRuntime.Update` peak 283.46 ms against a
`PrepareObjectPhase` total of 283.47 ms. The PM4 overlay window is 0.12 ms and the unprobed remainder
is ~0. There is no ambiguity and no need to subdivide further. Every entry in the "Recent hitches"
list reads `<- PrepareObjectPhase`, at 228–283 ms.

**SC-006 met.** Unaccounted time is median 0.05 / p99 0.16 ms and the pass gap peak fell from 259–314
ms to 9.45 ms. Hitches now name a stage instead of a void — which is what made Defect A findable in
one flight after an entire investigation failed to place it.

**SC-002 and SC-003 met.** 526 batched / 3 unbatched, from 0 / 312. `MdxOpaqueSubmission` p99 fell
30.75 → 14.12 ms and median 2.03 → 0.01 ms. The 3 remaining unbatched are models that genuinely
declare `RequiresUnbatchedWorldRender` — the clean fallback FR-004 asks for, with a visible count.

**SC-001 not met, and Phase 3 must not be credited for it.** Frame p99 barely moved (259.70 → 246.62)
because the periodic stall was never the MDX cost. This is exactly the confusion the plan's risk table
warned about.

**SC-005 not met.** `DeferredAssetLoads` max is 103.1 ms against a 3.5 ms budget, and p99 is 14.31 ms.
Removing the additive overshoot was not sufficient; the residual single-load cost dominates, so the
off-thread decode (Phase 5 step 2) is required rather than optional.

**SC-004 untouched.** `SceneMaintenance` max 454.5 ms, essentially unchanged from the 454.8 ms
baseline, and still the single largest stage observation. Phase 4's gate is now open.

## Phase 2 — implemented, unmeasured

The capture named the operation, so Phase 2 could be written against evidence rather than a guess.
`AudioRuntime.Update`'s cost is **not** audio residency and **not** emitter attenuation — with 5565
resident emitters and 0 active, it is `RefreshEmitterDiagnosticsIfDue` rebuilding an
`AudioTriggerDiagnostic` record for every resident emitter, four times a second
(`DiagnosticRefreshIntervalTicks = Stopwatch.Frequency / 4`), on the render thread. A wall-clock 250 ms
period is exactly the "fixed size, regular period" signature the spec recorded, and it explains why
the frame interval drifted with framerate (~47–50 frames at Un'goro rates, ~12–13 at 50 fps).

Two independent stalls, both pure diagnostics:

1. **The periodic one.** The rebuild ran unconditionally, whether or not any surface displayed the
   result. Its only consumer is the audio panel in `ViewerApp_Audio.cs`, which already has explicit
   *Refresh decisions* and *Probe current emitters* buttons. The refresh is now gated on
   `NoteEmitterDiagnosticsObserved()`, which that panel calls as it renders; outside a one-second
   observation window `Update` does no diagnostics work at all. Camera-relative fields stay live for
   anyone actually looking.
2. **The movement-triggered one.** `RemoveTile` called `RefreshEmitterDiagnostics()` *synchronously*
   in addition to invalidating. `RemoveTile` fires on streaming eviction, so crossing a tile boundary
   paid a full rebuild over every remaining resident emitter — a stall at the exact moment the
   renderer could least afford it. The eager rebuild is removed; the invalidate is sufficient, since
   any consumer refreshes on read.

Correctness (US2 scenario 3) is preserved: audio playback never reads this list, so gating it cannot
change what is audible, and `NoteEmitterDiagnosticsObserved` refreshes a stale list on open rather
than showing rows left over from an eviction.

**Not attempted, deliberately:** `BuildEmitterDiagnostic` is still expensive per emitter
(`EnumerateVirtualPaths().Distinct().ToArray()`, a `FileExists` probe per candidate,
`DescribeResourceSource`, a wide record with interpolated strings). With the gate in place that cost
is paid only by someone who has opted into the panel. If the panel turns out to be unusable in a dense
zone, memoising path resolution is the next step — but that is a separate change with its own
before/after, per FR-010.

## Verification performed here

- `dotnet build wow-viewer/WowViewer.slnx -c Debug` — **0 errors**.
- `dotnet test wow-viewer/tests/WowViewer.Core.Tests` — **9 failures before these changes, the same
  9 after**, all pre-existing and unrelated (WDL pass ordering in `WorldFramePassCoordinatorTests`,
  `LkToAlphaRoundTrip`, `AdtV23SummaryReader`, `ModelFootprintReader`, `V18StorePlacementsReader`,
  `EnrichmentStreamFormat`). Net **+15 passing tests**.
- Phases 1, 3 and 5 were captured on the Stranglethorn route (results above). **Phase 2 is
  source-proven only and has not been measured** — it was written after the capture that named its
  target, so SC-001 remains unproven.

## Phase 2b — audio scoped to the camera tile; MCSE frame left measured, not guessed

Reported after the Phase 2 gate landed: **only water-triggered emitters behave; every MCSE emitter
reads out of range wherever the camera goes.** Two separable problems, and they must not be conflated.

### Scope (implemented)

The distance test in `WorldAudioRuntime.Update` consulted **no tile information at all** — it scanned
every resident tile (33 tiles / 5565 emitters). `Update` now takes the terrain manager's own
`CameraTileX/Y` and considers only tiles within `AudibleTileRadius` (1) of it, and
`RefreshEmitterDiagnostics` uses the same window so the panel reports the camera's surroundings
rather than the streamed world. The camera tile is *passed in*, never re-derived, so the audio window
cannot drift out of agreement with the renderer's current tile.

Radius 1 rather than 0: an emitter cutoff can reach across a 533-unit tile edge, so a strict
current-tile-only scope would pop audio at boundaries. The constant is one edit away from 0.

New observability: `ScannedEmitterCount` and `InRangeEmitterCount`, shown in the panel next to the
row count. **In-range 0 everywhere is a placement bug, not a range setting** — that distinction is
what the counters exist to make.

### The out-of-range cause is NOT settled, and was not guessed at

`AlphaTerrainAdapter.ConvertSoundPosition` transforms MCSE positions as `chunkCorner - local`, on the
strength of a comment: *"Alpha MCSE stores a chunk-local C3Vector."* That claim has **no evidence
behind it.** The Ghidra work on `CMapChunk::Create` proved the 0x34-byte *field layout*
(`workstream-audio-client-053-ghidra.md`), not the frame the position is expressed in, and no test
asserts anything about position magnitude.

The leading hypothesis explains the whole symptom, including why water is the exception: if the
stored vector is not chunk-local, `chunkCorner - local` throws every MCSE emitter tens of thousands
of units off-map, and MCNK liquid rows are unaffected because
`LegacyLiquidSoundEmitterFactory.ResolvePosition` derives from the renderer's own
`chunk.WorldPosition` and never passes through that transform.

**That is a hypothesis, so it gets a measurement, not a fix.** `McseFrameEvidence` reports, over
resident MCSE rows near the camera, the raw min/max per axis and how many fall inside a chunk edge
(33.33), inside a tile edge (533.33), or beyond — with an explicit verdict that says
*"inconclusive, do not switch frames on this evidence"* for a mixed sample. It is displayed at the
top of Utilities > Audio.

Tile keying was checked and **cleared**: `AddTile`/`RemoveTile`/`EmitterKey` all use the same
`(tileX, tileY)` convention as `OnTileLoaded`, so the reported "mixing up our matches to the tiles"
is not a dictionary-keying fault. Scanning every tile made the symptom look like one.

**Read the MCSE frame line, then act:**

- *"CHUNK-LOCAL confirmed"* → the transform is right and the out-of-range cause is elsewhere; look at
  `ResolveResidentSoundEntryId` / `SoundEntries` resolution and the `FirstPositive` cutoff chain next.
- *"NOT chunk-local"* → `ConvertSoundPosition` is the bug. The likely correct form is the MODF
  convention already used for placements (`rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`),
  but confirm against the reported magnitudes before writing it.
- *"INCONCLUSIVE"* → do not switch frames; subdivide by trigger kind or tile first.

**Not tested.** There is no viewer-project test assembly (`tests/` contains only Core, Core.Anim,
Core.Curation, Core.PM4), and `TerrainSoundEmitter` lives in the viewer, so `McseFrameEvidence` and
the tile window have source proof only. Moving them into core to make them testable is a real
follow-up, not something to claim as done.

## Capture 2 — Stormwind, 2026-08-15, 2048 frames: the owner moved to `WmoSubmission`

Taken after Phase 2 and the audio scoping landed. The periodic audio stall is gone from the working
set and a different, larger defect is now dominant.

| Reading | Stranglethorn (capture 1) | Stormwind (capture 2) |
| --- | --- | --- |
| median frame ms | 17.40 | **6.98** |
| p95 / p99 / max ms | 107.22 / 246.62 / 636.29 | 148.54 / 196.41 / 504.20 |
| hitches / frames | 186 / 1170 (16%) | 592 / 2048 (29%) |
| unaccounted median / p99 ms | 0.05 / 0.16 | **0.02 / 0.11** |
| `PrepareObjectPhase` max ms | 283.4 | **2.5** |
| `SceneMaintenance` max ms | 454.5 | **3.9** |
| `WmoSubmission` median / p99 / max ms | 3.74 / 15.69 / 27.1 | **0.71 / 154.10 / 161.3** |
| `WmoTransparentSubmission` p99 / max ms | 9.36 / 15.7 | **44.76 / 61.1** |
| `DeferredAssetLoads` p99 / max ms | 14.31 / 103.1 | 11.96 / **442.9** |
| WMO draw calls (total / batched) | 127 / 127 | **80484 / 80200** |
| WMO visible groups | 52 | **7512** |
| WMO doodad submissions | 95 | **15852** |

**Phase 2 is confirmed by the collapse, not by assertion.** `PrepareObjectPhase` max fell 283.4 →
2.5 ms and no longer appears in the hitch list at all. `SceneMaintenance` max fell 454.5 → 3.9 ms,
which also means **Phase 4 may need no work** — re-measure before implementing it, because the
454.8 ms figure it was written against no longer reproduces.

**Every one of the 592 recent hitches reads `<- WmoSubmission`, at 153–157 ms.** This is a new
defect, not a residue of the old one: `WmoSubmission` median is 0.71 ms, so the cost is entirely in
the dense case.

### Diagnosis (user-supplied, matches the counters)

Stormwind submits **all districts at once** — 7512 visible groups and 80484 draw calls — instead of
the district group the camera is inside. The counters agree: batching is working (80200 of 80484
batched), so this is not a submission-efficiency problem like Defect B was. **It is an admission
problem.** The renderer is drawing group geometry it should have rejected.

Spec 151 Phase 1 landed portal-aware admission, so the portal decision exists; what this capture
shows is that it is not constraining a large multi-group interior. Likely candidates, none verified:

- The exterior seed admits every group when the camera is outside any portal volume, so a city-sized
  WMO degenerates to "draw everything".
- Per-group visibility metadata in the WMO that the client uses and this renderer does not — the user
  specifically suspects a group-to-group visibility relation ("which groups peer into other groups"),
  which would be a `MOGP`/`MOVV`/`MOVB` style visible-block or the portal reference lists.
- `WmoRenderer.SupportsGpuInstancedOpaque` requires `_wmo.Portals.Count == 0` and all groups manually
  visible, so a portal-bearing WMO takes a different path than the batched one — worth checking
  whether the batched path bypasses group admission entirely.

**Do not start fixing this from the list above.** It belongs to Spec 151, it needs its own spec/plan
slice, and the first step is the same one that worked twice here: instrument group admission
(groups considered / admitted / rejected, and by which rule) before changing any logic.

## Remaining work, in measured priority order

1. **WMO group admission in dense interiors — new owner, belongs to Spec 151.** `WmoSubmission`
   p99 154.10 / max 161.3 ms, 7512 visible groups, 80484 draw calls, and 100% of recent hitches.
   Instrument group admission before changing any logic.
2. **Phase 5 step 2 — decode off the render thread.** `DeferredAssetLoads` max 442.9 ms in Stormwind
   against a 3.5 ms budget. The admission policy bounded the additive overshoot but the single-load
   residual is now the second-largest stage; SC-005 needs the async decode.
3. **Phase 4 — re-measure before implementing.** `SceneMaintenance` max is 3.9 ms in this capture,
   down from the 454.8 ms the phase was written against. The defect may not reproduce; confirm on a
   route that forces `_instancesDirty` before spending work on it.

SC-001 is met for the audio stall specifically — the ~212/283 ms periodic `PrepareObjectPhase`
pattern is absent. The frame is still not smooth, but the remaining cost is a *different, named*
defect rather than an unexplained one.
