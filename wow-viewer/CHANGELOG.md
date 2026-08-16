# Changelog — parp-tools WoW Viewer

Release notes for each tagged version live in [`docs/releases/`](docs/releases/) and are what the
GitHub Release publishes. This file is the index and the short version.

## v0.5.2.1 — 2026-08-15

Out-of-band patch. **v0.5.2 shipped with known, unresolved rendering jank**; this fixes the causes
that were found, and names the one that was not.

### Fixed

- **Periodic ~283 ms render-thread stall, four times a second.** The audio *emitter diagnostics*
  panel was rebuilding a record for every resident sound emitter (5,565 in a dense zone) on a 250 ms
  timer, **whether or not anything was displaying it**, plus a second synchronous rebuild on every
  streaming tile eviction — i.e. while moving. Now gated on being observed, and the eviction path
  only invalidates. `PrepareObjectPhase` max **283.4 → 2.5 ms**, and it no longer appears in the
  hitch list.
- **Opaque MDX drew one call per instance.** The route planner's predicate body was a hardcoded
  `return true`, so 100% of instances took the per-instance fallback while the batching path sat
  unused. It now consumes the renderer's own `RequiresUnbatchedWorldRender` declaration — the same
  contract the WMO doodad path already used. **0 batched / 312 unbatched → 526 batched / 3
  unbatched**; `MdxOpaqueSubmission` p99 **30.75 → 14.12 ms**, median **2.03 → 0.01 ms**.
- **Batched and unbatched MDX output could diverge.** `RenderInstance` ignored the per-renderer
  wireframe flag that `RenderWithTransform` honours, so a wireframe-flagged model would have drawn
  filled once batched. Now declared as requiring the unbatched path.
- **Deferred asset loads checked their budget only *between* loads**, so a started load ran to
  completion — 58 ms against a 3.5 ms budget. Loads are now admitted against a learned per-asset-kind
  cost estimate before starting.
- **Audio range ignored which tile the camera was on**, testing distance against every streamed tile.
  Now scoped to the camera tile and its ring, using the terrain manager's own camera tile rather than
  re-deriving it.

### Added

- **`PrepareObjectPhase` has a stage timer.** It was the only one of eleven frame passes without one,
  which is why a 283 ms stall stayed invisible for an entire investigation. A test now fails the
  build if a pass is added without a timer, a stage is recorded by nothing, or a stage is
  double-counted. Unaccounted frame time: **259–314 ms pass gap → median 0.02 / p99 0.11 ms**.
- **`McseFrameEvidence`** — measures what coordinate frame decoded MCSE sound-emitter positions are
  actually in, reported in Utilities > Audio. Added instead of guessing at a fix (see Known issues).
- **Live on/off switch for opaque MDX batching** in Utilities > Perf, so a before/after comparison is
  one flight rather than two builds.
- Submission counters, scanned/in-range emitter counts, and deferred-load budget counters, so each of
  the above is verifiable rather than asserted.

### Known issues

- **Dense WMO interiors are slow; Stormwind is the worst case.** The city submits all districts at
  once — 7,512 visible groups, 80,484 draw calls — instead of the district the camera occupies. This
  is a group *admission* problem, not batching (80,200 of those calls are correctly batched). **Not
  fixed here**; it is the next work item.
- **Deferred loading can still spike ~443 ms on one load** in a dense zone. The new policy bounds the
  additive overshoot, but a single synchronous decode costs what it costs. Moving decode off the
  render thread is the real fix and is not done.
- **`MCSE` sound emitters read as permanently out of range**; only water-triggered emitters behave.
  The position transform assumes a chunk-local frame on the strength of an unevidenced code comment.
  This release ships the measurement, not a guessed fix.

### Notes

The leading theory going in — per-frame allocation churn in the scene graph — was **refuted by
measurement** (median world-render CPU 0.33–8.58 ms; traversal max 0.22 ms) and the planned scene
flattening work was suspended rather than continued on momentum. Restoring MDX batching did **not**
remove the hitching and is not credited with having done so.

## v0.5.2 — 2026-08-15

Portal-aware WMO visibility, per-ADT scene graph, bounded camera-centered streaming, audio runtime,
Alpha 0.5.3 time-of-day clock, LIT lighting decode, PM4 coordinate solve, and the five-destination
workbench UI. 238 commits since `v0.5.1-build1`.

Full notes: [`docs/releases/v0.5.2.md`](docs/releases/v0.5.2.md)

## v0.5.0

Full notes: [`docs/releases/v0.5.0.md`](docs/releases/v0.5.0.md)
