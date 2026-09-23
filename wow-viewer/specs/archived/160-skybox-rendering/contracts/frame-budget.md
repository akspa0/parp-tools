# Contract: Sky Frame-Cost Budget

**Spec**: [../spec.md](../spec.md) | **Satisfies**: FR-022, FR-023, SC-008, SC-009 | **Research**: R6

**Status**: ⛔ **UNFILLED — Phase 0 blocks on this.** The budget is set from a measured pre-change
baseline, not asserted in advance. Once any sky code changes, the baseline is unrecoverable.

---

## What is measured

Two stages, both already instrumented in `WorldRenderFrameHistory` — no new profiler (research R6):

| Stage | Enum | Covers |
|---|---|---|
| Sky | `WorldRenderStage.Sky` (3) | Procedural dome draw |
| Skybox backdrop | `WorldRenderStage.SkyboxBackdrop` (4) | Client skybox model draw |

For each: **p50, p99, max**, plus hitch attribution.

---

## Measurement validity

The window **must** report `CameraMovedDuringWindow` / `CanDemonstrateMovementBehavior` as true.

A static-camera capture is not valid evidence for this contract. The project has a recorded instance
of a renderer profiler producing false null results precisely because it used a static camera and no
frame-time distribution — a p99 and a max are the whole point here, and a still camera cannot
produce a meaningful one.

---

## Baseline capture (Phase 0, user-run)

Capture on **at least two maps** — one dense, one sparse — so the budget is not fitted to a single
scene. Record for each:

| Field | Value |
|---|---|
| Client build identity | _(to fill)_ |
| Configured client root | _(to fill — root reported, never hardcoded in source)_ |
| Map | _(to fill)_ |
| Camera moved during window | _(must be true)_ |
| Frames in window | _(to fill)_ |
| `Sky` p50 / p99 / max (ms) | _(to fill)_ |
| `SkyboxBackdrop` p50 / p99 / max (ms) | _(to fill)_ |
| Sky-attributed hitches | _(to fill)_ |

Commands are in [../quickstart.md](../quickstart.md).

---

## The budget

Filled from the table above at the end of Phase 0. Expressed as **both** a relative delta and an
absolute ceiling, because either alone is gameable:

- **Delta gate**: post-change `Sky` + `SkyboxBackdrop` p99 ≤ baseline p99 × _(factor, to fill)_.
- **Absolute gate**: post-change `Sky` + `SkyboxBackdrop` p99 ≤ _(ms, to fill)_.
- **Hitch gate**: zero new hitches attributed to either stage.

Phase 3 is expected to raise steady-state sky cost somewhat — a five-band interpolation does more
work per fragment than a two-colour `mix`. The budget must be set with that known increase in mind
rather than pinned to the current cost and then waived later.

---

## Disabled-sky gate (FR-023, SC-009)

With sky rendering disabled, **both** stages must measure zero — not "small". Non-zero cost with sky
off means source evaluation or draw submission is still running, which is the defect FR-023 names.

---

## Why this contract exists

Specs 151, 152, and 153 are actively working the renderer's frame time, and 151 currently owns the
remaining measured hitching. This spec touches the same per-frame path. Without a gate, sky work
could silently spend the budget those specs are trying to recover, and the regression would surface
as *their* problem rather than this one's.
