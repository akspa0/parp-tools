# Implementation Plan: Object Draw-Call Reduction

**Branch**: `207-object-draw-call-reduction` | **Date**: 2026-09-02 | **Spec**: [spec.md](./spec.md)

## Summary

Objects are ~97% of the frame (WMOs ≈ 55.5 ms, doodads ≈ 41.2 ms, terrain ≈ 3.3 ms) and the cost is
per-draw CPU overhead at ~4.6 µs per submission. Reduce draw calls, in the order of ratio-per-risk:
instance the faded doodads that are currently excluded from instancing, then cull WMO groups on their
own bounds instead of admitting them wholesale, then use the client's own skin profiles so distance
reduces geometry rather than only opacity.

## Technical Context

**Language/Version**: C# / .NET 10
**Primary Dependencies**: Silk.NET (OpenGL), ImGui.NET
**Testing**: xUnit (`tests/WowViewer.Core.Tests`) — note **no test project references the viewer**, so
logic that must be tested belongs in `WowViewer.Core*`
**Target Platform**: Windows desktop, OpenGL
**Performance Goals**: 8,989 opaque MDX draws → ≤900 (SC-001); `MdxOpaqueSubmission` median 10.66 →
≤3 ms; `WmoSubmission` median 30.00 → ≤15 ms; frames over 33.3 ms 98.9% → <50%
**Constraints**: no visual regression (pixel-compared); every change runtime-switchable (FR-010);
every requirement paired with a counter (FR-011)
**Scale/Scope**: 8,989 visible MDX of 43,715 resident; 215 of 664 WMOs; 3,584 terrain chunks

## Constitution Check

- **II — one mechanism, not a parallel one**: US1 extends the existing instancing path rather than
  adding a fifth batching route. US2 extends the existing admission evaluator.
- **III — real-data validation**: every SC is read from the operator's own flight on MoP 5.0.1,
  Valley of the Four Winds, against recorded baselines.
- **Measure before building toward**: T1.3 (distinct *visible* models) lands before the batching work
  is tuned, because it is the floor US1 is measured against and today's panel figure is a *resident*
  count.

## Phase 0 — Baseline (blocking, cheap)

Nothing here changes rendering. It exists so the before/after is readable, which spec 201 established
is otherwise impossible.

- Record on a fixed camera: opaque MDX submissions, instanced vs state-hoisted vs unbatched, the
  per-gate breakdown, WMO groups considered/admitted/rejected with per-rule attribution, and stage
  medians.
- **Gate**: if `GatedOpaqueFadeBelowThreshold` is a small share of submissions, the fade band is not
  where the cost is and US1 must be re-planned before it is built.

## Phase 1 — Doodad instancing (US1)

**The fix is to split the batch, not to delete the gate.** One instanced draw shares one blend state;
partially-transparent geometry in the opaque pass needs blend on and depth-write off. So emit two
instanced batches per model:

| batch | admits | blend | depth write |
|---|---|---|---|
| opaque | `fade >= 0.999` | off | on |
| faded | `fade < 0.999` | on | off, drawn after the opaque batch |

Per-instance fade is already uploaded (attribute 10, divisor 1) and already consumed by the fragment
shader, so no shader change is required — this is the CPU-side batching decision only.

Draw calls per model go from *N instances* to *at most 2*.

**Risk**: faded instances are unsorted within their batch. They are near-transparent by definition,
so the artifact should be invisible; if it is not, sort the faded batch back-to-front by model
centroid distance, which is one sort over a small list rather than per instance.

**Rollback**: the existing `GPU instancing for opaque models` toggle already disables the whole path.

## Phase 2 — WMO group admission (US2)

Today a group inherits its placement's admission: 80 considered, 80 admitted, 0 rejected. Evaluate
each group on its own bounds — frustum and projected size — **independently of any portal graph**,
which is what makes this work on the 62.5% of admissions currently coming from the conservative
fallback and the 0% coming from portal traversal.

Split the fallback counter by reason first (T2.1) so "file has no portal data" is distinguishable
from "portal data present, graph not built": the first needs group-bounds culling, the second is a
bug, and one counter today hides both.

**Risk**: over-culling interiors, visible as popping walls. Mitigated by reusing the terrain/MDX
projected-size thresholds rather than inventing new ones, and by keeping the whole thing behind a
toggle for A/B.

## Phase 3 — Skin-profile LOD (US3)

Select an authored skin profile by projected size. No mesh generation: MoP M2s ship several
(`nViews`/`ofsViews`), and spec 193 already covers extracting them. Assets with one profile are
unaffected, which is the FR-009 no-op requirement.

Once LOD exists, the 80% fade annulus — 36% of the visible disc, currently full-detail geometry
rendering nearly nothing — can be replaced by LOD plus a hard cull.

## Complexity Tracking

| Decision | Simpler alternative rejected | Why |
|---|---|---|
| Two instance batches per model | Delete the `>= 0.999` gate | One shared blend state would render faded geometry as opaque, trading draw calls for popping |
| Group culling without portals | Fix portal traversal first | Portal traversal currently admits **0**; group bounds work regardless of whether a graph exists |
| Authored skin profiles | Generate decimated LOD | The client already ships the levels; generating them is a separate pipeline and a separate spec |

## Progress Tracking

- [ ] Phase 0 — baseline recorded, gate evaluated
- [ ] Phase 1 — doodad instancing (US1)
- [ ] Phase 2 — WMO group admission (US2)
- [ ] Phase 3 — skin-profile LOD (US3)
