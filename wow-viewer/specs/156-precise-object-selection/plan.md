# Implementation Plan: Precise Object Selection, PM4 Match Confirmation, and a World-Space Cursor

**Branch**: `v0.5.3-dev` (this repo keeps all specs on one branch; no per-feature branch is created)
**Date**: 2026-08-16
**Spec**: [spec.md](./spec.md)

## Summary

Four independently-shippable capabilities: triangle-precise picking for regular placed objects (US1),
triangle-precise picking for PM4 overlay objects (US2 — the cheapest, since its geometry is already
assembled and cached in memory today), a durable human-confirmed PM4↔placement match library (US3), and
a world-space cursor marker (US4).

Phase ordering deliberately **inverts the spec's priority order for the first two phases**: US2 ships
first as Phase 1, because its geometry already exists in memory (`Pm4OverlayObject.Triangles`) and the
change is a single test-function swap, which proves the whole ray-vs-triangle approach end-to-end at
near-zero risk before US1's much larger mesh-availability problem is touched. US1 remains the higher-value
story and the reason this spec exists; it is simply not the cheapest one to prove the mechanism with.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: `WowViewer.Core.PM4` (assembled PM4 object geometry), `WowViewer.Core.IO` (WMO/M2/MDX readers), Silk.NET.OpenGL (marker rendering), existing `BoundingBoxRenderer` world-space marker primitives

**Storage**: One new persisted artifact — the confirmed-match library (US3). Identifiers, paths, and provenance only; never client asset bytes (Data Policy). Format decided in Phase 3, following this project's existing convention for generated records (JSON under `output/`, matching Spec 155's sweep reports).

**Testing**: xUnit (`WowViewer.Core.Tests`) for geometry math and match-library semantics; manual in-viewer verification for anything visual (US1/US2 picking behavior, US4 marker occlusion) — a headless test cannot confirm "clicking the visible surface selects it"

**Target Platform**: Desktop viewer (`WoWViewer` / `ParpToolsWoWViewer`)

**Project Type**: Desktop 3D viewer

**Performance Goals**: No perceptible added hover/click latency in a dense scene (SC-004). Note this is deliberately unquantified in spec.md — its checklist flags SC-004 as the one non-numeric success criterion, to be pinned to a real measured budget during Phase 1 rather than invented here.

**Constraints**: Precise testing must never run unbounded over the whole scene (FR-003); existing multi-candidate disambiguation UX must survive unchanged (FR-005); no PM4 geometry may become a regular object's live selection volume (FR-009).

**Scale/Scope**: Four user stories, one new geometry-math surface (ray-vs-triangle + candidate filtering), one new persisted artifact (match library), one new rendered element (world cursor marker).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design — see below.*

| Principle | Check | Status |
|---|---|---|
| I. Repo Independence | All new code under `wow-viewer/src/`; no outside paths | PASS |
| II. Library-First | Ray-vs-triangle math → `WowViewer.Core.Runtime` (or `Core`), not `ViewerApp`. Match library → `WowViewer.Core.PM4` (alongside the existing correlation extractor it reuses). `ViewerApp` partials stay orchestration/UI | PASS |
| III. Real-Data Validation | Every story validated against real staged clients and real loaded scenes; PM4 work against real PM4 tiles | PASS (pending per-phase execution) |
| Format Reader/Writer Ownership | No new parser. PM4 geometry is consumed as already assembled by `BuildPm4TileObjects`; WMO/M2 mesh data comes from existing readers. US3 explicitly reuses the existing `Pm4SurfaceCorrelationExtractor` fingerprint rather than defining a new one (FR-015) | PASS |
| Terrain Alpha Risk Area | Not touched — this feature changes picking and adds a marker; it does not alter MCAL decode, alpha packing, or terrain shader blending | PASS (N/A) |
| One Phase at a Time | Five phases (0-4), each with its own validation gate | PASS |
| Bite-Sized Plans | Each phase ≤10 steps | PASS |
| No Client Path Assumptions | Roots stay runtime configuration | PASS |
| Core.Anim exclusion | No phase touches `WowViewer.Core.Anim` | PASS |
| Data Policy | Match library stores identifiers/paths/provenance only, never asset bytes (FR-016) | PASS |

## Project Structure

### Documentation (this feature)

```text
specs/156-precise-object-selection/
├── spec.md                     # already written
├── checklists/requirements.md  # already written
├── plan.md                     # this file
├── research.md                 # Phase 0 output
├── data-model.md               # Phase 1 output
├── contracts/
│   ├── hit-result.md
│   └── confirmed-match.md
└── quickstart.md               # Phase 1 output
```

### Source Code (repository root: `wow-viewer/`)

```text
src/core/WowViewer.Core.Runtime/
└── Picking/
    ├── RayTriangleIntersector.cs      # NEW — Möller–Trumbore or equivalent; pure math, unit-testable
    └── PreciseHitResult.cs            # NEW — distance, world point, which test kind produced it

src/core/WowViewer.Core.PM4/
└── Matching/
    ├── ConfirmedMatch.cs              # NEW — the record (PM4 identity ↔ placement identity + provenance)
    └── ConfirmedMatchLibrary.cs       # NEW — load/save/query/retract; reuses existing
                                       #        Pm4SurfaceCorrelationExtractor for candidate surfacing

src/viewer/WoWViewer/
├── Terrain/WorldScene.cs              # EXTEND — swap PM4 pick from RayAABBIntersect to triangle test
│                                      #          (Phase 1); add precise pass for scene objects (Phase 2)
├── Terrain/WorldAssetManager.cs       # EXTEND — retain/serve pickable mesh geometry (Phase 2)
├── ViewerApp_ClickSelection.cs        # EXTEND — unchanged disambiguation UX, precise hits feeding it
├── ViewerApp_MatchConfirmation.cs     # NEW partial — confirm/retract/review UI (Phase 3)
└── ViewerApp_WorldCursor.cs           # NEW partial — per-frame marker (Phase 4)

tests/WowViewer.Core.Tests/
├── RayTriangleIntersectorTests.cs     # NEW
└── ConfirmedMatchLibraryTests.cs      # NEW
```

**Structure Decision**: Geometry math goes in `Core.Runtime/Picking/` as pure, unit-testable functions
with no scene dependency — this is the part that must be provably correct in isolation. Scene integration
stays in `WorldScene.cs` where the existing pick methods already live (rather than a speculative refactor
of a 15k-line file). The match library lives in `Core.PM4/Matching/` next to the correlation code it
reuses.

## Phases

### Phase 0 — Establish whether pickable mesh data is actually reachable

The spec's single largest open assumption, deferred explicitly to planning: *"The render mesh data …
is assumed to be reachable from the code path that performs picking."* Partially answered already during
planning (see research.md) — **the answer is "parsed, then discarded"**, which materially shapes Phase 2.
This phase closes the remainder.

1. Confirm what `WmoMeshSummary` / `MdxCollisionMeshSummary` retain vs discard (research.md §1 — done:
   counts, bounds, and *sampled* footprint vertices only; no full vertex/index arrays).
2. Confirm the existing re-read-on-demand-and-cache precedent (`TryGetMdxCollisionSummary`,
   `WorldAssetManager.cs:471-500`) — this is the pattern Phase 2 should follow rather than inventing one.
3. Measure the memory cost of retaining full pickable geometry for a realistic dense scene, versus
   re-reading on demand — decide retention policy from that measurement, not from assumption.
4. Confirm whether M2/MD20 models can supply pickable geometry at all on the paths that matter (the MDX
   collision path explicitly excludes MD20/MD21 today — `WorldAssetManager.cs:492`), and record which
   model kinds will therefore fall back to bounding-volume picking under FR-002.
5. Pin SC-004's performance budget to a real measured number from the existing frame-stats system, so
   "no perceptible added lag" becomes checkable (spec checklist flags this as its one unquantified SC).

**Exit gate**: research.md answers 1-5 with measurements, not estimates. No production code changes.

---

### Phase 1 (US2) — Triangle-precise PM4 picking

Ships first despite being P2: the geometry is already assembled and cached
(`Pm4OverlayObject.Triangles`), so this proves ray-vs-triangle end-to-end with essentially no data-plumbing
risk.

1. Implement `RayTriangleIntersector` in `Core.Runtime/Picking/` — pure function, no scene dependency.
2. Unit-test it hard: hit/miss, back-face behavior, degenerate (zero-area) triangles, grazing rays,
   ray origin inside geometry.
3. In `TryPickPm4ObjectByRay` (`WorldScene.cs:12670`), keep the existing AABB test as the cheap first-pass
   filter, then run the triangle test against the surviving candidates' `Triangles` (FR-003, FR-004).
4. Fall back to the existing AABB result when an object's triangle list is empty (FR-004's explicit
   requirement — a PM4 object with no decoded geometry must not become unpickable).
5. Verify the existing multi-candidate disambiguation overlay still appears for genuinely ambiguous hits
   (FR-005) — this must not regress.
6. Real-data validation: in a real PM4 overlay scene, click inside two objects' overlapping bounding boxes
   at a point on only one object's actual surface; confirm the correct object is selected (SC-002).

**Exit gate**: SC-002 met, demonstrated live in the viewer.

---

### Phase 2 (US1) — Triangle-precise picking for regular placed objects

The spec's headline story, and the larger one, because Phase 0 establishes the mesh data is not currently
retained.

1. Decide and implement the pickable-geometry source per Phase 0 step 3's measurement: either retain
   full vertex/index data alongside the existing mesh summaries, or re-read on demand and cache (following
   the existing `TryGetMdxCollisionSummary` precedent). Do not decide this here — decide it from the
   measurement.
2. Extend `WorldAssetManager` to serve pickable geometry per model key, cached, with an explicit
   "not available" result rather than an exception (FR-002's fallback depends on this being a normal state).
3. In `TryRayIntersectInstanceBounds` / `AppendSceneObjectPickHits` (`WorldScene.cs:13273`, `:12563`),
   keep the existing oriented-bounding-box test as the first-pass filter, then run the triangle test in
   the model's local space against surviving candidates (FR-001, FR-003).
4. Fall back to today's exact bounding-volume behavior whenever pickable geometry is unavailable —
   unloaded, failed, degenerate, or an unsupported model kind per Phase 0 step 4 (FR-002, FR-010).
5. Remove the constant WMO `(2,2,2)` / MDX `(1,1,1)` pad **only** on the precise path; the fallback path
   must keep today's padding exactly, since changing it would alter behavior for objects this feature
   doesn't cover (FR-010).
6. Verify the disambiguation overlay still behaves identically for same-distance hits (FR-005).
7. Unit tests: local-space ray transform correctness; fallback selection when geometry is absent;
   nearest-hit ordering across mixed precise/fallback candidates.
8. Real-data validation: on a known non-convex object (L-shaped building or sparse foliage), clicking
   inside the bounding volume but outside the visible mesh does not select it; clicking the visible surface
   does (SC-001).
9. Performance validation against Phase 0 step 5's measured budget in a dense scene (SC-004).

**Exit gate**: SC-001 and SC-004 met, both demonstrated against real data; SC-005 (no behavior change for
fallback-state objects) confirmed before/after.

---

### Phase 3 (US3) — Confirmed-match library

Independent of Phases 1-2 (spec.md's own Independent Test notes this works with today's box-based
selection), but ordered after them because a confirmation is only as trustworthy as the click that
produced it.

1. Define `ConfirmedMatch` (`Core.PM4/Matching/`) per data-model.md — PM4 identity, placement identity,
   provenance (who/when/why), status.
2. Implement `ConfirmedMatchLibrary`: load, save, query-by-PM4-identity (FR-014), confirm, retract.
3. Retraction is recorded as its own event, never a silent delete (FR-013) — the store is append-oriented
   for history, with current status derived.
4. Conflict handling: confirming a PM4 object that already has a *different* confirmed match surfaces the
   conflict rather than silently overwriting (spec.md edge case).
5. Candidate surfacing: reuse the existing `Pm4SurfaceCorrelationExtractor` fingerprint to list other
   unconfirmed PM4 objects sharing a confirmed object's fingerprint (FR-015) — candidates only, never
   auto-confirmed (FR-012).
6. Durable rejection: rejecting a surfaced candidate is recorded so it is not re-surfaced as new every
   session (spec.md edge case).
7. `ViewerApp_MatchConfirmation.cs`: select a PM4 object + a placement, confirm with a stated reason;
   view an object's existing match; retract.
8. Unit tests: confirm/query/retract round-trip; persistence across a simulated restart; no code path
   writes a confirmation from a score or proximity alone (SC-009, verifiable by inspection + test).
9. Real-data validation: confirm a real match in a real scene, restart the viewer, confirm it persists
   and answers a lookup (SC-007).

**Exit gate**: SC-007, SC-008, SC-009 met.

---

### Phase 4 (US4) — World-space cursor marker

1. Per-frame hover ray already exists (`ScreenToRay`, called every frame by `UpdateHoveredAssetInfo`) —
   reuse it; do not add a second ray computation.
2. Resolve the marker's world point from the existing terrain raycast (`TryRaycastTerrain`), and, once
   Phase 2 has landed, from the same precise object hit path rather than a separate one (FR-008).
3. Render the marker through `BoundingBoxRenderer`'s existing world-space marker primitives
   (`BatchPin`/`BatchOctahedron`), which already submit through the normal view*proj pipeline — this gives
   correct depth/occlusion for free (FR-006).
4. Explicit no-hit state: show nothing (or a clearly distinct indicator) when the cursor is over sky —
   never a marker parked at a misleading fixed distance (FR-007).
5. Verify the marker updates when the camera moves even if the mouse is still (spec.md edge case).
6. Real-data validation: in a scene with elevation, confirm the marker is correctly occluded by nearer
   terrain rather than drawn on top (SC-003).

**Exit gate**: SC-003 met, demonstrated live.

## Constitution Check — Post-Design Re-Check

Re-evaluated after Phase 1 design artifacts. No new violations: the one new persisted artifact (match
library) stores identifiers/paths/provenance only, satisfying the Data Policy gate it was checked against
pre-Phase-0; no new format reader is introduced (PM4 geometry and the correlation fingerprint are both
consumed as-is from existing owners); geometry math is isolated in a library with no scene dependency,
satisfying Library-First. The phase inversion (US2 before US1) is a sequencing decision for risk, not a
priority change — spec.md's stated priorities are unchanged, and each phase still ships independently.
PASS.

## Complexity Tracking

*No Constitution Check violations requiring justification.*

One deliberate deviation from spec priority order is recorded above rather than hidden: **Phase 1
implements US2 (P2) before Phase 2 implements US1 (P1)**. Justification: US2's geometry is already in
memory, so it validates the shared ray-vs-triangle mechanism at near-zero risk, while US1 additionally
requires solving mesh-data availability (Phase 0 established the data is currently parsed and discarded).
Building the risky data-plumbing work on top of an already-proven intersection path is strictly safer than
proving both at once. Both remain independently shippable, and US1 remains this spec's highest-value
outcome.
