# Tasks: WMO Portal Admission — Eliminating the Conservative Fallback

**Spec**: [spec.md](spec.md) | **Created**: 2026-09-01

Diagnose before changing any culling logic. Spec 151's own guidance — *"instrument group
admission before changing any logic"* — is why the reading in the spec exists at all, and it
applies again here.

Legend: `[ ]` open · `[x]` done · `[-]` in progress · **(operator)** = user-run.

---

## Phase 1 — Diagnose: which reason, and on what

- [ ] **T001** Aggregate fallback reasons across a flight instead of reporting the first.
      `WmoPortalVisibilityDiagnostics.FallbackReason` already carries the five named causes
      (`visibility_input_invalid`, `groups_absent`, `portal_data_absent`,
      `portal_edges_absent`, plus map-build errors); count each (FR-001).
- [ ] **T002** Record the WMO path per fallback reason, capped and deduplicated, so a reason
      can be traced to files (FR-002).
- [ ] **T003** Split `portal_data_absent` into "no portal chunks in the file" versus "portal
      chunks present, no traversable graph built" (FR-003). This is the distinction the whole
      spec turns on — the two have opposite fixes.
- [ ] **T004** Record, for admissions attributed to the GPU-instanced shell, whether portal
      culling was applicable (FR-006).
- [ ] **T005** **(operator)** Re-fly the Wandering Isle route with MDX batching **off**, and
      record the reason distribution. Baseline to beat: 145/145 placements, 806/806 groups,
      119 fallbacks of 290 evaluations.

**Phase 1 exit**: we know whether this is a data limitation or a graph-construction bug. **If
T005 shows portal data is genuinely absent across the corpus, the spec's honest outcome is
SC-002 as a finding and Phase 2 does not run.**

---

## Phase 2 — Fix whichever cause Phase 1 named

Do not start before T005. The two branches below are mutually exclusive; Phase 1 decides.

- [ ] **T101** *(if graph construction is at fault)* Fix the portal graph build so files with
      portal data produce a traversable adjacency, and the evaluator can reject (FR-004).
- [ ] **T102** *(if data is genuinely absent)* Record it in research and reduce scope to the
      GPU-instanced shell path (US3) and any non-portal rejection rule that is currently
      inert — `off-frustum+cone`, `distance`, `max-view`, `projected-size` and `not-resident`
      all reported **0 rejections**, which is itself suspicious on a 145-placement scene.
- [ ] **T103** Keep conservative admission wherever data cannot support culling, still counted
      as such (FR-007).
- [ ] **T104** Make the previous admission behaviour reachable at runtime for same-session
      before/after (FR-008), mirroring how spec 153 US3's toggle works.

---

## Phase 3 — Prove it

- [ ] **T201** **(operator)** Re-fly the same route. SC-003: admitted groups fall from 806/806.
- [ ] **T202** **(operator)** Capture comparison on the same route. SC-004: nothing visible
      before is missing after. A reduction that loses geometry is a regression.
- [ ] **T203** **(operator)** Frame-time distribution before/after. SC-005 — record it even if
      unchanged, which would mean admission was not the cost after all.
- [ ] **T204** Record the outcome in spec 151's workstream so the admission slice is closed
      there rather than left ambiguous between the two specs.

---

## Notes

- **Keep opaque batching off** during every comparison flight. With 13,180 unbatched opaque
  draw calls dominating frame time, an admission change measured with the toggle flipped would
  be unreadable. Note per spec 201 that this counter aggregates the M2 and MDX render paths,
  and that native-route M2 is unbatchable regardless of the toggle — so the toggle's effect on
  it is not yet a known quantity either.
- `frustum union only` accounted for just 4.7% and `portal traversal only` for **0** — the
  rules that should be doing the work are the ones doing least. Worth carrying into T102.
- Known test baseline: `WowViewer.Core.Tests` has **9 pre-existing failures**. Compare to 9.
