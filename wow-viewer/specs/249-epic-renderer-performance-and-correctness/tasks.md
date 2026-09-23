# Tasks — Epic 249 Renderer Performance, Lighting & Correctness

Implementation tasks are generated **after** triage. Nothing is checked without a §9.2 receipt.

## Phase 0 — Triage (operator)

- [ ] T001 Operator marks every R-xx item Want / Drop / Later in [TRIAGE.md](../TRIAGE.md).
- [ ] T002 Record decisions here as a dated amendment; Drop items move to a dropped table.
- [ ] T003 Generate phased tasks for Want items; pick the approach per plan.md.

## Operator verification sweep (shipped code — proof only)

- [ ] V001 236 T003 standalone-model shading + Gate 2 light spill onto WMO / terrain / doodads.
- [ ] V002 233 T015 Feature Tour + video witness (`FlybyUndead`, Warm Path).
- [ ] V003 136 T008/T011 doodad parity + real-client I/O comparison.
- [ ] V004 142 real-client movement captures (T069/T073/T077).
- [ ] V005 153 SC-007 interactive smoothness judgement.
- [ ] V006 226 wireframe/specular before-after captures (after R-30 if wanted).

## R-10 — Modern-data lighting performance (operator P1, 2026-09-23)

Blocked on operator approval of the proposed R-10 scope (spec.md amendment 2026-09-23).

- [x] R10-T001 Operator approves R-10a–e scope (or trims it). — Approved 2026-09-23: "Full bundle" (a–d now; e decided from re-measure). Receipt: operator answer in session.
- [x] R10-T002 Add frame counters (lights, query calls + candidates tested, WMO batched/lit-fallback/self-lit). Query time in ms deliberately not added (a stopwatch per query costs more than the query). Receipt: [evidence/r10-lighting-perf-code-2026-09-23.md](evidence/r10-lighting-perf-code-2026-09-23.md).
- [ ] R10-T003 Operator baseline capture on `wow_classic_beta` 1.60.1 `Azeroth` (FPS, frame ms, WMO draw calls, new counters).
- [x] R10-T004 Per-placement lit flag in `WorldObjectPassCoordinator` + tests; call-site change; batch path uploads 0 lights. Receipt: [evidence/r10-lighting-perf-code-2026-09-23.md](evidence/r10-lighting-perf-code-2026-09-23.md).
- [x] R10-T005 Reduce collected lights exactly. Spec-sync 2026-09-23: visible MDX is collected **after** the WMO pass, so instead of a visible-MDX restriction the rebuild keeps only lights whose sphere (+256 margin) touches the view (side + near planes). Receipt: [evidence/r10-lighting-perf-code-2026-09-23.md](evidence/r10-lighting-perf-code-2026-09-23.md).
- [x] R10-T006 `SceneLightManager` spatial index + equivalence tests. Receipt: [evidence/r10-lighting-perf-code-2026-09-23.md](evidence/r10-lighting-perf-code-2026-09-23.md).
- [ ] R10-T007 Build + focused tests; operator after-capture on the same map/camera; receipt in `evidence/`.
- [ ] R10-T008 Decide R-10e from the after-capture (self-lit WMO share).
