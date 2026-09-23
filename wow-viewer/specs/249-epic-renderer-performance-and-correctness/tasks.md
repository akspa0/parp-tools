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
