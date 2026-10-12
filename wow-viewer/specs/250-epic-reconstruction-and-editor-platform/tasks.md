# Tasks — Epic 250 Map Reconstruction, Composition & Editor Platform

Implementation tasks are generated **after** triage. Nothing is checked without a §9.2 receipt.

## Phase 0 — Triage (operator)

- [ ] T001 Operator marks every E-xx item Want / Drop / Later in [TRIAGE.md](../TRIAGE.md).
- [ ] T002 Record decisions here as a dated amendment; Drop items move to a dropped table.
- [ ] T003 Generate phased tasks for Want items; pick the approach per plan.md.

## Operator verification sweep (shipped code — proof only)

- [ ] V001 Spec 232 cartography witnesses: T015d rotated seam, T051 lock badge, T053 WL inspector,
      T054 default page, T056–T059 placement/Z/minimap, T064 edge-snap, T066 cell fine-tune.
- [ ] V002 222 Gate 3: add + align a layer in under 30 seconds.
- [ ] V003 191 T024 generated map in Alpha 0.5.3 and 3.3.5 clients.
- [ ] V004 194/196 runtime NFR proofs (≤2 ms, no allocation; boundary RMSE; ≥60 FPS).
