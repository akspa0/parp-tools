# Tasks — Spec 255 WorldScene Decomposition

Nothing is checked without a §9.2 receipt in `evidence/`. Runtime claims come only from operator smoke.

## Gate

- [x] T001 Operator approves the plan (step order W0–W9, targets, R-10 gate for W8/W9) or trims it.
  Receipt: 2026-09-27 operator: "Approve Spec 255 as written" (no trims).

## Independent steps (after T001)

- [x] T010 W0 frame record types to own files; build + audit receipt. `a5ca359`; Receipt: `evidence/w0-w7-extraction-2026-09-27.md`
- [x] T011 W1 hover/pick/wireframe → `SceneHoverPickController`; build, audit, tests, receipt. `215bcc1`; Receipt: `evidence/w0-w7-extraction-2026-09-27.md`
- [x] T012 W2 taxi actors & routes → `TaxiActorScene`. `b1341fb`; Receipt: `evidence/w0-w7-extraction-2026-09-27.md`
- [x] T013 W3 UniqueId/path filters + archaeology layers → `SceneObjectFilters`. `26e8507`; Receipt: `evidence/w0-w7-extraction-2026-09-27.md`
- [x] T014 W4 selected-object resolution + placement edit/move → `SceneSelectionState`. `733f7d1`; Receipt: `evidence/w0-w7-extraction-2026-09-27.md`
- [x] T015 W5 lighting/LIT/fog + skybox → `SceneAtmosphere`. `f046fda`; Receipt: `evidence/w0-w7-extraction-2026-09-27.md`
- [x] T016 W6 external spawns → `ExternalSpawnLayer`; PM4 MPRL yaw helpers → `Terrain/Pm4`. `0d8c2b2`, `6a8c6e4`; Receipt: `evidence/w0-w7-extraction-2026-09-27.md`
- [x] T017 W7 camera-path collision & terrain sampling → `SceneTerrainQueries`. `bced801`; Receipt: `evidence/w0-w7-extraction-2026-09-27.md`
- [ ] T018 Operator smoke for W0–W7 (checklist per step in plan.md; consolidated PowerShell checklist at the end of
  the W0–W7 receipt).

## Render-path steps (after R-10 after-capture, Epic 249 R10-T007)

- [ ] T020 W8 tile streaming, instance build, bounds/buckets → `SceneInstanceStore`.
- [ ] T021 W9 frame visibility, render-path planning, deferred loads, frame stats → `SceneFrameVisibility`, `SceneAssetStreaming`.
- [ ] T022 Operator flight on a legacy and a modern map before/after W8–W9 (frame counters, draw counts).
- [ ] T023 E2 `Render()` split (tracked as Epic 251 U01-T004/T005).

## Close

- [ ] T030 `WorldScene.cs` ≤ ~2,000 lines; update AGENTS.md §10 measurements; receipt.
