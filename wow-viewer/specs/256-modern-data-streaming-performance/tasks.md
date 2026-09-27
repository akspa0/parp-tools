# Tasks — Spec 256 Modern-Data Streaming & M2 Submission Performance

Nothing is checked without a §9.2 receipt in `evidence/`. FPS, load time and visuals come only from operator captures.

## Gate
- [x] T001 Operator approves the plan (phases P0–P3, order, revert-on-regression rule) or trims it.
  Receipt: 2026-09-27 operator "Approve Spec 256 plus the minimap budget" (reply to the offer to start with P0 and P1).

## P0 — Name the frame
- [ ] T010 Stage timings + unattributed remainder in Runtime Stats.
- [ ] T011 GC pause ms per frame and gen0/1/2 rates.
- [ ] T012 Per-load phase times (read / skin resolve / native parse / adapter parse / GPU create).
- [ ] T013 M2 batch-gate counts and CASC read/prefetch/cache counters shown.
- [ ] T014 Build + tests + receipt.
- [ ] T015 Operator capture at the baseline spot; A/B with `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=0`.

## P1 — Exact load-cost cuts (separate commit after P0; operator captures P0-only build and P1 build)
- [ ] T020 Skin lookup index + equivalence tests.
- [ ] T021 Skip the discarded adapter parse when the native renderer is chosen.
- [ ] T022 Build + tests + receipt; operator capture.

## P2 — Off-thread model loading (after T022)
- [ ] T030 Prepare/create split for M2/MDX loads (worker prepare, render-thread create).
- [ ] T031 Same for WMO loads.
- [ ] T032 Build + tests + receipt; operator capture (load time to settled scene, frame times while streaming).

## P3 — Native M2 instancing (after T032; scope from T015)
- [ ] T040 Decide: native instanced path vs legacy route default (operator).
- [ ] T041 Implement the chosen path with existing gates.
- [ ] T042 Build + tests + receipt; operator capture + visual A/B.

## Amendment — minimap upload budget
- [ ] T050 Minimap tile uploads by time budget (16 tiles / 2 ms; 32 / 6 ms with the minimap window open).
- [ ] T051 Operator: minimap pending count drains quickly on the baseline map; frame time unchanged.
