# Tasks — Spec 256 Modern-Data Streaming & M2 Submission Performance

Nothing is checked without a §9.2 receipt in `evidence/`. FPS, load time and visuals come only from operator captures.

## Gate
- [x] T001 Operator approves the plan (phases P0–P3, order, revert-on-regression rule) or trims it.
  Receipt: 2026-09-27 operator "Approve Spec 256 plus the minimap budget" (reply to the offer to start with P0 and P1).

## P0 — Name the frame
- [ ] T010 Stage timings + unattributed remainder in Runtime Stats. *(2026-09-27: existing Perf panel already shows the remainder and its in-Render breakdown; no new stage timer added until T015 shows where GC does not explain it.)*
- [x] T011 GC pause ms per frame and gen0/1/2 rates. `77f6f02`; Receipt: [evidence/p0-p1-minimap-2026-09-27.md](evidence/p0-p1-minimap-2026-09-27.md).
- [x] T012 Per-load phase times (read / skin resolve / native parse / adapter parse / GPU create). `77f6f02`; Receipt: [evidence/p0-p1-minimap-2026-09-27.md](evidence/p0-p1-minimap-2026-09-27.md).
- [x] T013 M2 batch-gate counts and CASC read/prefetch/cache counters shown. `77f6f02`; Receipt: [evidence/p0-p1-minimap-2026-09-27.md](evidence/p0-p1-minimap-2026-09-27.md).
- [x] T014 Build + tests + receipt. Receipt: [evidence/p0-p1-minimap-2026-09-27.md](evidence/p0-p1-minimap-2026-09-27.md).
- [ ] T015 Operator capture at the baseline spot; A/B with `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=0`.

## P1 — Exact load-cost cuts (separate commit after P0; operator captures P0-only build and P1 build)
- [x] T020 Skin lookup index + equivalence tests (scratch harness; no viewer test project exists). `4f1e6b4`; Receipt: [evidence/p0-p1-minimap-2026-09-27.md](evidence/p0-p1-minimap-2026-09-27.md).
- [x] T021 Skip the discarded adapter parse when the native renderer is chosen. `4f1e6b4`; Receipt: [evidence/p0-p1-minimap-2026-09-27.md](evidence/p0-p1-minimap-2026-09-27.md).
- [ ] T022 Build + tests + receipt; operator capture.

## P2 — (original) Off-thread model parsing — deferred by the 2026-09-27 amendment (parse ≈ 6 ms/M2)
- [ ] T030 Prepare/create split for M2/MDX loads (worker prepare, render-thread create).
- [ ] T031 Same for WMO loads.
- [ ] T032 Build + tests + receipt; operator capture (load time to settled scene, frame times while streaming).

## P3 — Native M2 instancing (after T032; scope from T015)
- [x] T040 Decide: native instanced path vs legacy route default (operator). Receipt: operator 2026-09-27 chose "Start P3 instancing" offered as "one instanced draw on the native renderer" → native instanced path.
- [x] T041 Implement the chosen path with existing gates. Receipt: [evidence/p3-native-instancing-2026-09-27.md](evidence/p3-native-instancing-2026-09-27.md).
- [ ] T042 Build + tests + receipt; operator capture + visual A/B.

## Amendment — minimap upload budget
- [x] T050 Minimap tile uploads by time budget (16 tiles / 2 ms; 32 / 6 ms with the minimap window open). `b15e553`; Receipt: [evidence/p0-p1-minimap-2026-09-27.md](evidence/p0-p1-minimap-2026-09-27.md).
- [x] T051 Operator: minimap pending count drains quickly on the baseline map. Receipt: operator 2026-09-27 "minimaps are quick now, at least, and fully load"; [evidence/capture-p1-2026-09-27.md](evidence/capture-p1-2026-09-27.md).

## File budget
- [ ] T060 Split `Terrain/WorldAssetManager.cs` (2,112 lines; over the ~2,000 budget before this spec, +24 timing/index lines here) — AGENTS.md §10.

## P2 amendment — native M2 textures (operator 2026-09-27: "Approve P2a and P2b and P2c")
- [x] T070 P2a shared reference-counted native M2 texture cache. `c9cf7b6`; Receipt: [evidence/p2abc-2026-09-27.md](evidence/p2abc-2026-09-27.md).
- [x] T071 P2b off-thread texture read/decode; render-thread upload. `6a326d3`; Receipt: [evidence/p2abc-2026-09-27.md](evidence/p2abc-2026-09-27.md).
- [x] T072 P2c DXT BLP compressed upload with authored mips. `bad4884`; Receipt: [evidence/p2abc-2026-09-27.md](evidence/p2abc-2026-09-27.md).
- [x] T073 Build + tests + receipt per step; operator capture (GPU create ms, DeferredAssetLoads, visual check). Receipt: [evidence/p2abc-2026-09-27.md](evidence/p2abc-2026-09-27.md); capture [evidence/capture-t073-2026-09-27.md](evidence/capture-t073-2026-09-27.md) (M2 GPU create ≈ 57 → ≈ 0.1 ms).

## Amendment — missing textures bind the error texture (operator 2026-09-27)
- [x] T080 Native M2: no naming-convention / `.blp`-scan / fallback-slot search; database resolution only; missing → `Textures\ShaneCube.blp` or generated checker (shared per data source); resolver `Resolve` drops its directory search. Receipt: [evidence/missing-texture-2026-09-27.md](evidence/missing-texture-2026-09-27.md).
- [ ] T081 Operator capture: M2 load / GPU create ms and deferred loads at the baseline spot; visual check of where the error texture now appears.

## Amendment D — WMO doodad loads (operator 2026-09-27: "Fix building doodad loads")
- [x] T090 D1–D5 per plan amendment D; build + tests + receipt. Receipt: [evidence/d-wmo-doodads-2026-09-27.md](evidence/d-wmo-doodads-2026-09-27.md).
- [x] T091 Operator capture: DeferredAssetLoads while streaming; doodad load phase line. Receipt: [evidence/capture-d-p3-2026-09-27.md](evidence/capture-d-p3-2026-09-27.md) (DeferredAssetLoads median 68.3 → 7.6 ms).
- [ ] T092 Split `Rendering/WmoRenderer.cs` (3,794 lines before D; over the ~2,000 budget) — AGENTS.md §10.
- [x] T093 Perf panel code moved out of `Pm4WorkbenchService` into `PerfPanelService` (operator 2026-09-27). Receipt: [evidence/perf-panel-move-2026-09-27.md](evidence/perf-panel-move-2026-09-27.md).

## Amendment A — M2 animation cost (operator 2026-09-27: "yes, the animation is eating everything for breakfast")
- [x] T100 Find the per-frame and first-use costs of native M2 animation (code + measurement). Receipt: [evidence/a-animation-2026-09-27.md](evidence/a-animation-2026-09-27.md) (code reading; measurement is T102).
- [x] T101 Implement the cuts; build + tests + receipt. Receipt: [evidence/a-animation-2026-09-27.md](evidence/a-animation-2026-09-27.md).
- [ ] T102 Operator capture: MdxAnimation stage, hitches when entering an animated area.
