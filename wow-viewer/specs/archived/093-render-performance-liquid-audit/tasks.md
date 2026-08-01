# Tasks: Spec 093 Render Performance And WMO Liquid Audit

**Input**: `spec.md`, `plan.md`

**Prerequisites**: Work only in `wow-viewer`. Do not edit `gillijimproject_refactor`.

**Tests**: Build validation is required for each source-changing phase. Manual dense-map capture is required before optimization rewrites.

## Phase 1: Honest Runtime Stats (P1)

**Goal**: Make Runtime Stats expose enough evidence to prove or reject the WMO batching/liquid hypothesis.

**Independent Test**: Build succeeds and Runtime Stats displays WMO opaque/transparent timing plus WMO draw composition fields.

- [x] T001 Add WMO per-render counters in `Rendering/WmoRenderer.cs`.
- [x] T002 Accumulate WMO draw-pressure counters in `Terrain/WorldScene.cs`.
- [x] T003 Split WMO transparent timing away from MDX transparent timing.
- [x] T004 Surface WMO draw composition in `ViewerApp_Sidebars.cs` Runtime Stats.
- [x] T005 Update `WorldRenderFrameStats` and advisor tests for the new fields.
- [x] T006 Build-check with `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- [x] T007 Focused-test `WorldRenderOptimizationAdvisorTests`.

## Phase 2: Dense-Map Capture (P1)

**Goal**: Capture actual numbers before choosing a performance rewrite.

- [ ] T008 Load staged `4_0_0_11927` Stormwind or dense city equivalent.
- [ ] T009 Record Runtime Stats with default object visibility and overlays off.
- [ ] T010 Record Runtime Stats with WMO hidden.
- [ ] T011 Record Runtime Stats with MDX hidden.
- [ ] T012 Record Runtime Stats with overlays/debug boxes hidden.
- [ ] T013 Decide whether the next source slice is WMO batching, MDX batching, overlay cleanup, asset loading, terrain, or liquid.

## Phase 3: WMO Liquid Audit (P1)

**Goal**: Fix WMO liquids from measured facts instead of guessing.

- [ ] T014 Pick a WMO with visible MLIQ and record WMO liquid draw count.
- [ ] T015 Verify whether basic GL blend state is active for WMO liquids.
- [ ] T016 Compare current flat MLIQ shader output against native-client liquid shader notes.
- [ ] T017 Implement one bounded visual correction only after T014-T016.
- [ ] T018 Build-check and manual screenshot-check the WMO liquid correction.

## Notes

- Current MDX "batched" counter means shared-shader submission, not true GPU instancing.
- Current WMO liquid shader is flat color and alpha; basic blend state exists, but material/shader/order behavior is still incomplete.
- Do not start renderer architecture rewrites until Phase 2 identifies the measured top cost.
