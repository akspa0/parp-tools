# Tasks: Renderer Culling Fix and Tile-Level Capture

## Phase 1: Fix Coordinate Bug in ComputeTilePlanarMin/Max

- [ ] T001 Fix `ComputeTilePlanarMin`/`ComputeTilePlanarMax` axis swap in `WowViewerWorldRuntimeBridge.cs`
- [ ] T002 Fix distance culling threshold to account for tile-level frustum (not full-map)
- [ ] T003 Add diagnostic output: camera pos, frustum bounds, culling distances, per-instance culling reason
- [ ] T004 Test: capture `Azeroth_30_48` on `3_3_5_12340` — verify `WMO visible > 0`

## Phase 2: Tile-Level Loading

- [ ] T005 Add `--tile <x> <y>` flag to validation-capture tool
- [ ] T006 Load only requested tile's ADT (not all 64x64)
- [ ] T007 Test: capture one tile, verify no other ADTs loaded

## Phase 3: Batch Capture

- [ ] T008 Accept multiple `--tile` flags or `--tile-range` in capture tool
- [ ] T009 Cache WDT and assets across tiles in batch
- [ ] T010 Output per-tile masks to structured output dir
- [ ] T011 Test: batch of 4 tiles matches sequential captures

## Phase 4: Alpha Client Support

- [ ] T012 Verify fix on `0_5_3_3368`
- [ ] T013 Fix any alpha-specific differences
- [ ] T014 Test: capture alpha tile with visible objects

## Phase 5: V16.2 Mask Generation

- [ ] T015 Run batch capture on target builds (3.0.1, 0.5.3)
- [ ] T016 Generate `object_visibility_mask` and `no_object_minimap`
- [ ] T017 Patch into V16.2 sidecar stores
