# Tasks: Object Mask Rendering Fix

**Plan**: `013-object-mask-rendering-fix/plan.md`

**Spec**: `013-object-mask-rendering-fix/spec.md`

---

Execution rule:

- do not start a later phase until the current phase validation is complete

## Phase 1: Diagnose Root Cause

**Goal**: Confirm the coordinate system mismatch between camera and placement data.

**Status**: COMPLETE

Diagnostic output from staged `3_3_5_12340 / Azeroth_30_48` confirmed:

- `wmoInstances.Count=1, mdxInstances.Count=53` — instances exist
- `First WMO: pos=(-8931.6,539.3) dist=13505.7` — objects 13,500 units from camera
- Camera at `(100,-9500)` looking at `(800,-8800)` — correct for engine tile (48,30)
- All instances culled by distance check (max cull distance 8192 units)

Root cause: `BuildStandardAdtVirtualPath` constructs ADT filename as `Azeroth_{tileY}_{tileX}.adt`. For engine tile (48,30), this reads `Azeroth_30_48.adt`. But the placement data in that file has positions at (-8931,539) which is in a completely different coordinate area.

## Phase 2: Fix Coordinate Mismatch and Deduplication

**Goal**: Fix the camera/placement coordinate swap and add uniqueID deduplication for adjacent tiles.

- [x] **2.1** Fix `ComputeTilePlanarMin/Max/Center` in `WowViewerWorldRuntimeBridge.cs`
  - Root cause confirmed: rendererX was computed from tileY instead of tileX, and vice versa
  - Fix: swap the axis mapping so `minX = MapOrigin - ((tileX + 1) * TileSize)` and `minY = MapOrigin - ((tileY + 1) * TileSize)`
  - This aligns with MdxViewer's convention: `rendererX = MapOrigin - tileX * ChunkSize`

- [x] **2.2** Add uniqueID deduplication for multi-tile instance building
  - ADTs can contain duplicated objects (same uniqueID) that straddle adjacent tile boundaries
  - Added `HashSet<int>` dedup for both WMO and MDX instances in the bridge's instance builder loop
  - First occurrence wins; subsequent duplicates are skipped

- [ ] **2.3** Run diagnostic capture to verify fix
  - Command: `dotnet run --project tools/validation-capture/WowViewer.Tool.ValidationCapture -- capture --client-root "I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft" --map-input Azeroth --dataset-root "%TEMP%\vc_dataset" --output-root "%TEMP%\vc_output" --tile-name azeroth_30_48 --tile-x 30 --tile-y 48 --build lk --real-scene-dry-run`
  - Verify: `WMO visible > 0` and `MDX visible > 0`
  - Verify: placement positions are within the tile's coordinate range

**Checkpoint**: Diagnostic shows non-zero visible counts and placement positions match camera area.

## Phase 3: Verify Artifact Quality

**Goal**: Confirm object masks show 3D mesh silhouettes.

- [ ] **3.1** Run full four-variant capture on LK client
  - Command: `dotnet run --project tools/validation-capture/WowViewer.Tool.ValidationCapture -- capture --client-root "I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft" --map-input Azeroth --dataset-root "%TEMP%\vc_dataset" --output-root "%TEMP%\vc_output" --tile-name azeroth_30_48 --tile-x 30 --tile-y 48 --build lk --gpu-viewer-style`
  - Verify all four variants complete

- [ ] **3.2** Inspect `object_visibility_mask.png`
  - Verify non-trivial pixel coverage (not all-black or all-white)
  - Verify coverage corresponds to 3D object silhouettes

- [ ] **3.3** Inspect `no_object_minimap.png`
  - Verify terrain without object silhouettes

- [ ] **3.4** Repeat on alpha client `0_5_3_3368`

**Checkpoint**: Artifacts show real 3D mesh silhouettes on both staged clients.

## Phase 4: Clean Up

**Goal**: Remove temporary diagnostics, update tracking.

- [ ] **4.1** Remove any temporary diagnostic code
- [ ] **4.2** Update `012-real-validation-batch-extraction/tasks.md` Phase 6 status
- [ ] **4.3** Update continuity docs

**Checkpoint**: Clean build, docs current.
