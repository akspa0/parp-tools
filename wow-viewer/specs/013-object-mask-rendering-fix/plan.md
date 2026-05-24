# Implementation Plan: Object Mask Rendering Fix

**Spec**: `013-object-mask-rendering-fix/spec.md`

**Created**: 2026-05-23

## Summary

Fix the visibility pipeline in the headless validation-capture path so that `WorldObjectVisibilityCollector` produces non-zero visible WMO/MDX counts, enabling accurate `object_visibility_mask` and `no_object_minimap` artifact generation for V16.1.1 and V16.2 training.

## Technical Context

- **Language**: C# / .NET 10
- **Key Files**:
  - `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs` — bridge that builds frames and calls visibility collector
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs` — visibility culling logic
  - `wow-viewer/src/viewer/WowViewer.App/WorldGpuPreviewRenderer.cs` — renderer that consumes visible objects
  - `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/ValidationWorldSceneAdapter.cs` — headless capture adapter
- **Validation anchors**: staged `0_5_3_3368 / Azeroth_30_48` and `3_3_5_12340 / Azeroth_30_48`

## Root Cause (Confirmed)

The `ComputeTilePlanarMin/Max` functions in `WowViewerWorldRuntimeBridge.cs` swap rendererX/rendererY relative to the placement reader's coordinate convention.

The engine convention defines: engine `tileX` = WDT Y (north-south), engine `tileY` = WDT X (east-west).
The renderer convention is: `rendererX = MapOrigin - tileX * TileSize`, `rendererY = MapOrigin - tileY * TileSize`.
But `ComputeTilePlanarMin` was computing `minX = MapOrigin - tileY * TileSize` and `minY = MapOrigin - tileX * TileSize` — the axes were swapped.

This caused the camera to be placed 13,500 units from the actual tile center for Azeroth_30_48, culling all placement instances by the distance check.

A secondary issue is that ADTs can contain duplicated objects (same uniqueID) that exist on adjacent tile boundaries. The runtime bridge needs uniqueID deduplication for multi-tile rendering.

## Implementation Phases

### Phase 1: Diagnose Root Cause

**Goal**: Determine whether the issue is empty instance lists or camera geometry culling.

**Step 1.1 — Add instance count diagnostics to bridge**

Target file: `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs`

The diagnostic at line 508 (after the visibility collection) already logs `wmoInstances.Count` and `mdxInstances.Count`. Verify these values are non-zero. If they are 0, the issue is in placement catalog resolution, not the visibility collector.

**Step 1.2 — Add per-stage cull diagnostics to collector**

Target file: `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs`

Add a diagnostic callback or counter to `CollectVisibleWmos` that tracks how many instances are culled at each stage (cone-distance, max-distance, asset-ready). This will tell us exactly which cull stage is rejecting all instances.

**Step 1.3 — Run bounded diagnostic capture**

Run `WowViewer.Tool.ValidationCapture capture --real-scene-dry-run` on staged `3_3_5_12340 / Azeroth_30_48` and read the diagnostic output to determine:
- Are `wmoInstances.Count` and `mdxInstances.Count` non-zero?
- If yes, which cull stage is rejecting them?
- What is the camera position vs first instance position?

**Validation**: Diagnostic output clearly identifies the root cause.

---

### Phase 2: Fix Root Cause

**Goal**: Fix the identified root cause so the visibility collector produces non-zero visible counts.

The fix depends on Phase 1 findings. Possible fixes:

**If empty instance lists (Step 2A)**:
- Investigate `ResolveTileAndPlacements` to understand why the placement catalog is empty for the target tile
- Check if the tile coordinate mapping is correct (the command swaps tile-x/tile-y)
- Fix the placement catalog resolution

**If camera geometry culling (Step 2B)**:
- Adjust the camera offset or forward vector so objects fall within the cone cull distance
- Or adjust `FogEnd` / `ObjectStreamingRangeMultiplier` to increase the cull distance
- Or fix the cone factor computation for the validation-capture camera pose

**Step 2.1 — Implement the fix**

Target file: depends on Phase 1 findings.

**Step 2.2 — Add regression diagnostic**

Add a one-line diagnostic that confirms the fix: `WMO visible=N, MDX visible=M` where N and M are non-zero.

**Validation**: Diagnostic output shows `WMO visible > 0` and `MDX visible > 0`.

---

### Phase 3: Verify Artifact Quality

**Goal**: Confirm that the visibility fix produces correct `object_visibility_mask` and `no_object_minimap` artifacts.

**Step 3.1 — Run full four-variant capture**

Run the complete four-variant capture on staged `3_3_5_12340 / Azeroth_30_48`:
- primary
- noliquids
- noobjects
- objectsonly

Verify each variant produces visually distinct output.

**Step 3.2 — Inspect object visibility mask**

Check that `object_visibility_mask.png` contains non-trivial pixel coverage corresponding to 3D object silhouettes.

**Step 3.3 — Inspect no-object minimap**

Check that `no_object_minimap.png` shows terrain without the objects visible in the primary capture.

**Step 3.4 — Repeat on alpha client**

Run the same bounded proof on staged `0_5_3_3368 / Azeroth_30_48` to confirm the fix works on both client eras.

**Validation**:
- `object_visibility_mask.png` is not all-black or all-white
- `no_object_minimap.png` shows terrain without object silhouettes
- Four variants are visually distinct
- Proof exists on both staged clients

---

### Phase 4: Clean Up Diagnostics

**Goal**: Remove temporary diagnostic code and update task tracking.

**Step 4.1 — Remove temporary diagnostic additions**

Clean up any temporary diagnostic code added during Phase 1, keeping only the regression-safe diagnostic line.

**Step 4.2 — Update 012 tasks.md**

Mark Phase 6 task 6.7 as verified (not just checked) with evidence of successful artifact generation.

**Step 4.3 — Update continuity docs**

Update `activeContext.md` and `progress.md` to reflect the object mask fix and its validation.

**Validation**: Clean build, no temporary code, docs are current.

## Complexity Tracking

No constitution violations. This is a focused fix within existing architecture.

## Success Criteria

- SC-001: Diagnostic shows `WMO visible > 0` and `MDX visible > 0`
- SC-002: `object_visibility_mask.png` contains non-trivial pixel coverage
- SC-003: `no_object_minimap.png` shows terrain without objects
- SC-004: Four variants are visually distinct on bounded proof anchors
- SC-005: Proof exists on both staged `0_5_3_3368` and `3_3_5_12340`
