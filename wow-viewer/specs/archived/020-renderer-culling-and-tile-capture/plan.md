# Implementation Plan: Renderer Culling Fix and Tile-Level Capture

**Spec**: `020-renderer-culling-and-tile-capture/spec.md`
**Created**: 2026-05-24

## Summary

Fix the renderer culling pipeline so single-tile capture produces valid object visibility masks, then add tile-level loading and batch capture to enable V16.2 dataset generation.

## Implementation Phases

### Phase 1: Fix Coordinate Bug in ComputeTilePlanarMin/Max

**Goal**: Single-tile capture shows non-zero object visibility counts.

1. Fix `ComputeTilePlanarMin` and `ComputeTilePlanarMax` in `WowViewerWorldRuntimeBridge.cs` to not swap rendererX/rendererY axes
2. Fix the distance culling threshold — ensure it's large enough for objects within the tile's actual bounds plus a margin
3. Add diagnostic output: camera position, frustum bounds, culling distances, per-instance culling reason
4. Test: capture `Azeroth_30_48` on `3_3_5_12340` — verify `WMO visible > 0`

### Phase 2: Tile-Level Loading

1. Add `--tile <x> <y>` flag (distinct from existing `--tile-coord`)
2. Load only the requested tile's ADT, not all 64×64 tiles
3. Verify the camera frustum is sized for one tile, not the full map
4. Test: capture one tile and verify no other ADT files are read

### Phase 3: Batch Capture

1. Accept `--tile <x> <y>` multiple times or `--tile-range <x0>:<y0>-<x1>:<y1>`
2. Cache WDT and loaded assets across tiles in the batch
3. Output per-tile masks to a structured output directory
4. Test: batch of 4 tiles, verify masks match sequential single-tile captures

### Phase 4: Alpha Client Support

1. Verify the same fix works on `0_5_3_3368` (pre-2003 era)
2. Fix any alpha-specific coordinate or culling differences
3. Test: capture alpha tile with visible objects

### Phase 5: V16.2 Mask Generation

1. Run batch capture on target builds (3.0.1 dev map, 0.5.3 pre-2003)
2. Generate `object_visibility_mask` and `no_object_minimap` per tile
3. Patch into V16.2 sidecar stores (per spec 011)
