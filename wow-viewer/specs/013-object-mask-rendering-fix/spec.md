# Feature Specification: Object Mask Rendering Fix

**Feature Branch**: `013-object-mask-rendering-fix`

**Created**: 2026-05-23

**Status**: Draft

**Input**: Diagnostic evidence from staged `3_3_5_12340 / Azeroth_30_48` showing that the headless validation-capture path reports `WMO visible=0, MDX visible=0` despite placement instances being present, producing empty or marker-only object masks instead of high-fidelity 3D mesh silhouettes.

## Problem Statement

Phase 6 of `012-real-validation-batch-extraction` marked WMO/MDX mesh rendering as complete, but diagnostic output shows the visibility collector produces zero visible objects:

```
Objects: WMO visible=0 loaded=0, MDX visible=0 loaded=0
```

Diagnostic from staged `3_3_5_12340 / Azeroth_30_48`:

```
WmosVisible=True DoodadsVisible=True
wmoInstances.Count=1 mdxInstances.Count=53
culledWmoCount=1 culledMdxCount=53
VisibleWmos.Count=0 VisibleMdx.Count=0
cameraPosition=(100.0,-9500.0,319.5)
cameraTarget=(800.0,-8800.0,59.5)
First WMO: key=WORLD\WMO\AZEROTH\BUILDINGS\STORMWIND\STORMWIND.WMO pos=(-8931.6,539.3,102.0) dist=13505.7
```

The instances exist but are 13,500 units from the camera. The camera is at engine tile (48,30) covering X [533,1067] Y [-9067,-8533], but the WMO position (-8931,539) is in a completely different coordinate area. All instances are culled by the distance check (max cull distance 8192 units).

This blocks:
- `object_visibility_mask` generation for V16.1.1 normal-lane no-object guidance
- `no_object_minimap` generation for V16.2 patched-signal expansion
- Any downstream training that depends on accurate terrain/object separation

The object masks produced by the current path are empty or contain only point markers, not the 3D mesh silhouettes required by the dataset contract.

## Goal

Fix the visibility pipeline so that the headless validation-capture path produces accurate `object_visibility_mask` and `no_object_minimap` artifacts with real 3D WMO/MDX mesh silhouettes, enabling V16.1.1 and V16.2 dataset workflows.

## Current Implementation Status

- Phase 6 tasks 6.1-6.7 are marked complete in `012-real-validation-batch-extraction/tasks.md`
- `WorldGpuPreviewRenderer.cs` has WMO/MDX shader code, cache structures, and buffer construction
- The visibility collector runs but returns 0 visible objects
- Instances exist (1 WMO, 53 MDX) but are at wrong coordinates
- The issue is a coordinate system mismatch between camera computation and ADT placement data

## Root Cause (Confirmed)

The `ComputeTilePlanarMin/Max` functions in `WowViewerWorldRuntimeBridge.cs` swap rendererX/rendererY relative to the placement reader's coordinate convention.

**Engine convention**: engine `tileX` = WDT Y (north-south), engine `tileY` = WDT X (east-west).  
**Renderer convention**: `rendererX = MapOrigin - tileX * TileSize` (from WDT Y), `rendererY = MapOrigin - tileY * TileSize` (from WDT X).  
**Bug**: `ComputeTilePlanarMin` computed `minX = MapOrigin - tileY * TileSize` and `minY = MapOrigin - tileX * TileSize` — the axes were swapped.

This caused the camera to be placed 13,500 units from the actual tile center, culling all placement instances.

**Secondary issue — duplicated uniqueIDs across adjacent tiles**: ADTs can contain duplicated objects (same uniqueID) that exist on tile boundaries. When rendering multiple adjacent tiles, the instance builder must deduplicate by uniqueID to avoid rendering the same object twice. The `LkToAlphaCommand` already does this for conversion; the runtime bridge now also deduplicates.

## User Scenarios & Testing

### User Story 1 - Object Visibility Produces Non-Zero Counts (Priority: P1)

A terrain researcher runs the headless validation-capture tool and sees non-zero `WMO visible` and `MDX visible` counts in the diagnostic output, confirming the visibility collector is actually adding instances to the visible set.

**Why this priority**: If visibility is zero, nothing downstream works. This is the blocking gate.

**Independent Test**: Run `WowViewer.Tool.ValidationCapture capture --real-scene-dry-run` on staged `3_3_5_12340 / Azeroth_30_48` and verify diagnostic output shows `WMO visible > 0` and `MDX visible > 0`.

**Acceptance Scenarios**:

1. **Given** a staged LK client root with the development map, **When** the validation-capture tool builds a frame, **Then** `WMO visible` is greater than zero.
2. **Given** the same staged client, **When** the frame is built, **Then** `MDX visible` is greater than zero.
3. **Given** the visibility counts are non-zero, **When** the counts are compared to the instance counts, **Then** the visible count is a reasonable fraction of the total instance count (not 0% and not 100% of all instances in all cases).

---

### User Story 2 - Object Visibility Mask Shows 3D Mesh Silhouettes (Priority: P1)

A terrain researcher generates `object_visibility_mask.png` from a headless capture run and sees accurate 3D mesh outlines of buildings and doodads, not empty images or point markers.

**Why this priority**: The object mask is the primary artifact needed by V16.1.1 and V16.2 training.

**Independent Test**: Run the full four-variant capture on staged `3_3_5_12340 / Azeroth_30_48` and verify the `object_visibility_mask.png` contains non-trivial pixel coverage matching expected building footprints.

**Acceptance Scenarios**:

1. **Given** a tile with visible WMOs, **When** the four-variant capture completes, **Then** `object_visibility_mask.png` contains pixel coverage corresponding to WMO building footprints.
2. **Given** a tile with visible doodads, **When** the four-variant capture completes, **Then** `object_visibility_mask.png` contains pixel coverage corresponding to doodad positions.
3. **Given** the no-objects variant, **When** the `no_object_minimap.png` is generated, **Then** the image shows terrain without the objects visible in the primary capture.

---

### User Story 3 - Visibility Pipeline Is Debuggable (Priority: P2)

A terrain researcher can add or read diagnostic output that explains why specific instances were culled or kept, so future visibility regressions can be diagnosed without code archaeology.

**Why this priority**: The current debugging session took significant effort because visibility silently returned zero with no explanation.

**Independent Test**: Run a bounded capture with verbose logging and verify the diagnostic file contains per-instance or per-stage visibility decision data.

**Acceptance Scenarios**:

1. **Given** a visibility pipeline issue, **When** the tool runs, **Then** diagnostic output identifies which visibility stage culled instances (distance, cone, bounds, asset-ready).
2. **Given** instances are visible, **When** diagnostics are enabled, **Then** the output includes the first few visible instance positions and distances for sanity checking.
3. **Given** the diagnostic output exists, **When** reviewed, **Then** it identifies the camera position, target, and forward vector used for visibility decisions.

### Edge Cases

- What happens when the camera is positioned such that all objects are behind it? The visibility collector should cull them, but the diagnostic should explain why.
- What happens when placement instances exist but have zero-size bounds? The collector should handle degenerate bounds gracefully.
- What happens when the WDT or ADT uses a different coordinate convention than expected? The coordinate system must be verified end-to-end.

## Requirements

### Functional Requirements

- **FR-001**: The visibility pipeline MUST produce non-zero visible counts for tiles that have placement instances.
- **FR-002**: The `object_visibility_mask` artifact MUST contain 3D mesh silhouettes, not point markers or empty images.
- **FR-003**: The `no_object_minimap` artifact MUST show terrain without the objects visible in the primary capture.
- **FR-004**: The four-variant capture (primary, noliquids, noobjects, objectsonly) MUST produce visually distinct outputs when objects are present.
- **FR-005**: The visibility pipeline MUST correctly handle the coordinate system used by ADT placement readers (MapOrigin-based coordinate transform).
- **FR-006**: Diagnostic output MUST identify which visibility stage culled instances when verbose logging is enabled.
- **FR-007**: The fix MUST work on both staged alpha (`0_5_3_3368`) and LK (`3_3_5_12340`) clients.
- **FR-008**: The fix MUST not break terrain rendering or the existing terrain-only capture variants.
- **FR-009**: When rendering multiple adjacent tiles, the visibility pipeline MUST deduplicate placement instances by uniqueID so that objects straddling tile boundaries are not rendered twice.

### Key Entities

- **Visibility Pipeline**: the chain from placement instances through `WorldObjectVisibilityCollector` to `WorldVisibilityFrame.VisibleWmos`/`VisibleMdx`.
- **Object Visibility Mask**: a binary or grayscale image where pixel coverage corresponds to 3D object silhouettes.
- **No-Object Minimap**: a rendered terrain image with objects removed, showing terrain shading and structure.
- **Pass Options**: the per-variant visibility flags (`WmosVisible`, `DoodadsVisible`) that control whether the collector runs.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Diagnostic output shows `WMO visible > 0` and `MDX visible > 0` for staged `3_3_5_12340 / Azeroth_30_48`.
- **SC-002**: `object_visibility_mask.png` contains non-trivial pixel coverage (not all-black or all-white).
- **SC-003**: `no_object_minimap.png` shows terrain without object silhouettes visible in the primary capture.
- **SC-004**: The four-variant capture produces visually distinct outputs on the bounded proof anchors.
- **SC-005**: Bounded proof exists on both staged `0_5_3_3368` and `3_3_5_12340` for `Azeroth_30_48`.

## Assumptions

- The ADT placement data is loaded correctly (confirmed by 1816 markers).
- The issue is in the visibility pipeline, not in data loading.
- The fix is likely a configuration or coordinate-system issue, not a fundamental architecture problem.
- The existing Phase 6 shader and buffer code is correct and just needs instances to be visible.
- Diagnostic output will be sufficient to identify the root cause without deeper reverse engineering.

## Relationship to Other Specs

- **Depends on**: `012-real-validation-batch-extraction` (Phase 6 completion)
- **Enables**: `007-v16-1-1-curated-normal-acceleration` (object masks for no-object guidance)
- **Enables**: `011-v16-2-patched-signal-expansion` (precise masks for sidecar signals)
- **Blocks**: V16.1.1 training with no-object guidance, V16.2 dataset patching
