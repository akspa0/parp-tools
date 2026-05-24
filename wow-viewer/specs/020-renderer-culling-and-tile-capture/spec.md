# Feature Specification: Renderer Culling Fix and Tile-Level Capture

**Feature Branch**: `020-renderer-culling-and-tile-capture`

**Created**: 2026-05-24

**Status**: Draft

**Input**: The V16.2 dataset needs precise object visibility masks (see `011-v16-2-patched-signal-expansion`), which require rendering terrain tiles with correct WMO/MDX visibility. The renderer currently culls all objects because the camera frustum and distance culling use wrong coordinates. Fixing this cascade unlocks: single-tile capture, batched tile capture, V16.2 object mask generation, and proper object-aware loss weighting in V16.2 training.

## Problem Statement

Spec `013-object-mask-rendering-fix` diagnosed the root cause: `ComputeTilePlanarMin/Max` in `WowViewerWorldRuntimeBridge.cs` swaps rendererX/rendererY relative to the placement reader's coordinate convention. However the full fix also requires addressing a secondary culling issue when the camera frustum is set up for an entire map but only a single tile is loaded — the distance-based culling rejects instances that are within the actual tile bounds.

The current capture pipeline is thus trapped:
- Cannot capture a single tile with objects visible (culling rejects everything)
- Cannot batch tiles because single-tile capture is broken
- Cannot generate V16.2 precise object masks
- Cannot train with proper object-aware loss weighting

## User Scenarios & Testing

### User Story 1 — Single-Tile Capture Shows Non-Zero Object Visibility (Priority: P1)

A terrain researcher renders a single tile with the validation-capture tool and sees `WMO visible > 0` and `MDX visible > 0` in the diagnostic output. Objects within the tile bounds appear in the rendered output.

**Why this priority**: All downstream work (batching, V16.2 masks, training) is blocked until a single tile renders correctly.

**Independent Test**: Run `WowViewer.Tool.ValidationCapture capture --tile 30 48 --build 3_3_5_12340` on staged `Azeroth` and verify the output mask contains non-zero pixel coverage.

**Acceptance Scenarios**:

1. **Given** a staged `3_3_5_12340` client with the development map, **When** capture runs for tile `(30,48)`, **Then** `WMO visible > 0` and `MDX visible > 0` in diagnostics.
2. **Given** a single-tile capture with visible objects, **When** the `object_visibility_mask` output is inspected, **Then** it contains non-zero pixel coverage matching building footprints.
3. **Given** the same capture, **When** the `no_object_minimap` is compared to the primary capture, **Then** object silhouettes are removed.

---

### User Story 2 — Single-Tile Capture Works Without Loading the Whole Map (Priority: P1)

The renderer loads only the ADT/WDT data for the requested tile, not the entire map's 4096×4096 coordinate space. This makes capture fast enough to be practical for dataset generation.

**Why this priority**: Even with correct culling, loading the whole map wastes time and memory. Tile-level loading is essential for batching.

**Independent Test**: Capture tile `(30,48)` and verify the process reads only the files for that tile, not all 64×64 tiles.

**Acceptance Scenarios**:

1. **Given** a single-tile capture command, **When** the tool runs, **Then** it loads only the WDT and the specific ADT `(30,48)` rather than scanning all tiles.
2. **Given** tile-level loading, **When** capture completes, **Then** the time is proportional to one tile (not proportional to map size).

---

### User Story 3 — Batch Capture Processes Multiple Tiles Efficiently (Priority: P2)

A terrain researcher can specify a tile list or range and capture them in one process, reusing the WDT and shared assets across tiles. The per-tile time is similar to single-tile capture (no per-tile overhead from loading WDT repeatedly).

**Why this priority**: Batching is the path to generating V16.2 masks for the full corpus.

**Independent Test**: Capture tiles `(30,48),(31,48),(30,49),(31,49)` in one batch and verify they all produce non-zero object masks in comparable time to four sequential single-tile captures without WDT reloading.

**Acceptance Scenarios**:

1. **Given** a batch of 4 tiles, **When** capture runs, **Then** all 4 produce valid object masks.
2. **Given** batch capture, **When** the total time is measured, **Then** it is less than 4× the time of the first tile (because WDT and assets are cached).

---

### Edge Cases

- What if a tile has no ADT (ocean tile with no terrain data)? The capture must handle missing ADT gracefully.
- What if a tile has zero placement instances? The capture must produce an empty object mask rather than an error.
- What if the camera is positioned exactly at a tile boundary? Objects on adjacent tiles should be visible if within the frustum (or clearly culled by distance, whichever is correct).

## Requirements

### Functional Requirements

- **FR-001**: The camera frustum MUST be computed from the tile's actual world-space bounds, not from the full-map origin.
- **FR-002**: The distance culling threshold MUST be large enough to include objects within the tile bounds plus a configurable margin.
- **FR-003**: `ComputeTilePlanarMin/Max` MUST use the correct coordinate convention (rendererX = WDT Y, rendererY = WDT X).
- **FR-004**: The capture tool MUST support a `--tile <x> <y>` flag for single-tile capture in addition to the existing `--tile-coord <x> <y>`.
- **FR-005**: The capture tool MUST only load ADT data for the requested tile(s), not the entire map.
- **FR-006**: The batch capture MUST accept a list of tile coordinates or a range.
- **FR-007**: Batch capture MUST reload WDT at most once per map per batch invocation.
- **FR-008**: Diagnostic output MUST report camera position, frustum bounds, and culling distances per capture.
- **FR-009**: The fix MUST work on both alpha (`0_5_3_3368`) and LK (`3_3_5_12340`) clients.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Single-tile capture of `Azeroth_30_48` on `3_3_5_12340` produces non-zero object mask coverage.
- **SC-002**: The same capture runs without loading all 4096 map tiles.
- **SC-003**: Batch capture of 4 tiles runs faster than 4 sequential invocations.
- **SC-004**: Output masks from batch capture are pixel-identical to masks from sequential single-tile captures.

## Relationship to Other Specs

- **Fixes**: `013-object-mask-rendering-fix` (implements the fix that spec's diagnostic described)
- **Enables**: `011-v16-2-patched-signal-expansion` (precise object masks become feasible to generate)
- **Enables**: V16.2 model training with proper object-aware loss weighting
- **Prerequisite**: V16.1.4 combined model training (already implemented, needs object signals to improve)
- **Future**: MotherShip game-engine restructuring (the plugin architecture this work will eventually live in)
