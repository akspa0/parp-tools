# Feature Specification: Capture Batch Tuning and Object-Roof Library Outputs

**Feature Branch**: `026-capture-batch-tuning`

**Created**: 2026-05-28

**Status**: Draft

**Input**: User request: improve MdxViewer validation capture batch throughput by removing artificial settle slowdowns, add multi-tile batch-fast-settle within one loaded session, add per-tile capture metadata, and port the improvements to wow-viewer capture.

## Problem Statement

The real MdxViewer validation capture pipeline has two bottlenecks that make broad proof impractical:

1. **Artificial settle throttles**: `RequiredSettledFrames=48` and `MaxFramesBeforeCapture=2400` are hardcoded, forcing every tile to wait ~48 frames after the scene is genuinely ready. For a batch of N tiles sharing the same loaded world, this means N × 48 wasted frames even though the world was settled after the first tile.

2. **No batch-aware settle**: After the first tile in a batch settles, subsequent tiles that only move the camera within the same loaded scene should use a much shorter settle because no new asset loading is needed.

3. **Missing per-tile metadata**: Captures produce PNGs but no structured metadata about what was captured, how long it took, or whether it timed out.

4. **No object-roof per-asset outputs**: The current `GenerateMkHarvestViewerValidationObjectArtifacts` only produces tile-family diff masks, not per-asset object crops.

## Goal

Deliver a bounded MdxViewer hotfix that:

1. Exposes `RequiredSettledFrames` and `MaxFramesBeforeCapture` as configurable validation-batch knobs (not hardcoded constants).
2. Adds `BatchSettledFrames` (shorter settle after the batch is first confirmed ready).
3. Tracks batch-settle state and uses the fast settle for subsequent tiles.
4. Reduces proof-defaults to `RequiredSettledFrames=12`, `MaxFramesBeforeCapture=480`, `BatchSettledFrames=2`.
5. Emits per-tile capture metadata JSON alongside each PNG.
6. Preserves full backwards compatibility — old defaults are still available via explicit config.

Then port the same improvements to `WowViewer.Tool.ValidationCapture`.

## Scope

### In scope — MdxViewer hotfix (gillijimproject_refactor)

- Replace hardcoded `MkHarvestViewerValidationRequiredSettledFrames` / `MkHarvestViewerValidationMaxFramesBeforeCapture` with plan-level fields on `MkHarvestViewerValidationCapturePlan`.
- Add `BatchSettledFrames` (short settle after first batch-settle confirmation).
- Add `bool FastSettleAfterBatchReady` to enable/disable fast-settle.
- Track `_batchHasSettled` flag on `ActiveMkHarvestViewerValidationBatch`.
- Use `BatchSettledFrames` for subsequent tiles once `_batchHasSettled` is true.
- Emit `{tile_name}_capture_metadata.json` per captured variant.
- Reduce proof-defaults.
- Build succeeds, no regressions.

### In scope — wow-viewer port

- Mirror the same knobs and batch-fast-settle logic in `ValidationCaptureCommand` / `HeadlessValidationCaptureSession`.
- Add `--settled-frames`, `--max-frames`, `--batch-settled-frames` CLI flags.
- Emit per-tile capture metadata from the headless capture path.

### Out of scope

- Object-roof per-asset rendering (that's T002 in spec 025, a deeper feature).
- Changing the terrain/renderer pipeline.
- Changing the wow-viewer GPU renderer.

## User Scenarios

### US-1 — Batch capture runs faster per tile after the first settle

**Given** a validation capture batch of 8 tiles on the same map,
**When** the first tile's scene finishes settling,
**Then** subsequent tiles settle in `BatchSettledFrames` frames (default 2) instead of `RequiredSettledFrames` frames (default 12).

### US-2 — Capture knobs are configurable

**Given** a researcher wants faster proof runs with shorter settle,
**When** they set `RequiredSettledFrames=4`,
**Then** the batch uses 4 frames for first-settle and `BatchSettledFrames` for subsequent tiles.

### US-3 — Per-tile metadata is emitted

**Given** a capture completes for tile `Azeroth_30_48`,
**When** the batch finishes,
**Then** a `{tile_name}_{variant}_capture_metadata.json` file exists in the output directory containing build, map, tileX, tileY, variant, settledFrames, maxFramesBeforeCapture, timedOut.

### US-4 — Backwards compatibility is preserved

**Given** existing code that does not set the new fields on `MkHarvestViewerValidationCapturePlan`,
**When** the plan is created with default values,
**Then** the batch uses `RequiredSettledFrames=12` (not 48), `MaxFramesBeforeCapture=480` (not 2400), and `BatchSettledFrames=2`.

## Success Criteria

- **SC-1**: Build `gillijimproject_refactor/src/MdxViewer` succeeds.
- **SC-2**: Build `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture` succeeds.
- **SC-3**: The hardcoded constants `MkHarvestViewerValidationRequiredSettledFrames` and `MkHarvestViewerValidationMaxFramesBeforeCapture` no longer control batch settle timing; plan-level fields do.
- **SC-4**: `ActiveMkHarvestViewerValidationBatch` tracks `_batchHasSettled` state.
- **SC-5**: Per-tile metadata JSON is emitted for each captured variant in the MdxViewer path.
- **SC-6**: Wow-viewer CLI exposes `--settled-frames`, `--max-frames`, `--batch-settled-frames`.

## Constitution

- **C-01**: All changes to MdxViewer are bounded hotfixes to the existing capture automation path. No new rendering features.
- **C-02**: All wow-viewer changes go in `wow-viewer/tools/validation-capture/`.
- **C-03**: Defaults must be safe for automated batch proof (no infinite waits, no zero-settle).
- **C-04**: Backwards compatibility: old code paths that don't set the new fields get sensible defaults.