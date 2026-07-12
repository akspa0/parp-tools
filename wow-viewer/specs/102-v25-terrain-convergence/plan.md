# Implementation Plan: Spec 102 Minimap-Only Reset

**Status**: Phase 0 only; previous V25 plan invalidated
**Specification**: [spec.md](spec.md)

## Technical Context

The only deployment input is RGB minimap pixels. Existing V18/V25 terrain arrays are labels and evaluation facts, never model inputs. The current unified trainer is permanently fail-closed. Replacement work is a sequence of tiny, independent residual models.

## Constitution Check

- New Python stays under `data-harvester/` and uses the existing environment.
- No parser, client reader, or dataset-builder duplication.
- CUDA is explicit; silent CPU fallback is a failure.
- One model, one residual signal, one checkpoint, and one gate.
- No shared weights or joint training between stages.
- Each slice is independently testable and no phase exceeds ten tasks.

## Phase 0 — Contract and Baselines

1. Generate a frozen split manifest holding out complete maps and one era.
2. Generate an input-manifest audit proving every forward tensor derives from RGB pixels.
3. Evaluate zero-height, train-global-mean, and RGB-derived flat-height baselines on the frozen split. Never use per-tile target statistics as inputs.
4. Record the historical minimap-only checkpoint only if it can run on the identical split; otherwise label it non-comparable.

## Phase 1 — H0 Tile Offset Residual

1. Predict one scalar correction residual over the frozen deployable RGB-flat baseline.
2. Train with a dedicated H0 trainer and checkpoint.
3. Run at most three epochs and stop unless held-out offset error beats the RGB-flat baseline.

## Phase 2 — H1 Coarse Relief Residual

This phase opens only after H0 passes.

1. Freeze H0 and materialize its predictions for the frozen split.
2. Predict one 33×33 low-frequency relief residual from RGB plus H0 output.
3. Run at most three epochs and stop unless coarse-relief metrics beat the H0 plane.

## Phase 3 — H2 Terrain Detail Residual

This phase opens only after H1 passes.

1. Freeze H0/H1 and materialize the upsampled coarse prediction.
2. Predict one 257×257 detail residual from RGB plus the frozen coarse prediction.
3. Run at most three epochs and stop unless height/slope metrics beat H1 upsampling.

## Phase 4 — H3 Border Residual

This phase opens only after H2 passes.

1. Consume adjacent RGB tiles and frozen H2 border predictions.
2. Predict one shared-border correction residual.
3. Validate raw continuity before any deterministic stitching.

## Phase 5 — U1 Uncertainty

This phase opens only after H2 passes and is trained separately from height.

1. Consume RGB and frozen height outputs.
2. Predict one uncertainty map.
3. Validate calibration against held-out H2 error.

## Deferred Independent Phases

WDL export, objects, textures, alpha, liquids, PM4, and binary writers each require separate single-output models or deterministic stages and independent gates.
