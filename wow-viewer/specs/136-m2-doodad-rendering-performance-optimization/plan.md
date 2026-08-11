# Implementation Plan: M2 and WMO Doodad Submission Batching

## Status

The safe CPU/state-submission slice is complete on 2026-08-10. The viewer still requires a user-run
real-scene capture before any FPS or GPU claim is promoted.

## Phase 1 — Static M2 Batch Eligibility

Use the existing `BeginBatch()` / `RenderInstance()` contract for static legacy-backed M2 renderers.
Keep particle and ribbon emitters on the unbatched path. Keep native-runtime M2 renderers isolated
until their backend state can be represented by an explicit batch key.

## Phase 2 — Opaque WMO Doodad Grouping

Group visible opaque WMO doodads by their shared `IModelRenderer`, call `BeginBatch()` once per
renderer group, and submit each placement through `RenderInstance()`. Preserve transparent
back-to-front order and unbatched particle/ribbon behavior.

## Phase 3 — GPU Batch Investigation (Not Started)

Measure whether true instancing or multi-draw submission is justified. The candidate key must include
geometry, material, texture, render backend, and any per-instance data required for visual parity.
This phase requires synthetic scaling plus user-witnessed real-client capture and must not be inferred
from CPU submission counts alone.
