# Implementation Plan: M2 and WMO Doodad Submission Batching

## Status

The CPU/state-submission slice and the first GPU-instancing implementation are complete on
2026-08-10. The viewer still requires a user-run real-scene capture and visual parity check before
any FPS claim is promoted.

## Phase 1 — Static M2 Batch Eligibility

Use the existing `BeginBatch()` / `RenderInstance()` contract for static legacy-backed M2 renderers.
Keep particle and ribbon emitters on the unbatched path. Keep native-runtime M2 renderers isolated
until their backend state can be represented by an explicit batch key.

## Phase 2 — Opaque WMO Doodad Grouping

Group visible opaque WMO doodads by their shared `IModelRenderer`, call `BeginBatch()` once per
renderer group, and submit each placement through `RenderInstance()`. Preserve transparent
back-to-front order and unbatched particle/ribbon behavior.

## Phase 3 — GPU Instanced Opaque Submission

Use a renderer-scoped compatibility contract for opaque M2/MDX batches. Upload model matrices and
fade values through a dynamic instance VBO, add per-instance vertex attributes to the existing shader,
and issue one `glDrawElementsInstanced` call per compatible geoset. Keep native-runtime state,
transparent layers, particle/ribbon renderers, and unsupported fade/material states on the fallback.
The implementation is complete; synthetic scaling, visual parity, and user-witnessed real-client
capture remain the proof gate.

## Phase 4 — Client I/O Contention Containment

**Status**: Complete for the bounded ownership slice on 2026-08-10. Deferred WMO doodad loading
now advances once per `WorldScene` frame through `WorldAssetManager`, rather than once for every
visible WMO placement. Minimap archive reads are serialized to one background reader because the
loader shares the active `IDataSource` with terrain and object streaming. This is an I/O fan-out
fix, not a claim that all model parsing or GPU work is solved.

The proof gate is a user-run real-client comparison that records deferred asset CPU time, pending
WMO doodad count, minimap pending/uploaded/failed counts, and the existing render-stage timings.
