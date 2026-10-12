# Implementation Plan: M2 and WMO Doodad Submission Batching

## Status

The CPU/state-submission slice and the first GPU-instancing implementation are present, but the
production world MDX route is held on per-instance `RenderWithTransform()` until a real-client
visual regression check proves the batch path safe. The first GPU shader caused MDX model
construction failures on the user's OpenGL runtime, so the compatibility shader is restored and
GPU-instanced MDX is explicitly disabled pending a separate shader proof. The MDX fragment path
now falls back to a minimal texture/alpha/color shader when the richer fragment program is
rejected by a driver. The viewer still requires a user-run real-scene capture and visual parity
check before any FPS claim is promoted.

## Phase 1 — Static M2 Batch Eligibility

Keep the established per-instance `RenderWithTransform()` route for all world MDX/M2 renderers
while the batching regression is investigated. The shared `BeginBatch()` / `RenderInstance()` and
GPU instance routes remain implementation work, not the active correctness path. Keep native-runtime
M2 renderers isolated until their backend state can be represented by an explicit batch key.

## Phase 2 — Opaque WMO Doodad Grouping

Group visible opaque WMO doodads by their shared `IModelRenderer`, call `BeginBatch()` once per
renderer group, and submit each placement through `RenderInstance()`. Preserve transparent
back-to-front order and unbatched particle/ribbon behavior.

## Phase 3 — GPU Instanced Opaque Submission

Use a renderer-scoped compatibility contract for opaque adapted-M2 batches only after the active
per-instance route has visual parity proof. Upload model matrices
and fade values through a dynamic instance VBO, add per-instance vertex attributes to the existing
shader, and issue one `glDrawElementsInstanced` call per compatible geoset. Keep direct Alpha MDX
on the proven shared CPU/state batch path until legacy material and vertex-state parity is proven.
Keep native-runtime state, transparent layers, particle/ribbon renderers, and unsupported
fade/material states on the fallback.
The implementation remains available as dormant infrastructure, but its MDX shader inputs are not
active in the production route. Synthetic scaling, shader compilation, visual parity, and
user-witnessed real-client capture remain the proof gate before re-enabling it.

## Phase 4 — Client I/O Contention Containment

**Status**: Complete for the bounded ownership slice on 2026-08-10. Deferred WMO doodad loading
now advances once per `WorldScene` frame through `WorldAssetManager`, rather than once for every
visible WMO placement. Minimap archive reads are serialized to one background reader because the
loader shares the active `IDataSource` with terrain and object streaming. This is an I/O fan-out
fix, not a claim that all model parsing or GPU work is solved.

The proof gate is a user-run real-client comparison that records deferred asset CPU time, pending
WMO doodad count, minimap pending/uploaded/failed counts, and the existing render-stage timings.
