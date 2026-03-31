---
description: "Implement or plan the M2 runtime slice for scene submission and batching. Use when render-entry family classification, state-sort ordering, doodad/particle/ribbon path splits, or M2 runtime knobs such as clip planes or additive sort need a wow-viewer-owned coordinator."
name: "wow-viewer M2 Runtime 04 Scene Submission And Batching"
argument-hint: "Optional render family, batching hotspot, native comparator seam, or runtime flag to prioritize"
agent: "codex"
---

Implement or plan the M2 runtime slice that owns classified scene submission and batching policy in wow-viewer.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
4. `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
5. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
6. `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
7. `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs`
8. `wow-viewer/README.md`
9. `AGENTS.md`

## Goal

Create an explicit wow-viewer-owned M2 scene/submission model that can express:

- render-entry family classification
- family-specific submission paths for doodads, particles, ribbons, and special callbacks
- state-aware sort/batch grouping
- explicit runtime flags/policies such as clip planes, z-fill, doodad batching, particle batching, and additive particle sorting

## Non-Negotiable Constraints

- Do not turn this slice into the full world-runtime extraction; use the world-runtime prompt set if the real problem is broader `WorldScene` ownership.
- Do not hide every M2 draw behind one generic model renderer abstraction.
- Keep batch boundaries and runtime knobs explicit.
- Do not claim full render parity from one coordinator/batcher seam.
- Keep family-specific behavior concrete instead of introducing a speculative backend abstraction.

## What The Work Must Produce

1. the exact render-entry family/coordinator seam to add
2. the exact files that should own batching and runtime policy in wow-viewer
3. the narrowest proof that classified submission is real
4. the temporary host/compatibility code that can remain in `MdxViewer`
5. the follow-on boundary for consumer cutover in slice 05

## Deliverables

Return all items:

1. the exact scene-submission/batching seam to implement
2. why it is the correct fourth slice
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. what still remains consumer-owned afterward

## First Output

Start with:

1. the scene-submission boundary you are assuming now
2. the single first coordinator/batching seam you would land
3. the narrowest proof that would make that seam real
4. what you are explicitly not claiming yet