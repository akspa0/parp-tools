---
description: "Reduce per-frame CPU work in WorldScene by tightening culling, broad-phase rejection, and draw submission without breaking debug workflows."
name: "Scene Culling Batching Performance Plan"
argument-hint: "Optional camera scenario, object family, culling symptom, or map to prioritize"
agent: "codex"
---

Implement the scene culling and batching performance slice in `gillijimproject_refactor/src/MdxViewer` with lower per-frame CPU work and more stable draw submission as the runtime goal.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
5. `gillijimproject_refactor/documentation/wow-400-engine-performance-recovery-guide.md`
6. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
7. `gillijimproject_refactor/src/MdxViewer/Rendering/FrustumCuller.cs`
8. `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs`
9. `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`

## Goal

Reduce broad-phase visibility cost and unnecessary draw submission work in the main scene without regressing selection, PM4 debugging, or recent world-object fixes.

## Why This Slice Exists

- The 4.0.0 client exposes `CWorldOccluder`, `CWorldAntiOccluder`, and `CWorldOcclusionVolume`, which confirms that visibility is treated as a dedicated subsystem.
- The active viewer already has frustum and distance culling, but much of the work still happens in large per-frame loops with repeated AABB checks and local state setup.
- The next useful slice is not “build a full occlusion engine.” It is to make the current broad-phase cheaper and the current submission path more stable.

## Scope

- In scope:
  - `WorldScene` broad-phase rejection and submission order
  - culling and batching behavior in terrain/WMO/MDX/PM4 passes
  - counters or instrumentation that make the effect measurable
  - preserving debug overlays as opt-in cost
- Out of scope unless direct evidence forces it:
  - full occlusion-volume implementation
  - terrain decode changes
  - unrelated UI rewrites

## Non-Negotiable Constraints

- Do not break PM4 picking, PM4 bounds, or other debug workflows while optimizing.
- Do not hide visible regressions behind more aggressive cull distances.
- Do not start with a speculative occlusion system when broad-phase and batching waste still exist.
- Keep the change minimal and measurable.
- Update memory-bank notes if culling behavior, counters, or known tradeoffs materially change.

## Required Implementation Order

1. Identify the hottest per-frame scene loops and the broad-phase checks they perform.
2. Confirm whether the biggest cost is repeated culling, sorting, or draw submission fragmentation.
3. Land the smallest change that removes one confirmed source of repeated work.
4. Preserve existing correctness-sensitive exceptions such as nearby no-cull handling and PM4 debug visibility.
5. Add or update counters so the impact is visible.
6. Update memory-bank notes with the exact behavior change and remaining limitations.

## Investigation Checklist

- Verify how often the same objects are re-tested in multiple passes.
- Verify whether expensive bounds transforms or sort keys can be reused per frame.
- Verify which debug overlays are still paying cost even when disabled.
- Check whether any current batching opportunity exists before deeper culling work.
- Prefer one confirmed hot path over a broad rewrite.

## Validation Rules

- Build the changed viewer solution.
- If runtime validation on real data is not run, say so explicitly.
- If automated tests are not added or run, say so explicitly.
- Do not claim “occlusion” gains unless the change actually implements more than current frustum/distance rejection.

## Deliverables

Return all items:

1. exact culling/batching behavior changed
2. files changed and why
3. what repeated work was removed versus what remains
4. build status
5. automated-test status
6. runtime-validation status
7. memory-bank updates made

## First Output

Start with:

1. the current per-frame hot loop with the highest likely performance value
2. what evidence from the engine and current code supports that choice
3. what files will be changed first
4. what visibility behavior must not regress during the pass