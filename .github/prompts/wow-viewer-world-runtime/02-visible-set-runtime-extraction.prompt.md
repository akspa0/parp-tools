---
description: "Implement or plan the next wow-viewer world-runtime slice after asset-miss stabilization. Use when extracting visible WMO or MDX or taxi-actor collection, render-frame scratch ownership, culling buckets, or visible-set contracts out of WorldScene and into WowViewer.Core.Runtime."
name: "wow-viewer World Runtime 02 Visible Set Runtime Extraction"
argument-hint: "Optional visible-set seam, culling bucket, or scratch-ownership hotspot to prioritize"
agent: "agent"
---

Implement or plan the next `WorldScene` split slice: move visible-set logic and frame scratch ownership into `wow-viewer` runtime code.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
4. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
5. `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs`
6. `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderOptimizationAdvisor.cs`
7. `wow-viewer/README.md`

## Goal

Extract visible WMO and MDX and taxi visible-set collection plus frame scratch ownership into `WowViewer.Core.Runtime` without yet moving full render-pass ownership.

## Current Working Boundary

- `WorldRenderFrameStats` and the optimization advisor already live in `WowViewer.Core.Runtime`.
- `WorldScene` still owns `VisibleWmoInstance`, `VisibleMdxInstance`, render-frame scratch lists, and `CollectVisibleWmoInstances` or `CollectVisibleMdxInstances`.
- this slice should move collection and scratch ownership first, not all draw submission.

## Non-Negotiable Constraints

- Do not introduce a fake graphics abstraction layer here.
- Do not move terrain or WMO or MDX rendering itself yet.
- Keep the slice narrow enough that `WorldScene` still compiles as the host after the extraction.
- Prefer runtime-owned contracts and pure helpers that can be tested without GL.

## What The Work Must Produce

1. the exact visible-set and scratch seam to extract
2. the exact `WowViewer.Core.Runtime` files that should own it
3. the exact `WorldScene` code that should remain host-only afterward
4. the narrowest tests or proofs that make the extraction real
5. the next pass-service follow-up for slice 03

## Deliverables

Return all items:

1. the exact visible-set seam to extract
2. why it is the correct next step after slice 01
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. what new runtime contracts or services should exist afterward

## First Output

Start with:

1. the visible-set boundary you are assuming today
2. the single next extraction seam
3. the narrow proof that would show `WorldScene` got smaller for real
4. what you are explicitly not claiming yet