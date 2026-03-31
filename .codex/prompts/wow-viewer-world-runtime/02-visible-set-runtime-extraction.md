---
description: "Implement or plan the next wow-viewer world-runtime slice after asset-miss stabilization. Use when extracting visible WMO or MDX or taxi-actor collection, render-frame scratch ownership, culling buckets, or visible-set contracts out of WorldScene and into WowViewer.Core.Runtime."
name: "wow-viewer World Runtime 02 Visible Set Runtime Extraction"
argument-hint: "Optional visible-set seam, culling bucket, or scratch-ownership hotspot to prioritize"
agent: "codex"
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