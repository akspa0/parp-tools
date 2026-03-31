---
description: "Implement or plan the main wow-viewer world-runtime service split. Use when extracting explicit terrain or WMO or MDX or overlay pass services, world-pass sequencing, or a WorldPassCoordinator from WorldScene into WowViewer.Core.Runtime without jumping straight to full app migration."
name: "wow-viewer World Runtime 03 World Pass Service Extraction"
argument-hint: "Optional pass family or service boundary to prioritize"
agent: "codex"
---

Implement or plan the central world-runtime split: explicit pass services and coordinator ownership in `WowViewer.Core.Runtime`.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
4. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
5. `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs`
6. `wow-viewer/README.md`

## Goal

Split the monolithic world render sequence into explicit terrain, WMO, MDX, and overlay runtime services plus a coordinator in `wow-viewer`, while still letting `MdxViewer` own the current renderer implementations.