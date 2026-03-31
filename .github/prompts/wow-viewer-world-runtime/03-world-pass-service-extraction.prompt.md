---
description: "Implement or plan the main wow-viewer world-runtime service split. Use when extracting explicit terrain or WMO or MDX or overlay pass services, world-pass sequencing, or a WorldPassCoordinator from WorldScene into WowViewer.Core.Runtime without jumping straight to full app migration."
name: "wow-viewer World Runtime 03 World Pass Service Extraction"
argument-hint: "Optional pass family or service boundary to prioritize"
agent: "agent"
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

## Current Working Boundary

- telemetry exists in `Core.Runtime`
- visible-set extraction should already be landed or treated as an immediate precursor
- `WorldScene.Render(...)` still sequences lighting, sky, WDL, terrain, WMO, MDX, liquid, and overlay work inline

## Non-Negotiable Constraints

- Do not invent a broad rendering backend abstraction just to make the split look cleaner.
- Keep the services concrete and scoped to the current pass families.
- Keep `MdxViewer` renderers and terrain managers as host-side dependencies until a later app cutover.
- Avoid one-shot full-file moves that make rollback impossible.

## What The Work Must Produce

1. the exact pass services to create first
2. the exact coordinator or orchestration seam to add in `WowViewer.Core.Runtime`
3. the exact `WorldScene` code to retain temporarily as the host adapter
4. the proof that the pass sequence is runtime-owned instead of still being hidden inline
5. the follow-on boundary for slice 04

## Deliverables

Return all items:

1. the first pass-service slice to implement
2. why that is the right first decomposition seam
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. what still remains app-owned after this slice

## First Output

Start with:

1. the pass boundary you are assuming now
2. the single first service extraction you would land
3. the narrow proof that the coordinator is becoming real
4. what you are explicitly not claiming yet