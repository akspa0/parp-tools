---
description: "Implement or plan the next world-runtime slice needed for minimap generation. Use when moving tile-job planning, filter reuse, capture framing, or minimap-generation orchestration out of ViewerApp or WorldScene and into wow-viewer runtime code that both the active viewer and future CLI can consume."
name: "wow-viewer Minimap 03 Runtime-Owned Generation Extraction"
argument-hint: "Optional runtime seam, WorldScene hotspot, or shared-service gap to prioritize"
agent: "agent"
---

Implement or plan the next minimap-generation extraction seam from `WorldScene` and `ViewerApp` into `wow-viewer` runtime code.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/plans/wow_viewer_minimap_generation_plan_2026-04-08.md`
5. `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
6. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
7. `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
8. `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
9. `wow-viewer/src/core/WowViewer.Core.Runtime/World/ObjectPathFilterEntry.cs`
10. `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs`
11. `wow-viewer/README.md`

## Goal

Extract the next shared minimap-generation seam so the active viewer and future CLI can consume the same tile-job, filter, and framing logic without leaving that ownership buried in `WorldScene` and `ViewerApp` forever.

## Current Working Boundary

- the active viewer now owns the practical capture host, fog-range expansion, and filter UI
- only the object-path filter primitive has crossed into shared runtime so far for this minimap work
- the broader world-runtime split already has visibility and pass-coordinator seams in progress, but minimap-generation ownership is not yet one of those explicit services
- the user wants the minimap push to be one of the forcing functions that makes `WorldScene` smaller for real

## Non-Negotiable Constraints

- do not invent a fake graphics abstraction layer here
- do not attempt a giant renderer rewrite in one slice
- keep the extraction narrow enough that `MdxViewer` remains a working compatibility host afterward
- prefer pure contracts, planners, and reusable services that can be tested without GL when practical
- keep path-family filtering shared between viewer and CLI instead of creating separate filter models
- do not imply active-viewer runtime signoff from library-only proof

## What The Work Must Produce

1. the single minimap-generation seam to extract next
2. the exact `wow-viewer` files that should own it
3. the exact `WorldScene` or `ViewerApp` code that should remain host-only afterward
4. the narrowest tests or proofs that make the extraction real
5. the next ordered follow-up after this slice

## Deliverables

Return all items:

1. the exact runtime seam to extract
2. why it is the correct next step after the first shared filter primitive
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. what new runtime contracts or services should exist afterward

## First Output

Start with:

1. the minimap-generation boundary you are assuming today
2. the single next extraction seam
3. the narrow proof that would show `WorldScene` or `ViewerApp` got smaller for real
4. what you are explicitly not claiming yet