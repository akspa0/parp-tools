---
description: "Route library-first wow-viewer viewer cutover work to the correct ordered prompt. Use when choosing the next implementation slice for fast source access, rendering library extraction, camera/minimap parity, terrain adapter extraction, asset streaming, terrain GPU rendering, world-object rendering, WorldScene service split, or thin viewer-host cutover."
name: "wow-viewer GPU Viewer Plan Set"
argument-hint: "Describe the next viewer migration problem such as map-open latency, GPU ownership, terrain rendering, world-object rendering, or thin-host cutover"
agent: "agent"
---

# wow-viewer GPU Viewer Plan Set

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. gillijimproject_refactor/plans/wow_viewer_mdxviewer_cutaway_reset_plan_2026-04-24.md
4. gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md
5. gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md
6. gillijimproject_refactor/plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md
7. wow-viewer/README.md
8. .github/copilot-instructions.md

## Goal

Choose the correct next prompt so migration stays library-first, GPU-first, and performance-first.

- Do not route implementation ownership back into gillijimproject_refactor/MdxViewer.
- Do not keep deepening wow-viewer/src/viewer/WowViewer.App as the long-term architecture.
- Keep wow-viewer libraries as canonical owners.

## Ordered Prompts

1. wow-viewer-gpu-viewer/01-fast-source-path.prompt.md
2. wow-viewer-gpu-viewer/02-rendering-library-bootstrap.prompt.md
3. wow-viewer-gpu-viewer/03-camera-and-minimap-foundations.prompt.md
4. wow-viewer-gpu-viewer/04-terrain-adapter-extraction.prompt.md
5. wow-viewer-gpu-viewer/05-asset-streaming-and-residency.prompt.md
6. wow-viewer-gpu-viewer/06-terrain-gpu-renderer-port.prompt.md
7. wow-viewer-gpu-viewer/07-world-object-renderer-port.prompt.md
8. wow-viewer-gpu-viewer/08-world-scene-service-split.prompt.md
9. wow-viewer-gpu-viewer/09-thin-viewer-host-cutover.prompt.md

## Routing Rules

- Use 01 when the pain is world-open latency, archive bootstrap cost, repeated MPQ probing, or shell-owned source access.
- Use 02 when GPU renderers are still app-local and need a dedicated library home.
- Use 03 when the next slice is camera behavior or minimap coordinate parity.
- Use 04 when the next slice is ADT/WDT/split-ADT tile ownership and terrain adapter behavior.
- Use 05 when the next slice is asset queueing, residency, cache reuse, negative lookup suppression, or background-ready loading.
- Use 06 when the next slice is terrain mesh generation and terrain texturing on GPU.
- Use 07 when the next slice is WMO/MDX/M2 world-object rendering on GPU.
- Use 08 when the next slice is splitting WorldScene into reusable library services.
- Use 09 when libraries are ready and the next step is building the new thin viewer host.

## Deliverables

Return all items:

1. the single best next prompt to run
2. why it is correct now
3. exact file/project scope to include
4. exact out-of-scope boundaries
5. realistic proof level for this slice
6. which prompt should run immediately after

## First Output

Start with:

1. the exact viewer migration problem you think is being solved now
2. the single best next prompt from the ordered set
3. the smallest implementation scope that makes this slice real
4. the narrowest proof that validates it
5. what you are explicitly not claiming yet
