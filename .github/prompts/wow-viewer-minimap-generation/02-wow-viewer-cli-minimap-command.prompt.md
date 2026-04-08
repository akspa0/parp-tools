---
description: "Implement or plan the wow-viewer CLI minimap surface after deterministic capture rules exist. Use when adding minimap commands, shared job planning, batch map runs, manifest output, resume or retry behavior, or deciding how the CLI should consume shared minimap-generation services instead of reusing viewer-only code paths."
name: "wow-viewer Minimap 02 CLI Command"
argument-hint: "Optional CLI surface, command verb, or batch-job concern to prioritize"
agent: "agent"
---

Implement or plan the `wow-viewer` CLI surface for minimap generation.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_minimap_generation_plan_2026-04-08.md`
4. `wow-viewer/README.md`
5. `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs`
6. `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs`
7. `wow-viewer/src/core/WowViewer.Core.IO/Files/MinimapService.cs`
8. `wow-viewer/src/core/WowViewer.Core.Runtime/World/ObjectPathFilterEntry.cs`

## Goal

Add a real `wow-viewer` command surface for minimap generation that can batch over map tiles using shared services instead of leaving the workflow stranded in `MdxViewer` automation.

## Current Working Boundary

- the active viewer is still the only place that can currently prove real-scene capture behavior
- path-family filter primitives already have a shared runtime contract
- shared minimap helpers already exist in `Core.IO`, but there is no job planner or CLI verb for tile generation yet
- the command surface should be thin over shared services, not a second parser or renderer ownership fork

## Non-Negotiable Constraints

- do not hard-wire the CLI to one viewer-only state file format
- keep archive-root or data-root and output-root arguments explicit
- keep tile naming, manifest output, and resume behavior deterministic
- prefer reusable shared service seams in `wow-viewer` over command-local logic
- do not claim full headless renderer closure if the slice still relies on active-viewer-hosted capture behavior or partial compatibility bridges

## What The Work Must Produce

1. the exact command or subcommand shape
2. the exact shared services or contracts the CLI must depend on
3. the exact files to change
4. the narrowest real-data proof that the command is not fake
5. what should remain out of scope until the runtime-extraction slice

## Deliverables

Return all items:

1. the exact CLI surface to add
2. why it is the correct next step after deterministic capture rules exist
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. what shared services or manifests should exist afterward

## First Output

Start with:

1. the command boundary you are assuming today
2. the single next CLI seam to implement
3. the narrow proof that would show the CLI surface is real
4. what you are explicitly not claiming yet