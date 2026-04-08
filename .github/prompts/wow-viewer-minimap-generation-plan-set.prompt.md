---
description: "Route staged wow-viewer minimap-generation work to the right ordered prompt. Use when planning or implementing deterministic one-PNG-per-ADT capture, object-family filtering for minimap output, wow-viewer CLI minimap commands, resumable tile jobs, or the runtime extraction needed to stop keeping minimap generation trapped in MdxViewer.WorldScene."
name: "wow-viewer Minimap Generation Plan Set"
argument-hint: "Describe the minimap capture, CLI, or WorldScene extraction seam you want to attack next"
agent: "agent"
---

Choose the right detailed prompt for the staged minimap-generation and capture-runtime work.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/plans/wow_viewer_minimap_generation_plan_2026-04-08.md`
5. `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
6. `wow-viewer/README.md`
7. `.github/copilot-instructions.md`

## Goal

Route the current request to the correct ordered prompt in `.github/prompts/wow-viewer-minimap-generation/` so minimap generation lands as narrow, validated slices instead of dissolving into another giant `WorldScene` rewrite.

## Ordered Prompts

- `wow-viewer-minimap-generation/01-deterministic-adt-capture-queue.prompt.md`
- `wow-viewer-minimap-generation/02-wow-viewer-cli-minimap-command.prompt.md`
- `wow-viewer-minimap-generation/03-runtime-owned-minimap-generation-extraction.prompt.md`

## Routing Rules

- Use `01-deterministic-adt-capture-queue.prompt.md` when the problem is one-PNG-per-ADT capture, deterministic camera framing, queue state, capture presets, tile naming, resumable viewer-side capture, or proving the current active viewer can harvest real minimap tiles instead of just taking ad hoc screenshots.
- Use `02-wow-viewer-cli-minimap-command.prompt.md` when the problem is a real `wow-viewer` CLI surface for minimap generation, shared job planning, archive-root or data-root inputs, output manifests, batch map runs, retry or resume behavior, or keeping the command thin over reusable services.
- Use `03-runtime-owned-minimap-generation-extraction.prompt.md` when the problem is moving minimap-generation ownership out of `ViewerApp` or `WorldScene`, reusing path-family filters across viewer and CLI, extracting tile-job or framing contracts into `wow-viewer`, or sequencing the minimap work with the broader world-runtime split.

## Deliverables

Return all items:

1. the best next prompt to run
2. why it is the correct slice now
3. which ordered prompt should follow after it
4. what concrete repo and file scope the next slice should include
5. what should stay out of scope for that slice
6. what proof level is realistic for that slice

## First Output

Start with:

1. the exact minimap-generation problem you think the user is actually trying to solve
2. the single best next prompt from the ordered set
3. the narrow proof that would make that slice real
4. what you are explicitly not claiming yet