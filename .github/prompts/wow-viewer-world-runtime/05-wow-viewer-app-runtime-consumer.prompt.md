---
description: "Implement or plan the first wow-viewer app consumer for the extracted world runtime services. Use when the earlier world-runtime slices are already real and the goal is to make WowViewer.App consume the shared runtime seams without pretending the full production viewer has already been replaced."
name: "wow-viewer World Runtime 05 wow-viewer App Runtime Consumer"
argument-hint: "Optional wow-viewer app seam or consumer milestone to prioritize"
agent: "agent"
---

Implement or plan the first `wow-viewer` app-side consumer for the extracted world runtime services.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
4. `wow-viewer/src/viewer/WowViewer.App/Program.cs`
5. `wow-viewer/README.md`
6. `.github/copilot-instructions.md`

## Goal

Create the first `wow-viewer` app consumer for extracted world-runtime services after the earlier slices are already established.

## Current Working Boundary

- `WowViewer.App` is still minimal
- this slice comes after runtime service ownership exists
- the proof target is a real app consumer seam, not production viewer parity

## Non-Negotiable Constraints

- Do not re-implement `MdxViewer` UI breadth here.
- Do not claim runtime viewer replacement from a minimal app shell.
- Keep the consumer scope narrow and centered on already-extracted runtime services.

## What The Work Must Produce

1. the exact first app consumer seam to add
2. the exact files that should own it in `wow-viewer`
3. the exact validation required
4. the exact parity gaps that remain afterward

## Deliverables

Return all items:

1. the first app consumer seam to implement
2. why it is the correct post-extraction milestone
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. what is still not proven afterward

## First Output

Start with:

1. the consumer boundary you are assuming today
2. the single first app seam you would add
3. the narrow proof that `wow-viewer` now exercises the runtime services directly
4. what you are explicitly not claiming yet