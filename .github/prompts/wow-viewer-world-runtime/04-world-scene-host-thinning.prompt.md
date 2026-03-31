---
description: "Implement or plan the WorldScene host-thinning slice after the pass services exist. Use when reducing WorldScene to a thin host or adapter over wow-viewer runtime services without yet claiming the full world renderer has moved into wow-viewer."
name: "wow-viewer World Runtime 04 WorldScene Host Thinning"
argument-hint: "Optional host responsibility or adapter seam to prioritize"
agent: "agent"
---

Implement or plan the `WorldScene` host-thinning slice after the main runtime services are already real.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
4. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
5. `wow-viewer/README.md`

## Goal

Reduce `WorldScene` to scene-host and adapter responsibilities while keeping the live viewer stable and honest about what is still app-owned.

## Current Working Boundary

- telemetry and pass ownership should already have moved into `wow-viewer` runtime services
- `WorldScene` should still own some host-only concerns such as UI-facing toggles, direct GL-facing object references, or viewer integration glue
- this slice is about deleting orchestration weight, not promising a finished wow-viewer viewer app

## Non-Negotiable Constraints

- Do not perform a giant rename-only move that leaves the same behavior tangled in a different file.
- Keep all remaining host responsibilities explicit.
- Do not claim the renderer is fully migrated until the runtime services are the clear design owner.

## What The Work Must Produce

1. the exact `WorldScene` responsibilities to keep
2. the exact responsibilities to delete or delegate
3. the exact adapter or host contracts that remain in `MdxViewer`
4. the proof that `WorldScene` is materially smaller and less central
5. the hand-off boundary for slice 05

## Deliverables

Return all items:

1. the exact host-thinning seam to land
2. why it is the right next step
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. what runtime-ownership claim is justified afterward

## First Output

Start with:

1. the host boundary you are assuming today
2. the single next delegation seam
3. the narrow proof that `WorldScene` stopped being the design owner for that behavior
4. what you are explicitly not claiming yet