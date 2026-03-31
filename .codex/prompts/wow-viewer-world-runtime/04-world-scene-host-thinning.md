---
description: "Implement or plan the WorldScene host-thinning slice after the pass services exist. Use when reducing WorldScene to a thin host or adapter over wow-viewer runtime services without yet claiming the full world renderer has moved into wow-viewer."
name: "wow-viewer World Runtime 04 WorldScene Host Thinning"
argument-hint: "Optional host responsibility or adapter seam to prioritize"
agent: "codex"
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