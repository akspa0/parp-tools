---
description: "Triage runtime failures where WDL heightmap spawn chooser is unavailable, disabled, or no-ops across client versions in gillijimproject_refactor."
name: "WDL Spawn Chooser Runtime Triage"
argument-hint: "Optional map name, screenshot clue, or log excerpt"
agent: "codex"
---

Use this prompt when the map spawn chooser path is reported broken on real data.

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md
3. gillijimproject_refactor/src/MdxViewer/ViewerApp_WdlPreview.cs

## Goal

Determine why spawn chooser does not function and classify the failure stage:

1. disabled before open (warm-state gate)
2. open path falls back to default load
3. chooser UI interaction does not set selection
4. spawn commit does not apply selected tile
5. cache state/error path is stale or inconsistent

## Required Inspection Files

1. gillijimproject_refactor/src/MdxViewer/ViewerApp.cs
2. gillijimproject_refactor/src/MdxViewer/ViewerApp_WdlPreview.cs
3. gillijimproject_refactor/src/MdxViewer/Terrain/WdlPreviewCacheService.cs
4. gillijimproject_refactor/src/MdxViewer/Terrain/WdlPreviewRenderer.cs
5. gillijimproject_refactor/src/MdxViewer/Terrain/WdlDataSourceResolver.cs

## Required Triage Steps

1. Verify map-row `Spawn` enablement logic and exact `WdlPreviewWarmState` transitions.
2. Trace `OpenWdlPreview(...)` and all fallback branches that call default spawn load.
3. Trace chooser selection state updates (`_selectedSpawnTile`) and click hit logic.
4. Trace `LoadSelectedPreviewMapAtSpawn()` and confirm camera/world spawn application.
5. Confirm whether warmup/cache failures are true data failures or stale state propagation.
6. Keep build evidence separate from runtime evidence.

## Deliverables

Return all items:

1. exact failing stage
2. likely root cause location (function + condition)
3. smallest safe code slice to change first
4. explicit runtime validation plan for Alpha-era and 3.x maps
5. gaps where evidence is still missing

## Validation Rules

- Do not claim fix based on build alone.
- If no automated tests were added or run, state it explicitly.
- If runtime validation was not performed, state it explicitly.
