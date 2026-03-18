---
name: terrain-alpha-regression
description: 'Use when debugging terrain alpha-mask regressions, MCAL or MCLY decoding issues, split ADT texture problems, terrain blend artifacts, shadow-mask mismatches, or post-baseline regressions in gillijimproject_refactor. Includes a concrete investigation checklist, file map, and test recommendations.'
argument-hint: 'Describe the symptom, file, map, tile, or commit range to investigate'
user-invocable: true
---

# Terrain Alpha Regression

## When To Use

- Terrain texture blending changed unexpectedly.
- Alpha masks look inverted, smeared, or shifted.
- A fix touched `MCAL`, `MCLY`, `TerrainRenderer`, `TerrainTileMeshBuilder`, `StandardTerrainAdapter`, or terrain import/export.
- You need to compare behavior before and after commit `343dadfa27df08d384614737b6c5921efe6409c8`.

## Procedure

1. Read the project memory first.
   Read `gillijimproject_refactor/memory-bank/activeContext.md`, `gillijimproject_refactor/memory-bank/progress.md`, `gillijimproject_refactor/memory-bank/data-paths.md`, and `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`.

2. Confirm the active pipeline before proposing fixes.
   Inspect `Mcal.cs`, `StandardTerrainAdapter.cs`, `TerrainChunkData.cs`, `TerrainTileMeshBuilder.cs`, `TerrainRenderer.cs`, `TerrainImageIo.cs`, and `ViewerApp.cs`.

3. Diff against the baseline.
   Compare the current file set against commit `343dadfa27df08d384614737b6c5921efe6409c8` before changing decode rules or alpha packing.

4. Separate decode bugs from render bugs.
   Determine whether the bad output starts in parser output, chunk storage, mesh packing, or shader/render usage.

5. Validate with real data.
   Use the fixed paths in the memory bank. Do not ask the user for alternate paths unless the fixed paths are missing.

6. Make testing concrete.
   If you recommend tests, target the smallest high-value seam first instead of adding broad placeholder coverage. Use the checklist in [alpha-regression-checklist.md](./references/alpha-regression-checklist.md).

7. Report findings first.
   Lead with bugs, risky heuristics, missing validation, and missing tests. Only then summarize changes or next steps.

## Build Commands

- Parser and format work: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
- Viewer work: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`

## Guardrails

- Do not silently replace strict alpha decode with relaxed heuristics.
- Do not claim a terrain fix is verified without a real-data check.
- Do not use library or archived tests as proof that the active viewer terrain path is safe.