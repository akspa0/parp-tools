# Terrain Alpha Regression

## When To Use

- Terrain texture blending changed unexpectedly.
- Alpha masks look inverted, smeared, or shifted.
- A fix touched `MCAL`, `MCLY`, `TerrainRenderer`, `TerrainTileMeshBuilder`, `StandardTerrainAdapter`, or terrain import or export.
- You need to compare behavior before and after commit `343dadfa27df08d384614737b6c5921efe6409c8`.

## Procedure

1. Read the project memory first.
2. Confirm the active pipeline before proposing fixes.
3. Diff against the baseline.
4. Separate decode bugs from render bugs.
5. Validate with real data.
6. Make testing concrete.
7. Report findings first.

## Read First

- `gillijimproject_refactor/memory-bank/activeContext.md`
- `gillijimproject_refactor/memory-bank/progress.md`
- `gillijimproject_refactor/memory-bank/data-paths.md`
- `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`
- `AGENTS.md`

## Build Commands

- Parser and format work: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
- Viewer work: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`

## Guardrails

- Do not silently replace strict alpha decode with relaxed heuristics.
- Do not claim a terrain fix is verified without a real-data check.
- Do not use library or archived tests as proof that the active viewer terrain path is safe.
