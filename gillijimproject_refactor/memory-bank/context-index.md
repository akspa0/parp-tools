# Context Index (Concise)

Purpose: fast entry points for current active work in gillijimproject_refactor.

## Read First (Always)

1. memory-bank/activeContext.md
2. memory-bank/progress.md
3. memory-bank/data-paths.md
4. src/MdxViewer/memory-bank/activeContext.md

## Terrain Recovery Focus

- Baseline commit: 343dadfa27df08d384614737b6c5921efe6409c8
- High-risk files:
  - src/MdxViewer/Terrain/StandardTerrainAdapter.cs
  - src/MdxViewer/Terrain/TerrainRenderer.cs
  - src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs
  - src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs
- Primary strategy:
  - protect Alpha path
  - isolate 1.x/2.x and 3.x/4.x paths
  - avoid silent version guessing

## Validation Ground Rules

- Runtime target beats fixture-only checks.
- "Build passes" is not signoff.
- "Unit tests pass" is not signoff.
- Use fixed paths from memory-bank/data-paths.md unless missing.

## Quick Commands

- Viewer build:
  - dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug
- Converter core build:
  - dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug
- Doc lookup:
  - python gillijimproject_refactor/tools/doc_lookup.py build
  - python gillijimproject_refactor/tools/doc_lookup.py query "terrain alpha baseline"

## Related Docs

- docs/CONTEXT_LOOKUP.md
- .github/copilot-instructions.md
- .github/prompts/brokenasfuck-3x-support.md
