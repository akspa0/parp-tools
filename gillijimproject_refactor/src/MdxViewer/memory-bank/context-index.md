# MdxViewer Context Index (Concise)

Purpose: fast routing for viewer-side debugging and terrain recovery.

## Read First

1. src/MdxViewer/memory-bank/activeContext.md
2. src/MdxViewer/memory-bank/progress.md
3. src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md
4. ../../memory-bank/context-index.md

## Current Critical Themes

- Terrain alpha/shadow regressions around post-baseline changes
- UI complexity drift reducing debugging clarity
- Need for explicit version-family routing before terrain decode/render path selection

## Implementation Prompts

- src/MdxViewer/memory-bank/implementation_prompts.md — paste-ready Copilot
  prompts for TXAN animation, animation UI, MH2O liquid, ribbon emitters,
  detail doodads, GPU instancing, async textures, debug overlays

## Viewer High-Risk Files

- src/MdxViewer/Terrain/TerrainRenderer.cs
- src/MdxViewer/Terrain/StandardTerrainAdapter.cs
- src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs
- src/MdxViewer/ViewerApp.cs

## Recovery Rules

- Keep Alpha behavior stable and regression-checked.
- Do not combine era-specific terrain semantics in one ambiguous path.
- Prefer isolated path logic over heuristics that guess client family.

## Useful Commands

- Build viewer:
  - dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug
- Run viewer:
  - dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj
- Query docs quickly:
  - python i:/parp/parp-tools/gillijimproject_refactor/tools/doc_lookup.py query "alpha debug overlay"
