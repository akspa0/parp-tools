---
description: "Use when editing terrain alpha masks, MCAL or MCLY parsing, split ADT texture sourcing, terrain blending, shadow masks, or alpha import/export in gillijimproject_refactor. Covers the current alpha pipeline, regression hotspots, and validation requirements."
name: "Terrain Alpha Guardrails"
applyTo: "gillijimproject_refactor/src/MdxViewer/Terrain/**/*.cs, gillijimproject_refactor/src/MdxViewer/Export/Terrain*.cs, gillijimproject_refactor/src/MdxViewer/ViewerApp.cs, gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/*.cs, gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/AlphaMapService.cs"
---
# Terrain Alpha Guardrails

- Treat commit `343dadfa27df08d384614737b6c5921efe6409c8` as the baseline for regression analysis when terrain blending changed after that point.
- Read `gillijimproject_refactor/memory-bank/data-paths.md`, `gillijimproject_refactor/memory-bank/activeContext.md`, and `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md` before changing this pipeline.

## Current Pipeline

- `Mcal.GetAlphaMapForLayer()` is the strict LK decoder for flagged layers. It distinguishes compressed alpha, big alpha, and 4-bit alpha, and only applies the edge fix to the 4-bit path.
- `StandardTerrainAdapter.ExtractAlphaMaps()` is still the active viewer-side decode path. It currently uses a strict LK path when `MCLY` flags indicate alpha and falls back to sequential 4-bit decode when no LK flags are present.
- `AlphaMapService.ReadAlpha()` exists as a reusable implementation, but it is not the current source of truth for the active viewer terrain path. Do not assume it is already wired into `StandardTerrainAdapter`.
- `TerrainChunkData.AlphaMaps` stores expanded `64x64` byte maps per overlay layer.
- `TerrainTileMeshBuilder` packs alpha and shadow into RGBA texture-array slices. `TerrainRenderer` then binds and blends those textures.
- For split ADTs, texture layers and alpha data can come from `*_tex0.adt`, not the root ADT. Do not read only the root file and assume the layer ordering still matches.

## Non-Negotiable Rules

- Do not add relaxed alpha heuristics or “recover from bad flags” behavior without a concrete failing sample and a regression check for both Alpha and LK data.
- Preserve `DoNotFixAlphaMap` semantics. The edge fix applies only to the 4-bit path and should not be blindly applied to big alpha or compressed alpha.
- Treat MPHD `0x4` and `0x80` as big-alpha indicators.
- Treat `MCLY 0x200` as compressed alpha only in the LK path. Do not reuse that assumption for older formats without evidence.
- Keep alpha-layer indexing stable: layer `0` is the base texture, layers `1..3` are overlay alpha maps.
- When editing alpha behavior, inspect both decode and render stages. Many regressions come from changing only one side.

## Minimum Validation

- Build `WoWMapConverter.Core` after parser changes.
- Build `MdxViewer.sln` after viewer or terrain changes.
- Diff the changed files against baseline commit `343dadfa27df08d384614737b6c5921efe6409c8` when the change affects alpha rendering behavior.
- Validate with real data from `test_data/development/World/Maps/development` or the other fixed paths in the memory bank.
- If import/export or debug visualization changed, verify the corresponding `ViewerApp` menu path and `TerrainImageIo` output still match the runtime behavior.