---
description: "Recover 4.0.0.11927 terrain texturing in the active viewer path using the latest wow.exe Ghidra findings, with documentation and memory-bank updates as part of the change."
name: "WoW 4.0 Terrain Blend Recovery"
argument-hint: "Optional symptom, tile, file, or 4.0 terrain behavior to prioritize"
agent: "codex"
---

Recover 4.0.0.11927 terrain texturing in `gillijimproject_refactor`.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
5. `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`
6. `gillijimproject_refactor/docs/archive/WoW_400_ADT_Analysis.md`
7. `gillijimproject_refactor/documentation/wow-400-terrain-blend-wow-exe-guide.md`
8. `gillijimproject_refactor/docs/ADT_WDT_Format_Specification.md`

## Goal

Make concrete progress on the active viewer's 4.0.0 terrain blending so it matches the wow.exe behavior more closely on the fixed `development` dataset.

## Current Reverse-Engineered Model

- 4.0.0 is still pre-split on disk, but runtime terrain texturing is not just local MCAL decoding.
- `CMapChunk_UnpackAlphaBits` handles local 4-bit / 8-bit / compressed decode.
- In 8-bit mode, layers without direct alpha payload can be synthesized as residual coverage from sibling layers.
- `CMapChunk_UnpackChunkAlphaSet` stitches the current chunk with three linked neighbors and matches neighbor layers by texture id.
- `CMapChunk_RefreshBlendTextures` then creates a `TerrainBlend` resource from those decoded results.

## Non-Negotiable Constraints

- Do not revert back to broad relaxed alpha heuristics.
- Do not claim 4.0 terrain is fixed from build-only validation.
- Do not treat old archive notes that say "4.0 is identical to 3.3.5" as authoritative.
- Keep the active implementation minimal and viewer-compatible before attempting a full `TerrainBlend` resource port.

## Required Implementation Order

1. Confirm whether the current bug is missing local decode, missing residual synthesis, or missing edge stitching.
2. Prefer the smallest active-viewer change that maps directly to the wow.exe findings.
3. Implement residual-alpha synthesis for 8-bit 4.0 layers before broader renderer surgery.
4. Implement texture-id-based neighbor edge stitching as the next step.
5. Update documentation and memory-bank files in the same change set so future sessions do not lose the model.

## Required Deliverables

Return all items:

1. exact behavior implemented
2. files changed and why
3. documentation and memory-bank files updated
4. build status
5. automated-test status
6. real-data runtime-validation status
7. remaining gaps versus the wow.exe model

## First Output

Start with:

1. the current 4.0 terrain texturing gap in the active viewer
2. the minimal next implementation slice worth landing
3. what docs and memory-bank files will be updated along with the code