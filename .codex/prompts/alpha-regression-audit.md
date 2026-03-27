---
description: "Audit terrain alpha-mask regressions in gillijimproject_refactor. Use for MCAL, MCLY, terrain blend, split ADT, or shadow-mask regressions, especially after the baseline commit."
name: "Alpha Regression Audit"
argument-hint: "Optional symptom, file, tile, or commit range to focus on"
agent: "codex"
---
Audit the terrain alpha-mask pipeline in `gillijimproject_refactor`.

Requirements:
- Treat commit `343dadfa27df08d384614737b6c5921efe6409c8` as the default baseline unless I specify a different one.
- Read the relevant memory-bank files before drawing conclusions.
- Inspect the active pipeline across `Mcal.cs`, `StandardTerrainAdapter.cs`, `TerrainChunkData.cs`, `TerrainTileMeshBuilder.cs`, `TerrainRenderer.cs`, `TerrainImageIo.cs`, and `ViewerApp.cs`.
- Explicitly compare `TerrainImageIo` atlas import/export semantics against both per-chunk upload and batched tile-array packing semantics.
- Distinguish decode bugs from packing bugs, shader bugs, and UI/import-export bugs.
- Report findings first, ordered by severity, with concrete file references.
- Then provide a minimal validation plan and the smallest high-value tests that should exist.
- Do not claim the issue is resolved without real-data validation using the fixed development paths from the memory bank.