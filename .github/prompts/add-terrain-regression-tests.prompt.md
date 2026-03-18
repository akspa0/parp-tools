---
description: "Add or design first-party regression tests for terrain alpha masks, MCAL decoding, split ADT texture sourcing, or terrain texture-array packing in gillijimproject_refactor."
name: "Add Terrain Regression Tests"
argument-hint: "Describe the bug, file, or seam that needs tests"
agent: "agent"
---
Add first-party regression coverage for the active terrain pipeline in `gillijimproject_refactor`.

Requirements:
- Prefer test projects next to the active code under `gillijimproject_refactor/src`, not under `lib`, `next`, or archived folders.
- Prioritize the smallest high-value seam first.
- Good starting seams are `Mcal` decode variants, `StandardTerrainAdapter.ExtractAlphaMaps()`, `TerrainTileMeshBuilder` alpha-shadow packing, and `TerrainImageIo` atlas roundtrip parity.
- If a real-data integration check is needed, use the fixed development paths from the memory bank and say exactly what still requires manual validation.
- Build the changed production project and the changed test project.
- If the repo has no suitable first-party test project yet, scaffold the minimum viable one instead of writing pseudo-tests in docs.