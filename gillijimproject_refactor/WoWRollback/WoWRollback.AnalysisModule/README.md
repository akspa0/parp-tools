# WoWRollback.AnalysisModule

## Overview
Analysis and overlay generation library for WoWRollback.
- UniqueID analysis (layers, gaps, timelines)
- Terrain CSVs and per-tile overlays (objects, terrain, clusters)
- Outputs feed the interactive web viewer

## Quick Start
- Add a `ProjectReference` to `WoWRollback.AnalysisModule`.
- Call the module from your tool or see `WoWRollback.Cli` for integration examples.

Key entry points:
- `AnalysisOrchestrator` – top-level coordination
- `UniqueIdAnalyzer` – UniqueID distribution and layer detection
- `TerrainCsvGenerator` – MCNK terrain metadata
- `OverlayGenerator` – per-tile overlay JSONs

## See Also
- `../README.md` (Architecture)
- `../docs/DataVisualizationTool-Design.md`
- `../docs/Known-Limitations.md`
- `../docs/SMOKE-TEST.md`
