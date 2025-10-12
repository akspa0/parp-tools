# Technical Context — WoWRollback (Rewrite)

## Runtime
- **Target**: .NET 9.0, Windows x64, dotnet CLI + MSBuild.
- **Key libs**: `System.Text.Json`. Game parsing comes from existing libs in the repo (no re‑implementation here).

## Modules and classes to reuse
- **`WoWRollback.AnalysisModule.AnalysisOrchestrator.RunAnalysis()`**
  - Staging pipeline, placements CSV probing (prefer `<map>_placements.csv`).
  - Writes root `05_viewer/index.json` after overlays.
- **`WoWRollback.AnalysisModule.OverlayGenerator`**
  - `GenerateFromIndex(...)`: produces object overlays grouped by (row,col).
  - `GenerateObjectsFromPlacementsCsv(...)`: master‑index fallback, then CSV; emits layered schema.
  - `WorldToPixel(...)`: transforms world → 0..512 pixel space using constants below.
- **`WoWRollback.Orchestrator.AnalysisStageRunner`**
  - Ensures `GenerateOverlays` is enabled during analysis.

## Overlay math and paths
- **Constants**: `TILE_SIZE = 533.33333f`, `MAP_HALF_SIZE = 32 * TILE_SIZE`.
- **Paths**: `05_viewer/overlays/<version>/<map>/combined/tile_r{row}_c{col}.json`.
- **Schema**: `layers[0].kinds[{M2,WMO}].points[].{world{x,y,z}, pixel{x,y}, assetPath, fileName, uniqueId}`.

## Index building
- **Source of truth**: Scan `05_viewer/overlays/<version>/<map>/combined/` for `tile_r*_c*.json`.
- **Output**: Root `05_viewer/index.json` with `{comparisonKey, versions, defaultVersion, maps[{map, tiles[{row,col,versions}]}]}`.

## File discovery rules
- **Placements CSV probe order**:
  1) `04_analysis/<version>/objects/<map>_placements.csv`
  2) `<adtOutputDir>/analysis/csv/<map>_placements.csv`
  3) `<adtOutputDir>/analysis/csv/placements.csv` (legacy)
- **Overlays source order**: master index `<map>_master_index.json` → placements CSV.

## Known gaps (document reality)
- **Alpha Parsing (Critical)**: The `AlphaWdtAnalyzer` currently has placeholder implementations:
  - Alpha MCNK chunk structure analysis
  - UniqueID field locations within Alpha format
  - WDT tile presence detection
- **Required Research**:
  - Study existing `AlphaWDTAnalysisTool` parsing logic
  - Verify Alpha format compatibility with current parsers
  - Map Alpha chunk structures to UniqueID locations

## Output contract (stable)
