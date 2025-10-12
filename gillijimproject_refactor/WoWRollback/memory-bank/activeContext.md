# Active Context - WoWRollback Unified Orchestrator Refactor

## What to cherry-pick
- **[overlay schema]** Layered JSON output: `layers[0].kinds[{M2,WMO}].points[].{world,pixel,assetPath,fileName,uniqueId}`.
- **[overlay paths]** `05_viewer/overlays/<version>/<map>/combined/tile_r{row}_c{col}.json`.
- **[viewer index]** Root `05_viewer/index.json` derived from overlays present on disk with `{comparisonKey, versions, defaultVersion, maps[{map, tiles[{row,col,versions}]}]}`.
- **[placements CSV rule]** Prefer `<map>_placements.csv`; probe in `04_analysis/<version>/objects/` then `<adt>/analysis/csv/`; legacy `placements.csv` last.
- **[master fallback]** Generate object overlays from `04_analysis/<version>/master/<map>_master_index.json` if CSV is missing; otherwise from CSV.
- **[world→pixel]** Use constants `TILE_SIZE=533.33333`, `MAP_HALF_SIZE=32*TILE_SIZE`; NW anchor per tile; scale to 0..512.
- **[session layout]** `01_dbcs/ … 05_viewer/`, single session root.

## Known working modules (by file/class)
- **`AnalysisOrchestrator.RunAnalysis()`**: stages, CSV probing, index builder writing `05_viewer/index.json`.
- **`OverlayGenerator.GenerateFromIndex()`**: index-based object overlays.
- **`OverlayGenerator.GenerateObjectsFromPlacementsCsv()`**: CSV + master fallback, layered schema emission.
- **`AnalysisStageRunner`**: ensures overlays are enabled during analysis.

## What to avoid
## Open issues (document reality)
- **[terrain/shadow]** Analyze-only on split ADTs yields empty terrain/shadow overlays/CSVs — do not depend on them.
- **[unique IDs]** When `analysis/index.json` exists but has zero placements, run UniqueID analysis from `<map>_placements.csv` for real results.

## Non-negotiables
- **[schema + paths]** Keep the overlay JSON schema and file layout unchanged to maximize portability.
- **[probe order]** `<map>_placements.csv` → master index → legacy names; never the reverse.
- **[skip-if-missing]** Non-fatal missing inputs; emit what’s available.
