# WoWRollback — Project Brief (Rewrite)

## Purpose
Provide a reliable analysis + viewer pipeline that emits actionable overlays and CSV diagnostics for per‑tile inspection and cross‑version cherry‑picking.

## Concrete Outcomes
- Objects overlays per tile at `05_viewer/overlays/<version>/<map>/combined/tile_r{row}_c{col}.json`.
- Root `05_viewer/index.json` with `{ comparisonKey, versions, defaultVersion, maps[{ map, tiles[{ row, col, versions }] }] }` based on actual overlays present.
- UniqueID CSVs and layers JSON produced from the richest available source (master index preferred, then placements CSV).

## Inputs
- Converted LK ADTs directory (supports analyze‑only).
- Optional `analysis/index.json` with placements.
- Generated placements CSV named `<map>_placements.csv` (found under `04_analysis/<version>/objects/` or `<adt>/analysis/csv/`).
- Master index `<map>_master_index.json` under `04_analysis/<version>/master/`.

## Outputs
- `04_analysis/<version>/objects/<map>_placements.csv`
- `04_analysis/<version>/master/<map>_master_index.json`
- `04_analysis/<version>/uniqueids/*`
- `05_viewer/overlays/<version>/<map>/combined/tile_r{row}_c{col}.json`
- `05_viewer/index.json`

## High‑Value Modules to Cherry‑Pick
- `AnalysisOrchestrator.RunAnalysis()` staging, including correct probing for `<map>_placements.csv` and session paths.
- `OverlayGenerator.GenerateFromIndex()` and `GenerateObjectsFromPlacementsCsv()` emitting layered schema and grouping by (row,col).
- Master‑index fallback for overlays when CSV is absent.
- Minimal viewer index builder driven by overlays found on disk.
- World→pixel conversion constants and method used by overlay generation.
