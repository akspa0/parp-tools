# Progress - WoWRollback Unified Orchestrator

## What works to reuse
- **[placements CSV probing]** Correctly finds `<map>_placements.csv` (objects) in `04_analysis/<version>/objects/` or `<adt>/analysis/csv/` with legacy fallback.
- **[overlay generation]** Objects overlays emitted at `05_viewer/overlays/<version>/<map>/combined/tile_r{row}_c{col}.json` in layered schema from either master index or CSV.
- **[viewer index]** Root `05_viewer/index.json` is built from overlays found on disk and lists tiles for the default version/map.
- **[world→pixel math]** Static transform aligned with viewer (tile NW anchor, 512×512 target).
- **[session layout]** Numbered subdirs under a timestamped session root.

## Known issues (keep documented)
- No unit tests yet for modules (DbcModule, AdtModule, ViewerModule)
- No integration test yet (Shadowfang end-to-end)
- ViewerStageRunner generates basic HTML; full interactive viewer TBD
- Need to verify real pipeline execution with actual data

## Next steps for cherry-picking
- **[copy modules]**
  - `WoWRollback.AnalysisModule/AnalysisOrchestrator.cs` (staging, CSV probing, index builder call)
  - `WoWRollback.AnalysisModule/OverlayGenerator.cs` (index + CSV overlays, master fallback, world→pixel)
  - `WoWRollback.Orchestrator/AnalysisStageRunner.cs` (ensure overlays enabled)
- **[preserve contracts]** Keep overlay schema and file paths unchanged.
- **[minimal verify]** After copy, run a small subset and confirm:
  - Overlays present in `05_viewer/overlays/.../combined/`
  - `05_viewer/index.json` has non-empty tiles
- **[document gaps]** Carry over the Known issues above into your target branch docs.
