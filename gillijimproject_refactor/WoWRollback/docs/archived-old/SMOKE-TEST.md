# WoWRollback Orchestrator Smoke Test

This checklist validates the unified C# orchestrator using local sample data. Run it after significant refactors or before sharing builds.

## Prerequisites

- Alpha data present under `test_data/<version>/tree/DBFilesClient/`
- Lich King DBCs under `test_data/3.3.5/tree/DBFilesClient/`
- WoWDBDefs definitions in `lib/WoWDBDefs/definitions/`
- Sample map WDTs (e.g., `Shadowfang.wdt`) inside `test_data/<version>/World/Maps/<map>/`

## Command

```powershell
pwsh ./run-smoke-orchestrator.ps1 `
    -AlphaRoot ../test_data `
    -OutputRoot ./parp_out `
    -Maps Shadowfang `
    -Versions 0.5.3 `
    -Verbose
```

## Validation Checklist

- **Shared DBC Cache**: Inspect `parp_out/shared_outputs/dbc/<alias>/<build>/DBC/` for `AreaTable_*.csv` entries. Cached exports should persist between runs.
- **Shared Crosswalk Cache**: Verify `parp_out/shared_outputs/crosswalks/<alias>/<build>/compare/v2/` contains `area_patch_crosswalk.csv` and companion audit reports.
- **Session Folder**: Confirm `parp_out/session_YYYYMMDD_HHmmss/` exists with `adt/`, `analysis/`, `viewer/`, `logs/`, and `manifest.json`.
- **ADT Exports**: Under `session.../adt/<version>/World/Maps/<Map>/`, ensure LK ADT files were generated.
- **CSV Analysis**: Check `session.../adt/<version>/csv/maps/<Map>/` for `asset_fixups.csv` and related exports.
- **Manifest**: Open `session.../manifest.json` to confirm success flags and shared cache paths.
- **Logs**: Review `session.../logs/` for pipeline diagnostics (future enhancement).
- **Viewer**: Placeholder stage still reports “Viewer asset generation not yet implemented.” Deliverable directory is `session.../viewer/`.

## Troubleshooting Tips

- **Missing WDT**: Ensure `--maps` and `--versions` match data under `AlphaRoot`
- **DBD Directory Errors**: Confirm `-DbdDir` points to the definitions folder
- **Non-zero Exit Codes**: Check console output and generated logs for stack traces

Document updates should include real map/version combinations once broader datasets are integrated.
