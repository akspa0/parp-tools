# WoWRollback.Orchestrator

## Overview
Runs the end-to-end Alpha→LK pipeline in one command. Coordinates DBC dumps/crosswalks, ADT conversion, analysis (WIP), and viewer generation.

## Quick Start
```powershell
dotnet run --project . -- ^
  --maps Azeroth,Kalimdor ^
  --versions 0.5.3,0.5.5 ^
  --alpha-root ..\test_data ^
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient ^
  --serve --port 8080
```

## Notes
- Reads loose Alpha files from `{alpha-root}/{version}/tree/World/Maps/...`.
- Uses LK AreaTable/Map DBCs from `--lk-dbc-dir`.
- Viewer served via `ViewerModule` when `--serve` is provided.

## See Also
- Detailed pipeline in root `README.md` (Orchestrator Command section).
- `docs/SMOKE-TEST.md` for validation.
- `WoWRollback.Cli` for direct Alpha↔LK commands.
