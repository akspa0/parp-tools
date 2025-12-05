# Tech Context

## Languages & Frameworks
- C# (.NET 9)
- Console app CLI

## Key Projects
- `AlphaWdtInspector/` (standalone diagnostics; no WoWRollback dependencies)
- `AlphaLkToAlphaStandalone/` – dedicated LK→Alpha converter + Alpha↔LK roundtrip validator:
  - `convert` command: LK ADT → stub Alpha WDT (occupancy-only), supports filesystem trees and MPQ 3.3.5 client roots.
  - `roundtrip` command: Alpha WDT → LK ADTs (via `AdtExportPipeline` + DBCTool.V2 crosswalk CSVs) → LK→Alpha stub + AreaID CSV diagnostics.

## Data & Path Auto-detection
- `roundtrip` infers:
  - Repo root by scanning upward for `test_data/` and `DBCTool.V2/`.
  - `DBD` dir from `lib/WoWDBDefs/definitions`.
  - Alpha DBC root from `test_data/<alias>/tree/DBFilesClient`.
  - LK DBC root from `test_data/3.3.5/tree/DBFilesClient`.
  - DBCTool.V2 outputs from `DBCTool.V2/dbctool_outputs`.

## External Libraries
- Uses `WoWRollback.Core` MPQ/StormLib stack to support a pristine 3.3.5 install as LK input.
- Uses `AlphaWdtAnalyzer.Core.AdtExportPipeline` and DBCTool.V2 loaders/crosswalks for Alpha→LK export and AreaID mapping.

## Test Utilities
- Golden-file checks for CSVs and `tile-diff` outputs on representative tiles.
- Example constants used throughout:
  - `TILESIZE = 533.33333f`
  - `WORLD_BASE = 32 * TILESIZE`
- Packaging: single-file publish with simple run scripts; outputs in `out/<session_ts>/...`.
